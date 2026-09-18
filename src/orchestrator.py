"""Pipeline orchestration with revision loop for the AI newsletter agent."""

import logging

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import RunUsage, UsageLimits

from src.agents import AgentBundle, default_bundle
from src.config import Settings, get_settings
from src.email_report import EmailResult, EmailStatus, send_report_email
from src.output import write_report
from src.prompts import get_date_range
from src.schemas import (
    DimensionScore,
    EvalResult,
    ReportOutput,
    ReportResult,
    SectionResult,
    Source,
    TokenUsage,
)

logger = logging.getLogger(__name__)

# Section definitions: (agent, name, prompt_template)
SECTIONS: list[tuple[str, str]] = [
    (
        "Industry Overview",
        "Summarize the week's most significant AI industry developments.",
    ),
    (
        "Research Updates",
        "Select and summarize 2 influential recent AI/ML papers.",
    ),
    (
        "Community Updates",
        "Surface important discussions and news from the AI community.",
    ),
    (
        "Coding Agents & Best Practices",
        "Cover coding agent best practices, tool updates, and community highlights.",
    ),
]


async def run_report(
    settings: Settings | None = None,
    *,
    bundle: AgentBundle | None = None,
    date_range: str | None = None,
) -> ReportResult:
    """Run the full newsletter report pipeline with evaluation and revision loop.

    Args:
        settings: Application settings. Uses defaults if not provided.
        bundle: Agent bundle to run with. If not provided, the module-level
            default bundle (built from get_settings()) is used. Pass an
            explicit bundle to run with a different model configuration.
        date_range: Fixed date range for the report. When None, the last
            7 days are computed at run time.

    Returns:
        ReportResult with the final report, evaluation scores, and metadata.
    """
    if settings is None:
        settings = get_settings()
    if bundle is None:
        bundle = default_bundle
    if date_range is None:
        date_range = get_date_range()

    logger.info("Starting report generation for %s", date_range)

    _setup_logfire(settings)

    # Shared usage tracking across all agent calls so the request_limit
    # applies to the combined total of sub-agent + manager + evaluator calls.
    shared_usage = RunUsage()
    usage_limits = UsageLimits(request_limit=settings.request_limit)

    # ── Gather sections from sub-agents ────────────────────────────────
    sections = await _gather_sections(bundle, date_range, shared_usage, usage_limits)

    # ── Manager synthesizes the final report ────────────────────────────
    synthesis_prompt = _build_synthesis_prompt(sections, date_range)

    try:
        result = await bundle.manager.run(
            synthesis_prompt,
            usage=shared_usage,
            usage_limits=usage_limits,
        )
    except UsageLimitExceeded as exc:
        logger.warning("Manager synthesis hit usage limit, building fallback report: %s", exc)
        return _fallback_report_result(sections, date_range, settings, shared_usage)
    except Exception as exc:
        logger.error("Manager synthesis failed, building fallback report: %s", exc)
        return _fallback_report_result(sections, date_range, settings, shared_usage)

    report_output: ReportOutput = result.output  # ty:ignore[invalid-assignment]
    report_markdown = report_output.content
    sources = report_output.sources
    message_history = result.all_messages()

    # ── Evaluation and revision loop ────────────────────────────────────
    revision_count = 0
    eval_passed = False
    eval_score = 0.0

    for cycle in range(settings.max_revision_cycles + 1):
        logger.info("Evaluation cycle %d", cycle + 1)

        # Evaluate the report
        eval_prompt = f"Evaluate this newsletter report:\n\n{report_markdown}"
        try:
            eval_result = await bundle.evaluator.run(
                eval_prompt,
                usage=shared_usage,
                usage_limits=usage_limits,
            )
        except UsageLimitExceeded as exc:
            logger.warning("Evaluation hit usage limit, skipping: %s", exc)
            break
        except Exception as exc:
            logger.error("Evaluation failed, using current report: %s", exc)
            break

        eval_output: EvalResult = eval_result.output  # ty:ignore[invalid-assignment]
        eval_score = _compute_average_score(eval_output.dimensions)
        eval_passed = eval_output.overall_pass

        logger.info(
            "Evaluation score: %.2f (threshold: %.2f) — %s",
            eval_score,
            settings.eval_threshold,
            "PASS" if eval_passed else "FAIL",
        )

        if eval_passed:
            break

        if cycle < settings.max_revision_cycles:
            # Build revision feedback
            feedback = _build_revision_feedback(eval_output)
            logger.info("Revising report based on evaluation feedback")

            # Re-run manager with feedback and previous message history
            revision_prompt = (
                f"The previous report scored {eval_score:.2f}/1.0 and did not pass evaluation.\n\n"
                f"Feedback:\n{feedback}\n\n"
                f"Please revise the report addressing these issues."
            )

            try:
                result = await bundle.manager.run(
                    revision_prompt,
                    message_history=message_history,
                    usage=shared_usage,
                    usage_limits=usage_limits,
                )
            except UsageLimitExceeded as exc:
                logger.warning("Revision hit usage limit, using current report: %s", exc)
                break
            except Exception as exc:
                logger.error("Revision failed, using current report: %s", exc)
                break

            report_output = result.output  # ty:ignore[invalid-assignment]
            report_markdown = report_output.content
            sources = report_output.sources
            message_history = result.all_messages()
            revision_count += 1

    # Write the report to file
    output_path = write_report(report_markdown, date_range, settings, sources)

    logger.info(
        "Report written to %s (score: %.2f, passed: %s)",
        output_path,
        eval_score,
        eval_passed,
    )

    # Best-effort email delivery (never affects report success)
    _email_result = send_report_email(
        report_markdown, date_range, eval_score, eval_passed, settings
    )
    _log_email_result(_email_result)

    return ReportResult(
        date_range=date_range,
        report_markdown=report_markdown,
        sources=sources,
        eval_passed=eval_passed,
        eval_score=eval_score,
        revision_count=revision_count,
        tokens=_usage_tokens(shared_usage),
    )


def _usage_tokens(usage: RunUsage) -> TokenUsage:
    """Convert a RunUsage accumulator into a TokenUsage schema.

    Args:
        usage: RunUsage with accumulated counts across all agent calls.

    Returns:
        TokenUsage with None counts treated as zero.
    """
    return TokenUsage(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
    )


async def _gather_sections(
    bundle: AgentBundle,
    date_range: str,
    usage: RunUsage,
    usage_limits: UsageLimits,
) -> list[SectionResult | None]:
    """Call each sub-agent sequentially and collect results.

    Each sub-agent call shares the same RunUsage so the request_limit applies
    across all calls. If a sub-agent fails for any reason (usage limit,
    search-backend error, transient API failure), it is recorded as None and
    the remaining sections are still attempted.

    Args:
        bundle: Agent bundle providing the sub-agents to run.
        date_range: The date range string for the report.
        usage: Shared RunUsage object for tracking requests across all agents.
        usage_limits: UsageLimits to enforce on each agent call.

    Returns:
        List of SectionResult or None (one per section, in order).
    """
    sub_agents = [
        bundle.industry_overview,
        bundle.research,
        bundle.community_news,
        bundle.coding_agents,
    ]

    sections: list[SectionResult | None] = []

    for (section_name, task_desc), agent in zip(SECTIONS, sub_agents, strict=True):
        prompt = (
            f"Generate the {section_name} section for the AI Industry Weekly "
            f"newsletter ({date_range}). {task_desc}"
        )
        try:
            result = await agent.run(prompt, usage=usage, usage_limits=usage_limits)
        except UsageLimitExceeded as exc:
            logger.warning("%s section skipped due to usage limit: %s", section_name, exc)
            sections.append(None)
            continue
        except Exception as exc:
            logger.error("%s section failed, skipping: %s", section_name, exc)
            sections.append(None)
            continue

        section: SectionResult = result.output  # ty:ignore[invalid-assignment]
        sections.append(section)
        logger.info("%s section gathered", section_name)

    return sections


def _build_synthesis_prompt(sections: list[SectionResult | None], date_range: str) -> str:
    """Build a prompt for the manager to synthesize collected sections.

    Args:
        sections: List of SectionResult or None (one per section, in order).
        date_range: The date range string for the report.

    Returns:
        Formatted prompt string containing all section content for the manager.
    """
    parts: list[str] = [
        f"Assemble the following pre-collected section outputs into the final "
        f"AI Industry Weekly newsletter for {date_range}."
    ]

    for (section_name, _), section in zip(SECTIONS, sections, strict=True):
        if section is not None:
            source_lines = "\n".join(
                f"[{i + 1}] {s.title} — {s.url}" for i, s in enumerate(section.sources)
            )
            parts.append(
                f"## {section_name}\n\n{section.content}\n\nSources:\n{source_lines}"
            )
        else:
            parts.append(
                f"## {section_name}\n\n"
                f"Section skipped: usage limit reached. Note this briefly in the report."
            )

    return "\n\n---\n\n".join(parts)


def _build_fallback_report(
    sections: list[SectionResult | None], date_range: str
) -> tuple[str, list[Source]]:
    """Build a fallback report from collected sections when manager synthesis fails.

    Concatenates the section content directly with per-section sources inline.
    Citations are NOT globally renumbered (they remain per-section).

    Args:
        sections: List of SectionResult or None (one per section, in order).
        date_range: The date range string for the report.

    Returns:
        Tuple of (report_markdown, sources) where sources is an empty list
        since sources are included inline per section.
    """
    parts: list[str] = [f"# AI Industry Weekly - {date_range}"]

    for (section_name, _), section in zip(SECTIONS, sections, strict=True):
        if section is not None:
            source_lines = "\n".join(
                f"- [{s.title}]({s.url})" for s in section.sources
            )
            block = f"## {section_name}\n\n{section.content}"
            if source_lines:
                block += f"\n\n**Sources:**\n{source_lines}"
            parts.append(block)
        else:
            parts.append(f"## {section_name}\n\n*Section skipped: usage limit reached.*")

    return "\n\n---\n\n".join(parts), []


def _fallback_report_result(
    sections: list[SectionResult | None],
    date_range: str,
    settings: Settings,
    shared_usage: RunUsage,
) -> ReportResult:
    """Build a fallback report from collected sections and deliver it.

    Used when manager synthesis fails (usage limit or unexpected error).
    Writes the report to disk and attempts email delivery; evaluation is
    marked as failed since no evaluation was performed.

    Args:
        sections: List of SectionResult or None (one per section, in order).
        date_range: The date range string for the report.
        settings: Application settings (for output dir and email config).
        shared_usage: Shared RunUsage accumulated across all agent calls.

    Returns:
        ReportResult describing the fallback report.
    """
    report_markdown, sources = _build_fallback_report(sections, date_range)
    output_path = write_report(report_markdown, date_range, settings, sources)
    logger.warning("Fallback report written to %s", output_path)
    _email_result = send_report_email(report_markdown, date_range, 0.0, False, settings)
    _log_email_result(_email_result)
    return ReportResult(
        date_range=date_range,
        report_markdown=report_markdown,
        sources=sources,
        eval_passed=False,
        eval_score=0.0,
        revision_count=0,
        tokens=_usage_tokens(shared_usage),
    )


def _compute_average_score(dimensions: list[DimensionScore]) -> float:
    """Compute the average score across all evaluation dimensions.

    Args:
        dimensions: List of DimensionScore objects.

    Returns:
        Average score as a float between 0.0 and 1.0.
    """
    if not dimensions:
        return 0.0
    return sum(d.score for d in dimensions) / len(dimensions)


def _build_revision_feedback(eval_output: EvalResult) -> str:
    """Build structured feedback from evaluation results for the revision loop.

    Args:
        eval_output: EvalResult object with dimension scores.

    Returns:
        Formatted feedback string for the manager agent.
    """
    feedback_parts: list[str] = []

    for dim in eval_output.dimensions:
        feedback_parts.append(
            f"**{dim.dimension}** (score: {dim.score:.2f}):\n"
            f"  Justification: {dim.justification}\n"
            f"  Suggestions: {'; '.join(dim.improvement_suggestions)}"
        )

    return "\n\n".join(feedback_parts)


def _setup_logfire(settings: Settings) -> None:
    """Configure Logfire observability if a token is provided.

    Args:
        settings: Application settings with logfire configuration.
    """
    try:
        import logfire

        logfire.configure(
            environment=settings.log_environment,
            token=settings.logfire_token or None,
        )
        logfire.instrument_pydantic_ai()
        logger.info("Logfire instrumentation enabled")
    except Exception as e:
        logger.warning("Failed to set up Logfire: %s", e)


def _log_email_result(result: EmailResult) -> None:
    """Log the outcome of an email delivery attempt at the appropriate level.

    Args:
        result: EmailResult from send_report_email.
    """
    if result.status == EmailStatus.SENT:
        logger.info("Email sent successfully")
    elif result.status == EmailStatus.DRY_RUN:
        logger.info("Dry-run HTML written to %s", result.html_path)
    elif result.status == EmailStatus.FAILED:
        logger.error("Email delivery failed: %s", result.error)
    # DISABLED and SKIPPED are already logged inside send_report_email