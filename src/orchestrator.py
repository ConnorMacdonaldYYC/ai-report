"""Pipeline orchestration with revision loop for the AI newsletter agent."""

import logging

from src.agents import evaluator_agent, manager_agent
from src.config import Settings, get_settings
from src.output import write_report
from src.prompts import get_date_range
from src.schemas import DimensionScore, EvalResult, ReportOutput, ReportResult

logger = logging.getLogger(__name__)


async def run_report(settings: Settings | None = None) -> ReportResult:
    """Run the full newsletter report pipeline with evaluation and revision loop.

    Args:
        settings: Application settings. Uses defaults if not provided.

    Returns:
        ReportResult with the final report, evaluation scores, and metadata.
    """
    if settings is None:
        settings = get_settings()

    date_range = get_date_range()
    logger.info("Starting report generation for %s", date_range)

    _setup_logfire(settings)

    # Initial report generation
    prompt = f"Generate the AI Industry Weekly newsletter for {date_range}."
    result = await manager_agent.run(prompt)
    report_output: ReportOutput = result.output
    report_markdown = report_output.content
    sources = report_output.sources
    message_history = result.all_messages()

    # Evaluation and revision loop
    revision_count = 0
    eval_passed = False
    eval_score = 0.0

    for cycle in range(settings.max_revision_cycles + 1):
        logger.info("Evaluation cycle %d", cycle + 1)

        # Evaluate the report
        eval_prompt = f"Evaluate this newsletter report:\n\n{report_markdown}"
        eval_result = await evaluator_agent.run(eval_prompt)

        eval_output = eval_result.output
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

            result = await manager_agent.run(
                revision_prompt,
                message_history=message_history,
            )
            report_output = result.output
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

    return ReportResult(
        date_range=date_range,
        report_markdown=report_markdown,
        sources=sources,
        eval_passed=eval_passed,
        eval_score=eval_score,
        revision_count=revision_count,
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
        )
        logfire.instrument_pydantic_ai()
        logger.info("Logfire instrumentation enabled")
    except Exception as e:
        logger.warning("Failed to set up Logfire: %s", e)