"""Single generation run for the evaluation framework.

run_single generates one report with a given generator model config and
returns a RunResult with token usage and wall-clock time captured.
"""

import tempfile
import time
from collections.abc import Iterable
from datetime import UTC, datetime

from pydantic_ai.run import AgentRunResult

from src.agents import AgentBundle, build_agents
from src.config import Settings
from src.eval.results import RunResult, TokenUsage
from src.orchestrator import run_report

# AgentRunResult is generic over the output type; usage extraction works
# for any output type.
AnyRunResult = AgentRunResult[object]


def extract_usage(
    results: AnyRunResult | Iterable[AnyRunResult],
) -> TokenUsage:
    """Extract token usage from one or more pydantic-ai run results.

    Args:
        results: A single AgentRunResult or an iterable of them.

    Returns:
        TokenUsage with summed input/output token counts. Token counts
        reported as None by a model are treated as zero.
    """
    if isinstance(results, AgentRunResult):
        results = [results]

    input_tokens = 0
    output_tokens = 0
    for result in results:
        usage = result.usage
        input_tokens += usage.input_tokens or 0
        output_tokens += usage.output_tokens or 0

    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _elapsed_seconds(start: float) -> float:
    """Return seconds elapsed since a time.perf_counter() start value."""
    return time.perf_counter() - start


async def run_single(
    settings: Settings,
    generator_model: str,
    seed: int,
    *,
    in_pipeline_evaluator_model: str | None = None,
    bundle: AgentBundle | None = None,
    output_dir: str | None = None,
    date_range: str | None = None,
) -> RunResult:
    """Run one report generation with a given generator model and seed.

    Clones the settings with the generator model applied to the sub-agents
    and report manager, disables email, and runs the full pipeline once.
    Token usage and wall-clock duration are captured in the RunResult.

    Args:
        settings: Base application settings to clone for this run.
        generator_model: Model name to use for the sub-agents and report
            manager (the generator under evaluation).
        seed: Seed for this run, recorded in the RunResult.
        in_pipeline_evaluator_model: Optional model name for the in-pipeline
            evaluator agent. When None, the evaluator model from settings is
            kept.
        bundle: Optional AgentBundle to run with (test-injection path). When
            None, a fresh bundle is built from the cloned settings.
        output_dir: Directory to write the report to. When None, a fresh
            temporary directory is created so concurrent runs never collide.
        date_range: Fixed date range for the report. When None, the last
            7 days are computed at run time.

    Returns:
        RunResult with the generated report, token usage, and timing.
    """
    update: dict[str, object] = {
        "sub_agent_model": generator_model,
        "report_manager_model": generator_model,
        "email_enabled": False,
    }
    if in_pipeline_evaluator_model is not None:
        update["evaluator_model"] = in_pipeline_evaluator_model
    if output_dir is not None:
        update["output_dir"] = output_dir
    else:
        update["output_dir"] = tempfile.mkdtemp(prefix="eval-run-")

    cloned_settings = settings.model_copy(update=update)

    if bundle is None:
        bundle = build_agents(cloned_settings)

    start = time.perf_counter()
    report = await run_report(
        cloned_settings, bundle=bundle, date_range=date_range
    )
    duration_seconds = _elapsed_seconds(start)

    return RunResult(
        generator_model=generator_model,
        seed=seed,
        date_range=report.date_range,
        report_markdown=report.report_markdown,
        sources=report.sources,
        tokens=report.tokens,
        duration_seconds=duration_seconds,
        revision_count=report.revision_count,
        eval_passed=report.eval_passed,
        eval_score=report.eval_score,
        timestamp=datetime.now(UTC).isoformat(),
    )
