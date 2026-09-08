"""Single evaluation runner for the evaluation framework.

evaluate_single evaluates one fixed report with one evaluator model and
returns an EvalRun with the average dimension score, pass flag, token
usage, and wall-clock time captured.
"""

import time
from datetime import UTC, datetime

from src.agents import AgentBundle, build_agents
from src.config import Settings
from src.eval.results import EvalRun
from src.eval.runner import extract_usage
from src.schemas import DimensionScore


async def evaluate_single(
    settings: Settings,
    report_markdown: str,
    evaluator_model: str,
    generator_model: str,
    seed: int,
    *,
    bundle: AgentBundle | None = None,
) -> EvalRun:
    """Evaluate a fixed report with a single evaluator model.

    The settings are cloned with the evaluator model swapped in and email
    delivery disabled, so the caller's settings object is never mutated.
    If no bundle is provided, a fresh one is built from the cloned settings.

    Args:
        settings: Base application settings (not mutated).
        report_markdown: The fixed report to evaluate.
        evaluator_model: Model name to use as the evaluator.
        generator_model: Model name that generated the report.
        seed: Seed used to generate the report.
        bundle: Optional pre-built AgentBundle to run with (test injection).
            If not provided, a bundle is built from the cloned settings.

    Returns:
        EvalRun with the average dimension score, pass flag, dimensions,
        token usage, duration, and timestamp.
    """
    cloned = settings.model_copy(
        update={"evaluator_model": evaluator_model, "email_enabled": False}
    )
    if bundle is None:
        bundle = build_agents(cloned)

    start = time.perf_counter()
    result = await bundle.evaluator.run(
        f"Evaluate this newsletter report:\n\n{report_markdown}"
    )
    duration_seconds = time.perf_counter() - start

    eval_output = result.output
    score = _average_score(eval_output.dimensions)

    return EvalRun(
        evaluator_model=evaluator_model,
        generator_model=generator_model,
        seed=seed,
        score=score,
        eval_passed=eval_output.overall_pass,
        dimensions=eval_output.dimensions,
        tokens=extract_usage(result),
        duration_seconds=duration_seconds,
        timestamp=datetime.now(UTC).isoformat(),
    )


def _average_score(dimensions: list[DimensionScore]) -> float:
    """Return the average score across all dimensions (0.0 for empty).

    Args:
        dimensions: Dimension scores to average.

    Returns:
        The mean score, or 0.0 when the list is empty.
    """
    if not dimensions:
        return 0.0
    return sum(d.score for d in dimensions) / len(dimensions)