"""Matrix orchestration for the evaluation framework.

run_matrix executes the full generators x evaluators x seeds matrix with
bounded concurrency, saves every generated report, and writes the matrix
JSON plus rendered summaries to a timestamped output directory.

Failed generation or evaluation runs are logged and skipped so one bad
model does not abort the whole matrix.
"""

import asyncio
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from src.agents import AgentBundle
from src.config import Settings
from src.eval.evaluator import evaluate_single
from src.eval.report import (
    render_evaluator_agreement,
    render_summary_csv,
    render_summary_markdown,
)
from src.eval.results import EvalRun, MatrixResult, RunResult
from src.eval.runner import run_single

logger = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    """Make a model name safe to use as a filename component.

    Args:
        name: Model name (e.g. "openai/gpt-4o").

    Returns:
        Name with filesystem-unsafe characters replaced by dashes.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "-", name)


async def run_matrix(
    settings: Settings,
    generators: list[str],
    evaluators: list[str],
    seeds: int,
    *,
    concurrency: int = 3,
    output_dir: str = "./eval_results",
    run_bundle: AgentBundle | None = None,
    eval_bundle: AgentBundle | None = None,
    date_range: str | None = None,
) -> MatrixResult:
    """Run the full generators x evaluators x seeds evaluation matrix.

    Each (generator, seed) pair produces one report via run_single. Every
    report is then evaluated by every evaluator model via evaluate_single.
    The in-pipeline evaluator (revision-loop referee) is fixed to
    evaluators[0] so the revision loop behaves identically for all
    generators.

    Failed runs are logged and skipped; the matrix always completes.

    Args:
        settings: Base application settings (never mutated).
        generators: Generator model names to compare.
        evaluators: Evaluator model names forming the judge panel.
        seeds: Number of repeated runs per generator (>= 1).
        concurrency: Maximum number of concurrent agent runs.
        output_dir: Root directory for results (a timestamped
            subdirectory is created inside it).
        run_bundle: Optional AgentBundle injected into every generation
            run (test path). Real runs build fresh bundles per run.
        eval_bundle: Optional AgentBundle injected into every evaluation
            run (test path).
        date_range: Fixed date range passed to every generation run so all
            reports in the matrix share the same window label. When None,
            each run computes the last 7 days at run time.

    Returns:
        MatrixResult with all runs, evals, and metadata.

    Raises:
        ValueError: If generators or evaluators is empty, or seeds < 1.
    """
    if not generators:
        msg = "generators must not be empty"
        raise ValueError(msg)
    if not evaluators:
        msg = "evaluators must not be empty"
        raise ValueError(msg)
    if seeds < 1:
        msg = "seeds must be >= 1"
        raise ValueError(msg)

    run_dir = Path(output_dir) / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(concurrency)
    referee = evaluators[0]

    async def generate(generator: str, seed: int) -> RunResult | None:
        """Generate one report, skipping (with a warning) on failure."""
        safe_name = _sanitize_filename(generator)
        async with semaphore:
            try:
                run = await run_single(
                    settings,
                    generator,
                    seed,
                    in_pipeline_evaluator_model=referee,
                    bundle=run_bundle,
                    output_dir=str(reports_dir / f"{safe_name}__seed{seed}"),
                    date_range=date_range,
                )
            except Exception as exc:
                logger.warning(
                    "Generation run failed for %s seed %d, skipping: %s",
                    generator,
                    seed,
                    exc,
                )
                return None
        report_path = reports_dir / f"{safe_name}__seed{seed}.md"
        report_path.write_text(run.report_markdown)
        logger.info("Generated report for %s seed %d -> %s", generator, seed, report_path)
        return run

    async def evaluate(run: RunResult, evaluator: str) -> EvalRun | None:
        """Evaluate one report with one evaluator, skipping on failure."""
        async with semaphore:
            try:
                return await evaluate_single(
                    settings,
                    run.report_markdown,
                    evaluator,
                    run.generator_model,
                    run.seed,
                    bundle=eval_bundle,
                )
            except Exception as exc:
                logger.warning(
                    "Evaluation failed for %s seed %d with %s, skipping: %s",
                    run.generator_model,
                    run.seed,
                    evaluator,
                    exc,
                )
                return None

    logger.info(
        "Starting eval matrix: %d generators x %d evaluators x %d seeds",
        len(generators),
        len(evaluators),
        seeds,
    )

    gen_tasks = [
        generate(generator, seed)
        for generator in generators
        for seed in range(1, seeds + 1)
    ]
    gen_results = await asyncio.gather(*gen_tasks)
    runs = [run for run in gen_results if run is not None]
    logger.info("Generation phase complete: %d/%d runs succeeded", len(runs), len(gen_tasks))

    eval_tasks = [evaluate(run, evaluator) for run in runs for evaluator in evaluators]
    eval_results = await asyncio.gather(*eval_tasks)
    evals = [eval_run for eval_run in eval_results if eval_run is not None]
    logger.info(
        "Evaluation phase complete: %d/%d evals succeeded", len(evals), len(eval_tasks)
    )

    matrix = MatrixResult(
        generators=generators,
        evaluators=evaluators,
        seeds=seeds,
        runs=runs,
        evals=evals,
        created_at=datetime.now(UTC).isoformat(),
    )

    matrix.save(run_dir / "matrix.json")
    (run_dir / "summary.md").write_text(render_summary_markdown(matrix))
    (run_dir / "summary.csv").write_text(render_summary_csv(matrix))
    (run_dir / "evaluator_agreement.md").write_text(render_evaluator_agreement(matrix))
    logger.info("Matrix results written to %s", run_dir)

    return matrix
