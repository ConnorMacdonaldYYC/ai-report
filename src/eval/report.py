"""Rendering for eval matrix results: aggregation and human-readable output.

Turns a MatrixResult into per-generator summaries plus ranked markdown/CSV
tables and a generator x evaluator agreement table.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from src.eval.results import MatrixResult

__all__ = [
    "GeneratorSummary",
    "aggregate_per_generator",
    "render_evaluator_agreement",
    "render_summary_csv",
    "render_summary_markdown",
]


@dataclass(frozen=True)
class GeneratorSummary:
    """Aggregated metrics for one generator model across the matrix."""

    generator: str
    mean_score: float
    std_score: float
    mean_tokens: int
    mean_time_seconds: float
    mean_revisions: float
    n_runs: int
    n_evals: int


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean of values, or 0.0 when empty.

    Args:
        values: Numbers to average.

    Returns:
        The mean, or 0.0 for an empty list.
    """
    return sum(values) / len(values) if values else 0.0


def aggregate_per_generator(matrix: MatrixResult) -> list[GeneratorSummary]:
    """One summary per generator (in matrix.generators order).

    Args:
        matrix: The evaluation matrix to aggregate.

    Returns:
        A list of GeneratorSummary, one per generator in matrix.generators order.
    """
    summaries: list[GeneratorSummary] = []
    for generator in matrix.generators:
        scores = [e.score for e in matrix.evals if e.generator_model == generator]
        runs = [r for r in matrix.runs if r.generator_model == generator]
        summaries.append(
            GeneratorSummary(
                generator=generator,
                mean_score=_mean(scores),
                std_score=statistics.stdev(scores) if len(scores) >= 2 else 0.0,
                mean_tokens=round(_mean([float(r.tokens.total_tokens) for r in runs])),
                mean_time_seconds=_mean([r.duration_seconds for r in runs]),
                mean_revisions=_mean([float(r.revision_count) for r in runs]),
                n_runs=len(runs),
                n_evals=len(scores),
            )
        )
    return summaries


def _ranked(matrix: MatrixResult) -> list[GeneratorSummary]:
    """Return per-generator summaries sorted by mean score descending."""
    return sorted(
        aggregate_per_generator(matrix),
        key=lambda s: s.mean_score,
        reverse=True,
    )


def render_summary_markdown(matrix: MatrixResult) -> str:
    """Render a ranked markdown summary table of generator performance.

    Args:
        matrix: The evaluation matrix to render.

    Returns:
        A markdown string with a metadata header line and a ranked table
        sorted by mean score descending.
    """
    header = (
        f"# Eval Matrix Summary - {len(matrix.generators)} generators x "
        f"{len(matrix.evaluators)} evaluators x {matrix.seeds} seeds "
        f"(created {matrix.created_at})"
    )
    lines = [
        header,
        "",
        "| Generator | Mean Score | Score Std | Mean Tokens | Mean Time (s) | "
        "Mean Revisions | Runs | Evals |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in _ranked(matrix):
        lines.append(
            f"| {s.generator} | {s.mean_score:.3f} | {s.std_score:.3f} | "
            f"{s.mean_tokens} | {s.mean_time_seconds:.3f} | "
            f"{s.mean_revisions:.3f} | {s.n_runs} | {s.n_evals} |"
        )
    return "\n".join(lines) + "\n"


def render_summary_csv(matrix: MatrixResult) -> str:
    """Render a CSV summary of generator performance.

    Args:
        matrix: The evaluation matrix to render.

    Returns:
        A CSV string with a header row and one plain-number row per generator,
        sorted by mean score descending.
    """
    header = (
        "generator,mean_score,score_std,mean_tokens,mean_time_seconds,"
        "mean_revisions,n_runs,n_evals"
    )
    rows = [header]
    for s in _ranked(matrix):
        rows.append(
            f"{s.generator},{s.mean_score},{s.std_score},{s.mean_tokens},"
            f"{s.mean_time_seconds},{s.mean_revisions},{s.n_runs},{s.n_evals}"
        )
    return "\n".join(rows) + "\n"


def render_evaluator_agreement(matrix: MatrixResult) -> str:
    """Render a markdown table of mean scores per generator x evaluator.

    Args:
        matrix: The evaluation matrix to render.

    Returns:
        A markdown table with generators as rows and evaluators as columns;
        cells show the mean score for that combination or "n/a" when absent.
    """
    lines = [
        "| Generator | " + " | ".join(matrix.evaluators) + " |",
        "|---|" + "---|" * len(matrix.evaluators),
    ]
    for generator in matrix.generators:
        cells: list[str] = []
        for evaluator in matrix.evaluators:
            scores = [
                e.score
                for e in matrix.evals
                if e.generator_model == generator and e.evaluator_model == evaluator
            ]
            cells.append(f"{_mean(scores):.3f}" if scores else "n/a")
        lines.append(f"| {generator} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"