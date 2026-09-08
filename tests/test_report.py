"""Tests for eval matrix report rendering (aggregation + markdown/CSV output)."""

import pytest

from src.eval.report import (
    GeneratorSummary,
    aggregate_per_generator,
    render_evaluator_agreement,
    render_summary_csv,
    render_summary_markdown,
)
from src.eval.results import EvalRun, MatrixResult, RunResult
from src.schemas import TokenUsage
from tests.utils import build_eval_run, build_run_result


def _matrix(
    generators: list[str],
    evaluators: list[str],
    seeds: int,
    runs: list[RunResult],
    evals: list[EvalRun],
) -> MatrixResult:
    """Build a MatrixResult with a fixed created_at timestamp."""
    return MatrixResult(
        generators=generators,
        evaluators=evaluators,
        seeds=seeds,
        runs=runs,
        evals=evals,
        created_at="2026-07-28T12:00:00Z",
    )


class TestAggregatePerGenerator:
    """Tests for aggregate_per_generator."""

    def test_aggregate_computes_mean_and_std(self) -> None:
        """Should compute mean/std of eval scores and means of run metrics."""
        matrix = _matrix(
            generators=["gen-a"],
            evaluators=["eval-x"],
            seeds=2,
            runs=[
                build_run_result("gen-a", seed=1),
                build_run_result(
                    "gen-a",
                    seed=2,
                    tokens=TokenUsage(input_tokens=2000, output_tokens=1000),
                    duration_seconds=7.5,
                    revision_count=2,
                ),
            ],
            evals=[
                build_eval_run("eval-x", "gen-a", 1, 0.8),
                build_eval_run("eval-x", "gen-a", 2, 0.6),
            ],
        )
        summaries = aggregate_per_generator(matrix)
        assert len(summaries) == 1
        summary: GeneratorSummary = summaries[0]
        assert summary.generator == "gen-a"
        assert summary.mean_score == pytest.approx(0.7)
        assert summary.std_score == pytest.approx(0.1414, abs=1e-3)
        assert summary.mean_tokens == 2250
        assert summary.mean_time_seconds == pytest.approx(10.0)
        assert summary.mean_revisions == pytest.approx(1.0)
        assert summary.n_runs == 2
        assert summary.n_evals == 2

    def test_aggregate_generator_with_no_evals(self) -> None:
        """Should still include generators with no evals, using zero defaults."""
        matrix = _matrix(
            generators=["gen-a", "gen-b"],
            evaluators=["eval-x"],
            seeds=1,
            runs=[build_run_result("gen-a", seed=1)],
            evals=[build_eval_run("eval-x", "gen-a", 1, 0.8)],
        )
        summaries = aggregate_per_generator(matrix)
        by_name = {s.generator: s for s in summaries}
        assert "gen-b" in by_name
        summary = by_name["gen-b"]
        assert summary.mean_score == 0.0
        assert summary.std_score == 0.0
        assert summary.mean_tokens == 0
        assert summary.mean_time_seconds == 0.0
        assert summary.mean_revisions == 0.0
        assert summary.n_runs == 0
        assert summary.n_evals == 0

    def test_aggregate_single_eval_zero_std(self) -> None:
        """Should report std_score 0.0 when there is only one eval."""
        matrix = _matrix(
            generators=["gen-a"],
            evaluators=["eval-x"],
            seeds=1,
            runs=[],
            evals=[build_eval_run("eval-x", "gen-a", 1, 0.8)],
        )
        summaries = aggregate_per_generator(matrix)
        assert summaries[0].std_score == 0.0
        assert summaries[0].mean_score == pytest.approx(0.8)


class TestRenderSummaryMarkdown:
    """Tests for render_summary_markdown."""

    def test_render_summary_markdown_ranked(self) -> None:
        """Should rank rows by mean score descending and include metadata."""
        matrix = _matrix(
            generators=["gen-a", "gen-b"],
            evaluators=["eval-x"],
            seeds=2,
            runs=[
                build_run_result("gen-a", seed=1),
                build_run_result("gen-b", seed=1),
            ],
            evals=[
                build_eval_run("eval-x", "gen-a", 1, 0.6),
                build_eval_run("eval-x", "gen-b", 1, 0.9),
            ],
        )
        md = render_summary_markdown(matrix)
        lines = md.splitlines()

        assert "2 seeds" in md
        assert "Generator" in md
        assert "Mean Score" in md
        assert "Mean Tokens" in md
        assert "gen-a" in md
        assert "gen-b" in md

        gen_b_idx = next(i for i, line in enumerate(lines) if "gen-b" in line)
        gen_a_idx = next(i for i, line in enumerate(lines) if "gen-a" in line)
        assert gen_b_idx < gen_a_idx


class TestRenderSummaryCsv:
    """Tests for render_summary_csv."""

    def test_render_summary_csv_rows(self) -> None:
        """Should emit a header plus one plain-number row per generator."""
        matrix = _matrix(
            generators=["gen-a", "gen-b"],
            evaluators=["eval-x"],
            seeds=2,
            runs=[
                build_run_result("gen-a", seed=1),
                build_run_result("gen-b", seed=1),
            ],
            evals=[
                build_eval_run("eval-x", "gen-a", 1, 0.6),
                build_eval_run("eval-x", "gen-b", 1, 0.9),
            ],
        )
        csv_text = render_summary_csv(matrix)
        lines = csv_text.strip().splitlines()
        assert len(lines) == 1 + len(matrix.generators)
        assert lines[0] == (
            "generator,mean_score,score_std,mean_tokens,mean_time_seconds,"
            "mean_revisions,n_runs,n_evals"
        )
        assert "±" not in csv_text


class TestRenderEvaluatorAgreement:
    """Tests for render_evaluator_agreement."""

    def test_render_evaluator_agreement_table(self) -> None:
        """Should show one column per evaluator and mean scores per cell."""
        matrix = _matrix(
            generators=["gen-a", "gen-b"],
            evaluators=["eval-x", "eval-y"],
            seeds=2,
            runs=[],
            evals=[
                build_eval_run("eval-x", "gen-a", 1, 0.9),
                build_eval_run("eval-x", "gen-a", 2, 0.7),
                build_eval_run("eval-x", "gen-b", 1, 0.5),
                build_eval_run("eval-y", "gen-b", 1, 0.6),
            ],
        )
        table = render_evaluator_agreement(matrix)
        assert "eval-x" in table
        assert "eval-y" in table
        assert "gen-a" in table
        assert "gen-b" in table
        assert "0.800" in table
        assert "n/a" in table