"""Tests for eval framework result schemas (TokenUsage, RunResult, EvalRun, MatrixResult)."""

from pathlib import Path

from src.eval.results import EvalRun, MatrixResult, RunResult, TokenUsage
from tests.utils import build_eval_run, build_run_result


class TestTokenUsage:
    """Tests for the TokenUsage schema."""

    def test_token_usage_defaults_to_zero(self) -> None:
        """Should default all counts to zero."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_total_is_sum_of_input_and_output(self) -> None:
        """total_tokens should be the sum of input and output tokens."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_total_tokens_is_serialized(self) -> None:
        """total_tokens should appear in JSON output (computed field)."""
        usage = TokenUsage(input_tokens=10, output_tokens=5)
        data = usage.model_dump()
        assert data["total_tokens"] == 15


class TestRunResult:
    """Tests for the RunResult schema (one generation run)."""

    def test_run_result_holds_report_and_metrics(self) -> None:
        """Should hold the report content plus token/time metrics."""
        run = build_run_result("gpt-4o", seed=1)
        assert run.generator_model == "gpt-4o"
        assert run.seed == 1
        assert run.tokens.total_tokens == 1500
        assert run.duration_seconds == 12.5
        assert run.eval_passed is True
        assert "AI Industry Weekly" in run.report_markdown

    def test_run_result_round_trips_json(self) -> None:
        """Should survive a JSON dump/load round trip."""
        run = build_run_result("gpt-4o", seed=2)
        loaded = RunResult.model_validate_json(run.model_dump_json())
        assert loaded == run
        assert loaded.tokens.total_tokens == 1500


class TestEvalRun:
    """Tests for the EvalRun schema (one evaluation of a fixed report)."""

    def test_eval_run_holds_score_and_tokens(self) -> None:
        """Should hold the score, dimensions, and token metrics."""
        eval_run = build_eval_run("claude-sonnet-4-6", "gpt-4o", seed=1, score=0.82)
        assert eval_run.evaluator_model == "claude-sonnet-4-6"
        assert eval_run.generator_model == "gpt-4o"
        assert eval_run.score == 0.82
        assert eval_run.tokens.total_tokens == 300
        assert len(eval_run.dimensions) == 4

    def test_eval_run_round_trips_json(self) -> None:
        """Should survive a JSON dump/load round trip."""
        eval_run = build_eval_run("claude-sonnet-4-6", "gpt-4o", seed=1, score=0.5)
        loaded = EvalRun.model_validate_json(eval_run.model_dump_json())
        assert loaded == eval_run
        assert loaded.eval_passed is False


class TestMatrixResult:
    """Tests for the MatrixResult schema (full G x E x seeds matrix)."""

    def test_matrix_result_round_trips_json(self, tmp_path: Path) -> None:
        """Should save to JSON and load back with all data intact."""
        matrix = MatrixResult(
            generators=["gpt-4o", "claude-sonnet-4-6"],
            evaluators=["gpt-4o-mini", "llama-3.1-70b"],
            seeds=2,
            runs=[
                build_run_result("gpt-4o", seed=1),
                build_run_result("gpt-4o", seed=2),
                build_run_result("claude-sonnet-4-6", seed=1),
                build_run_result("claude-sonnet-4-6", seed=2),
            ],
            evals=[
                build_eval_run("gpt-4o-mini", "gpt-4o", 1, 0.8),
                build_eval_run("llama-3.1-70b", "gpt-4o", 1, 0.75),
                build_eval_run("gpt-4o-mini", "gpt-4o", 2, 0.7),
                build_eval_run("llama-3.1-70b", "gpt-4o", 2, 0.65),
                build_eval_run("gpt-4o-mini", "claude-sonnet-4-6", 1, 0.9),
                build_eval_run("llama-3.1-70b", "claude-sonnet-4-6", 1, 0.85),
                build_eval_run("gpt-4o-mini", "claude-sonnet-4-6", 2, 0.88),
                build_eval_run("llama-3.1-70b", "claude-sonnet-4-6", 2, 0.83),
            ],
            created_at="2026-07-28T12:00:00Z",
        )
        path = tmp_path / "matrix.json"
        saved_path = matrix.save(path)
        assert saved_path == path
        assert path.exists()

        loaded = MatrixResult.load(path)
        assert loaded.generators == ["gpt-4o", "claude-sonnet-4-6"]
        assert loaded.evaluators == ["gpt-4o-mini", "llama-3.1-70b"]
        assert loaded.seeds == 2
        assert len(loaded.runs) == 4
        assert len(loaded.evals) == 8
        assert loaded.runs[0].tokens.total_tokens == 1500

    def test_matrix_result_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save() should create missing parent directories."""
        matrix = MatrixResult(
            generators=["gpt-4o"],
            evaluators=["gpt-4o-mini"],
            seeds=1,
            runs=[build_run_result("gpt-4o", seed=1)],
            evals=[build_eval_run("gpt-4o-mini", "gpt-4o", 1, 0.8)],
            created_at="2026-07-28T12:00:00Z",
        )
        nested = tmp_path / "a" / "b" / "matrix.json"
        matrix.save(nested)
        assert nested.exists()
