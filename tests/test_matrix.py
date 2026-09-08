"""Tests for the matrix orchestrator (run_matrix)."""

import asyncio
from typing import Any

import pytest
from pydantic_ai import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo

import src.eval.matrix as matrix_module
from src.agents import build_agents
from src.eval.results import EvalRun, MatrixResult, RunResult
from tests.utils import (
    ModelFn,
    build_eval_run,
    build_run_result,
    build_test_settings,
    default_sub_agent_fns,
    make_evaluator_fn,
    make_manager_fn,
    make_section_fn,
    override_all_agents,
)


def _make_bundles() -> tuple[Any, Any]:
    """Return (run_bundle, eval_bundle) built from test settings."""
    settings = build_test_settings()
    return build_agents(settings), build_agents(settings)


class TestRunMatrix:
    """Integration tests for run_matrix with FunctionModel-backed bundles."""

    @pytest.mark.asyncio
    async def test_run_matrix_small(self, tmp_path: Any) -> None:
        """Should produce runs and evals for every (generator, seed, evaluator)."""
        settings = build_test_settings()
        run_bundle, eval_bundle = _make_bundles()

        with (
            override_all_agents(
                default_sub_agent_fns(),
                make_manager_fn(),
                make_evaluator_fn(score=0.9),
                bundle=run_bundle,
            ),
            override_all_agents(
                default_sub_agent_fns(),
                make_manager_fn(),
                make_evaluator_fn(score=0.8),
                bundle=eval_bundle,
            ),
        ):
            matrix = await matrix_module.run_matrix(
                settings,
                generators=["gen-a", "gen-b"],
                evaluators=["eval-x", "eval-y"],
                seeds=1,
                concurrency=2,
                output_dir=str(tmp_path),
                run_bundle=run_bundle,
                eval_bundle=eval_bundle,
            )

        assert isinstance(matrix, MatrixResult)
        assert matrix.generators == ["gen-a", "gen-b"]
        assert matrix.evaluators == ["eval-x", "eval-y"]
        assert matrix.seeds == 1
        assert len(matrix.runs) == 2
        assert len(matrix.evals) == 4

        combos = {(e.generator_model, e.evaluator_model) for e in matrix.evals}
        expected = {
            ("gen-a", "eval-x"),
            ("gen-a", "eval-y"),
            ("gen-b", "eval-x"),
            ("gen-b", "eval-y"),
        }
        assert combos == expected

        # All evals come from the eval bundle's evaluator (score 0.8)
        assert all(e.score == pytest.approx(0.8) for e in matrix.evals)

        # Output artifacts exist under the timestamped run directory
        assert list(tmp_path.rglob("matrix.json"))
        assert list(tmp_path.rglob("summary.md"))
        assert list(tmp_path.rglob("summary.csv"))
        assert list(tmp_path.rglob("evaluator_agreement.md"))

        # Canonical report files exist
        reports_dir = next(iter(tmp_path.rglob("reports")))
        report_names = {
            p.name for p in reports_dir.iterdir() if p.name.endswith(".md")
        }
        assert "gen-a__seed1.md" in report_names
        assert "gen-b__seed1.md" in report_names

    @pytest.mark.asyncio
    async def test_run_matrix_respects_concurrency_limit(
        self, tmp_path: Any
    ) -> None:
        """At most `concurrency` generation runs should be in flight at once."""
        active = 0
        max_active = 0

        def slow_tracked(fn: ModelFn) -> ModelFn:
            """Wrap a sync model fn: track concurrency and sleep briefly."""

            async def slow_fn(
                messages: list[ModelMessage], info: AgentInfo
            ) -> ModelResponse:
                nonlocal active, max_active
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.05)
                active -= 1
                result = fn(messages, info)
                assert isinstance(result, ModelResponse)  # our fns are sync
                return result

            return slow_fn

        settings = build_test_settings()
        run_bundle, eval_bundle = _make_bundles()

        section_slow = slow_tracked(make_section_fn("Section"))
        manager_slow = slow_tracked(make_manager_fn())
        evaluator_slow = slow_tracked(make_evaluator_fn(score=0.9))

        with (
            override_all_agents(
                [section_slow] * 4,
                manager_slow,
                evaluator_slow,
                bundle=run_bundle,
            ),
            override_all_agents(
                default_sub_agent_fns(),
                make_manager_fn(),
                make_evaluator_fn(score=0.9),
                bundle=eval_bundle,
            ),
        ):
            matrix = await matrix_module.run_matrix(
                settings,
                generators=["gen-a", "gen-b"],
                evaluators=["eval-x"],
                seeds=2,
                concurrency=2,
                output_dir=str(tmp_path),
                run_bundle=run_bundle,
                eval_bundle=eval_bundle,
            )

        assert len(matrix.runs) == 4
        assert max_active <= 2
        # With 4 sleeping runs and a limit of 2, the limit should be reached
        assert max_active == 2

    @pytest.mark.asyncio
    async def test_run_matrix_skips_failed_runs(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed generation run should be skipped, not crash the matrix."""

        async def fake_run_single(
            settings: Any,
            generator_model: str,
            seed: int,
            **kwargs: Any,
        ) -> RunResult:
            if generator_model == "bad-gen":
                msg = "simulated API failure"
                raise RuntimeError(msg)
            return build_run_result(generator_model, seed)

        async def fake_evaluate_single(
            settings: Any,
            report_markdown: str,
            evaluator_model: str,
            generator_model: str,
            seed: int,
            **kwargs: Any,
        ) -> EvalRun:
            return build_eval_run(evaluator_model, generator_model, seed, 0.8)

        monkeypatch.setattr(matrix_module, "run_single", fake_run_single)
        monkeypatch.setattr(matrix_module, "evaluate_single", fake_evaluate_single)

        settings = build_test_settings()
        matrix = await matrix_module.run_matrix(
            settings,
            generators=["good-gen", "bad-gen"],
            evaluators=["eval-x"],
            seeds=1,
            output_dir=str(tmp_path),
        )

        assert [r.generator_model for r in matrix.runs] == ["good-gen"]
        assert [e.generator_model for e in matrix.evals] == ["good-gen"]

    @pytest.mark.asyncio
    async def test_run_matrix_uses_first_evaluator_as_referee(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The in-pipeline evaluator should be fixed to evaluators[0]."""
        captured: dict[str, Any] = {}

        async def fake_run_single(
            settings: Any,
            generator_model: str,
            seed: int,
            **kwargs: Any,
        ) -> RunResult:
            captured[generator_model] = kwargs
            return build_run_result(generator_model, seed)

        async def fake_evaluate_single(
            settings: Any,
            report_markdown: str,
            evaluator_model: str,
            generator_model: str,
            seed: int,
            **kwargs: Any,
        ) -> EvalRun:
            return build_eval_run(evaluator_model, generator_model, seed, 0.8)

        monkeypatch.setattr(matrix_module, "run_single", fake_run_single)
        monkeypatch.setattr(matrix_module, "evaluate_single", fake_evaluate_single)

        settings = build_test_settings()
        await matrix_module.run_matrix(
            settings,
            generators=["gen-a", "gen-b"],
            evaluators=["eval-x", "eval-y"],
            seeds=1,
            output_dir=str(tmp_path),
        )

        for generator in ("gen-a", "gen-b"):
            assert captured[generator]["in_pipeline_evaluator_model"] == "eval-x"

    @pytest.mark.asyncio
    async def test_run_matrix_passes_date_range(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fixed date range should flow through to every generation run."""
        captured: dict[str, Any] = {}

        async def fake_run_single(
            settings: Any,
            generator_model: str,
            seed: int,
            **kwargs: Any,
        ) -> RunResult:
            captured[f"{generator_model}:{seed}"] = kwargs
            return build_run_result(generator_model, seed)

        async def fake_evaluate_single(
            settings: Any,
            report_markdown: str,
            evaluator_model: str,
            generator_model: str,
            seed: int,
            **kwargs: Any,
        ) -> EvalRun:
            return build_eval_run(evaluator_model, generator_model, seed, 0.8)

        monkeypatch.setattr(matrix_module, "run_single", fake_run_single)
        monkeypatch.setattr(matrix_module, "evaluate_single", fake_evaluate_single)

        settings = build_test_settings()
        await matrix_module.run_matrix(
            settings,
            generators=["gen-a"],
            evaluators=["eval-x"],
            seeds=2,
            output_dir=str(tmp_path),
            date_range="Jan 1 - Jan 8, 2026",
        )

        assert captured["gen-a:1"]["date_range"] == "Jan 1 - Jan 8, 2026"
        assert captured["gen-a:2"]["date_range"] == "Jan 1 - Jan 8, 2026"

    @pytest.mark.asyncio
    async def test_run_matrix_validates_arguments(self, tmp_path: Any) -> None:
        """Should reject empty generators/evaluators and seeds < 1."""
        settings = build_test_settings()

        with pytest.raises(ValueError, match="generators"):
            await matrix_module.run_matrix(
                settings, generators=[], evaluators=["eval-x"], seeds=1,
                output_dir=str(tmp_path),
            )
        with pytest.raises(ValueError, match="evaluators"):
            await matrix_module.run_matrix(
                settings, generators=["gen-a"], evaluators=[], seeds=1,
                output_dir=str(tmp_path),
            )
        with pytest.raises(ValueError, match="seeds"):
            await matrix_module.run_matrix(
                settings, generators=["gen-a"], evaluators=["eval-x"], seeds=0,
                output_dir=str(tmp_path),
            )
