"""Tests for the single evaluation runner (evaluate_single).

evaluate_single evaluates a fixed report with one evaluator model and
returns an EvalRun with score, pass flag, dimensions, tokens, and timing.
"""

from contextlib import AbstractContextManager

import pytest
from pydantic_ai import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from src.agents import AgentBundle, build_agents
from src.config import Settings
from src.eval.evaluator import evaluate_single
from src.eval.results import EvalRun
from src.schemas import DimensionScore, EvalResult
from tests.utils import (
    ModelFn,
    build_dimension,
    build_test_settings,
    make_evaluator_fn,
)


def _make_eval_fn(
    dimensions: list[DimensionScore], overall_pass: bool
) -> ModelFn:
    """Return a FunctionModel callback producing the given EvalResult.

    Args:
        dimensions: Dimension scores to return.
        overall_pass: Pass flag to return.

    Returns:
        A FunctionModel callback for use with agent.override().
    """

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        output = EvalResult(
            dimensions=dimensions,
            overall_pass=overall_pass,
            summary="Test evaluation.",
        )
        return ModelResponse(parts=[TextPart(content=output.model_dump_json())])

    return fn


async def _run_with_evaluator_fn(
    fn: ModelFn,
    *,
    evaluator_model: str = "claude-eval",
    generator_model: str = "gpt-4o",
    seed: int = 1,
) -> EvalRun:
    """Run evaluate_single with a bundle whose evaluator is overridden.

    Args:
        fn: FunctionModel callback for the evaluator agent.
        evaluator_model: Evaluator model name to pass through.
        generator_model: Generator model name to pass through.
        seed: Seed to pass through.

    Returns:
        The EvalRun produced by evaluate_single.
    """
    bundle = build_agents(build_test_settings())
    with bundle.evaluator.override(model=FunctionModel(fn)):
        return await evaluate_single(
            build_test_settings(),
            "# Test report",
            evaluator_model=evaluator_model,
            generator_model=generator_model,
            seed=seed,
            bundle=bundle,
        )


class TestEvaluateSingle:
    """Tests for the evaluate_single runner."""

    @pytest.mark.asyncio
    async def test_evaluate_single_returns_eval_run(self) -> None:
        """Should return a complete EvalRun with score, tokens, and timing."""
        result = await _run_with_evaluator_fn(
            make_evaluator_fn(score=0.8),
            evaluator_model="claude-eval",
            generator_model="gpt-4o",
            seed=3,
        )

        assert result.evaluator_model == "claude-eval"
        assert result.generator_model == "gpt-4o"
        assert result.seed == 3
        assert result.score == 0.8
        assert result.eval_passed is True
        assert len(result.dimensions) == 1
        assert result.tokens.total_tokens > 0
        assert result.duration_seconds > 0
        assert result.timestamp != ""

    @pytest.mark.asyncio
    async def test_evaluate_single_averages_dimensions(self) -> None:
        """Should average all dimension scores into the overall score."""
        dimensions = [
            build_dimension("Relevance", 0.9),
            build_dimension("Coverage", 0.8),
            build_dimension("Insight", 0.7),
            build_dimension("Readability", 0.9),
        ]
        result = await _run_with_evaluator_fn(
            _make_eval_fn(dimensions, overall_pass=True)
        )

        assert result.score == pytest.approx(0.825)

    @pytest.mark.asyncio
    async def test_evaluate_single_empty_dimensions_score_zero(self) -> None:
        """Should score 0.0 when the evaluator returns no dimensions."""
        result = await _run_with_evaluator_fn(
            _make_eval_fn([], overall_pass=False)
        )

        assert result.score == 0.0
        assert result.eval_passed is False

    @pytest.mark.asyncio
    async def test_evaluate_single_pass_flag_from_overall_pass(self) -> None:
        """Should take the pass flag from the evaluator, not a score threshold."""
        dimensions = [build_dimension("Relevance", 0.5)]
        result = await _run_with_evaluator_fn(
            _make_eval_fn(dimensions, overall_pass=True)
        )

        assert result.score == 0.5
        assert result.eval_passed is True

    @pytest.mark.asyncio
    async def test_evaluate_single_clones_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should clone settings with the evaluator model and email disabled."""
        original = build_test_settings(
            evaluator_model="original-eval", email_enabled=True
        )
        recorded: list[Settings] = []
        entered_contexts: list[AbstractContextManager[None]] = []

        def fake_build_agents(cloned: Settings) -> AgentBundle:
            recorded.append(cloned)
            bundle = build_agents(cloned)
            ctx = bundle.evaluator.override(
                model=FunctionModel(make_evaluator_fn(score=0.8))
            )
            ctx.__enter__()
            entered_contexts.append(ctx)
            return bundle

        monkeypatch.setattr("src.eval.evaluator.build_agents", fake_build_agents)

        try:
            await evaluate_single(
                original,
                "# Test report",
                evaluator_model="claude-eval",
                generator_model="gpt-4o",
                seed=5,
            )
        finally:
            for ctx in entered_contexts:
                ctx.__exit__(None, None, None)

        assert len(recorded) == 1
        assert recorded[0].evaluator_model == "claude-eval"
        assert recorded[0].email_enabled is False
        # The original settings object must be unmutated.
        assert original.evaluator_model == "original-eval"
        assert original.email_enabled is True