"""Tests for the eval generation runner (extract_usage, run_single)."""

from pathlib import Path

import pytest
from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from src.agents import AgentBundle, build_agents
from src.config import Settings
from src.eval.results import RunResult, TokenUsage
from src.eval.runner import extract_usage, run_single
from tests.utils import (
    build_test_settings,
    default_sub_agent_fns,
    make_evaluator_fn,
    make_manager_fn,
    override_all_agents,
)


def _echo_model() -> FunctionModel:
    """Return a FunctionModel that echoes a fixed text response."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="hello world")])

    return FunctionModel(model_fn)


class TestExtractUsage:
    """Tests for extracting TokenUsage from a pydantic-ai run result."""

    async def test_extract_usage_from_agent_result(self) -> None:
        """Should return a TokenUsage with non-negative, consistent counts."""
        agent = Agent(_echo_model())
        result = await agent.run("say hello")
        usage = extract_usage(result)
        assert isinstance(usage, TokenUsage)
        assert usage.input_tokens >= 0
        assert usage.output_tokens >= 0
        assert usage.total_tokens == usage.input_tokens + usage.output_tokens

    async def test_extract_usage_accumulates_across_runs(self) -> None:
        """Should sum usage across multiple agent runs when given a list."""
        agent = Agent(_echo_model())
        result1 = await agent.run("first prompt")
        result2 = await agent.run("second prompt")
        combined = extract_usage([result1, result2])
        single1 = extract_usage(result1)
        single2 = extract_usage(result2)
        assert combined.input_tokens == single1.input_tokens + single2.input_tokens
        assert combined.output_tokens == single1.output_tokens + single2.output_tokens


class TestRunSingle:
    """Tests for running a single generation via run_single."""

    @pytest.mark.asyncio
    async def test_run_single_returns_run_result(self) -> None:
        """Should return a RunResult with populated fields."""
        settings = build_test_settings()
        generator_model = "test-generator-model"
        seed = 42

        bundle = build_agents(settings)
        with override_all_agents(
            default_sub_agent_fns(),
            make_manager_fn(),
            make_evaluator_fn(score=0.9),
            bundle=bundle,
        ):
            result = await run_single(
                settings,
                generator_model=generator_model,
                seed=seed,
                bundle=bundle,
            )

        assert isinstance(result, RunResult)
        assert result.generator_model == generator_model
        assert result.seed == seed
        assert result.report_markdown
        assert result.eval_passed is True
        assert result.tokens.total_tokens > 0
        assert result.duration_seconds > 0
        assert result.timestamp
        assert result.revision_count == 0

    @pytest.mark.asyncio
    async def test_run_single_does_not_mutate_original_settings(self) -> None:
        """Should not mutate the original settings object."""
        settings = build_test_settings()
        generator_model = "test-generator-model"
        seed = 42

        bundle = build_agents(settings)
        with override_all_agents(
            default_sub_agent_fns(),
            make_manager_fn(),
            make_evaluator_fn(score=0.9),
            bundle=bundle,
        ):
            await run_single(
                settings,
                generator_model=generator_model,
                seed=seed,
                bundle=bundle,
            )

        assert settings.sub_agent_model == "test-sub-model"
        assert settings.report_manager_model == "test-manager-model"
        assert settings.evaluator_model == "test-eval-model"

    @pytest.mark.asyncio
    async def test_run_single_clones_settings_for_agents(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should clone settings with generator model applied before building agents."""
        recorded: list[Settings] = []
        real_build_agents = build_agents

        # Build the bundle the patched factory will return and override its
        # agents so the run makes no real API calls.
        bundle = real_build_agents(build_test_settings())

        def recording_build_agents(settings: Settings) -> AgentBundle:
            recorded.append(settings)
            return bundle

        monkeypatch.setattr("src.eval.runner.build_agents", recording_build_agents)

        settings = build_test_settings()
        generator_model = "test-generator-model"
        evaluator_model = "test-in-pipeline-evaluator"

        with override_all_agents(
            default_sub_agent_fns(),
            make_manager_fn(),
            make_evaluator_fn(score=0.9),
            bundle=bundle,
        ):
            await run_single(
                settings,
                generator_model=generator_model,
                seed=7,
                in_pipeline_evaluator_model=evaluator_model,
            )

        assert len(recorded) == 1
        cloned = recorded[0]
        assert cloned.sub_agent_model == generator_model
        assert cloned.report_manager_model == generator_model
        assert cloned.evaluator_model == evaluator_model
        assert cloned.email_enabled is False

    @pytest.mark.asyncio
    async def test_run_single_writes_to_given_output_dir(self, tmp_path: Path) -> None:
        """Should write the report to the given output directory."""
        settings = build_test_settings()
        generator_model = "test-generator-model"
        seed = 42

        bundle = build_agents(settings)
        with override_all_agents(
            default_sub_agent_fns(),
            make_manager_fn(),
            make_evaluator_fn(score=0.9),
            bundle=bundle,
        ):
            await run_single(
                settings,
                generator_model=generator_model,
                seed=seed,
                bundle=bundle,
                output_dir=str(tmp_path),
            )

        md_files = list(tmp_path.glob("*.md"))
        assert len(md_files) >= 1

    @pytest.mark.asyncio
    async def test_run_single_passes_fixed_date_range(
        self, tmp_path: Path
    ) -> None:
        """Should pass the fixed date range through to the pipeline."""
        settings = build_test_settings()

        bundle = build_agents(settings)
        with override_all_agents(
            default_sub_agent_fns(),
            make_manager_fn(),
            make_evaluator_fn(score=0.9),
            bundle=bundle,
        ):
            result = await run_single(
                settings,
                generator_model="test-generator-model",
                seed=1,
                bundle=bundle,
                output_dir=str(tmp_path),
                date_range="Jan 1 - Jan 8, 2026",
            )

        assert result.date_range == "Jan 1 - Jan 8, 2026"
