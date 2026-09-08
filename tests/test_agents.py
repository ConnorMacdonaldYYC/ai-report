"""Tests for the agent factory (build_agents / AgentBundle).

Verifies that agents can be built from arbitrary Settings in-process,
with fresh instances per call and tools correctly attached.
"""

from collections.abc import Callable
from typing import Any

import pytest
from pydantic_ai import Agent, ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo

from src.agents import AgentBundle, build_agents
from src.schemas import EvalResult, ReportOutput, SectionResult
from tests.utils import build_test_settings

ModelFn = Callable[[list[ModelMessage], AgentInfo], ModelResponse]


class TestAgentBundle:
    """Tests for the build_agents factory and AgentBundle structure."""

    def test_build_agents_returns_bundle_with_all_agents(self) -> None:
        """Should return a bundle containing all six agents."""
        bundle = build_agents(build_test_settings())
        assert isinstance(bundle, AgentBundle)
        assert bundle.industry_overview is not None
        assert bundle.research is not None
        assert bundle.community_news is not None
        assert bundle.coding_agents is not None
        assert bundle.manager is not None
        assert bundle.evaluator is not None

    def test_bundle_agents_are_agent_instances(self) -> None:
        """Every bundle field should be a pydantic-ai Agent."""
        bundle = build_agents(build_test_settings())
        agents = [
            bundle.industry_overview,
            bundle.research,
            bundle.community_news,
            bundle.coding_agents,
            bundle.manager,
            bundle.evaluator,
        ]
        assert all(isinstance(a, Agent) for a in agents)

    def test_bundle_agents_are_distinct_instances_per_call(self) -> None:
        """Each build_agents call should produce fresh agent instances."""
        b1 = build_agents(build_test_settings())
        b2 = build_agents(build_test_settings())
        assert b1.manager is not b2.manager
        assert b1.industry_overview is not b2.industry_overview
        assert b1.evaluator is not b2.evaluator

    def test_bundle_uses_settings_model_strings(self) -> None:
        """Agents should be built from the provided settings' model strings."""
        settings = build_test_settings(
            report_manager_model="mgr-x",
            sub_agent_model="sub-x",
            evaluator_model="eval-x",
        )
        bundle = build_agents(settings)
        # defer_model_check=True means the model string is stored as given.
        assert bundle.manager.model == "openai:mgr-x"
        assert bundle.industry_overview.model == "openai:sub-x"
        assert bundle.evaluator.model == "openai:eval-x"


class TestBuildAgentsTools:
    """Functional tests verifying tools are attached to factory-built agents."""

    @pytest.mark.asyncio
    async def test_sub_agent_tool_is_attached_and_callable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Industry overview agent should have a working read_rss_feeds tool."""
        calls: list[str] = []

        async def fake_rss() -> str:
            calls.append("rss")
            return "fake rss content"

        monkeypatch.setattr("src.agents.rss_reader", fake_rss)

        bundle = build_agents(build_test_settings())
        section = SectionResult(content="test content [1]", sources=[])
        model = _tool_then_output_model("read_rss_feeds", section)

        with bundle.industry_overview.override(model=model):
            result = await bundle.industry_overview.run("generate section")

        assert calls == ["rss"]
        assert isinstance(result.output, SectionResult)
        assert result.output.content == "test content [1]"

    @pytest.mark.asyncio
    async def test_research_agent_tool_is_attached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Research agent should have a working search_arxiv tool."""
        calls: list[str] = []

        async def fake_arxiv(query: str, max_results: int = 10) -> str:
            calls.append(query)
            return "fake arxiv results"

        monkeypatch.setattr("src.agents.arxiv_search", fake_arxiv)

        bundle = build_agents(build_test_settings())
        section = SectionResult(content="paper summary [1]", sources=[])
        model = _tool_then_output_model("search_arxiv", section, {"query": "transformers"})

        with bundle.research.override(model=model):
            result = await bundle.research.run("generate section")

        assert calls == ["transformers"]
        assert isinstance(result.output, SectionResult)


def _tool_then_output_model(
    tool_name: str,
    output: SectionResult | ReportOutput | EvalResult,
    tool_args: dict[str, Any] | None = None,
) -> Any:
    """Return a FunctionModel that calls a tool, then returns structured output."""
    from pydantic_ai import TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name, tool_args or {})]
            )
        return ModelResponse(parts=[TextPart(content=output.model_dump_json())])

    return FunctionModel(model_fn)
