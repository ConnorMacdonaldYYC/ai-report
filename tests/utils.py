"""FunctionModel test helpers for testing pydantic-ai agents.

Provides utilities for creating test doubles that simulate agent responses
without making real API calls, following the mining-report testing pattern.

Also provides shared builders for eval-framework result schemas so test
files don't duplicate fixture construction.
"""

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any

from pydantic_ai import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from src.agents import AgentBundle
from src.config import Settings
from src.eval.results import EvalRun, RunResult
from src.schemas import (
    DimensionScore,
    EvalResult,
    ReportOutput,
    SectionResult,
    Source,
    TokenUsage,
)

# FunctionModel accepts sync and async callbacks.
ModelFn = Callable[[list[ModelMessage], AgentInfo], ModelResponse | Awaitable[ModelResponse]]


def create_simple_model(response: str) -> FunctionModel:
    """Create a FunctionModel that always returns the given text response.

    Args:
        response: The text response to return.

    Returns:
        A FunctionModel instance for use with agent.override().
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=response)])

    return FunctionModel(model_fn)


def create_tool_calling_model(
    tool_name: str,
    tool_args: dict[str, Any],
    final_response: str,
) -> FunctionModel:
    """Create a FunctionModel that first calls a tool, then returns a final response.

    Args:
        tool_name: Name of the tool to call.
        tool_args: Arguments for the tool call.
        final_response: The text response after the tool call.

    Returns:
        A FunctionModel instance for use with agent.override().
    """
    from pydantic_ai import ToolCallPart

    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name, tool_args)])
        return ModelResponse(parts=[TextPart(content=final_response)])

    return FunctionModel(model_fn)


def create_sequential_model(responses: list[str]) -> FunctionModel:
    """Create a FunctionModel that returns responses in sequence.

    Args:
        responses: List of text responses to return in order.

    Returns:
        A FunctionModel instance for use with agent.override().
    """
    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        return ModelResponse(parts=[TextPart(content=responses[idx])])

    return FunctionModel(model_fn)


# ── Pipeline agent FunctionModel factories ──────────────────────────────────


def make_section_fn(name: str) -> ModelFn:
    """Return a FunctionModel callback that produces a SectionResult."""

    def section_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        output = SectionResult(
            content=f"### {name}\n\n- Test content for {name} [1]",
            sources=[
                Source(
                    url=f"https://example.com/{name.lower().replace(' ', '-')}",
                    title=f"{name} Source",
                    source_type="web_search",
                ),
            ],
        )
        return ModelResponse(parts=[TextPart(content=output.model_dump_json())])

    return section_fn


def make_failing_fn() -> ModelFn:
    """Return a FunctionModel callback that raises UsageLimitExceeded."""

    def fail_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        from pydantic_ai.exceptions import UsageLimitExceeded

        raise UsageLimitExceeded(
            "The next request would exceed the request_limit of 50"
        )

    return fail_fn


def make_manager_fn(content: str | None = None) -> ModelFn:
    """Return a FunctionModel callback that produces a ReportOutput.

    Args:
        content: Optional report markdown. Defaults to a simple test report.
    """

    def manager_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        output = ReportOutput(
            content=content
            or "# AI Industry Weekly - Test\n\n## Industry Overview\n\nTest content [1]",
            sources=[
                Source(
                    url="https://example.com",
                    title="Test Source",
                    source_type="web_search",
                ),
            ],
        )
        return ModelResponse(parts=[TextPart(content=output.model_dump_json())])

    return manager_fn


def make_evaluator_fn(
    score: float = 0.9, overall_pass: bool | None = None
) -> ModelFn:
    """Return a FunctionModel callback that produces an EvalResult.

    Args:
        score: Score for every dimension.
        overall_pass: Explicit pass flag. Defaults to score >= 0.7.
    """

    def evaluator_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        passed = score >= 0.7 if overall_pass is None else overall_pass
        output = EvalResult(
            dimensions=[
                DimensionScore(
                    dimension="Relevance",
                    score=score,
                    justification="Test justification.",
                    improvement_suggestions=[],
                ),
            ],
            overall_pass=passed,
            summary="Test evaluation.",
        )
        return ModelResponse(parts=[TextPart(content=output.model_dump_json())])

    return evaluator_fn


def default_sub_agent_fns() -> list[ModelFn]:
    """Return section FunctionModel callbacks for all 4 sub-agents."""
    names = [
        "Industry Overview",
        "Research Updates",
        "Community Updates",
        "Coding Agents",
    ]
    return [make_section_fn(name) for name in names]


@contextmanager
def override_all_agents(
    sub_agent_fns: list[ModelFn],
    manager_fn: ModelFn,
    evaluator_fn: ModelFn,
    bundle: AgentBundle | None = None,
) -> Iterator[None]:
    """Override all agents with FunctionModel mocks.

    Args:
        sub_agent_fns: List of 4 FunctionModel callbacks for sub-agents.
        manager_fn: FunctionModel callback for the manager agent.
        evaluator_fn: FunctionModel callback for the evaluator agent.
        bundle: Optional AgentBundle whose agents should be overridden.
            Defaults to the module-level agent singletons.
    """
    from src.agents import (
        coding_agents_agent,
        community_news_agent,
        evaluator_agent,
        industry_overview_agent,
        manager_agent,
        research_agent,
    )

    if bundle is None:
        sub_agents = [
            industry_overview_agent,
            research_agent,
            community_news_agent,
            coding_agents_agent,
        ]
        manager = manager_agent
        evaluator = evaluator_agent
    else:
        sub_agents = [
            bundle.industry_overview,
            bundle.research,
            bundle.community_news,
            bundle.coding_agents,
        ]
        manager = bundle.manager
        evaluator = bundle.evaluator

    overrides = [
        agent.override(model=FunctionModel(fn))
        for agent, fn in zip(sub_agents, sub_agent_fns, strict=True)
    ]
    overrides.append(manager.override(model=FunctionModel(manager_fn)))
    overrides.append(evaluator.override(model=FunctionModel(evaluator_fn)))

    for ctx in overrides:
        ctx.__enter__()
    try:
        yield
    finally:
        for ctx in reversed(overrides):
            ctx.__exit__(None, None, None)


# ── Eval-framework result schema builders ───────────────────────────────────


def build_test_settings(**overrides: Any) -> Settings:
    """Return a Settings instance with test-friendly defaults.

    Args:
        **overrides: Optional field overrides.
    """
    defaults: dict[str, Any] = {
        "_env_file": None,
        "model_provider": "openai",
        "openai_api_key": "test-key",
        "base_url": "https://test.example.com/v1",
        "report_manager_model": "test-manager-model",
        "sub_agent_model": "test-sub-model",
        "evaluator_model": "test-eval-model",
        "output_dir": "./test_output",
        "request_limit": 50,
        "rss_feeds": ["https://example.com/feed.xml"],
        "hn_search_keywords": ["AI"],
    }
    defaults.update(overrides)
    return Settings(**defaults)


def build_dimension(name: str, score: float) -> DimensionScore:
    """Return a DimensionScore with test-friendly defaults."""
    return DimensionScore(
        dimension=name,
        score=score,
        justification=f"{name} justification.",
        improvement_suggestions=[],
    )


def build_run_result(generator: str, seed: int, **overrides: object) -> RunResult:
    """Return a RunResult with test-friendly defaults.

    Args:
        generator: Generator model name.
        seed: Seed number.
        **overrides: Optional field overrides.
    """
    defaults: dict[str, object] = {
        "generator_model": generator,
        "seed": seed,
        "date_range": "Jul 19 - Jul 26, 2026",
        "report_markdown": "# AI Industry Weekly\n\n## Content [1]",
        "sources": [
            Source(
                url="https://example.com",
                title="Example Source",
                source_type="web_search",
            )
        ],
        "tokens": TokenUsage(input_tokens=1000, output_tokens=500),
        "duration_seconds": 12.5,
        "revision_count": 0,
        "eval_passed": True,
        "eval_score": 0.85,
        "timestamp": "2026-07-28T10:00:00Z",
    }
    defaults.update(overrides)
    return RunResult(**defaults)  # type: ignore[arg-type]


def build_eval_run(
    evaluator: str, generator: str, seed: int, score: float
) -> EvalRun:
    """Return an EvalRun with test-friendly defaults.

    Args:
        evaluator: Evaluator model name.
        generator: Generator model name.
        seed: Seed number.
        score: Score for every dimension.
    """
    return EvalRun(
        evaluator_model=evaluator,
        generator_model=generator,
        seed=seed,
        score=score,
        eval_passed=score >= 0.7,
        dimensions=[
            build_dimension("Relevance", score),
            build_dimension("Coverage", score),
            build_dimension("Insight", score),
            build_dimension("Readability", score),
        ],
        tokens=TokenUsage(input_tokens=200, output_tokens=100),
        duration_seconds=3.2,
        timestamp="2026-07-28T10:01:00Z",
    )
