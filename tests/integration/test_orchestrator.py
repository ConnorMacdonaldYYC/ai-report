"""Integration tests for the orchestrator pipeline.

These tests use FunctionModel to mock agent responses
without making real API calls.
"""

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
from pydantic_ai import ModelMessage, ModelResponse, TextPart
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.function import AgentInfo, FunctionModel

from src.agents import (
    coding_agents_agent,
    community_news_agent,
    evaluator_agent,
    industry_overview_agent,
    manager_agent,
    research_agent,
)
from src.config import Settings
from src.orchestrator import _build_revision_feedback, _compute_average_score, run_report
from src.schemas import DimensionScore, EvalResult, ReportOutput, SectionResult, Source

SECTION_NAMES = [
    "Industry Overview",
    "Research Updates",
    "Community Updates",
    "Coding Agents",
]

ModelFn = Callable[[list[ModelMessage], AgentInfo], ModelResponse]


def _make_section_fn(name: str) -> ModelFn:
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


def _make_failing_fn() -> ModelFn:
    """Return a FunctionModel callback that raises UsageLimitExceeded."""

    def fail_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise UsageLimitExceeded(
            "The next request would exceed the request_limit of 50"
        )

    return fail_fn


@contextmanager
def _override_all_agents(
    sub_agent_fns: list[ModelFn],
    manager_fn: ModelFn,
    evaluator_fn: ModelFn,
) -> Iterator[None]:
    """Override all agents with FunctionModel mocks.

    Args:
        sub_agent_fns: List of 4 FunctionModel callbacks for sub-agents.
        manager_fn: FunctionModel callback for the manager agent.
        evaluator_fn: FunctionModel callback for the evaluator agent.
    """
    sub_agents = [
        industry_overview_agent,
        research_agent,
        community_news_agent,
        coding_agents_agent,
    ]
    overrides = [
        agent.override(model=FunctionModel(fn))
        for agent, fn in zip(sub_agents, sub_agent_fns, strict=True)
    ]
    overrides.append(manager_agent.override(model=FunctionModel(manager_fn)))
    overrides.append(evaluator_agent.override(model=FunctionModel(evaluator_fn)))

    for ctx in overrides:
        ctx.__enter__()
    try:
        yield
    finally:
        for ctx in reversed(overrides):
            ctx.__exit__(None, None, None)


def _default_sub_agent_fns() -> list[ModelFn]:
    """Return default section FunctionModel callbacks for all 4 sub-agents."""
    return [_make_section_fn(name) for name in SECTION_NAMES]


class TestComputeAverageScore:
    """Tests for the average score computation."""

    def test_average_of_dimensions(self) -> None:
        """Should compute the average of all dimension scores."""
        dimensions = [
            DimensionScore(
                dimension="Relevance",
                score=0.9,
                justification="",
                improvement_suggestions=[],
            ),
            DimensionScore(
                dimension="Coverage",
                score=0.8,
                justification="",
                improvement_suggestions=[],
            ),
            DimensionScore(
                dimension="Insight",
                score=0.7,
                justification="",
                improvement_suggestions=[],
            ),
            DimensionScore(
                dimension="Readability",
                score=0.85,
                justification="",
                improvement_suggestions=[],
            ),
        ]
        result = _compute_average_score(dimensions)
        assert abs(result - 0.8125) < 0.001

    def test_empty_dimensions(self) -> None:
        """Should return 0.0 for empty dimensions."""
        result = _compute_average_score([])
        assert result == 0.0


class TestBuildRevisionFeedback:
    """Tests for the revision feedback builder."""

    def test_builds_structured_feedback(self) -> None:
        """Should build structured feedback from eval results."""
        eval_result = EvalResult(
            dimensions=[
                DimensionScore(
                    dimension="Relevance",
                    score=0.5,
                    justification="Missing key topics.",
                    improvement_suggestions=["Add more recent updates."],
                ),
                DimensionScore(
                    dimension="Coverage",
                    score=0.6,
                    justification="Some sections thin.",
                    improvement_suggestions=["Expand coding agents section."],
                ),
            ],
            overall_pass=False,
            summary="Needs improvement.",
        )
        feedback = _build_revision_feedback(eval_result)
        assert "Relevance" in feedback
        assert "0.50" in feedback
        assert "Add more recent updates" in feedback
        assert "Coverage" in feedback

    def test_feedback_includes_all_dimensions(self) -> None:
        """Should include all dimensions in the feedback."""
        eval_result = EvalResult(
            dimensions=[
                DimensionScore(
                    dimension="Relevance",
                    score=0.8,
                    justification="Good.",
                    improvement_suggestions=[],
                ),
                DimensionScore(
                    dimension="Coverage",
                    score=0.7,
                    justification="OK.",
                    improvement_suggestions=[],
                ),
                DimensionScore(
                    dimension="Insight",
                    score=0.6,
                    justification="Fair.",
                    improvement_suggestions=[],
                ),
                DimensionScore(
                    dimension="Readability",
                    score=0.9,
                    justification="Great.",
                    improvement_suggestions=[],
                ),
            ],
            overall_pass=True,
            summary="Good report.",
        )
        feedback = _build_revision_feedback(eval_result)
        assert "Relevance" in feedback
        assert "Coverage" in feedback
        assert "Insight" in feedback
        assert "Readability" in feedback


class TestRunReport:
    """Integration tests for the full run_report pipeline using FunctionModel."""

    @pytest.mark.asyncio
    async def test_run_report_passes_evaluation(self, tmp_path: str) -> None:
        """Should produce a report that passes evaluation on first try."""

        def manager_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            output = ReportOutput(
                content="# AI Industry Weekly - Test\n\n"
                "## Industry Overview\n\nTest content [1]",
                sources=[
                    Source(
                        url="https://example.com",
                        title="Test Source",
                        source_type="web_search",
                    ),
                ],
            )
            return ModelResponse(
                parts=[TextPart(content=output.model_dump_json())]
            )

        def evaluator_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            output = EvalResult(
                dimensions=[
                    DimensionScore(
                        dimension="Relevance",
                        score=0.9,
                        justification="Timely topics.",
                        improvement_suggestions=[],
                    ),
                    DimensionScore(
                        dimension="Coverage",
                        score=0.85,
                        justification="All sections covered.",
                        improvement_suggestions=[],
                    ),
                    DimensionScore(
                        dimension="Insight",
                        score=0.8,
                        justification="Good analysis.",
                        improvement_suggestions=[],
                    ),
                    DimensionScore(
                        dimension="Readability",
                        score=0.9,
                        justification="Well-structured.",
                        improvement_suggestions=[],
                    ),
                ],
                overall_pass=True,
                summary="Solid newsletter.",
            )
            return ModelResponse(
                parts=[TextPart(content=output.model_dump_json())]
            )

        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            report_manager_model="test-model",
            sub_agent_model="test-model",
            evaluator_model="test-model",
            output_dir=str(tmp_path),
            max_revision_cycles=0,
        )

        with _override_all_agents(
            _default_sub_agent_fns(), manager_fn, evaluator_fn
        ):
            result = await run_report(settings)

        assert result.eval_passed is True
        assert result.eval_score > 0.7
        assert result.revision_count == 0
        assert "AI Industry Weekly" in result.report_markdown
        assert len(result.sources) >= 1

        output_files = os.listdir(str(tmp_path))
        md_files = [f for f in output_files if f.endswith(".md")]
        assert len(md_files) >= 1

    @pytest.mark.asyncio
    async def test_run_report_fails_evaluation(self, tmp_path: str) -> None:
        """Should produce a report that fails evaluation when scores are low."""

        def manager_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            output = ReportOutput(
                content="# AI Industry Weekly - Test\n\n"
                "## Industry Overview\n\nThin content",
                sources=[],
            )
            return ModelResponse(
                parts=[TextPart(content=output.model_dump_json())]
            )

        def evaluator_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            output = EvalResult(
                dimensions=[
                    DimensionScore(
                        dimension="Relevance",
                        score=0.3,
                        justification="Off-topic.",
                        improvement_suggestions=[
                            "Focus on recent AI developments."
                        ],
                    ),
                    DimensionScore(
                        dimension="Coverage",
                        score=0.2,
                        justification="Missing sections.",
                        improvement_suggestions=["Add all required sections."],
                    ),
                    DimensionScore(
                        dimension="Insight",
                        score=0.1,
                        justification="No analysis.",
                        improvement_suggestions=["Add deeper analysis."],
                    ),
                    DimensionScore(
                        dimension="Readability",
                        score=0.4,
                        justification="Poorly structured.",
                        improvement_suggestions=["Improve formatting."],
                    ),
                ],
                overall_pass=False,
                summary="Needs major improvement.",
            )
            return ModelResponse(
                parts=[TextPart(content=output.model_dump_json())]
            )

        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            report_manager_model="test-model",
            sub_agent_model="test-model",
            evaluator_model="test-model",
            output_dir=str(tmp_path),
            max_revision_cycles=0,
        )

        with _override_all_agents(
            _default_sub_agent_fns(), manager_fn, evaluator_fn
        ):
            result = await run_report(settings)

        assert result.eval_passed is False
        assert result.eval_score < 0.7
        assert result.revision_count == 0

    @pytest.mark.asyncio
    async def test_run_report_with_revision(self, tmp_path: str) -> None:
        """Should revise the report when evaluation fails on first cycle."""

        call_count = 0

        def manager_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            output = ReportOutput(
                content=f"# AI Industry Weekly - Test (revision {call_count})\n\n"
                f"## Content",
                sources=[
                    Source(
                        url="https://example.com",
                        title=f"Source {call_count}",
                        source_type="web_search",
                    ),
                ],
            )
            return ModelResponse(
                parts=[TextPart(content=output.model_dump_json())]
            )

        eval_call_count = 0

        def evaluator_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            nonlocal eval_call_count
            eval_call_count += 1
            if eval_call_count == 1:
                output = EvalResult(
                    dimensions=[
                        DimensionScore(
                            dimension="Relevance",
                            score=0.4,
                            justification="Weak.",
                            improvement_suggestions=["Improve relevance."],
                        ),
                        DimensionScore(
                            dimension="Coverage",
                            score=0.5,
                            justification="Missing sections.",
                            improvement_suggestions=["Add missing sections."],
                        ),
                        DimensionScore(
                            dimension="Insight",
                            score=0.3,
                            justification="No analysis.",
                            improvement_suggestions=["Add analysis."],
                        ),
                        DimensionScore(
                            dimension="Readability",
                            score=0.6,
                            justification="OK.",
                            improvement_suggestions=[],
                        ),
                    ],
                    overall_pass=False,
                    summary="Needs revision.",
                )
            else:
                output = EvalResult(
                    dimensions=[
                        DimensionScore(
                            dimension="Relevance",
                            score=0.85,
                            justification="Good.",
                            improvement_suggestions=[],
                        ),
                        DimensionScore(
                            dimension="Coverage",
                            score=0.8,
                            justification="All sections present.",
                            improvement_suggestions=[],
                        ),
                        DimensionScore(
                            dimension="Insight",
                            score=0.75,
                            justification="Good analysis.",
                            improvement_suggestions=[],
                        ),
                        DimensionScore(
                            dimension="Readability",
                            score=0.9,
                            justification="Well-structured.",
                            improvement_suggestions=[],
                        ),
                    ],
                    overall_pass=True,
                    summary="Good after revision.",
                )
            return ModelResponse(
                parts=[TextPart(content=output.model_dump_json())]
            )

        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            report_manager_model="test-model",
            sub_agent_model="test-model",
            evaluator_model="test-model",
            output_dir=str(tmp_path),
            max_revision_cycles=1,
        )

        with _override_all_agents(
            _default_sub_agent_fns(), manager_fn, evaluator_fn
        ):
            result = await run_report(settings)

        assert result.eval_passed is True
        assert result.revision_count == 1
        assert call_count == 2
        assert eval_call_count == 2

    @pytest.mark.asyncio
    async def test_run_report_graceful_usage_limit(self, tmp_path: str) -> None:
        """Should write a fallback report when the manager hits the usage limit."""

        def evaluator_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            output = EvalResult(
                dimensions=[
                    DimensionScore(
                        dimension="Relevance",
                        score=0.5,
                        justification="OK.",
                        improvement_suggestions=[],
                    ),
                ],
                overall_pass=False,
                summary="Partial report.",
            )
            return ModelResponse(
                parts=[TextPart(content=output.model_dump_json())]
            )

        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            report_manager_model="test-model",
            sub_agent_model="test-model",
            evaluator_model="test-model",
            output_dir=str(tmp_path),
            max_revision_cycles=0,
        )

        with _override_all_agents(
            _default_sub_agent_fns(), _make_failing_fn(), evaluator_fn
        ):
            result = await run_report(settings)

        assert result.eval_passed is False
        assert result.eval_score == 0.0
        assert result.revision_count == 0
        assert "AI Industry Weekly" in result.report_markdown
        assert "Industry Overview" in result.report_markdown
        assert "Research Updates" in result.report_markdown

        output_files = os.listdir(str(tmp_path))
        md_files = [f for f in output_files if f.endswith(".md")]
        assert len(md_files) >= 1

    @pytest.mark.asyncio
    async def test_run_report_partial_sections_usage_limit(
        self, tmp_path: str
    ) -> None:
        """Should produce a fallback report when some sub-agents hit the limit."""

        def evaluator_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            output = EvalResult(
                dimensions=[
                    DimensionScore(
                        dimension="Relevance",
                        score=0.5,
                        justification="OK.",
                        improvement_suggestions=[],
                    ),
                ],
                overall_pass=False,
                summary="Partial report.",
            )
            return ModelResponse(
                parts=[TextPart(content=output.model_dump_json())]
            )

        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            report_manager_model="test-model",
            sub_agent_model="test-model",
            evaluator_model="test-model",
            output_dir=str(tmp_path),
            max_revision_cycles=0,
        )

        # First two sub-agents succeed, last two hit the usage limit
        sub_fns = [
            _make_section_fn("Industry Overview"),
            _make_section_fn("Research Updates"),
            _make_failing_fn(),
            _make_failing_fn(),
        ]

        with _override_all_agents(sub_fns, _make_failing_fn(), evaluator_fn):
            result = await run_report(settings)

        assert "Industry Overview" in result.report_markdown
        assert "Research Updates" in result.report_markdown
        assert "usage limit reached" in result.report_markdown.lower()
        assert result.eval_passed is False

        output_files = os.listdir(str(tmp_path))
        md_files = [f for f in output_files if f.endswith(".md")]
        assert len(md_files) >= 1

    @pytest.mark.asyncio
    async def test_run_report_email_disabled(self, tmp_path: str) -> None:
        """Should produce a report normally when email is disabled (default)."""

        def manager_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            output = ReportOutput(
                content="# AI Industry Weekly - Test\n\n"
                "## Industry Overview\n\nTest content [1]",
                sources=[
                    Source(
                        url="https://example.com",
                        title="Test Source",
                        source_type="web_search",
                    ),
                ],
            )
            return ModelResponse(
                parts=[TextPart(content=output.model_dump_json())]
            )

        def evaluator_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            output = EvalResult(
                dimensions=[
                    DimensionScore(
                        dimension="Relevance",
                        score=0.9,
                        justification="Timely topics.",
                        improvement_suggestions=[],
                    ),
                    DimensionScore(
                        dimension="Coverage",
                        score=0.85,
                        justification="All sections covered.",
                        improvement_suggestions=[],
                    ),
                    DimensionScore(
                        dimension="Insight",
                        score=0.8,
                        justification="Good analysis.",
                        improvement_suggestions=[],
                    ),
                    DimensionScore(
                        dimension="Readability",
                        score=0.9,
                        justification="Well-structured.",
                        improvement_suggestions=[],
                    ),
                ],
                overall_pass=True,
                summary="Solid newsletter.",
            )
            return ModelResponse(
                parts=[TextPart(content=output.model_dump_json())]
            )

        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            output_dir=str(tmp_path),
            max_revision_cycles=0,
            # email is disabled by default — explicit for clarity
            email_enabled=False,
        )

        with _override_all_agents(
            _default_sub_agent_fns(), manager_fn, evaluator_fn
        ):
            result = await run_report(settings)

        assert result.eval_passed is True
        assert result.eval_score > 0.7

        # Only .md files should exist — no HTML dry-run file
        output_files = os.listdir(str(tmp_path))
        html_files = [f for f in output_files if f.endswith(".html")]
        assert len(html_files) == 0

    @pytest.mark.asyncio
    async def test_run_report_with_dry_run_email(self, tmp_path: str) -> None:
        """Should write an HTML email preview file when dry-run is enabled."""

        def manager_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            output = ReportOutput(
                content="# AI Industry Weekly - Test\n\n"
                "## Industry Overview\n\nTest content [1]",
                sources=[
                    Source(
                        url="https://example.com",
                        title="Test Source",
                        source_type="web_search",
                    ),
                ],
            )
            return ModelResponse(
                parts=[TextPart(content=output.model_dump_json())]
            )

        def evaluator_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            output = EvalResult(
                dimensions=[
                    DimensionScore(
                        dimension="Relevance",
                        score=0.9,
                        justification="Timely topics.",
                        improvement_suggestions=[],
                    ),
                    DimensionScore(
                        dimension="Coverage",
                        score=0.85,
                        justification="All sections covered.",
                        improvement_suggestions=[],
                    ),
                    DimensionScore(
                        dimension="Insight",
                        score=0.8,
                        justification="Good analysis.",
                        improvement_suggestions=[],
                    ),
                    DimensionScore(
                        dimension="Readability",
                        score=0.9,
                        justification="Well-structured.",
                        improvement_suggestions=[],
                    ),
                ],
                overall_pass=True,
                summary="Solid newsletter.",
            )
            return ModelResponse(
                parts=[TextPart(content=output.model_dump_json())]
            )

        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            output_dir=str(tmp_path),
            max_revision_cycles=0,
            email_enabled=True,
            email_dry_run=True,
            email_to="test@example.com",
            email_from="sender@example.com",
        )

        with _override_all_agents(
            _default_sub_agent_fns(), manager_fn, evaluator_fn
        ):
            result = await run_report(settings)

        assert result.eval_passed is True
        assert result.eval_score > 0.7

        # Should have both the .md report and the .html dry-run email
        output_files = os.listdir(str(tmp_path))
        md_files = [f for f in output_files if f.endswith(".md")]
        html_files = [f for f in output_files if f.endswith(".html")]
        assert len(md_files) >= 1
        assert len(html_files) >= 1
