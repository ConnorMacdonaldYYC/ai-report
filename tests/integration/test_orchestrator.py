"""Integration tests for the orchestrator pipeline.

These tests use FunctionModel to mock agent responses
without making real API calls.
"""

import os

import pytest

from src.agents import evaluator_agent, manager_agent
from src.config import Settings
from src.orchestrator import _build_revision_feedback, _compute_average_score, run_report
from src.schemas import DimensionScore, EvalResult, ReportOutput, Source


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
        from pydantic_ai import ModelMessage, ModelResponse, TextPart
        from pydantic_ai.models.function import AgentInfo, FunctionModel

        # Track which agent is being called
        call_count = 0

        def manager_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            output = ReportOutput(
                content="# AI Industry Weekly - Test\n\n## Industry Overview\n\nTest content [1]",
                sources=[
                    Source(
                        url="https://example.com",
                        title="Test Source",
                        source_type="web_search",
                    ),
                ],
            )
            return ModelResponse(parts=[TextPart(content=output.model_dump_json())])

        def evaluator_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
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
            return ModelResponse(parts=[TextPart(content=output.model_dump_json())])

        settings = Settings(
            model_provider="openai",
            openai_api_key="test-key",
            report_manager_model="test-model",
            sub_agent_model="test-model",
            evaluator_model="test-model",
            output_dir=str(tmp_path),
            max_revision_cycles=0,
        )

        with (
            manager_agent.override(model=FunctionModel(manager_fn)),
            evaluator_agent.override(model=FunctionModel(evaluator_fn)),
        ):
            result = await run_report(settings)

        assert result.eval_passed is True
        assert result.eval_score > 0.7
        assert result.revision_count == 0
        assert "AI Industry Weekly" in result.report_markdown
        assert len(result.sources) >= 1

        # Verify the output file was created
        output_files = os.listdir(str(tmp_path))
        md_files = [f for f in output_files if f.endswith(".md")]
        assert len(md_files) >= 1

    @pytest.mark.asyncio
    async def test_run_report_fails_evaluation(self, tmp_path: str) -> None:
        """Should produce a report that fails evaluation when scores are low."""
        from pydantic_ai import ModelMessage, ModelResponse, TextPart
        from pydantic_ai.models.function import AgentInfo, FunctionModel

        def manager_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            output = ReportOutput(
                content="# AI Industry Weekly - Test\n\n## Industry Overview\n\nThin content",
                sources=[],
            )
            return ModelResponse(parts=[TextPart(content=output.model_dump_json())])

        def evaluator_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            output = EvalResult(
                dimensions=[
                    DimensionScore(
                        dimension="Relevance",
                        score=0.3,
                        justification="Off-topic.",
                        improvement_suggestions=["Focus on recent AI developments."],
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
            return ModelResponse(parts=[TextPart(content=output.model_dump_json())])

        settings = Settings(
            model_provider="openai",
            openai_api_key="test-key",
            report_manager_model="test-model",
            sub_agent_model="test-model",
            evaluator_model="test-model",
            output_dir=str(tmp_path),
            max_revision_cycles=0,
        )

        with (
            manager_agent.override(model=FunctionModel(manager_fn)),
            evaluator_agent.override(model=FunctionModel(evaluator_fn)),
        ):
            result = await run_report(settings)

        assert result.eval_passed is False
        assert result.eval_score < 0.7
        assert result.revision_count == 0

    @pytest.mark.asyncio
    async def test_run_report_with_revision(self, tmp_path: str) -> None:
        """Should revise the report when evaluation fails on first cycle."""
        from pydantic_ai import ModelMessage, ModelResponse, TextPart
        from pydantic_ai.models.function import AgentInfo, FunctionModel

        call_count = 0

        def manager_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            output = ReportOutput(
                content=f"# AI Industry Weekly - Test (revision {call_count})\n\n## Content",
                sources=[
                    Source(
                        url="https://example.com",
                        title=f"Source {call_count}",
                        source_type="web_search",
                    ),
                ],
            )
            return ModelResponse(parts=[TextPart(content=output.model_dump_json())])

        eval_call_count = 0

        def evaluator_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal eval_call_count
            eval_call_count += 1
            # Fail first evaluation, pass second
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
            return ModelResponse(parts=[TextPart(content=output.model_dump_json())])

        settings = Settings(
            model_provider="openai",
            openai_api_key="test-key",
            report_manager_model="test-model",
            sub_agent_model="test-model",
            evaluator_model="test-model",
            output_dir=str(tmp_path),
            max_revision_cycles=1,
        )

        with (
            manager_agent.override(model=FunctionModel(manager_fn)),
            evaluator_agent.override(model=FunctionModel(evaluator_fn)),
        ):
            result = await run_report(settings)

        assert result.eval_passed is True
        assert result.revision_count == 1
        assert call_count == 2  # Initial + 1 revision
        assert eval_call_count == 2  # Failed + passed