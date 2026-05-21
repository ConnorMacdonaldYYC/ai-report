"""Integration tests for the orchestrator pipeline.

These tests use FunctionModel to mock agent responses
without making real API calls.
"""

from src.orchestrator import _build_revision_feedback, _compute_average_score
from src.schemas import DimensionScore, EvalResult


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