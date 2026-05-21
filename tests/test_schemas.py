"""Tests for the schemas module."""

import pytest
from pydantic import ValidationError

from src.schemas import (
    DimensionScore,
    EvalResult,
    ReportOutput,
    ReportRequest,
    ReportResult,
    SectionResult,
    Source,
)


class TestReportRequest:
    """Tests for the ReportRequest TypedDict."""

    def test_create_report_request(self) -> None:
        """Should create a ReportRequest with date_range."""
        request: ReportRequest = {"date_range": "May 11 - May 18, 2026"}
        assert request["date_range"] == "May 11 - May 18, 2026"


class TestReportResult:
    """Tests for the ReportResult BaseModel."""

    def test_create_report_result(self) -> None:
        """Should create a ReportResult with all fields."""
        result = ReportResult(
            date_range="May 11 - May 18, 2026",
            report_markdown="# Report content",
            eval_passed=True,
            eval_score=0.85,
            revision_count=0,
        )
        assert result.date_range == "May 11 - May 18, 2026"
        assert result.eval_passed is True
        assert result.eval_score == 0.85
        assert result.revision_count == 0
        assert result.sources == []

    def test_report_result_with_sources(self) -> None:
        """Should create a ReportResult with sources."""
        sources = [
            Source(url="https://example.com", title="Example", source_type="web_search"),
        ]
        result = ReportResult(
            date_range="May 11 - May 18, 2026",
            report_markdown="# Report content",
            eval_passed=True,
            eval_score=0.85,
            revision_count=0,
            sources=sources,
        )
        assert len(result.sources) == 1
        assert result.sources[0].title == "Example"

    def test_report_result_serialization(self) -> None:
        """Should serialize and deserialize correctly."""
        result = ReportResult(
            date_range="Test Range",
            report_markdown="# Test",
            eval_passed=False,
            eval_score=0.5,
            revision_count=2,
        )
        data = result.model_dump()
        restored = ReportResult.model_validate(data)
        assert restored.date_range == result.date_range
        assert restored.revision_count == result.revision_count


class TestDimensionScore:
    """Tests for the DimensionScore Pydantic model."""

    def test_create_dimension_score(self) -> None:
        """Should create a DimensionScore with all fields."""
        score = DimensionScore(
            dimension="Relevance",
            score=0.85,
            justification="Topics are timely and significant.",
            improvement_suggestions=["Add more recent updates."],
        )
        assert score.dimension == "Relevance"
        assert score.score == 0.85
        assert len(score.improvement_suggestions) == 1

    def test_dimension_score_serialization(self) -> None:
        """Should serialize and deserialize correctly."""
        score = DimensionScore(
            dimension="Coverage",
            score=0.5,
            justification="Some gaps.",
            improvement_suggestions=["Expand section.", "Add examples."],
        )
        data = score.model_dump()
        restored = DimensionScore.model_validate(data)
        assert restored.dimension == score.dimension
        assert restored.score == score.score


class TestEvalResult:
    """Tests for the EvalResult Pydantic model."""

    def test_create_eval_result(self) -> None:
        """Should create an EvalResult with dimensions."""
        result = EvalResult(
            dimensions=[
                DimensionScore(
                    dimension="Relevance",
                    score=0.9,
                    justification="Good.",
                    improvement_suggestions=[],
                ),
                DimensionScore(
                    dimension="Coverage",
                    score=0.8,
                    justification="Good.",
                    improvement_suggestions=[],
                ),
                DimensionScore(
                    dimension="Insight",
                    score=0.7,
                    justification="OK.",
                    improvement_suggestions=[],
                ),
                DimensionScore(
                    dimension="Readability",
                    score=0.85,
                    justification="Good.",
                    improvement_suggestions=[],
                ),
            ],
            overall_pass=True,
            summary="A solid newsletter.",
        )
        assert result.overall_pass is True
        assert len(result.dimensions) == 4

    def test_eval_result_serialization(self) -> None:
        """Should serialize and deserialize correctly."""
        result = EvalResult(
            dimensions=[
                DimensionScore(
                    dimension="Relevance",
                    score=0.9,
                    justification="Good.",
                    improvement_suggestions=[],
                ),
                DimensionScore(
                    dimension="Coverage",
                    score=0.8,
                    justification="Good.",
                    improvement_suggestions=[],
                ),
                DimensionScore(
                    dimension="Insight",
                    score=0.7,
                    justification="OK.",
                    improvement_suggestions=[],
                ),
                DimensionScore(
                    dimension="Readability",
                    score=0.85,
                    justification="Good.",
                    improvement_suggestions=[],
                ),
            ],
            overall_pass=True,
            summary="A solid newsletter.",
        )
        data = result.model_dump()
        restored = EvalResult.model_validate(data)
        assert restored.overall_pass == result.overall_pass
        assert len(restored.dimensions) == 4


class TestSource:
    """Tests for the Source Pydantic model."""

    def test_create_source(self) -> None:
        """Should create a Source with all fields."""
        source = Source(
            url="https://example.com/article",
            title="Example Article",
            source_type="web_search",
        )
        assert source.url == "https://example.com/article"
        assert source.title == "Example Article"
        assert source.source_type == "web_search"

    def test_source_serialization(self) -> None:
        """Should serialize and deserialize correctly."""
        source = Source(
            url="https://arxiv.org/abs/2401.12345",
            title="Test Paper",
            source_type="arxiv",
        )
        data = source.model_dump()
        restored = Source.model_validate(data)
        assert restored.url == source.url
        assert restored.source_type == source.source_type

    def test_source_invalid_type(self) -> None:
        """Should reject invalid source_type."""
        with pytest.raises(ValidationError):
            Source(
                url="https://example.com",
                title="Bad Source",
                source_type="invalid_type",  # type: ignore[arg-type]
            )


class TestSectionResult:
    """Tests for the SectionResult Pydantic model."""

    def test_create_section_result(self) -> None:
        """Should create a SectionResult with content and sources."""
        result = SectionResult(
            content="Google announced Gemini 3.0 [1] and OpenAI released GPT-5 [2].",
            sources=[
                Source(
                    url="https://blog.google/gemini",
                    title="Gemini Blog",
                    source_type="web_search",
                ),
                Source(
                    url="https://openai.com/gpt5",
                    title="GPT-5 Release",
                    source_type="web_search",
                ),
            ],
        )
        assert "[1]" in result.content
        assert "[2]" in result.content
        assert len(result.sources) == 2

    def test_section_result_serialization(self) -> None:
        """Should serialize and deserialize correctly."""
        result = SectionResult(
            content="Test content [1].",
            sources=[
                Source(url="https://example.com", title="Example", source_type="web_search"),
            ],
        )
        data = result.model_dump()
        restored = SectionResult.model_validate(data)
        assert restored.content == result.content
        assert len(restored.sources) == 1
        assert restored.sources[0].title == "Example"


class TestReportOutput:
    """Tests for the ReportOutput Pydantic model."""

    def test_create_report_output(self) -> None:
        """Should create a ReportOutput with content and sources."""
        output = ReportOutput(
            content="Full report markdown with citation [1].",
            sources=[
                Source(url="https://example.com", title="Example", source_type="web_search"),
            ],
        )
        assert "Full report markdown" in output.content
        assert len(output.sources) == 1

    def test_report_output_serialization(self) -> None:
        """Should serialize and deserialize correctly."""
        output = ReportOutput(
            content="Content [1] and [2].",
            sources=[
                Source(url="https://a.com", title="Source A", source_type="web_search"),
                Source(url="https://b.com", title="Source B", source_type="arxiv"),
            ],
        )
        data = output.model_dump()
        restored = ReportOutput.model_validate(data)
        assert len(restored.sources) == 2
        assert restored.sources[0].title == "Source A"
        assert restored.sources[1].source_type == "arxiv"

    def test_report_output_empty_sources(self) -> None:
        """Should create a ReportOutput with no sources."""
        output = ReportOutput(
            content="No citations here.",
            sources=[],
        )
        assert output.sources == []