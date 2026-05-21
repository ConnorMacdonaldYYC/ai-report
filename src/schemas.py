"""Data schemas for the AI newsletter agent system.

Uses TypedDict for simple request data and BaseModel for structured
agent outputs and evaluation schemas that need validation and
field descriptions for Pydantic AI structured output.
"""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

# ── Report request (TypedDict — simple data bag) ──────────────────────────


class ReportRequest(TypedDict):
    """Input to the report pipeline."""

    date_range: str  # e.g. "May 11 - May 18, 2026"


# ── Source tracking (BaseModel — used in agent structured output) ──────────


class Source(BaseModel):
    """A source referenced in the report."""

    url: str = Field(description="URL of the source article or page")
    title: str = Field(description="Title or headline of the source")
    source_type: Literal["web_search", "arxiv", "hackernews", "rss"] = Field(
        description="Origin of this source"
    )


# ── Agent output schemas (BaseModel — structured output for Pydantic AI) ────


class SectionResult(BaseModel):
    """Structured output from a sub-agent section.

    Content uses [1], [2], etc. citation markers that reference
    entries in the sources list by index ([1] = sources[0]).
    """

    content: str = Field(
        description="Markdown content with [1], [2] etc. citation markers "
        "referencing entries in the sources list"
    )
    sources: list[Source] = Field(
        description="Sources referenced in content, ordered by citation number. "
        "[1] = sources[0], [2] = sources[1], etc."
    )


class ReportOutput(BaseModel):
    """Final assembled report from the manager agent.

    Content uses globally-numbered [1], [2], etc. citation markers
    that reference entries in the sources list.
    """

    content: str = Field(
        description="Full report markdown with [N] citation markers"
    )
    sources: list[Source] = Field(
        description="All sources, globally numbered. [1] = sources[0], etc."
    )


# ── Report result (BaseModel — structured pipeline output) ─────────────────


class ReportResult(BaseModel):
    """Output of the report pipeline."""

    date_range: str
    report_markdown: str
    sources: list[Source] = Field(default_factory=list)
    eval_passed: bool
    eval_score: float
    revision_count: int


# ── Evaluation schemas (BaseModel, following mining-report pattern) ──


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    dimension: str
    score: float  # 0.0-1.0
    justification: str
    improvement_suggestions: list[str]


class EvalResult(BaseModel):
    """Result from the evaluation agent."""

    dimensions: list[DimensionScore]  # exactly 4
    overall_pass: bool
    summary: str