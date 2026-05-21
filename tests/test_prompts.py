"""Tests for the prompts module."""

from datetime import date

from src.config import Settings
from src.prompts import (
    CODING_AGENTS_SYSTEM_PROMPT,
    COMMUNITY_NEWS_SYSTEM_PROMPT,
    INDUSTRY_OVERVIEW_SYSTEM_PROMPT,
    MANAGER_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    get_date_range,
    get_evaluator_system_prompt,
    get_manager_instructions,
)


class TestSystemPrompts:
    """Tests for system prompt constants."""

    def test_manager_system_prompt_is_not_empty(self) -> None:
        """Manager system prompt should be non-empty."""
        assert len(MANAGER_SYSTEM_PROMPT) > 0

    def test_industry_overview_prompt_contains_key_terms(self) -> None:
        """Industry overview prompt should mention key topics."""
        assert "Google" in INDUSTRY_OVERVIEW_SYSTEM_PROMPT
        assert "OpenAI" in INDUSTRY_OVERVIEW_SYSTEM_PROMPT
        assert "Anthropic" in INDUSTRY_OVERVIEW_SYSTEM_PROMPT
        assert "Regulatory" in INDUSTRY_OVERVIEW_SYSTEM_PROMPT

    def test_research_prompt_contains_scoring_rubric(self) -> None:
        """Research prompt should contain the scoring rubric."""
        assert "Novelty" in RESEARCH_SYSTEM_PROMPT
        assert "Practical Impact" in RESEARCH_SYSTEM_PROMPT
        assert "Citation Velocity" in RESEARCH_SYSTEM_PROMPT
        assert "Author Prominence" in RESEARCH_SYSTEM_PROMPT

    def test_community_news_prompt_is_not_empty(self) -> None:
        """Community news prompt should be non-empty."""
        assert len(COMMUNITY_NEWS_SYSTEM_PROMPT) > 0

    def test_coding_agents_prompt_contains_key_terms(self) -> None:
        """Coding agents prompt should mention key topics."""
        assert "coding agent" in CODING_AGENTS_SYSTEM_PROMPT.lower()
        assert "best practices" in CODING_AGENTS_SYSTEM_PROMPT.lower()


class TestGetDateRange:
    """Tests for the get_date_range function."""

    def test_returns_date_range_string(self) -> None:
        """Should return a date range string."""
        result = get_date_range()
        assert "-" in result
        # Should contain a year
        from datetime import date

        assert str(date.today().year) in result


class TestGetManagerInstructions:
    """Tests for the get_manager_instructions function."""

    def test_includes_date_range(self) -> None:
        """Should include the date range in instructions."""
        settings = Settings()
        instructions = get_manager_instructions(settings)
        assert (
            "date range" in instructions.lower()
            or str(date.today().year) in instructions
        )

    def test_includes_sub_agent_calls(self) -> None:
        """Should mention all sub-agent tools."""
        settings = Settings()
        instructions = get_manager_instructions(settings)
        assert "industry_overview" in instructions
        assert "research" in instructions
        assert "community_news" in instructions
        assert "coding_agents" in instructions


class TestGetEvaluatorSystemPrompt:
    """Tests for the get_evaluator_system_prompt function."""

    def test_includes_threshold(self) -> None:
        """Should include the evaluation threshold."""
        settings = Settings(eval_threshold=0.7)
        prompt = get_evaluator_system_prompt(settings)
        assert "0.7" in prompt

    def test_includes_all_dimensions(self) -> None:
        """Should include all 4 evaluation dimensions."""
        settings = Settings()
        prompt = get_evaluator_system_prompt(settings)
        assert "Relevance" in prompt
        assert "Coverage" in prompt
        assert "Insight" in prompt
        assert "Readability" in prompt