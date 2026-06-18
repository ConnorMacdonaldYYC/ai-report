"""Shared test fixtures for the AI newsletter agent system."""

import pytest

from src.config import Settings
from src.schemas import DimensionScore, EvalResult, Source


@pytest.fixture
def sample_settings() -> Settings:
    """Return a Settings instance with test-friendly defaults."""
    return Settings(
        model_provider="openai",
        openai_api_key="test-key",
        openai_base_url="https://test.example.com/v1",
        report_manager_model="deepseek-v4-flash",
        sub_agent_model="deepseek-v4-flash",
        evaluator_model="deepseek-v4-flash",
        output_dir="./test_output",
        eval_threshold=0.7,
        max_revision_cycles=1,
        max_tool_calls=5,
        max_output_tokens=50000,
        log_environment="dev",
        rss_feeds=["https://example.com/feed.xml"],
        hn_min_score=50,
        hn_search_keywords=["AI", "LLM"],
    )


@pytest.fixture
def sample_eval_result() -> EvalResult:
    """Return a sample EvalResult for testing."""
    return EvalResult(
        dimensions=[
            DimensionScore(
                dimension="Relevance",
                score=0.85,
                justification="Topics are timely and significant.",
                improvement_suggestions=["Add more recent regulatory updates."],
            ),
            DimensionScore(
                dimension="Coverage",
                score=0.80,
                justification="All sections present, some thin.",
                improvement_suggestions=["Expand the coding agents section."],
            ),
            DimensionScore(
                dimension="Insight",
                score=0.75,
                justification="Good analysis with some context.",
                improvement_suggestions=["Add more forward-looking analysis."],
            ),
            DimensionScore(
                dimension="Readability",
                score=0.90,
                justification="Well-structured and concise.",
                improvement_suggestions=["Minor formatting improvements."],
            ),
        ],
        overall_pass=True,
        summary="A solid newsletter with good coverage and readability.",
    )


@pytest.fixture
def sample_sources() -> list[Source]:
    """Return a list of sample Source objects for testing."""
    return [
        Source(
            url="https://openai.com/blog/gpt-5/",
            title="OpenAI Announces GPT-5",
            source_type="web_search",
        ),
        Source(
            url="https://arxiv.org/abs/2401.12345",
            title="Scaling Laws for In-Context Learning",
            source_type="arxiv",
        ),
        Source(
            url="https://news.ycombinator.com/item?id=12345",
            title="HN Discussion on AI Coding Assistants",
            source_type="hackernews",
        ),
    ]


@pytest.fixture
def sample_report_markdown() -> str:
    """Return a sample newsletter markdown for testing."""
    return """# AI Industry Weekly - May 11 - May 18, 2026

## Industry Overview

### Big AI Labs
- **Google**: Announced Gemini 3.0 with improved reasoning capabilities [1]
- **OpenAI**: Released GPT-5 with enhanced coding abilities [2]
- **Anthropic**: Launched Claude 4 with better tool use [3]

### Smaller Labs
- Mistral released a new open-weights model
- DeepSeek announced breakthroughs in efficiency

### Regulatory Updates
- EU AI Act enforcement began this week [3]
- US Senate held hearings on AI safety

---

## Research Updates

### Scaling Laws for In-Context Learning [2]
- **Authors**: Smith et al.
- **Summary**: This paper demonstrates that in-context learning follows predictable scaling laws.
- **Why it matters**: Provides theoretical foundation for understanding
  how LLMs learn from examples.

### Efficient Attention Mechanisms for Long Context
- **Authors**: Johnson et al.
- **Summary**: A new attention mechanism that reduces memory usage by 10x for long sequences.
- **Why it matters**: Enables practical long-context processing on consumer hardware.

---

## Community Updates

### Top Discussions
- HN thread on AI coding assistants reached 500+ comments [4]
- Debate on open-weights vs proprietary models continues

### Other News
- New AI chip announced by startup
- Major university launches AI ethics center

---

## Coding Agents & Best Practices

### Best Practices
- Use structured prompts for coding agents
- Always review generated code before committing

### Tool Updates
- Cursor 2.0 released with improved multi-file editing
- Aider v0.50 adds support for new models

### Community Highlights
- Discussion on best practices for AI-assisted code review
- Tips for using coding agents in large codebases
"""