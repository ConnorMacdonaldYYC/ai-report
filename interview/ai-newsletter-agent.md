# AIReport — AI Newsletter Agent Specification

## Overview

AIReport is a Pydantic AI agent system that compiles a weekly AI industry newsletter. It follows the same hierarchical delegation pattern as `mining-report`, using a manager agent that coordinates specialized sub-agents via tool calls, with an evaluator-driven revision loop for quality control.

**Trigger**: Manual local execution  
**Output**: Markdown file in `output/` directory  
**Time window**: Auto-detects last 7 days from run date  
**Models**: Gemini family (low cost priority), configured via pydantic-settings

---

## Architecture

```
Report Manager Agent
├── tool: industry_overview  → Industry Overview Agent
├── tool: research           → Research Agent
├── tool: community_news     → Community News Agent
├── tool: coding_agents      → Coding Agents Agent
└── revision loop ← Evaluation Agent
```

**Pattern**: Hierarchical tool-delegation (same as mining-report). The manager agent calls each sub-agent as a tool sequentially, assembles the sections into a final report, and submits to the evaluator. If the evaluator score falls below threshold, the manager revises using the feedback.

**Execution**: Sequential — sub-agents are called one at a time via tool wrappers, matching mining-report's pattern.

---

## Sub-Agents

### 1. Industry Overview Agent

**Purpose**: Summarize the week's major developments from big AI labs and regulatory landscape.

**Newsletter section**:
- Highlights from big AI labs (Google, OpenAI, Anthropic)
- Selected highlights from smaller labs
- Regulatory updates (new laws, court cases, policy changes)

**Tools**: HackerNews API, RSS reader, web search

### 2. Research Agent

**Purpose**: Select and summarize 2 influential recent papers.

**Newsletter section**:
- 2 paper summaries with analysis

**Paper selection strategy**: Scoring rubric defined in the system prompt. The agent evaluates candidates on:
- Novelty — does it introduce a new technique or paradigm?
- Practical impact — likelihood of real-world adoption
- Citation velocity — early signs of community uptake
- Author prominence — from well-known labs or researchers

The agent scores candidates against these criteria and selects the top 2.

**Tools**: arXiv API, web search

### 3. Community News Agent

**Purpose**: Surface important discussions and developments from the broader AI community.

**Newsletter section**:
- Important topics from HackerNews, blogs, etc.
- Other recent news or developments not covered in other sections

**Tools**: HackerNews API, RSS reader, web search

### 4. Coding Agents Agent

**Purpose**: Cover coding agent discussion, best practices, and shared experiences.

**Newsletter section**:
- Shared best practices on coding agents from blogs, forums
- Notable coding agent tool releases or updates
- Community discussion highlights

**Tools**: RSS reader, web search

---

## Evaluation Agent

**Standalone agent** (not a tool on the manager) that scores the assembled report on 4 dimensions:

| Dimension | Description |
|-----------|-------------|
| **Relevance** | Are the topics timely and significant to the AI industry this week? |
| **Coverage** | Are all newsletter sections adequately addressed? No major gaps? |
| **Insight** | Does the report add analysis and context beyond raw news aggregation? |
| **Readability** | Is the report well-structured, concise, and easy to read? |

**Scoring**: Each dimension scored 0.0–1.0 with rubric descriptions at 0.0, 0.25, 0.50, 0.75, 1.0 (same pattern as mining-report). Overall pass if average ≥ threshold (default 0.7).

**Revision loop**: If the report fails evaluation, the manager receives structured feedback and re-invokes relevant sub-agents to improve weak sections. Configurable `max_revision_cycles` (default: 1).

**Output schema**:
```python
class DimensionScore(BaseModel):
    dimension: str
    score: float  # 0.0-1.0
    justification: str
    improvement_suggestions: list[str]

class EvalResult(BaseModel):
    dimensions: list[DimensionScore]  # exactly 4
    overall_pass: bool
    summary: str
```

---

## Tools

### arXiv API

- **Used by**: Research agent
- **Purpose**: Search and retrieve recent AI/ML papers with metadata
- **Returns**: Paper title, authors, abstract, submission date, arXiv ID, category
- **Implementation**: HTTP client to arXiv API, filtered by cs.AI, cs.LG, cs.CL categories, sorted by submission date within the reporting window

### HackerNews API

- **Used by**: Industry Overview agent, Community News agent
- **Purpose**: Surface trending AI-related stories and discussions
- **Returns**: Story title, URL, score, comment count, top comments
- **Implementation**: Algolia HN Search API, filtered by AI-related keywords, minimum score threshold configurable in settings

### RSS Reader

- **Used by**: Industry Overview agent, Community News agent, Coding Agents agent
- **Purpose**: Fetch recent posts from configured AI blogs and news sources
- **Returns**: Article title, URL, publication date, summary/excerpt
- **Implementation**: Feedparser-based RSS/Atom reader, feeds configured in pydantic-settings

### Web Search

- **Used by**: All sub-agents (fallback and supplementary)
- **Purpose**: General web search for anything not covered by dedicated tools
- **Implementation**: Pydantic AI's built-in WebSearch tool (same as mining-report)

---

## Configuration

All settings via pydantic-settings with environment variable support, following mining-report's pattern:

```python
class Settings(BaseModel):
    # Model configuration
    model_provider: str = "google-vertex"
    report_manager_model: str = "gemini-2.0-flash"
    sub_agent_model: str = "gemini-2.0-flash"
    evaluator_model: str = "gemini-2.0-flash"

    # Output
    output_dir: str = "./output"

    # Observability
    logfire_token: str | None = None
    log_environment: str = "dev"

    # Quality control
    eval_threshold: float = 0.7
    max_revision_cycles: int = 1
    max_tool_calls: int = 5
    max_output_tokens: int = 50000

    # RSS feeds (configurable list)
    rss_feeds: list[str] = [
        "https://openai.com/blog/rss.xml",
        "https://www.anthropic.com/rss",
        "https://ai.googleblog.com/feeds/posts/default",
        "https://simonwillison.net/atom/everything/"
        "https://chrisloy.dev/rss.xml"
        # Additional feeds added as discovered
    ]

    # HN search
    hn_min_score: int = 50
    hn_search_keywords: list[str] = [
        "AI", "LLM", "GPT", "Claude", "Gemini", "Mistral", "Deepseek", "Open weights"
        "machine learning", "deep learning", "coding agent",
    ]
```

---

## Newsletter Layout

The final markdown report follows this structure:

```markdown
# AI Industry Weekly — {date_range}

## Industry Overview

### Big AI Labs
- **Google**: [highlights]
- **OpenAI**: [highlights]
- **Anthropic**: [highlights]

### Smaller Labs
- [selected highlights]

### Regulatory Updates
- [new laws, court cases, policy changes]

---

## Research Updates

### {Paper 1 Title}
- **Authors**: ...
- **Summary**: ...
- **Why it matters**: ...

### {Paper 2 Title}
- **Authors**: ...
- **Summary**: ...
- **Why it matters**: ...

---

## Community Updates

### Top Discussions
- [important topics from HN, blogs, etc.]

### Other News
- [notable developments not covered above]

---

## Coding Agents & Best Practices

### Best Practices
- [shared practices from blogs, forums]

### Tool Updates
- [notable coding agent releases or updates]

### Community Highlights
- [discussion highlights]
```

---

## Project Structure

```
ai-report/
├── src/
│   ├── __init__.py
│   ├── agents.py              # Agent definitions (manager, sub-agents, evaluator)
│   ├── orchestrator.py         # Pipeline orchestration with revision loop
│   ├── config.py              # pydantic-settings configuration
│   ├── prompts.py             # All system prompts + scoring rubrics
│   ├── schemas.py             # TypedDict request/result + Pydantic eval schemas
│   ├── output.py              # Report formatting and file writing
│   └── tools/
│       ├── __init__.py
│       ├── arxiv_search.py     # arXiv API tool
│       ├── hackernews.py       # HackerNews API tool
│       ├── rss_reader.py       # RSS/Atom feed reader tool
│       └── web_search.py       # Web search (Pydantic AI built-in)
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Shared fixtures
│   ├── utils.py                # FunctionModel test helpers
│   ├── test_arxiv_search.py
│   ├── test_hackernews.py
│   ├── test_rss_reader.py
│   └── integration/
│       └── test_orchestrator.py
├── run.py                      # CLI entry point
├── pyproject.toml
├── Makefile
└── AGENTS.md
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent framework | Pydantic AI |
| Models | Gemini family (via google-vertex), low cost priority |
| Configuration | pydantic-settings |
| Observability | Logfire (local dev) |
| Testing | pytest + pytest-asyncio |
| Linting | ruff + mypy |
| Package manager | uv / hatchling |

### Dependencies

```toml
[project]
dependencies = [
    "pydantic-ai>=1.89.1",
    "pydantic-settings>=2.0.0",
    "logfire>=3.0.0",
    "feedparser>=6.0.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.6.0",
    "mypy>=1.10.0",
]
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data sources | Curated RSS/API feeds + web search | Structured, reliable data from known sources with web search for coverage |
| Sub-agent structure | Section-based (4 agents) | Mirrors newsletter sections directly, focused prompts per domain |
| Execution model | Sequential | Matches mining-report pattern, simpler debugging, low cost with Flash |
| Revision loop | Yes, with evaluator | Proven pattern from mining-report, quality matters for weekly reading |
| Paper selection | Scoring rubric in prompt | Consistent, explainable selections without extra tool complexity |
| RSS configuration | pydantic-settings | Easy to add/remove feeds without code changes |
| Time window | Auto-detect last 7 days | Convenient for weekly runs, no manual input needed |
| Community section | Split into community_news + coding_agents | Tighter focus per agent, better output quality |
| Eval dimensions | Relevance, coverage, insight, readability | Tailored to newsletter evaluation, not financial analysis |
| GitHub trending tool | No | Start with 4 tools, add later if needed |
| Output delivery | Markdown file only | Start simple, add delivery channels later |

---

## Implementation Notes

- Follow mining-report's patterns exactly: module-level `Agent` instances, tool wrappers that call `agent.run()` and return `str(result.output)`, `defer_model_check=True` on sub-agents
- Use `TypedDict` for `ReportRequest` and `ReportResult` (matching mining-report's state pattern)
- Use `BaseModel` for `EvalResult` and `DimensionScore` (matching mining-report's eval schema pattern)
- The orchestrator manages message history across revision cycles using `result.all_messages()`
- All functions must have full type annotations (mypy `disallow_untyped_defs=true`)
- Use modern union syntax: `str | None` not `Optional[str]`
- Test with `FunctionModel` overrides via `agent.override(model=...)` context manager (same as mining-report)
- Run linting and testing via `make lint` and `make test`
