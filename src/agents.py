"""Agent definitions for the AI newsletter system.

Module-level Agent instances following the mining-report pattern:
- Sub-agents are defined with defer_model_check=True
- Manager agent calls sub-agents via tool wrappers
- Evaluator agent is standalone (not a tool on the manager)
"""

from pydantic_ai import Agent, RunContext

from src.config import get_settings
from src.prompts import (
    CODING_AGENTS_SYSTEM_PROMPT,
    COMMUNITY_NEWS_SYSTEM_PROMPT,
    INDUSTRY_OVERVIEW_SYSTEM_PROMPT,
    MANAGER_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    get_evaluator_system_prompt,
    get_manager_instructions,
)
from src.schemas import EvalResult, ReportOutput, SectionResult
from src.tools.arxiv_search import arxiv_search
from src.tools.hackernews import hackernews_search, hackernews_top_comments
from src.tools.rss_reader import rss_reader
from src.tools.web_search import web_search

# ── Shared settings ────────────────────────────────────────────────────────

settings = get_settings()


# ── Sub-agents (module-level, defer_model_check=True) ─────────────────────


# Industry Overview Agent
industry_overview_agent = Agent(
    settings.sub_agent_model_string,
    output_type=SectionResult,
    system_prompt=INDUSTRY_OVERVIEW_SYSTEM_PROMPT,
    output_retries=3,
    defer_model_check=True,
)


@industry_overview_agent.tool
async def search_hackernews(ctx: RunContext[None], query: str) -> str:
    """Search HackerNews for AI-related stories.

    Args:
        query: Search query string.
    """
    return await hackernews_search(query)


@industry_overview_agent.tool
async def get_hn_comments(ctx: RunContext[None], story_id: str) -> str:
    """Fetch top comments for a HackerNews story.

    Args:
        story_id: The HN story object ID.
    """
    return await hackernews_top_comments(story_id)


@industry_overview_agent.tool
async def read_rss_feeds(ctx: RunContext[None]) -> str:
    """Read recent posts from configured RSS feeds."""
    return await rss_reader()


@industry_overview_agent.tool
async def search_web_industry(ctx: RunContext[None], query: str) -> str:
    """Search the web for additional information.

    Args:
        query: Search query string.
    """
    return await web_search(query, usage=ctx.usage)


# Research Agent
research_agent = Agent(
    settings.sub_agent_model_string,
    output_type=SectionResult,
    system_prompt=RESEARCH_SYSTEM_PROMPT,
    output_retries=3,
    defer_model_check=True,
)


@research_agent.tool
async def search_arxiv(ctx: RunContext[None], query: str, max_results: int = 10) -> str:
    """Search arXiv for recent AI/ML papers.

    Args:
        query: Search query (e.g. "transformer", "reinforcement learning").
        max_results: Maximum number of results to return.
    """
    return await arxiv_search(query, max_results=max_results)


@research_agent.tool
async def search_web_research(ctx: RunContext[None], query: str) -> str:
    """Search the web for additional research information.

    Args:
        query: Search query string.
    """
    return await web_search(query, usage=ctx.usage)


# Community News Agent
community_news_agent = Agent(
    settings.sub_agent_model_string,
    output_type=SectionResult,
    system_prompt=COMMUNITY_NEWS_SYSTEM_PROMPT,
    output_retries=3,
    defer_model_check=True,
)


@community_news_agent.tool
async def search_hn_community(ctx: RunContext[None], query: str) -> str:
    """Search HackerNews for community discussions.

    Args:
        query: Search query string.
    """
    return await hackernews_search(query)


@community_news_agent.tool
async def get_hn_comments_community(ctx: RunContext[None], story_id: str) -> str:
    """Fetch top comments for a HackerNews story.

    Args:
        story_id: The HN story object ID.
    """
    return await hackernews_top_comments(story_id)


@community_news_agent.tool
async def read_community_rss(ctx: RunContext[None]) -> str:
    """Read recent posts from configured RSS feeds."""
    return await rss_reader()


@community_news_agent.tool
async def search_web_community(ctx: RunContext[None], query: str) -> str:
    """Search the web for additional community news.

    Args:
        query: Search query string.
    """
    return await web_search(query, usage=ctx.usage)


# Coding Agents Agent
coding_agents_agent = Agent(
    settings.sub_agent_model_string,
    output_type=SectionResult,
    system_prompt=CODING_AGENTS_SYSTEM_PROMPT,
    output_retries=3,
    defer_model_check=True,
)


@coding_agents_agent.tool
async def read_coding_rss(ctx: RunContext[None]) -> str:
    """Read recent posts from configured RSS feeds about coding agents."""
    return await rss_reader()


@coding_agents_agent.tool
async def search_web_coding(ctx: RunContext[None], query: str) -> str:
    """Search the web for coding agent news and best practices.

    Args:
        query: Search query string.
    """
    return await web_search(query, usage=ctx.usage)


# ── Manager Agent ─────────────────────────────────────────────────────────


manager_agent = Agent(
    settings.manager_model_string,
    output_type=ReportOutput,
    system_prompt=MANAGER_SYSTEM_PROMPT,
    instructions=get_manager_instructions(settings),
    model_settings={"max_tokens": settings.max_output_tokens},
    output_retries=3,
    defer_model_check=True,
)


@manager_agent.tool
async def industry_overview(ctx: RunContext[None], task: str) -> str:
    """Get the industry overview section of the newsletter.

    Covers big AI lab highlights, smaller lab highlights, and regulatory updates.

    Args:
        task: Specific focus or instructions for this section.
    """
    result = await industry_overview_agent.run(task, usage=ctx.usage)
    section: SectionResult = result.output  # ty:ignore[invalid-assignment]
    return f"{section.content}\n\nSources:\n" + "\n".join(f"[{i + 1}] {s.title} — {s.url}" for i, s in enumerate(section.sources))


@manager_agent.tool
async def research(ctx: RunContext[None], task: str) -> str:
    """Get the research updates section of the newsletter.

    Selects and summarizes 2 influential recent papers.

    Args:
        task: Specific focus or instructions for this section.
    """
    result = await research_agent.run(task, usage=ctx.usage)
    section: SectionResult = result.output  # ty:ignore[invalid-assignment]
    return f"{section.content}\n\nSources:\n" + "\n".join(f"[{i + 1}] {s.title} — {s.url}" for i, s in enumerate(section.sources))


@manager_agent.tool
async def community_news(ctx: RunContext[None], task: str) -> str:
    """Get the community updates section of the newsletter.

    Covers important discussions and other news from the AI community.

    Args:
        task: Specific focus or instructions for this section.
    """
    result = await community_news_agent.run(task, usage=ctx.usage)
    section: SectionResult = result.output  # ty:ignore[invalid-assignment]
    return f"{section.content}\n\nSources:\n" + "\n".join(f"[{i + 1}] {s.title} — {s.url}" for i, s in enumerate(section.sources))


@manager_agent.tool
async def coding_agents(ctx: RunContext[None], task: str) -> str:
    """Get the coding agents section of the newsletter.

    Covers best practices, tool updates, and community highlights.

    Args:
        task: Specific focus or instructions for this section.
    """
    result = await coding_agents_agent.run(task, usage=ctx.usage)
    section: SectionResult = result.output  # ty:ignore[invalid-assignment]
    return f"{section.content}\n\nSources:\n" + "\n".join(f"[{i + 1}] {s.title} — {s.url}" for i, s in enumerate(section.sources))


# ── Evaluator Agent (standalone, not a tool on the manager) ────────────────

evaluator_agent = Agent(
    settings.evaluator_model_string,
    output_type=EvalResult,
    system_prompt=get_evaluator_system_prompt(settings),
    output_retries=3,
    defer_model_check=True,
)
