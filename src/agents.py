"""Agent definitions for the AI newsletter system.

Agents are built via the build_agents() factory so that multiple model
configurations can coexist in one process (used by the eval framework).

Module-level singletons (built once from get_settings()) remain for
backward compatibility with existing imports and the standard CLI path:
- Sub-agents are defined with defer_model_check=True
- Manager agent is a synthesis-only agent (no tool wrappers)
- Evaluator agent is standalone (not a tool on the manager)
"""

import logging
from dataclasses import dataclass

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from src.config import Settings, get_settings
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

logger = logging.getLogger(__name__)


# ── Agent bundle ───────────────────────────────────────────────────────────


@dataclass
class AgentBundle:
    """All agents for one report pipeline run, built from a single Settings.

    Building a fresh bundle per run allows different model configurations
    to coexist in one process (used by the eval framework).
    """

    industry_overview: Agent[None, SectionResult]
    research: Agent[None, SectionResult]
    community_news: Agent[None, SectionResult]
    coding_agents: Agent[None, SectionResult]
    manager: Agent[None, ReportOutput]
    evaluator: Agent[None, EvalResult]


def _build_model(model_name: str, settings: Settings) -> Model:
    """Build a pydantic-ai Model with the opencode-go session header wired in.

    The header is applied through the underlying SDK client's ``default_headers``,
    so it rides on every chat-completions / messages request sent through pydantic-ai.

    Args:
        model_name: Bare model name (e.g. ``"deepseek-v4-flash"``, ``"claude-sonnet-4-6"``).
        settings: Application settings providing provider, base_url, api_key.

    Returns:
        An OpenAIChatModel or AnthropicModel configured for the chosen provider.
    """
    headers = {"x-opencode-session": settings.opencode_session_id}
    if settings.model_provider == "openai":
        openai_client = AsyncOpenAI(
            base_url=settings.base_url,
            api_key=settings.openai_api_key,
            default_headers=headers,
        )
        return OpenAIChatModel(model_name, provider=OpenAIProvider(openai_client=openai_client))
    if settings.model_provider == "anthropic":
        anthropic_client = AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            base_url=settings.base_url,
            default_headers=headers,
        )
        return AnthropicModel(
            model_name,
            provider=AnthropicProvider(anthropic_client=anthropic_client),
        )
    msg = f"Unsupported model_provider: {settings.model_provider!r}"
    raise ValueError(msg)


def build_agents(settings: Settings) -> AgentBundle:
    """Build a fresh set of agents from the given settings.

    Args:
        settings: Application settings providing model strings and limits.

    Returns:
        AgentBundle with all six agents, tools attached.
    """

    # ── Industry Overview Agent ─────────────────────────────────────────
    industry_overview = Agent(
        _build_model(settings.sub_agent_model, settings),
        output_type=SectionResult,
        system_prompt=INDUSTRY_OVERVIEW_SYSTEM_PROMPT,
        output_retries=3,
        defer_model_check=True,
    )

    @industry_overview.tool
    async def search_hackernews(ctx: RunContext[None], query: str) -> str:
        """Search HackerNews for AI-related stories.

        Args:
            query: Search query string.
        """
        return await hackernews_search(query)

    @industry_overview.tool
    async def get_hn_comments(ctx: RunContext[None], story_id: str) -> str:
        """Fetch top comments for a HackerNews story.

        Args:
            story_id: The HN story object ID.
        """
        return await hackernews_top_comments(story_id)

    @industry_overview.tool
    async def read_rss_feeds(ctx: RunContext[None]) -> str:
        """Read recent posts from configured RSS feeds."""
        return await rss_reader()

    @industry_overview.tool
    async def search_web_industry(ctx: RunContext[None], query: str) -> str:
        """Search the web for additional information.

        Args:
            query: Search query string.
        """
        return await web_search(query, usage=ctx.usage)

    # ── Research Agent ──────────────────────────────────────────────────
    research = Agent(
        _build_model(settings.sub_agent_model, settings),
        output_type=SectionResult,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        output_retries=3,
        defer_model_check=True,
    )

    @research.tool
    async def search_arxiv(ctx: RunContext[None], query: str, max_results: int = 10) -> str:
        """Search arXiv for recent AI/ML papers.

        Args:
            query: Search query (e.g. "transformer", "reinforcement learning").
            max_results: Maximum number of results to return.
        """
        return await arxiv_search(query, max_results=max_results)

    @research.tool
    async def search_web_research(ctx: RunContext[None], query: str) -> str:
        """Search the web for additional research information.

        Args:
            query: Search query string.
        """
        return await web_search(query, usage=ctx.usage)

    # ── Community News Agent ────────────────────────────────────────────
    community_news = Agent(
        _build_model(settings.sub_agent_model, settings),
        output_type=SectionResult,
        system_prompt=COMMUNITY_NEWS_SYSTEM_PROMPT,
        output_retries=3,
        defer_model_check=True,
    )

    @community_news.tool
    async def search_hn_community(ctx: RunContext[None], query: str) -> str:
        """Search HackerNews for community discussions.

        Args:
            query: Search query string.
        """
        return await hackernews_search(query)

    @community_news.tool
    async def get_hn_comments_community(ctx: RunContext[None], story_id: str) -> str:
        """Fetch top comments for a HackerNews story.

        Args:
            story_id: The HN story object ID.
        """
        return await hackernews_top_comments(story_id)

    @community_news.tool
    async def read_community_rss(ctx: RunContext[None]) -> str:
        """Read recent posts from configured RSS feeds."""
        return await rss_reader()

    @community_news.tool
    async def search_web_community(ctx: RunContext[None], query: str) -> str:
        """Search the web for additional community news.

        Args:
            query: Search query string.
        """
        return await web_search(query, usage=ctx.usage)

    # ── Coding Agents Agent ─────────────────────────────────────────────
    coding_agents = Agent(
        _build_model(settings.sub_agent_model, settings),
        output_type=SectionResult,
        system_prompt=CODING_AGENTS_SYSTEM_PROMPT,
        output_retries=3,
        defer_model_check=True,
    )

    @coding_agents.tool
    async def read_coding_rss(ctx: RunContext[None]) -> str:
        """Read recent posts from configured RSS feeds about coding agents."""
        return await rss_reader()

    @coding_agents.tool
    async def search_web_coding(ctx: RunContext[None], query: str) -> str:
        """Search the web for coding agent news and best practices.

        Args:
            query: Search query string.
        """
        return await web_search(query, usage=ctx.usage)

    # ── Manager Agent (synthesis-only — no tool wrappers) ───────────────
    manager = Agent(
        _build_model(settings.report_manager_model, settings),
        output_type=ReportOutput,
        system_prompt=MANAGER_SYSTEM_PROMPT,
        instructions=get_manager_instructions(settings),
        model_settings={"max_tokens": settings.max_output_tokens},
        output_retries=3,
        defer_model_check=True,
    )

    # ── Evaluator Agent (standalone, not a tool on the manager) ─────────
    evaluator = Agent(
        _build_model(settings.evaluator_model, settings),
        output_type=EvalResult,
        system_prompt=get_evaluator_system_prompt(settings),
        output_retries=3,
        defer_model_check=True,
    )

    return AgentBundle(
        industry_overview=industry_overview,
        research=research,
        community_news=community_news,
        coding_agents=coding_agents,
        manager=manager,
        evaluator=evaluator,
    )


# ── Module-level default bundle (backward compatibility) ───────────────────
# Built once at import time from get_settings(). Existing imports of the
# module-level agent singletons keep working unchanged.

default_bundle = build_agents(get_settings())

industry_overview_agent = default_bundle.industry_overview
research_agent = default_bundle.research
community_news_agent = default_bundle.community_news
coding_agents_agent = default_bundle.coding_agents
manager_agent = default_bundle.manager
evaluator_agent = default_bundle.evaluator
