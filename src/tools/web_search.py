"""Web search tool using Pydantic AI's built-in WebSearch capability."""

import logging

from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import RunUsage

from src.config import Settings

logger = logging.getLogger(__name__)

# The web search is handled via the WebSearch capability on agents.
# This module provides a helper to create a search agent that can be used
# as a fallback tool by sub-agents.


def create_web_search_agent(settings: Settings) -> Agent[None, str]:
    """Create a web search agent using Pydantic AI's built-in WebSearch capability.

    Args:
        settings: Application settings.

    Returns:
        An agent configured with web search capability.
    """
    # Deferred import — src.agents imports src.tools, so pulling _build_model
    # at the top of this module would form a circular import.
    from pydantic_ai.capabilities import WebSearch

    from src.agents import _build_model

    return Agent(
        _build_model(settings.sub_agent_model, settings),
        output_type=str,
        output_retries=2,
        instructions=(
            "You are a web search assistant. Given a search query, "
            "use web search to find relevant, recent information and return "
            "a concise summary of the most important findings. "
            "Focus on recent developments within the past week. "
            "IMPORTANT: After using the web search tool, you MUST provide "
            "a text summary of the results. Do not make additional tool calls."
        ),
        capabilities=[WebSearch(native=False, local=True)],
        defer_model_check=True,
    )


async def web_search(
    query: str,
    settings: Settings | None = None,
    usage: RunUsage | None = None,
) -> str:
    """Perform a web search and return summarized results.

    This is a convenience function that creates a search agent and runs it.
    If the usage limit is exceeded, returns a fallback message instead of raising.

    Args:
        query: The search query string.
        settings: Application settings. Uses defaults if not provided.
        usage: Optional RunUsage object to accumulate token usage across agents.

    Returns:
        Summarized search results as a string, or a fallback message if the
        usage limit was exceeded.
    """
    if settings is None:
        from src.config import get_settings

        settings = get_settings()

    agent = create_web_search_agent(settings)
    try:
        result = await agent.run(query, usage=usage)
        return str(result.output)
    except UsageLimitExceeded as exc:
        logger.warning("Web search skipped for '%s': %s", query, exc)
        return (
            "Web search was skipped because the request usage limit was reached. "
            "Proceed using the information already gathered from other sources."
        )
