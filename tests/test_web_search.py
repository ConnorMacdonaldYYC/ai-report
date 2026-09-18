"""Unit tests for the web_search tool fallback behaviour."""

import importlib
from types import SimpleNamespace

import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import RunUsage

from src.tools.web_search import web_search
from tests.utils import build_test_settings

# src.tools re-exports the web_search function, which shadows the submodule
# on plain attribute access — import the module itself explicitly.
web_search_module = importlib.import_module("src.tools.web_search")


class _StubAgent:
    """Minimal stand-in for a pydantic-ai Agent.

    Args:
        outcome: A string returned as the run output, or an exception to raise.
    """

    def __init__(self, outcome: str | Exception) -> None:
        self._outcome = outcome

    async def run(self, prompt: str, usage: RunUsage | None = None) -> object:
        """Return the configured outcome, or raise it if it is an exception.

        Args:
            prompt: The prompt passed to the agent (ignored).
            usage: Optional usage accumulator (ignored).
        """
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return SimpleNamespace(output=self._outcome)


class TestWebSearch:
    """Tests for web_search degradation behaviour."""

    async def test_returns_summary_on_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return the agent's output when the search succeeds."""
        settings = build_test_settings()
        monkeypatch.setattr(
            web_search_module, "create_web_search_agent",
            lambda _: _StubAgent("summary of findings"),
        )

        result = await web_search("test query", settings=settings)

        assert result == "summary of findings"

    async def test_returns_fallback_on_usage_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return a fallback message when the usage limit is exceeded."""
        settings = build_test_settings()
        monkeypatch.setattr(
            web_search_module, "create_web_search_agent",
            lambda _: _StubAgent(UsageLimitExceeded("request_limit of 50 exceeded")),
        )

        result = await web_search("test query", settings=settings)

        assert "usage limit" in result
        assert "other sources" in result

    async def test_returns_fallback_on_backend_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return a fallback message instead of raising when the search backend fails."""
        settings = build_test_settings()
        monkeypatch.setattr(
            web_search_module, "create_web_search_agent",
            lambda _: _StubAgent(RuntimeError("DDGSException: No results found.")),
        )

        result = await web_search("test query", settings=settings)

        assert "unavailable" in result
        assert "other sources" in result
