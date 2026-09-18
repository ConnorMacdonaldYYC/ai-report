"""Unit tests for orchestrator section-gathering degradation."""

import pytest
from pydantic_ai import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.usage import RunUsage, UsageLimits

from src.agents import build_agents
from src.orchestrator import _gather_sections, _setup_logfire
from tests.utils import (
    ModelFn,
    build_test_settings,
    make_evaluator_fn,
    make_manager_fn,
    make_section_fn,
    override_all_agents,
)

DATE_RANGE = "Jan 1 - Jan 8, 2026"


def _failing_fn(exc: Exception) -> ModelFn:
    """Return a FunctionModel callback that raises the given exception."""

    def fail_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise exc

    return fail_fn


class TestGatherSections:
    """Tests for _gather_sections failure handling."""

    async def test_failed_section_recorded_as_none(self) -> None:
        """A sub-agent crash should record None and not stop the remaining sections."""
        settings = build_test_settings()
        bundle = build_agents(settings)

        sub_agent_fns = [
            make_section_fn("Industry Overview"),
            _failing_fn(RuntimeError("DDGSException: No results found.")),
            make_section_fn("Community Updates"),
            make_section_fn("Coding Agents"),
        ]

        with override_all_agents(
            sub_agent_fns, make_manager_fn(), make_evaluator_fn(), bundle=bundle
        ):
            sections = await _gather_sections(
                bundle, DATE_RANGE, RunUsage(), UsageLimits()
            )

        assert len(sections) == 4
        assert sections[0] is not None
        assert sections[1] is None
        assert sections[2] is not None
        assert sections[3] is not None

    async def test_all_sections_failing_returns_all_none(self) -> None:
        """Every sub-agent crashing should still return a 4-entry list of Nones."""
        settings = build_test_settings()
        bundle = build_agents(settings)

        sub_agent_fns = [
            _failing_fn(RuntimeError("backend down")) for _ in range(4)
        ]

        with override_all_agents(
            sub_agent_fns, make_manager_fn(), make_evaluator_fn(), bundle=bundle
        ):
            sections = await _gather_sections(
                bundle, DATE_RANGE, RunUsage(), UsageLimits()
            )

        assert sections == [None, None, None, None]


class TestSetupLogfire:
    """Tests for _setup_logfire token wiring."""

    def _patch_logfire(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> dict[str, object]:
        """Patch logfire.configure to record kwargs, and disable instrumentation.

        Args:
            monkeypatch: pytest monkeypatch fixture.

        Returns:
            Dict that fake configure() records its kwargs into.
        """
        recorded: dict[str, object] = {}
        monkeypatch.setattr(
            "logfire.configure", lambda **kwargs: recorded.update(kwargs)
        )
        monkeypatch.setattr("logfire.instrument_pydantic_ai", lambda: None)
        return recorded

    def test_passes_token_to_configure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-blank logfire_token should be passed to logfire.configure."""
        recorded = self._patch_logfire(monkeypatch)
        settings = build_test_settings(logfire_token="lf-test-token")

        _setup_logfire(settings)

        assert recorded["token"] == "lf-test-token"
        assert recorded["environment"] == "dev"

    def test_blank_token_passes_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A blank logfire_token should pass None so dev falls back to cached creds."""
        recorded = self._patch_logfire(monkeypatch)
        settings = build_test_settings()

        _setup_logfire(settings)

        assert recorded["token"] is None
