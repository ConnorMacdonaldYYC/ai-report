"""Tests for the config module."""

import os

import pytest

from src.config import Settings, get_settings


class TestSettings:
    """Tests for the Settings configuration class."""

    def test_default_values(self) -> None:
        """Should have sensible default values."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            report_manager_model="deepseek-v4-flash",
            sub_agent_model="deepseek-v4-flash",
            evaluator_model="deepseek-v4-flash",
        )
        assert settings.model_provider == "openai"
        assert settings.base_url == "https://opencode.ai/zen/go"
        assert settings.report_manager_model == "deepseek-v4-flash"
        assert settings.sub_agent_model == "deepseek-v4-flash"
        assert settings.evaluator_model == "deepseek-v4-flash"
        assert settings.output_dir == "./output"
        assert settings.eval_threshold == 0.7
        assert settings.max_revision_cycles == 1
        assert settings.max_tool_calls == 5
        assert settings.max_output_tokens == 50000

    def test_model_string_properties(self) -> None:
        """Should generate correct model strings."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            report_manager_model="deepseek-v4-flash",
        )
        assert settings.manager_model_string == "openai:deepseek-v4-flash"
        assert settings.sub_agent_model_string.startswith("openai:")
        assert settings.evaluator_model_string.startswith("openai:")

    def test_anthropic_model_string(self) -> None:
        """Should generate correct model strings for Anthropic provider."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="anthropic",
            report_manager_model="claude-sonnet-4-6",
            sub_agent_model="claude-sonnet-4-6",
            evaluator_model="claude-sonnet-4-6",
        )
        assert settings.manager_model_string == "anthropic:claude-sonnet-4-6"
        assert settings.sub_agent_model_string == "anthropic:claude-sonnet-4-6"
        assert settings.evaluator_model_string == "anthropic:claude-sonnet-4-6"

    def test_custom_rss_feeds(self) -> None:
        """Should accept custom RSS feeds."""
        settings = Settings(_env_file=None, rss_feeds=["https://example.com/feed.xml"])  # type: ignore[call-arg]
        assert settings.rss_feeds == ["https://example.com/feed.xml"]

    def test_custom_hn_settings(self) -> None:
        """Should accept custom HackerNews settings."""
        settings = Settings(_env_file=None, hn_min_score=100, hn_search_keywords=["AI", "LLM"])  # type: ignore[call-arg]
        assert settings.hn_min_score == 100
        assert settings.hn_search_keywords == ["AI", "LLM"]

    def test_env_prefix(self) -> None:
        """Should use AI_REPORT_ prefix for environment variables."""
        assert Settings.model_config.get("env_prefix") == "AI_REPORT_"

    def test_configure_openai_sets_env_vars(self) -> None:
        """Should set OPENAI env vars when provider is openai."""
        original_api_key = os.environ.pop("OPENAI_API_KEY", None)
        original_base_url = os.environ.pop("OPENAI_BASE_URL", None)
        try:
            settings = Settings(  # type: ignore[call-arg]
                _env_file=None,
                model_provider="openai",
                openai_api_key="test-key-123",
                base_url="https://custom.example.com/v1",
            )
            settings.configure()
            assert os.environ.get("OPENAI_API_KEY") == "test-key-123"
            assert os.environ.get("OPENAI_BASE_URL") == "https://custom.example.com/v1"
        finally:
            if original_api_key is not None:
                os.environ["OPENAI_API_KEY"] = original_api_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)
            if original_base_url is not None:
                os.environ["OPENAI_BASE_URL"] = original_base_url
            else:
                os.environ.pop("OPENAI_BASE_URL", None)

    def test_configure_anthropic_sets_env_vars(self) -> None:
        """Should set ANTHROPIC env vars when provider is anthropic."""
        original_api_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        original_base_url = os.environ.pop("ANTHROPIC_BASE_URL", None)
        try:
            settings = Settings(  # type: ignore[call-arg]
                _env_file=None,
                model_provider="anthropic",
                anthropic_api_key="sk-ant-test-key",
                base_url="https://custom.example.com/v1",
            )
            settings.configure()
            assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-test-key"
            assert os.environ.get("ANTHROPIC_BASE_URL") == "https://custom.example.com/v1"
        finally:
            if original_api_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = original_api_key
            else:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            if original_base_url is not None:
                os.environ["ANTHROPIC_BASE_URL"] = original_base_url
            else:
                os.environ.pop("ANTHROPIC_BASE_URL", None)

    def test_configure_raises_on_missing_openai_key(self) -> None:
        """Should raise ValueError if OPENAI_API_KEY is not set with openai provider."""
        settings = Settings(_env_file=None, model_provider="openai", openai_api_key="")  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            settings.configure()

    def test_configure_raises_on_missing_anthropic_key(self) -> None:
        """Should raise ValueError if ANTHROPIC_API_KEY is not set with anthropic provider."""
        settings = Settings(_env_file=None, model_provider="anthropic", anthropic_api_key="")  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            settings.configure()

    def test_invalid_provider_raises(self) -> None:
        """Should raise ValueError for unsupported model_provider."""
        with pytest.raises(ValueError, match="Unsupported model_provider"):
            Settings(_env_file=None, model_provider="google")  # type: ignore[call-arg]

    # ── opencode-go session ID ──────────────────────────────────────────

    def test_session_id_auto_generated_when_blank(self) -> None:
        """An empty opencode_session_id should be filled in with a UUID hex."""
        settings = Settings(_env_file=None, model_provider="openai")  # type: ignore[call-arg]
        assert settings.opencode_session_id
        # 32-char hex from uuid4().hex — stable within this instance.
        assert len(settings.opencode_session_id) == 32
        int(settings.opencode_session_id, 16)  # parses as hex

    def test_session_id_respects_explicit_value(self) -> None:
        """An explicit opencode_session_id should not be overwritten."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            opencode_session_id="pinned-session-id",
        )
        assert settings.opencode_session_id == "pinned-session-id"

    def test_openai_api_key_reads_from_env(self) -> None:
        """Should read OPENAI_API_KEY from environment variable."""
        original = os.environ.get("OPENAI_API_KEY")
        try:
            os.environ["OPENAI_API_KEY"] = "sk-test-env-key"
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
            assert settings.openai_api_key == "sk-test-env-key"
        finally:
            if original is not None:
                os.environ["OPENAI_API_KEY"] = original
            else:
                os.environ.pop("OPENAI_API_KEY", None)

    def test_anthropic_api_key_reads_from_env(self) -> None:
        """Should read ANTHROPIC_API_KEY from environment variable."""
        original = os.environ.get("ANTHROPIC_API_KEY")
        try:
            os.environ["ANTHROPIC_API_KEY"] = "sk-ant-env-key"
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
            assert settings.anthropic_api_key == "sk-ant-env-key"
        finally:
            if original is not None:
                os.environ["ANTHROPIC_API_KEY"] = original
            else:
                os.environ.pop("ANTHROPIC_API_KEY", None)

    # ── Email settings ──────────────────────────────────────────────────

    def test_email_defaults(self) -> None:
        """Email settings should have sensible defaults (opt-in, not opt-out)."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.email_enabled is False
        assert settings.email_to == []
        assert settings.email_from == ""
        assert settings.smtp_host == "smtp.gmail.com"
        assert settings.smtp_port == 587
        assert settings.smtp_username == ""
        assert settings.smtp_password == ""
        assert settings.smtp_use_tls is True
        assert settings.email_subject_prefix == "AI Industry Weekly"
        assert settings.email_dry_run is False

    def test_custom_email_settings(self) -> None:
        """Should accept custom email settings via constructor."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            email_enabled=True,
            email_to=["recipient@example.com"],
            email_from="sender@example.com",
            smtp_host="smtp.custom.com",
            smtp_port=465,
            smtp_username="custom_user",
            smtp_password="custom_pass",
            smtp_use_tls=False,
            email_subject_prefix="Custom Report",
            email_dry_run=True,
        )
        assert settings.email_enabled is True
        assert settings.email_to == ["recipient@example.com"]
        assert settings.email_from == "sender@example.com"
        assert settings.smtp_host == "smtp.custom.com"
        assert settings.smtp_port == 465
        assert settings.smtp_username == "custom_user"
        assert settings.smtp_password == "custom_pass"
        assert settings.smtp_use_tls is False
        assert settings.email_subject_prefix == "Custom Report"
        assert settings.email_dry_run is True


class TestGetSettings:
    """Tests for the get_settings factory function."""

    def test_get_settings_returns_settings(self) -> None:
        """Should return a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)
