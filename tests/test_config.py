"""Tests for the config module."""

import os

import pytest

from src.config import Settings, get_settings


class TestSettings:
    """Tests for the Settings configuration class."""

    def test_default_values(self) -> None:
        """Should have sensible default values."""
        settings = Settings()
        assert settings.model_provider == "openai"
        assert settings.openai_base_url == "https://opencode.ai/zen/go/v1"
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
        settings = Settings(
            model_provider="openai",
            report_manager_model="deepseek-v4-flash",
        )
        assert settings.manager_model_string == "openai:deepseek-v4-flash"
        assert settings.sub_agent_model_string.startswith("openai:")
        assert settings.evaluator_model_string.startswith("openai:")

    def test_anthropic_model_string(self) -> None:
        """Should generate correct model strings for Anthropic provider."""
        settings = Settings(
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
        settings = Settings(rss_feeds=["https://example.com/feed.xml"])
        assert settings.rss_feeds == ["https://example.com/feed.xml"]

    def test_custom_hn_settings(self) -> None:
        """Should accept custom HackerNews settings."""
        settings = Settings(hn_min_score=100, hn_search_keywords=["AI", "LLM"])
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
            settings = Settings(
                model_provider="openai",
                openai_api_key="test-key-123",
                openai_base_url="https://custom.example.com/v1",
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

    def test_configure_anthropic_sets_env_var(self) -> None:
        """Should set ANTHROPIC_API_KEY when provider is anthropic."""
        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            settings = Settings(
                model_provider="anthropic",
                anthropic_api_key="sk-ant-test-key",
            )
            settings.configure()
            assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-test-key"
        finally:
            if original is not None:
                os.environ["ANTHROPIC_API_KEY"] = original
            else:
                os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_configure_raises_on_missing_openai_key(self) -> None:
        """Should raise ValueError if OPENAI_API_KEY is not set with openai provider."""
        settings = Settings(model_provider="openai", openai_api_key="")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            settings.configure()

    def test_configure_raises_on_missing_anthropic_key(self) -> None:
        """Should raise ValueError if ANTHROPIC_API_KEY is not set with anthropic provider."""
        settings = Settings(model_provider="anthropic", anthropic_api_key="")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            settings.configure()

    def test_invalid_provider_raises(self) -> None:
        """Should raise ValueError for unsupported model_provider."""
        with pytest.raises(ValueError, match="Unsupported model_provider"):
            Settings(model_provider="google")

    def test_openai_api_key_reads_from_env(self) -> None:
        """Should read OPENAI_API_KEY from environment variable."""
        original = os.environ.get("OPENAI_API_KEY")
        try:
            os.environ["OPENAI_API_KEY"] = "sk-test-env-key"
            settings = Settings()
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
            settings = Settings()
            assert settings.anthropic_api_key == "sk-ant-env-key"
        finally:
            if original is not None:
                os.environ["ANTHROPIC_API_KEY"] = original
            else:
                os.environ.pop("ANTHROPIC_API_KEY", None)


class TestGetSettings:
    """Tests for the get_settings factory function."""

    def test_get_settings_returns_settings(self) -> None:
        """Should return a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)