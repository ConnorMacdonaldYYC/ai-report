"""Tests for the config module."""



from src.config import Settings, get_settings


class TestSettings:
    """Tests for the Settings configuration class."""

    def test_default_values(self) -> None:
        """Should have sensible default values."""
        settings = Settings()
        assert settings.model_provider == "google-cloud"
        assert settings.report_manager_model == "gemini-2.5-flash-lite"
        assert settings.sub_agent_model == "gemini-2.5-flash-lite"
        assert settings.evaluator_model == "gemini-2.5-flash-lite"
        assert settings.output_dir == "./output"
        assert settings.eval_threshold == 0.7
        assert settings.max_revision_cycles == 1
        assert settings.max_tool_calls == 5
        assert settings.max_output_tokens == 50000

    def test_model_string_properties(self) -> None:
        """Should generate correct model strings."""
        settings = Settings(
            model_provider="google-cloud",
            report_manager_model="gemini-2.5-flash-lite",
        )
        assert settings.manager_model_string == "google-cloud:gemini-2.5-flash-lite"
        assert settings.sub_agent_model_string == "google-cloud:gemini-2.5-flash-lite"
        assert settings.evaluator_model_string == "google-cloud:gemini-2.5-flash-lite"

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
        # Settings uses env_prefix="AI_REPORT_"
        # This test verifies the prefix is configured
        assert Settings.model_config.get("env_prefix") == "AI_REPORT_"


class TestGetSettings:
    """Tests for the get_settings factory function."""

    def test_get_settings_returns_settings(self) -> None:
        """Should return a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)