"""Application configuration via pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings, configurable via environment variables.

    Environment variables use the AI_REPORT_ prefix.
    For example: AI_REPORT_MODEL_PROVIDER=google-cloud
    """

    model_config = SettingsConfigDict(env_prefix="AI_REPORT_")

    # Model configuration
    model_provider: str = "google-cloud"
    report_manager_model: str = "gemini-2.5-flash-lite"
    sub_agent_model: str = "gemini-2.5-flash-lite"
    evaluator_model: str = "gemini-2.5-flash-lite"

    # Output
    output_dir: str = "./output"

    # Observability
    log_environment: str = "dev"

    # Quality control
    eval_threshold: float = 0.7
    max_revision_cycles: int = 1
    max_tool_calls: int = 5
    max_output_tokens: int = 50000

    # RSS feeds
    rss_feeds: list[str] = [
        "https://openai.com/blog/rss.xml",
        "https://www.anthropic.com/rss",
        "https://ai.googleblog.com/feeds/posts/default",
        "https://simonwillison.net/atom/everything/",
        "https://chrisloy.dev/rss.xml",
    ]

    # HackerNews search
    hn_min_score: int = 50
    hn_search_keywords: list[str] = [
        "AI",
        "LLM",
        "GPT",
        "Claude",
        "Gemini",
        "Mistral",
        "Deepseek",
        "Open weights",
        "machine learning",
        "deep learning",
        "coding agent",
    ]

    @property
    def manager_model_string(self) -> str:
        """Full model string for the report manager agent."""
        return f"{self.model_provider}:{self.report_manager_model}"

    @property
    def sub_agent_model_string(self) -> str:
        """Full model string for sub-agents."""
        return f"{self.model_provider}:{self.sub_agent_model}"

    @property
    def evaluator_model_string(self) -> str:
        """Full model string for the evaluator agent."""
        return f"{self.model_provider}:{self.evaluator_model}"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
