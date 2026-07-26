"""Application configuration via pydantic-settings."""

import os
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SUPPORTED_PROVIDERS = ("openai", "anthropic")


class Settings(BaseSettings):
    """All application settings, configurable via environment variables or .env file.

    Environment variables use the AI_REPORT_ prefix for app-specific settings.
    API credentials use their standard names (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).

    Supported providers: openai, anthropic

    Settings are loaded from (in order of priority):
    1. Explicit constructor arguments
    2. Environment variables
    3. .env file
    4. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="AI_REPORT_",
        env_file=".env",
        populate_by_name=True,
    )

    # Provider selection
    model_provider: str = "openai"

    # API keys — read from standard env var names (no prefix)
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")

    # Base URL — shared across providers (opencode.ai proxy)
    base_url: str = Field(
        default="https://opencode.ai/zen/go",
        validation_alias="OPENAI_BASE_URL",
    )

    # Model configuration
    report_manager_model: str = "deepseek-v4-flash"
    sub_agent_model: str = "deepseek-v4-flash"
    evaluator_model: str = "deepseek-v4-flash"

    # Output
    output_dir: str = "./output"

    # Observability
    log_environment: str = "dev"

    # Quality control
    eval_threshold: float = 0.7
    max_revision_cycles: int = 1
    max_tool_calls: int = 5
    max_output_tokens: int = 50000
    request_limit: int = 300

    # Email delivery (default-off; opt in via --send-email or AI_REPORT_EMAIL_ENABLED)
    email_enabled: bool = False
    email_to: str = ""
    email_from: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    email_subject_prefix: str = "AI Industry Weekly"
    # Dry-run: render HTML to output dir instead of sending (no SMTP creds needed)
    email_dry_run: bool = False

    # RSS feeds
    rss_feeds: list[str] = [
        "https://openai.com/blog/rss.xml",
        "https://www.anthropic.com/rss",
        "https://ai.googleblog.com/feeds/posts/default",
        "https://simonwillison.net/atom/everything/",
        "https://chrisloy.dev/rss.xml",
        "https://www.normaltech.ai/feed",
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

    @model_validator(mode="after")
    def validate_provider(self) -> "Settings":
        """Validate that model_provider is supported."""
        if self.model_provider not in SUPPORTED_PROVIDERS:
            msg = (
                f"Unsupported model_provider: {self.model_provider!r}. "
                f"Choose from: {', '.join(SUPPORTED_PROVIDERS)}"
            )
            raise ValueError(msg)
        return self

    def configure(self) -> None:
        """Set provider-specific env vars so pydantic_ai picks them up.

        Call this at application startup before running any agents.
        Raises ValueError if the required API key for the selected provider is missing.
        """
        if self.model_provider == "openai":
            if not self.openai_api_key:
                msg = (
                    "OPENAI_API_KEY is not set. "
                    "Please set it in .env or as an environment variable."
                )
                raise ValueError(msg)
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            os.environ["OPENAI_BASE_URL"] = self.base_url
        elif self.model_provider == "anthropic":
            if not self.anthropic_api_key:
                msg = (
                    "ANTHROPIC_API_KEY is not set. "
                    "Please set it in .env or as an environment variable."
                )
                raise ValueError(msg)
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key
            os.environ["ANTHROPIC_BASE_URL"] = self.base_url

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
