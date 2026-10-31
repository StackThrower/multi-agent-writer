"""Typed settings loaded from environment variables / .env (12-factor style)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RT_", extra="ignore")

    # "provider:model" strings understood by langchain.chat_models.init_chat_model
    main_model: str = "anthropic:claude-sonnet-5-5"
    """Planner, writer, reviewer, supervisor — tasks where reasoning quality matters."""

    fast_model: str = "anthropic:claude-haiku-5-5"
    """Researchers — many parallel, tool-heavy calls, so a cheaper/faster model."""

    temperature: float | None = None
    """None = provider default (some reasoning models only accept the default)."""

    max_subquestions: int = 4
    max_revisions: int = 2
    researcher_max_tool_calls: int = 6

    search_backend: Literal["auto", "tavily", "duckduckgo"] = "auto"
    search_results: int = 5
    fetch_max_chars: int = 6_000
    http_timeout_s: float = 15.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
