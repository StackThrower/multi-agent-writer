"""Provider-agnostic chat model factory.

`init_chat_model("anthropic:claude-sonnet-5-5")` / `init_chat_model("openai:gpt-5.5")` —
switching providers is a config change, not a code change.
"""

from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from research_team.config import Settings, get_settings


def _init(model: str, temperature: float | None) -> BaseChatModel:
    # Some reasoning models reject a custom temperature — pass it only when configured.
    kwargs = {} if temperature is None else {"temperature": temperature}
    return init_chat_model(model, **kwargs)


def main_llm(settings: Settings | None = None) -> BaseChatModel:
    s = settings or get_settings()
    return _init(s.main_model, s.temperature)


def fast_llm(settings: Settings | None = None) -> BaseChatModel:
    s = settings or get_settings()
    return _init(s.fast_model, s.temperature)


def structured(llm: BaseChatModel, schema: type) -> Runnable[Any, Any]:
    """`with_structured_output` with the best method for the given model.

    Models whose profile advertises native structured output (constrained decoding,
    e.g. recent Claude / GPT models) use `json_schema`; others fall back to tool calling.
    Some models do not support *forced* tool calling at all, so the choice matters.
    Model profiles: https://docs.langchain.com/oss/python/langchain/models#model-profiles
    """
    profile = getattr(llm, "profile", None) or {}
    method = "json_schema" if profile.get("structured_output") else "function_calling"
    return llm.with_structured_output(schema, method=method)
