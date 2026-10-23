"""LLM providers (OpenAI-compatible chat completions, SPEC 3/8)."""

from __future__ import annotations

from ..config import ConfigError, Settings
from ..retry import RetryConfig
from .base import LlmProvider
from .openai_compat import OpenAiCompatProvider

__all__ = [
    "LlmProvider",
    "OpenAiCompatProvider",
    "build_llm_provider",
]


def build_llm_provider(
    settings: Settings, *, model: str | None = None, on_retry=None, warn=None
) -> LlmProvider:
    """Build the single OpenAI-compatible client from settings (SPEC 3).

    This one interface covers OpenRouter (including free models), Groq,
    Gemini's OpenAI-compatible endpoint, and any other naive chat-completions
    server: only ``base_url``, ``api_key`` and ``model`` are configurable.
    """
    if not (settings.llm_api_key or "").strip():
        raise ConfigError(
            "LLM_API_KEY is empty. Copy .env.example to .env and set it "
            "(e.g. a free OpenRouter key) - see README.md."
        )
    resolved_model = (model or settings.llm_model or "").strip()
    if not resolved_model:
        raise ConfigError(
            "LLM_MODEL is empty. Set it in .env (or pass --llm-model), for "
            "example one of the free OpenRouter models documented in README.md."
        )
    return OpenAiCompatProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=resolved_model,
        timeout=settings.llm_timeout_seconds,
        retry_config=RetryConfig(
            max_tries=settings.max_retries,
            base_delay=settings.retry_base_delay,
            max_delay=settings.retry_max_delay,
        ),
        on_retry=on_retry,
        warn=warn,
    )
