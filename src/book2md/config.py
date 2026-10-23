"""Configuration loaded from environment variables / a ``.env`` file.

Settings mirror step 11 of SPEC.md. Every value can be overridden by the CLI
where it makes sense (e.g. ``--llm-model``), so the core package stays free of
UI code. API keys never live in code: they come from the environment.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: OCR provider implementations that ship with book2md.
AVAILABLE_OCR_PROVIDERS = ("ocrspace",)


class Settings(BaseSettings):
    """All configuration knobs, loaded from the environment or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OCR (free-tier API) -------------------------------------------------
    ocr_provider: str = "ocrspace"
    ocr_api_key: str = ""
    #: Language code for the OCR provider. OCR.space uses 3-letter codes.
    ocr_language: str = "pol"
    #: OCREngine used by OCR.space ("1" legacy, "2" newer/preferred).
    ocr_engine: str = "2"
    #: Language of the book itself (prompts are tuned for Polish).
    book_language: str = "pl"
    ocr_concurrency: int = Field(default=4, ge=1, le=32)
    # Image preparation for the provider's size limits (SPEC 6):
    ocr_max_long_side: int = 2000     # downscale long side to this many px
    ocr_jpeg_quality: int = 85       # initial JPEG re-encode quality
    ocr_max_bytes: int = 900_000     # OCR.space free tier: ~1 MB payload limit
    ocr_timeout_seconds: float = 90.0

    # --- LLM (OpenAI-compatible chat completions, e.g. OpenRouter) ----------
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_concurrency: int = Field(default=2, ge=1, le=16)
    llm_timeout_seconds: float = 300.0
    #: Target size of one LLM chunk in words (chunks are cut on page markers).
    chunk_words: int = Field(default=2000, ge=300, le=3000)

    # --- Retries (429 / 5xx, exponential backoff, SPEC 6/8.1) ---------------
    max_retries: int = Field(default=5, ge=1, le=10)
    retry_base_delay: float = 2.0
    retry_max_delay: float = 60.0


def load_settings() -> Settings:
    """Load settings from the environment / ``.env``.

    Raises a clear error if the environment file cannot be parsed.
    """
    return Settings()


def require_value(value: str, what: str) -> str:
    """Return ``value`` or raise :class:`ConfigError` with setup guidance."""
    if not (value or "").strip():
        raise ConfigError(
            f"{what} is not set. Copy .env.example to .env and fill it in "
            f"(see README.md)."
        )
    return value.strip()


class ConfigError(RuntimeError):
    """A configuration problem the user has to fix (missing key, etc.)."""
