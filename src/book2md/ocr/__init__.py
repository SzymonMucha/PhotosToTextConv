"""OCR providers (SPEC 6)."""

from __future__ import annotations

from ..config import ConfigError, Settings
from ..retry import RetryConfig
from .base import OcrProvider, OcrResult
from .ocrspace import OcrSpaceProvider

__all__ = ["OcrProvider", "OcrResult", "OcrSpaceProvider", "build_ocr_provider"]


def build_ocr_provider(name, settings: Settings, on_retry=None, warn=None) -> OcrProvider:
    """Factory: resolve a provider name against :class:`Settings`, or fail clearly.

    One construction path for CLI (and later web/desktop front-ends). Concrete
    providers map the pipeline's ``BOOK_LANGUAGE=pl`` into their own codes
    internally (SPEC 6).
    """
    if name != "ocrspace":
        raise ConfigError(
            f"Unknown OCR_PROVIDER {name!r}. Available providers: ocrspace."
        )
    return OcrSpaceProvider(
        api_key=settings.ocr_api_key,
        language=settings.ocr_language,
        engine=settings.ocr_engine,
        timeout=settings.ocr_timeout_seconds,
        retry_config=RetryConfig(
            max_tries=settings.max_retries,
            base_delay=settings.retry_base_delay,
            max_delay=settings.retry_max_delay,
        ),
        on_retry=on_retry,
        warn=warn,
    )
