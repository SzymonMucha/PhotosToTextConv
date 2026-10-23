"""OCR provider interface (SPEC 6).

One abstract provider, many implementations. Concrete providers translate the
pipeline's Polish language code (``BOOK_LANGUAGE=pl``) into their own codes
(e.g. ``pol`` for OCR.space) *inside* the provider class, as required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class OcrResult:
    """Typed output of one OCR call."""

    text: str
    confidence: float | None  # filled only if the provider supplies it
    raw: dict = field(default_factory=dict)  # raw provider response (debugging)


class OcrProvider(ABC):
    """Provider of text extraction for one page image.

    Implementations must raise :class:`~book2md.retry.RetriableError` for
    transient HTTP problems (429/5xx) and
    :class:`~book2md.retry.PermanentError` for non-retryable ones; the retry
    machinery lives in :mod:`book2md.retry`.
    """

    #: short machine-readable name, used in cache metadata and summary
    name: str = "ocr"

    @abstractmethod
    async def ocr_image(self, image_bytes: bytes, filename: str) -> OcrResult:
        """OCR one prepared image and return the result (may raise)."""
