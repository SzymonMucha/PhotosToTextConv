"""OCR.space provider (first concrete OCR implementation, SPEC 6).

The free OCR.space tier is a plain REST API with a ~1 MB file limit, which is
why images are pre-compressed by :mod:`book2md.imageprep` before upload.

Polish note (SPEC 6): OCR.space uses 3-letter language codes; Polish is
``pol``. The pipeline's ``BOOK_LANGUAGE=pl`` is mapped here, inside the
provider class. The language can be overridden per environment through
``OCR_LANGUAGE`` if the user's provider behaves differently.
"""

from __future__ import annotations

import httpx

from ..config import ConfigError
from ..retry import PermanentError, RetriableError, RetryConfig, retry_call
from .base import OcrProvider, OcrResult

#: POST target of the OCR.space engine (the engine default is 1; engine 2 is
#: newer and generally better on photos).
_API_URL = "https://api.ocr.space/parse/image"

# Words inside the provider payload that indicate quota/rate-limit errors.
_RATE_LIMIT_HINTS = ("rate limit", "daily", "quota", "requests limit", "too many")


class OcrSpaceProvider(OcrProvider):
    """Upload an image and return its text (with retries on 429/5xx)."""

    name = "ocrspace"

    def __init__(
        self,
        api_key: str,
        *,
        language: str = "pol",
        engine: str = "2",
        timeout: float = 90.0,
        retry_config: RetryConfig | None = None,
        on_retry=None,
        warn=None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ConfigError(
                "OCR_API_KEY is empty. Register at https://ocr.space/OCRAPI "
                "for a free key and put it in .env (see README.md)."
            )
        self.api_key = api_key.strip()
        self.language = language.strip() or "pol"
        self.engine = str(engine)
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self.on_retry = on_retry
        self.warn = warn  # used for soft problems, e.g. engine fallbacks
        self._client = http_client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ocr_image(self, image_bytes: bytes, filename: str) -> OcrResult:
        """OCR one image; raise on failure. Text is returned verbatim."""
        return await retry_call(
            lambda: self._send(image_bytes, filename),
            label=f"ocr[{filename}]",
            config=self.retry_config,
            on_retry=self.on_retry,
        )

    async def _send(self, image_bytes: bytes, filename: str) -> OcrResult:
        data: dict[str, str] = {
            "apikey": self.api_key,
            "language": self.language,  # 'pol' for Polish
            "OCREngine": self.engine,
            "isOverlayRequired": "false",  # smaller responses; we don't need boxes
            "detectOrientation": "true",
            "scale": "true",  # let the engine upscale low-res photos
        }
        files = {"file": (filename, image_bytes, "image/jpeg")}
        try:
            resp = await self._client.post(_API_URL, data=data, files=files)
        except httpx.TimeoutException as exc:
            raise RetriableError(f"OCR request timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise RetriableError(f"OCR network error: {exc}") from exc

        # HTTP-level problems first.
        if resp.status_code in (429,) or resp.status_code >= 500:
            raise RetriableError(
                f"OCR.space returned HTTP {resp.status_code}", status_code=resp.status_code
            )
        if resp.status_code in (401, 403):
            raise PermanentError(
                "OCR.space rejected the API key (HTTP {code}); check OCR_API_KEY.".format(
                    code=resp.status_code
                )
            )
        if resp.status_code >= 400:
            raise PermanentError(
                f"OCR.space rejected the request (HTTP {resp.status_code}): "
                f"{_body_snippet(resp.text)}"
            )

        try:
            payload = resp.json()
        except ValueError as exc:
            raise PermanentError(
                f"OCR.space returned non-JSON response: {_body_snippet(resp.text)}"
            ) from exc
        return self._parse(payload)

    def _parse(self, payload: dict) -> OcrResult:
        """Turn the OCR.space JSON into an :class:`OcrResult`.

        The API signals most failures with HTTP 200 plus
        ``IsErroredOnParsing`` / ``ErrorMessage``, so this is not just unwrapping.
        """
        errors: list[str] | str = payload.get("ErrorMessage") or []
        if isinstance(errors, str):
            errors = [errors] if errors else []
        errored = bool(payload.get("IsErroredOnParsing")) or bool(errors)
        if errored:
            msg = "; ".join(errors) or "unknown OCR.space error"
            lowered = msg.lower()
            if any(hint in lowered for hint in _RATE_LIMIT_HINTS):
                # Free tier: daily request limit reached -> retriable quota stop.
                raise RetriableError(f"OCR.space: {msg}", status_code=429)
            raise PermanentError(f"OCR.space: {msg}")

        results = payload.get("ParsedResults") or []
        exit_code = str(payload.get("OCRExitCode", "0"))
        if not results:
            text = ""
        else:
            first = results[0]
            text = (first.get("ParsedText") or "").strip()
            if str(first.get("FileParseExitCode", "1")) not in ("1", 1):
                # Engine parsed the file but produced no usable text.
                message = "; ".join(first.get("ErrorMessage") or "")
                if message:
                    raise PermanentError(f"OCR.space: {message}")
                text = ""
        if exit_code not in ("1", "2"):  # 1 = success, 2 = partial success
            if text:  # be forgiving: keep text when there is some
                if self.warn:
                    self.warn(f"OCR.space exit code {exit_code}, keeping the text it returned.")
            else:
                raise PermanentError(
                    f"OCR.space parsing failed (OCRExitCode={exit_code})."
                )
        return OcrResult(
            text=text,
            confidence=_average_confidence(results[0] if results else None),
            raw=payload,
        )


def _body_snippet(text: str, limit: int = 200) -> str:
    """Shorten a raw HTTP body for error messages."""
    return text.strip()[:limit]


def _average_confidence(result: dict | None) -> float | None:
    """Average the per-line confidence of TextOverlay when present.

    With ``isOverlayRequired=false`` the overlay is absent and we return
    ``None`` (SPEC: confidence is filled "if provider supplies it").
    """
    if not result:
        return None
    overlay = result.get("TextOverlay") or {}
    lines = overlay.get("Lines") or []
    confidences = [
        line.get("Confidence") for line in lines if line.get("Confidence") is not None
    ]
    if not confidences:
        return None
    return round(sum(confidences) / len(confidences), 2)