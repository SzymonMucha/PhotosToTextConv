"""Generic OpenAI-compatible chat-completions client (SPEC 3).

Speaks the ``POST {base_url}/chat/completions`` dialect used by OpenRouter,
Groq, Gemini's compatibility endpoint, local LM Studio/Ollama servers, etc.
Retries 429/5xx with exponential backoff (max tries per :class:`RetryConfig`);
persistent 429 becomes a :class:`QuotaExceededError` so ``--resume`` can be
suggested (SPEC 14).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ..retry import (
    PermanentError,
    RetriableError,
    RetryConfig,
    retry_call,
)
from .base import LlmProvider


class OpenAiCompatProvider(LlmProvider):
    """Minimal, dependency-free OpenAI-compatible client over ``httpx``."""

    name = "openai-compat"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 300.0,
        temperature: float = 0.0,
        retry_config: RetryConfig | None = None,
        on_retry=None,
        warn=None,
        extra_headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._model = model
        self.timeout = timeout
        self.temperature = temperature
        self.retry_config = retry_config or RetryConfig()
        self.on_retry = on_retry
        self.warn = warn
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Optional OpenRouter attribution headers (free-tier friendly).
            "HTTP-Referer": "https://github.com/book2md",
            "X-Title": "book2md",
        }
        if extra_headers:
            headers.update(extra_headers)
        self.headers = headers
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=timeout, write=60.0, pool=30.0)
        )

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self, *, system: str, user: str, temperature: float | None = None
    ) -> str:
        """One chat-completions call with retries; returns the message content."""
        resolved_temperature = self.temperature if temperature is None else temperature
        return await retry_call(
            lambda: self._send(system, user, resolved_temperature),
            label=f"llm[{self._model}]",
            config=self.retry_config,
            on_retry=self.on_retry,
        )

    async def _send(self, system: str, user: str, temperature: float) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        url = f"{self.base_url}/chat/completions"
        try:
            resp = await self._client.post(url, headers=self.headers, json=payload)
        except httpx.TimeoutException as exc:
            raise RetriableError(f"LLM request timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise RetriableError(f"LLM network error: {exc}") from exc

        text = resp.text
        if resp.status_code == 429 or resp.status_code >= 500:
            raise RetriableError(
                f"LLM endpoint returned HTTP {resp.status_code}: {_snippet(text)}",
                status_code=resp.status_code,
            )
        if resp.status_code >= 400:
            detail = _server_error_message(text, resp.status_code)
            raise PermanentError(f"LLM endpoint rejected the request: {detail}")

        try:
            data = resp.json()
        except ValueError as exc:
            raise PermanentError(f"LLM endpoint returned non-JSON body: {_snippet(text)}") from exc
        return self._extract_answer(data)

    @staticmethod
    def _extract_answer(data: dict) -> str:
        """Pull ``choices[0].message.content`` out of the response.

        ``finish_reason == "length"`` means truncation; content is returned
        anyway (saved as-is for the user to review in the output file).
        """
        error = data.get("error") or {}
        if isinstance(error, dict) and error.get("message"):
            raise PermanentError(f"LLM endpoint error: {error['message']}")
        choices = data.get("choices") or []
        if not choices:
            raise PermanentError(f"LLM response has no choices: {_snippet(json.dumps(data))}")
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if content is None:
            raise PermanentError("LLM response message has no content.")
        if not isinstance(content, str):
            raise PermanentError("LLM content is not text; unsupported provider response.")
        finish_reason = (choices[0] or {}).get("finish_reason")
        if finish_reason == "length":
            # Not fatal here: the user can see and finish the truncated part
            # in the output file; nothing is dropped silently.
            pass
        return content


def _snippet(text: str, limit: int = 300) -> str:
    return " ".join(text.split())[:limit]


def _server_error_message(text: str, status_code: int) -> str:
    """Best-effort human message out of a 4xx body."""
    try:
        data = json.loads(text)
    except ValueError:
        return f"HTTP {status_code}: {_snippet(text)}"
    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        if isinstance(err, dict) and err.get("message"):
            message = str(err["message"])
            lowered = message.lower()
            if status_code == 401 or "api key" in lowered or "auth" in lowered:
                return f"authentication problem: {message}"
            return message
    return f"HTTP {status_code}: {_snippet(text)}"
