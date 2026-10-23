"""Shared retry/backoff machinery for OCR and LLM HTTP calls.

Both stages must retry on 429 (rate limit) and 5xx with exponential backoff,
at most ``MAX_RETRIES`` tries (SPEC 6, 8.1). Free tiers throttle aggressively,
so backoff grows geometrically with jitter and callers can listen to retries.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")


class RetriableError(Exception):
    """A transient failure (429 rate limit, 5xx) worth retrying."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class QuotaExceededError(RetriableError):
    """The provider's daily/monthly quota is exhausted.

    The run stops early; cached work is kept, so ``--resume`` continues where
    the run stopped once the quota resets (SPEC 14).
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        hint = (
            f"{message} The free quota seems exhausted. Wait for the quota to "
            f"reset and re-run the same command with --resume to continue."
        )
        super().__init__(hint, status_code=status_code)


class PermanentError(Exception):
    """A non-retryable failure (invalid key, bad request, ...)."""


@dataclass
class RetryConfig:
    """Backoff parameters. Tests pass a tiny ``sleep`` and ``base_delay``."""

    max_tries: int = 5
    base_delay: float = 2.0
    max_delay: float = 60.0
    sleep: Callable[[float], Awaitable[None]] = field(default=asyncio.sleep)


async def retry_call(
    fn: Callable[[], Awaitable[T]],
    *,
    label: str,
    config: RetryConfig,
    on_retry: Callable[[str], None] | None = None,
) -> T:
    """Await ``fn()`` retrying :class:`RetriableError` with exponential backoff.

    - retries only on :class:`RetriableError` (429/5xx mapped by providers)
    - max ``config.max_tries`` total attempts
    - if the last failure was a 429, raise :class:`QuotaExceededError`, because
      persistent rate limiting on a free tier usually means an exhausted quota
    - any other exception (permanent errors) propagates immediately
    """
    last_exc: RetriableError | None = None
    for attempt in range(1, config.max_tries + 1):
        try:
            return await fn()
        except RetriableError as exc:  # noqa: PERF203 - retry loop by design
            last_exc = exc
            if attempt >= config.max_tries:
                break
            delay = min(
                config.max_delay,
                config.base_delay * (2 ** (attempt - 1)),
            )
            delay *= random.uniform(0.8, 1.2)  # jitter: avoid thundering herd
            if on_retry is not None:
                on_retry(
                    f"{label}: attempt {attempt}/{config.max_tries} failed "
                    f"({exc}); retrying in {delay:.1f}s"
                )
            await config.sleep(delay)
    assert last_exc is not None
    if last_exc.status_code == 429:
        raise QuotaExceededError(str(last_exc), status_code=429) from last_exc
    raise RetriableError(
        f"{label}: giving up after {config.max_tries} attempts: {last_exc}",
        status_code=last_exc.status_code,
    ) from last_exc
