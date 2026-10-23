"""Shared pytest fixtures and fake providers.

No test performs a network call: OCR and LLM providers are fakes scripted
in-memory (SPEC 3/16: "All tests pass without network access").
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from book2md.config import Settings

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass
class CollectingReporter:
    """In-memory :class:`book2md.pipeline.Reporter` for assertions."""

    events: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def stage_start(self, name: str, total: int | None = None) -> None:
        self.events.append(("start", name, total))

    def stage_progress(self, n: int = 1, detail: str | None = None) -> None:
        self.events.append(("progress", n, detail))

    def stage_done(self, detail: str | None = None) -> None:
        self.events.append(("done", detail))

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)
        self.events.append(("warning", msg))

    def info(self, msg: str) -> None:
        self.events.append(("info", msg))

    def warnings_of_stage(self) -> list[str]:
        return list(self.warnings)


@pytest.fixture
def reporter() -> CollectingReporter:
    return CollectingReporter()


def make_settings(tmp_path: Path | None = None, **overrides) -> Settings:
    """Settings for tests: dummy keys, tiny backoff, deterministic chunks."""
    kwargs = {
        "ocr_api_key": "dummy-ocr-key",
        "llm_api_key": "dummy-llm-key",
        "llm_base_url": "http://llm.invalid/v1",
        "llm_model": "fake-model",
        "chunk_words": 2000,
        "ocr_max_bytes": 900_000,
        "retry_base_delay": 0.001,  # no slow tests
        "retry_max_delay": 0.005,
        "max_retries": 5,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


# ---------------------------------------------------------------------------
# Fake providers
# ---------------------------------------------------------------------------
class FakeOcrProvider:
    """Scripted OCR provider: maps filename -> text; counts real API work."""

    name = "fake-ocr"
    calls: list[str] = field(default_factory=list)  # filenames actually OCR'd

    def __init__(self, responses: dict[str, str] | None = None, default_text: str = "") -> None:
        self.responses = dict(responses or {})
        self.default_text = default_text
        self.calls = []

    async def ocr_image(self, image_bytes: bytes, filename: str) -> object:
        from book2md.ocr.base import OcrResult

        self.calls.append(filename)
        text = self.responses.get(filename, self.default_text)
        if isinstance(text, Exception):
            raise text
        if callable(text):
            text = text(filename)
        return OcrResult(text=text, confidence=0.9, raw={"fake": True})


class FakeLlmProvider:
    """Scripted LLM: the first scripted answer is consumed per call.

    ``plan`` is a list of answers (or ``(label, answer)`` helpers); when it
    runs dry, ``default`` is returned. ``calls`` records prompts for asserts.
    """

    name = "fake-llm"

    def __init__(self, plan: list[str] | None = None, default: str | None = None) -> None:
        from collections import deque

        self._plan = deque(plan or [])
        self.default = default
        self.calls: list[dict] = []
        self.model = "fake-model"

    async def chat(self, *, system: str, user: str, temperature: float = 0.0):
        from book2md.retry import PermanentError

        self.calls.append({"system": system, "user": user})
        if self._plan:
            answer = self._plan.popleft()
            if isinstance(answer, Exception):
                raise answer
            return answer
        if self.default is None:
            raise PermanentError("fake-llm: no scripted answer left")
        return self.default


def create_image(
    path: Path,
    *,
    size: tuple[int, int] = (120, 160),
    color: str | None = None,
    exif_datetime: datetime | None = None,
    subsec: str | None = None,
) -> Path:
    """Write a tiny JPEG, optionally embedding EXIF DateTimeOriginal (tests).

    ``color=None`` derives a unique-but-stable color from the filename so two
    different files never accidentally produce identical JPEG bytes (which
    would make them look like duplicates by SHA-1).
    """
    import warnings
    from PIL import Image as _Image

    warnings.filterwarnings("ignore", category=UserWarning)
    if color is None:
        digest = sum(path.name.encode("utf-8"))
        color = (digest % 256, (digest // 7) % 256, (digest // 13) % 256)
    img = _Image.new("RGB", size, color=color)
    if exif_datetime is not None:
        exif = img.getexif()
        sub_ifd = exif.get_ifd(0x8769)  # Exif sub-IFD
        sub_ifd[36867] = exif_datetime.strftime("%Y:%m:%d %H:%M:%S")
        if subsec:
            sub_ifd[37520] = subsec
        img.save(path, "JPEG", exif=exif)
    else:
        img.save(path, "JPEG")
    return path


def set_mtime(path: Path, epoch: float) -> None:
    os.utime(path, (epoch, epoch))


@pytest.fixture
def photos_dir(tmp_path: Path) -> Path:
    """A photo folder with 3 plain JPEGs whose mtimes are in page order."""
    base = tmp_path / "photos"
    base.mkdir()
    create_image(base / "a.jpg")
    create_image(base / "b.jpg")
    create_image(base / "c.jpg")
    now = time.time()
    set_mtime(base / "a.jpg", now - 300)
    set_mtime(base / "b.jpg", now - 200)
    set_mtime(base / "c.jpg", now - 100)
    return base


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")
