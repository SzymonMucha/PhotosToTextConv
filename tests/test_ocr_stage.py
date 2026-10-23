"""OCR stage tests (SPEC 6/7): fake providers, caches, retries, merge."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from PIL import Image

from book2md import pipeline
from book2md.ocr.base import OcrResult
from book2md.ordering import order_folder
from book2md.retry import PermanentError, RetriableError, RetryConfig
from conftest import set_mtime


class CountingProvider:
    """Fake OCR provider that counts calls and returns per-filename text.

    Mirrors the real provider contract from SPEC 6: transient errors are
    retried with backoff *inside* the provider (see :func:`book2md.retry`).
    """

    name = "fake-ocr"

    def __init__(self, default_text: str = "") -> None:
        self.default_text = default_text
        self.calls: list[str] = []
        self.retry_config = RetryConfig(max_tries=5, base_delay=0.001)

    async def ocr_image(self, image_bytes: bytes, filename: str) -> OcrResult:
        self.calls.append(filename)
        retry = getattr(self, "retry_config", None) or RetryConfig(max_tries=3, base_delay=0.001)

        async def attempt():
            text = self.default_text.replace("{name}", filename)
            return OcrResult(text=text, confidence=0.8, raw={"fake": True})

        from book2md.retry import retry_call

        return await retry_call(attempt, label=f"fake-ocr[{filename}]", config=retry)


class FlakyProvider(CountingProvider):
    """Fails for specific files, optionally only on the first attempts."""

    def __init__(self, fail_files: set[str], fail_times: int = 10**9, error=None) -> None:
        super().__init__(default_text="ok {name}")
        self.fail_files = fail_files
        self.fail_times = fail_times
        self.attempts: dict[str, int] = {}
        self.error = error or PermanentError("boom")

    async def ocr_image(self, image_bytes: bytes, filename: str) -> OcrResult:
        self.calls.append(filename)

        async def attempt():
            self.attempts[filename] = self.attempts.get(filename, 0) + 1
            if filename in self.fail_files and self.attempts[filename] <= self.fail_times:
                raise self.error
            text = self.default_text.replace("{name}", filename)
            return OcrResult(text=text, confidence=0.8, raw={"fake": True})

        from book2md.retry import retry_call

        return await retry_call(
            attempt, label=f"fake-ocr[{filename}]", config=self.retry_config
        )


def staged(tmp_path: Path):
    """Order a 3-page photo folder and return (pages, settings, workdir)."""
    base = tmp_path / "photos"
    base.mkdir()
    for i, name in enumerate(("a.jpg", "b.jpg", "c.jpg")):
        # strongly distinct colors: JPEG would quantize subtle color
        # differences away, producing identical bytes (=> same SHA-1)
        Image.new("RGB", (40, 52), color=(i * 120, 40, 80)).save(base / name, "JPEG")
    now = time.time()
    set_mtime(base / "a.jpg", now - 300)
    set_mtime(base / "b.jpg", now - 200)
    set_mtime(base / "c.jpg", now - 100)
    pages, _strategy, _dup = order_folder(base)
    settings = make_settings(tmp_path)
    workdir = tmp_path / "workdir" / "photos"
    workdir.mkdir(parents=True)
    return pages, settings, workdir


def make_settings(tmp_path, **overrides):
    from conftest import make_settings as base_make

    return base_make(tmp_path, **overrides)


def run_ocr(pages, settings, workdir, reporter, provider):
    return asyncio.run(
        pipeline.stage_ocr(
            pages,
            provider,
            workdir=workdir,
            settings=settings,
            reporter=reporter,
        )
    )


def test_ocr_calls_provider_and_caches_per_image(tmp_path, reporter):
    pages, settings, workdir = staged(tmp_path)
    provider = CountingProvider(default_text="Tekst {name} z OCR.")
    statuses = run_ocr(pages, settings, workdir, reporter, provider)
    assert [s.status for s in statuses] == ["ok", "ok", "ok"]
    assert len(provider.calls) == 3
    cache_files = sorted((workdir / "ocr").iterdir())
    assert len(cache_files) == 3
    data = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert data["text"] == "Tekst a.jpg z OCR."
    assert data["sha1"] == pages[0].sha1


def test_second_run_serves_everything_from_cache(tmp_path, reporter):
    pages, settings, workdir = staged(tmp_path)
    provider = CountingProvider(default_text="Tekst {name}.")
    run_ocr(pages, settings, workdir, reporter, provider)
    statuses = run_ocr(pages, settings, workdir, reporter, provider)
    assert len(provider.calls) == 3  # zero new provider calls on re-run
    assert all(s.cached for s in statuses)
    assert [s.text for s in statuses] == [f"Tekst {n}." for n in ("a.jpg", "b.jpg", "c.jpg")]


def test_one_failed_page_never_aborts_run(tmp_path, reporter):
    pages, settings, workdir = staged(tmp_path)
    provider = FlakyProvider(fail_files={"b.jpg"})  # fails all 5 retries
    statuses = run_ocr(pages, settings, workdir, reporter, provider)
    failed = [s for s in statuses if s.status == "failed"]
    assert [s.page for s in failed] == [2]
    assert "boom" in failed[0].error
    # successful pages still ran (SPEC 6: continue with the others)
    assert sum(1 for s in statuses if s.status == "ok") == 2
    reporter.warnings  # at least one warning was surfaced (SPEC 6)
    assert any("b.jpg" in w for w in reporter.warnings)


def test_failed_page_is_not_cached_and_retried_next_run(tmp_path, reporter):
    pages, settings, workdir = staged(tmp_path)
    provider = FlakyProvider(fail_files={"b.jpg"})
    run_ocr(pages, settings, workdir, reporter, provider)
    page2 = pages[1]
    page2_cache = workdir / "ocr" / f"{page2.page:05d}_{page2.sha1}.json"
    assert not page2_cache.exists()  # failures stay uncached (SPEC 6)
    assert sum(1 for f in (workdir / "ocr").iterdir()) == 2  # pages 1, 3 cached
    # now let page 2 succeed; re-run must call it again and succeed
    provider2 = CountingProvider(default_text="Tekst {name}.")
    statuses = run_ocr(pages, settings, workdir, reporter, provider2)
    assert statuses[1].status == "ok"  # page 2 recovered
    assert ["b.jpg"] == [name for name in provider2.calls]  # only page 2 re-OCR'd


def test_retriable_error_retried_with_backoff_then_succeeds(tmp_path, reporter):
    pages, settings, workdir = staged(tmp_path)
    retriable = RetriableError("http 500", status_code=500)
    provider = FlakyProvider(
        fail_files={"b.jpg"}, fail_times=2, error=retriable  # succeed on 3rd try
    )
    provider.retry_config = RetryConfig(max_tries=5, base_delay=0.001)
    statuses = run_ocr(pages, settings, workdir, reporter, provider)
    assert statuses[1].status == "ok"
    assert provider.attempts["b.jpg"] == 3


def test_retry_exhaustion_yields_failed_pages_not_cache_files(tmp_path, reporter):
    pages, settings, workdir = staged(tmp_path)
    provider = FlakyProvider(
        fail_files={"b.jpg"}, error=RetriableError("http 503", status_code=503)
    )
    provider.retry_config = RetryConfig(max_tries=2, base_delay=0.001)
    statuses = run_ocr(pages, settings, workdir, reporter, provider)
    assert statuses[1].status == "failed"


def test_merge_writes_markers_and_fixes_ocr_output(tmp_path, reporter):
    pages, settings, workdir = staged(tmp_path)
    provider = CountingProvider(default_text="Zdanie {name}.")
    statuses = run_ocr(pages, settings, workdir, reporter, provider)
    raw_text = pipeline.stage_merge(statuses, workdir / "01_raw.txt", reporter)
    assert raw_text.startswith("<<<PAGE 1>>>\nZdanie a.jpg.")
    for n in (1, 2, 3):
        assert f"<<<PAGE {n}>>>" in raw_text
    # re-merge identical content must not rewrite the file
    raw_path = workdir / "01_raw.txt"
    mtime1 = raw_path.stat().st_mtime_ns
    pipeline.stage_merge(statuses, raw_path, reporter)
    assert raw_path.stat().st_mtime_ns == mtime1


def test_merge_marks_failed_by_placeholder(tmp_path, reporter):
    pages, settings, workdir = staged(tmp_path)
    provider = FlakyProvider(fail_files={"b.jpg"})
    statuses = run_ocr(pages, settings, workdir, reporter, provider)
    raw_text = pipeline.stage_merge(statuses, workdir / "01_raw.txt", reporter)
    assert "<<<PAGE 2>>>\n[[OCR FAILED]]" in raw_text


def test_empty_ocr_is_placeholder_not_error(tmp_path, reporter):
    pages, settings, workdir = staged(tmp_path)
    provider = CountingProvider(default_text="")  # blank pages
    statuses = run_ocr(pages, settings, workdir, reporter, provider)
    assert all(s.status == "empty" for s in statuses)
    raw_text = pipeline.stage_merge(statuses, workdir / "01_raw.txt", reporter)
    assert "[[EMPTY PAGE]]" in raw_text


def test_merge_normalizes_line_endings_and_whitespace(tmp_path, reporter):
    from book2md.pipeline import OcrPageStatus, stage_merge

    statuses = [
        OcrPageStatus(page=1, filename="a.jpg", sha1="x", status="ok", text="ala  \r\nma\rkota \r\n"),
    ]
    text = stage_merge(statuses, tmp_path / "01_raw.txt", reporter)
    assert text == "<<<PAGE 1>>>\nala\nma\nkota\n"
