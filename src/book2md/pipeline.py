"""Stage orchestration (SPEC 4: ``pipeline.py`` orchestrates all stages).

Public entry points:

* :func:`run` - the whole pipeline over a folder of photos
* :func:`ocr_only` - stages 1-3 (order + OCR + merge)
* :func:`convert_file` - stage 4 (one LLM pass: repair + format) on an
  existing raw text file (``book2md convert``)
* :func:`export_file` - stage 5 only (``book2md export``)

Every stage writes artifacts to disk so the user can inspect intermediate
results or hand-edit them; per-page OCR and per-chunk LLM caches make runs
idempotent and resumable. This module contains no UI code - progress is
reported through a tiny :class:`Reporter` interface which the CLI adapts to
rich progress bars.

The single LLM stage ("convert") asks the model to repair the OCR damage and
produce Markdown in one step. There is no validator stage any more: the
prompt itself carries the faithfulness contract (stick to the original
words). Each answer is only *salvaged* (code fences / thinking blocks
stripped) and stored; the sequential per-chunk flow provides the paragraph
continuity the prompt asks for.
"""

from __future__ import annotations

import asyncio
import json
import re as _re
import shutil as _shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from . import export as export_mod
from . import llm as llm_pkg
from . import ocr as ocr_pkg
from .chunking import split_chunks, take_words
from .config import Settings, load_settings
from .formatting import (
    collapse_blank_lines,
    merge_raw_text,
    normalize_heading_levels,
    sha1_text,
    strip_code_fences,
    strip_model_thinking,
    write_if_changed,
)
from .imageprep import prepare_file
from .ocr.base import OcrResult
from .ordering import PageImage, order_folder, write_order_json
from .prompts import CONVERT_SYSTEM_PROMPT, convert_user_prompt
from .retry import PermanentError, QuotaExceededError, RetriableError

#: Working-dir default: artifacts land in ``workdir/<book>/``.
DEFAULT_OUTPUT_DIR = "workdir"


# ---------------------------------------------------------------------------
# Progress reporting protocol (no UI code in the core!)
# ---------------------------------------------------------------------------
class Reporter(Protocol):
    """Progress hooks; the CLI adapts these to rich, tests collect events."""

    def stage_start(self, name: str, total: int | None = None) -> None: ...
    def stage_progress(self, n: int = 1, detail: str | None = None) -> None: ...
    def stage_done(self, detail: str | None = None) -> None: ...
    def warning(self, msg: str) -> None: ...
    def info(self, msg: str) -> None: ...


class SilentReporter:
    """No-op reporter for library use (import book2md, not the CLI)."""

    def stage_start(self, name: str, total: int | None = None) -> None:
        pass

    def stage_progress(self, n: int = 1, detail: str | None = None) -> None:
        pass

    def stage_done(self, detail: str | None = None) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass


# ---------------------------------------------------------------------------
# Run summary (final "pages OK/failed, chunks..., output paths")
# ---------------------------------------------------------------------------
@dataclass
class RunSummary:
    """Everything the CLI needs for the final summary print."""

    book_name: str = ""
    workdir: Path | None = None
    order_strategy: str = ""
    pages_total: int = 0
    pages_ocr_now: int = 0
    pages_cached: int = 0
    pages_failed: list[str] = field(default_factory=list)
    pages_empty: int = 0
    duplicates_skipped: int = 0
    convert_chunks: int = 0
    outputs: dict[str, str] = field(default_factory=dict)
    stopped_after: str | None = None


# ---------------------------------------------------------------------------
# Stage 2+3: OCR with per-page cache, concurrency, merge to 01_raw.txt
# ---------------------------------------------------------------------------
@dataclass
class OcrPageStatus:
    """Result of the OCR stage for one page (cached or fresh)."""

    page: int
    filename: str
    sha1: str
    status: str = "ok"  # "ok" | "empty" | "failed"
    text: str = ""
    confidence: float | None = None
    error: str | None = None
    cached: bool = False


def _ocr_cache_path(workdir: Path, page: PageImage) -> Path:
    """``ocr/<index>_<sha1>.json`` (SPEC 6)."""
    return workdir / "ocr" / f"{page.page:05d}_{page.sha1}.json"


async def stage_ocr(
    pages: list[PageImage],
    provider,
    *,
    workdir: Path,
    settings: Settings,
    reporter: Reporter,
    imageprep_kwargs: dict | None = None,
) -> list[OcrPageStatus]:
    """OCR all pages concurrently with caching; failures never abort the run.

    Per SPEC 6: OCR calls run concurrently behind a semaphore, each call is
    retried with backoff inside the provider, one page failing after retries
    is recorded as failed (``[[OCR FAILED]]`` at merge) but never aborts the
    run. An exhausted quota raises :class:`QuotaExceededError`, cancelling the
    sibling tasks via ``asyncio.TaskGroup`` so no further quota is burned.
    """
    ocr_dir = workdir / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(settings.ocr_concurrency)
    results: dict[int, OcrPageStatus] = {}
    reporter.stage_start("ocr", total=len(pages))

    async def one(page: PageImage) -> None:
        cache_file = _ocr_cache_path(workdir, page)
        if cache_file.is_file():  # idempotency: skip entirely (SPEC 2)
            entry = _load_ocr_cache_or_none(cache_file, page)
            if entry is not None:
                results[page.page] = entry
                reporter.stage_progress(1, detail=f"page {page.page}: cached")
                return

        # Image preparation is CPU work; do it before taking a worker slot.
        prepared = prepare_file(page.path, **(imageprep_kwargs or {}))
        async with semaphore:
            try:
                ocr_result = await provider.ocr_image(prepared.data, page.path.name)
            except QuotaExceededError:
                # Must be re-raised *before* the generic handler below: quota
                # exhaustion aborts the whole run (SPEC 14) instead of letting
                # sibling pages burn the remaining quota as "failed" pages.
                raise
            except (
                RetriableError,
                PermanentError,
            ) as exc:  # retries already exhausted inside the provider
                results[page.page] = OcrPageStatus(
                    page=page.page,
                    filename=page.path.name,
                    sha1=page.sha1,
                    status="failed",
                    error=str(exc),
                )
                reporter.stage_progress(1, detail=f"page {page.page}: FAILED")
                reporter.warning(
                    f"page {page.page} ({page.path.name}) OCR failed: {exc}"
                )
                return
        # QuotaExceededError is NOT caught: it aborts the run (TaskGroup
        # cancels the sibling tasks), the caches stay resumable (SPEC 14).
        results[page.page] = _save_ocr_cache(cache_file, page, ocr_result, provider)
        reporter.stage_progress(1, detail=f"page {page.page}: ocr ok")

    try:
        async with asyncio.TaskGroup() as group:
            for page in pages:
                group.create_task(one(page))
    except* QuotaExceededError as excgroup:  # re-raise typed, cleanly unwrapped
        raise excgroup.exceptions[0]

    return [results[p.page] for p in sorted(pages, key=lambda x: x.page) if p.page in results]


def _load_ocr_cache_or_none(cache_file: Path, page: PageImage) -> OcrPageStatus | None:
    """Read one per-page OCR cache entry; ``None`` when corrupt/stale."""
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        if str(data.get("sha1")) != page.sha1:
            return None  # image changed (or file reassigned): stale cache
        text = str(data.get("text", ""))
        status = str(data.get("status", "ok"))
        return OcrPageStatus(
            page=page.page,
            filename=page.filename,
            sha1=page.sha1,
            status=status if status in ("ok", "empty") else "ok",
            text=text,
            confidence=data.get("confidence"),
            cached=True,  # loaded from the per-page cache, no API call (SPEC 2)
        )
    except (ValueError, KeyError, OSError):
        return None  # unreadable/corrupt cache -> simply re-OCR this page


def _save_ocr_cache(
    cache_file: Path, page: PageImage, result: OcrResult, provider
) -> OcrPageStatus:
    """Persist one OCR result (``ocr/<idx>_<sha1>.json``, SPEC 6) and return
    the :class:`OcrPageStatus` describing it."""
    raw = result.raw if isinstance(result.raw, dict) else {"raw": result.raw}
    record = {
        "page": page.page,
        "sha1": page.sha1,
        "filename": page.filename,
        "provider": getattr(provider, "name", "ocr"),
        "text": result.text,
        "confidence": result.confidence,
        "raw": raw,  # kept for debugging; may be trimmed in a later version
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text = result.text
    return OcrPageStatus(
        page=page.page,
        filename=page.filename,
        sha1=page.sha1,
        status="ok" if text.strip() else "empty",
        text=text,
        confidence=result.confidence,
    )


def stage_merge(
    statuses: list[OcrPageStatus], raw_path: Path, reporter: Reporter
) -> str:
    """Stage 3: concatenate pages into ``01_raw.txt`` with markers (SPEC 7).

    Failed pages become ``[[OCR FAILED]]``; empty OCR pages ``[[EMPTY PAGE]]``
    (SPEC 14) - both computed here so the merge code itself stays dumb.
    """
    pairs = []
    for status in statuses:
        body = "[[OCR FAILED]]" if status.status == "failed" else status.text
        pairs.append((status.page, body))
    text = merge_raw_text(pairs)
    changed = write_if_changed(raw_path, text)
    reporter.stage_done(
        detail=f"{raw_path.name} ({'changed' if changed else 'unchanged'})"
    )
    return text


# ---------------------------------------------------------------------------
# Stage 4: the single LLM pass (repair + format in one), per-chunk cache
# ---------------------------------------------------------------------------
@dataclass
class ConvertStageResult:
    text: str
    chunks: int


async def convert_stage(
    raw_text: str,
    *,
    provider,
    cache_dir: Path,
    target_path: Path,
    settings: Settings,
    reporter: Reporter,
) -> ConvertStageResult:
    """Convert chunks of ``raw_text`` into Markdown in ONE LLM pass.

    Chunks are processed *sequentially* because each prompt carries the
    previous chunk's converted tail: paragraphs crossing the chunk boundary
    must flow together (the prompt asks for this explicitly). Free-tier
    quotas stay happier this way too. Each answer is salvage-cleaned (fences
    and thinking blocks removed) and stored in a per-chunk cache keyed by the
    input SHA-1, so re-runs make zero LLM calls.
    """
    chunks = split_chunks(raw_text, settings.chunk_words)
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    reporter.stage_start("convert", total=len(chunks))
    previous_tail = ""

    for chunk in chunks:
        cache_file = cache_dir / f"chunk_{chunk.index:04d}_{sha1_text(chunk.text)}.json"
        cached = _read_chunk_cache(cache_file) if cache_file.is_file() else None
        if cached is not None:
            output = str(cached["output"])
            reporter.stage_progress(1, detail=f"chunk {chunk.index}: cached")
        else:
            prompt = convert_user_prompt(
                text_to_convert=chunk.text, context_before=previous_tail
            )
            answer = strip_model_thinking(
                strip_code_fences(
                    await provider.chat(system=CONVERT_SYSTEM_PROMPT, user=prompt)
                )
            )
            output = answer
            reporter.stage_progress(1, detail=f"chunk {chunk.index}: ok")
            cache_file.write_text(
                json.dumps(
                    {
                        "chunk": chunk.index,
                        "pages": [chunk.first_page, chunk.last_page],
                        "input_sha1": sha1_text(chunk.text),
                        "output": output,
                        "model": getattr(provider, "model", ""),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        joined = output.strip("\n")
        outputs.append(joined)
        previous_tail = take_words(joined, 300, end=True)

    markdown = "\n\n".join(outputs) + "\n"
    # Global cleanups after all chunks: heading-level normalization (no
    # '#'-to-'###' jumps) and blank-line collapsing.
    markdown = normalize_heading_levels(markdown)
    markdown = collapse_blank_lines(markdown)
    changed = write_if_changed(target_path, markdown)
    reporter.stage_done(
        detail=f"{target_path.name} ({'changed' if changed else 'unchanged'})"
    )
    return ConvertStageResult(text=markdown, chunks=len(chunks))


def _read_chunk_cache(cache_file: Path) -> dict | None:
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return data if "output" in data else None
    except (ValueError, KeyError, OSError):
        return None


# ---------------------------------------------------------------------------
# Stage orchestration: the public API
# ---------------------------------------------------------------------------
def sanitize_book_name(folder_name: str) -> str:
    """Folder name -> filesystem-friendly book workdir name."""
    cleaned = _re.sub(r"[^\w.-]+", "_", folder_name.strip()).strip("._-")
    return cleaned or "book"


def _clear_for_no_resume(workdir: Path, reporter: Reporter) -> None:
    """``--no-resume``: drop caches so every page/chunk is recomputed."""
    for sub in ("ocr", "cache"):
        target = workdir / sub
        if target.exists():
            _shutil.rmtree(target)
    reporter.info("no-resume: cleared ocr/ and cache/ - everything re-processed")


async def run(
    input_dir: Path,
    *,
    out_dir: Path | None = None,
    title: str | None = None,
    order_by: str | None = None,
    reverse: bool = False,
    keep_duplicates: bool = False,
    stop_after: str | None = None,
    resume: bool = True,
    ocr_provider=None,
    ocr_provider_name: str | None = None,
    llm_provider=None,
    llm_model: str | None = None,
    settings: Settings | None = None,
    reporter: Reporter | None = None,
) -> RunSummary:
    """Run the full pipeline: order -> OCR -> merge -> convert -> export.

    ``stop_after`` bounds work: ``order`` | ``ocr`` (stages 1-3) | ``convert``
    (through ``03_formatted.md`` on raw text). Provider instances may be
    injected for tests; otherwise they are built from settings (.env) lazily
    per stage.
    """
    if stop_after not in (None, "order", "ocr", "convert"):
        raise ValueError(f"invalid --stop-after {stop_after!r} (order|ocr|convert)")
    settings = settings or load_settings()
    reporter = reporter or SilentReporter()
    summary = RunSummary()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    # Book folder name/title defaults come from the photo folder (SPEC 10/11).
    book_name = sanitize_book_name(input_dir.name)
    workdir = (Path(out_dir) if out_dir else Path(DEFAULT_OUTPUT_DIR)) / book_name
    workdir.mkdir(parents=True, exist_ok=True)
    if not resume:
        _clear_for_no_resume(workdir, reporter)
    summary.book_name = book_name
    summary.workdir = workdir

    # ----- Stage 1: ordering (SPEC 5) --------------------------------------
    reporter.stage_start("order")
    pages, strategy, duplicates = order_folder(
        input_dir,
        order_by=order_by,
        reverse=reverse,
        keep_duplicates=keep_duplicates,
        on_warning=reporter.warning,
    )
    write_order_json(workdir / "order.json", pages, strategy=strategy, reverse=reverse)
    reporter.stage_done(detail=f"{len(pages)} pages ordered by {strategy}")
    summary.order_strategy = strategy
    summary.pages_total = len(pages)
    summary.duplicates_skipped = duplicates
    summary.outputs["order.json"] = str(workdir / "order.json")
    if stop_after == "order":
        summary.stopped_after = stop_after
        return summary

    # ----- Stage 2: OCR (SPEC 6) -------------------------------------------
    provider = ocr_provider or ocr_pkg.build_ocr_provider(
        ocr_provider_name or settings.ocr_provider,
        settings,
        on_retry=lambda msg: reporter.warning(msg),
        warn=reporter.warning,
    )
    statuses = await stage_ocr(
        pages,
        provider,
        workdir=workdir,
        settings=settings,
        reporter=reporter,
        imageprep_kwargs={
            "max_long_side": settings.ocr_max_long_side,
            "jpeg_quality": settings.ocr_jpeg_quality,
            "max_bytes": settings.ocr_max_bytes,
        },
    )
    summary.pages_ocr_now = sum(1 for s in statuses if not s.cached)
    summary.pages_cached = sum(1 for s in statuses if s.cached)
    summary.pages_failed = [
        f"page {s.page} ({s.filename}): {s.error}" for s in statuses if s.status == "failed"
    ]
    summary.pages_empty = sum(1 for s in statuses if s.status == "empty")

    # ----- Stage 3: merge to 01_raw.txt (SPEC 7) ----------------------------
    reporter.stage_start("merge")
    raw_text = stage_merge(statuses, workdir / "01_raw.txt", reporter)
    summary.outputs["01_raw.txt"] = str(workdir / "01_raw.txt")
    if stop_after == "ocr":
        summary.stopped_after = stop_after
        await _close_providers(provider)
        return summary

    # ----- Stage 4: one LLM pass -> 03_formatted.md (was SPEC 8 + 9) --------
    llm = llm_provider or llm_pkg.build_llm_provider(
        settings,
        model=llm_model,
        on_retry=lambda msg: reporter.warning(msg),
        warn=reporter.warning,
    )
    convert_result = await convert_stage(
        raw_text,
        provider=llm,
        cache_dir=workdir / "cache" / "convert",
        target_path=workdir / "03_formatted.md",
        settings=settings,
        reporter=reporter,
    )
    summary.convert_chunks = convert_result.chunks
    summary.outputs["03_formatted.md"] = str(workdir / "03_formatted.md")
    if stop_after == "convert":
        summary.stopped_after = stop_after
        await _close_providers(provider, llm)
        return summary

    # ----- Stage 5: export .md and .docx from the same source (SPEC 10) -----
    reporter.stage_start("export")
    book_md, book_docx, stats = export_mod.export_book(
        convert_result.text,
        out_dir=workdir,
        title=title or input_dir.name,
        language=settings.book_language,
        source_pages=sum(1 for s in statuses if s.status != "failed"),
        reference_docx=None,
    )
    reporter.stage_done(
        detail=(
            f"md: {stats.md_headings} headings / {stats.md_paragraphs} paragraphs; "
            f"docx: {stats.docx_headings} headings / {stats.docx_paragraphs} paragraphs"
        )
    )
    summary.outputs["book.md"] = str(book_md)
    summary.outputs["book.docx"] = str(book_docx)
    await _close_providers(provider, llm)
    return summary


async def _close_providers(*providers) -> None:
    """Async HTTP clients keep sockets; close those that offer ``aclose``."""
    for provider in providers:
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            try:
                result = aclose()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # pragma: no cover - closing must never break runs
                pass


# --- sub-pipeline entry points used by the dedicated CLI commands ----------
async def ocr_only(
    input_dir: Path,
    *,
    out_dir: Path | None = None,
    order_by: str | None = None,
    reverse: bool = False,
    keep_duplicates: bool = False,
    resume: bool = True,
    ocr_provider=None,
    ocr_provider_name: str | None = None,
    settings: Settings | None = None,
    reporter: Reporter | None = None,
) -> RunSummary:
    """``book2md ocr-only``: stages 1-3 (order, OCR, merge) as one command."""
    return await run(
        input_dir,
        out_dir=out_dir,
        order_by=order_by,
        reverse=reverse,
        keep_duplicates=keep_duplicates,
        stop_after="ocr",
        resume=resume,
        ocr_provider=ocr_provider,
        ocr_provider_name=ocr_provider_name,
        settings=settings,
        reporter=reporter,
    )


async def convert_file(
    raw_path: Path,
    *,
    out_dir: Path | None = None,
    llm_provider=None,
    llm_model: str | None = None,
    settings: Settings | None = None,
    reporter: Reporter | None = None,
) -> RunSummary:
    """``book2md convert RAW.txt``: the single LLM pass on an existing raw file.

    Writes ``03_formatted.md`` (Markdown) next to the raw file (or into
    ``out_dir``), with the per-chunk cache in ``.cache/`` of that directory.
    """
    settings = settings or load_settings()
    reporter = reporter or SilentReporter()
    raw_path = Path(raw_path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw text file not found: {raw_path}")
    out_dir = Path(out_dir) if out_dir else raw_path.parent
    summary = RunSummary(book_name=raw_path.stem)
    llm = llm_provider or llm_pkg.build_llm_provider(
        settings,
        model=llm_model,
        on_retry=lambda msg: reporter.warning(msg),
        warn=reporter.warning,
    )
    result = await convert_stage(
        raw_path.read_text(encoding="utf-8"),
        provider=llm,
        cache_dir=out_dir / ".cache" / "convert",
        target_path=out_dir / "03_formatted.md",
        settings=settings,
        reporter=reporter,
    )
    summary.convert_chunks = result.chunks
    summary.outputs = {"03_formatted.md": str(out_dir / "03_formatted.md")}
    await _close_providers(llm)
    return summary


async def export_file(
    md_path: Path,
    *,
    title: str | None = None,
    language: str = "pl",
    out_dir: Path | None = None,
    source_pages: int | None = None,
    reference_docx: Path | None = None,
    reporter: Reporter | None = None,
) -> RunSummary:
    """``book2md export MD``: export stage only (SPEC 11)."""
    reporter = reporter or SilentReporter()
    md_path = Path(md_path)
    if not md_path.is_file():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")
    out_dir = Path(out_dir) if out_dir else md_path.parent
    summary = RunSummary(book_name=md_path.stem)
    if source_pages is None:
        source_pages = _infer_source_pages(md_path)
    export_mod.export_book(
        md_path.read_text(encoding="utf-8"),
        out_dir=out_dir,
        title=title or (md_path.parent.name or md_path.stem),
        language=language,
        source_pages=source_pages,
        reference_docx=reference_docx,
    )
    summary.outputs = {
        "book.md": str(out_dir / "book.md"),
        "book.docx": str(out_dir / "book.docx"),
    }
    return summary


def _infer_source_pages(md_path: Path) -> int:
    """Standalone export: count markers from sibling artifacts when present."""
    for sibling_name in ("01_raw.txt",):
        sibling = md_path.parent / sibling_name
        if sibling.is_file():
            return len(
                _re.findall(
                    r"^<<<PAGE \d+>>>",
                    sibling.read_text(encoding="utf-8"),
                    flags=_re.MULTILINE,
                )
            )
    return 0
