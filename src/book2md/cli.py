"""Typer CLI (Phase 1 interface, SPEC 11).

Commands::

    book2md run       PHOTOS   [--title ...] [--out DIR] [--order-by ...]
                               [--reverse] [--ocr-provider ...] [--llm-model ...]
                               [--stop-after order|ocr|repair|format]
                               [--resume/--no-resume] [--keep-duplicates]
    book2md ocr-only  PHOTOS   # stages 1-3
    book2md order     PHOTOS   # stage 1 only, prints + writes order.json
    book2md repair    RAW.txt  # stage 4 on an existing raw file
    book2md format    TXT      # stage 5 on an existing text file
    book2md export    MD       # stage 6 only

Every command shows rich progress per stage and a final summary: pages
OK/failed, chunks repaired/fell back, output paths (SPEC 11).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from book2md import __version__, pipeline
from book2md.config import AVAILABLE_OCR_PROVIDERS, ConfigError
from book2md.pipeline import RunSummary
from book2md.retry import QuotaExceededError

app = typer.Typer(
    help="book2md - turn photographed book pages into Markdown + DOCX.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_ORDER_BY = typer.Option(None, "--order-by", help="Order override: exif | mtime | name.")
_REVERSE = typer.Option(False, "--reverse", help="Reverse the detected order.")
_OUT = typer.Option(None, "--out", help="Output root directory (default: workdir).")
_RESUME = typer.Option(True, "--resume/--no-resume", help="Reuse caches (default: yes).")
_KEEP_DUPES = typer.Option(False, "--keep-duplicates", help="Keep duplicate photos (same SHA-1).")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"book2md {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Print the version and exit.",
    ),
) -> None:
    """book2md: book photos -> OCR -> LLM cleanup -> Markdown + DOCX."""


class RichReporter:
    """Adapts :class:`book2md.pipeline.Reporter` hooks onto rich widgets."""

    def __init__(self) -> None:
        self._progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        self._current_task = None

    def stage_start(self, name: str, total: int | None = None) -> None:
        self._progress.start()
        self._current_task = self._progress.add_task(name, total=total)

    def stage_progress(self, n: int = 1, detail: str | None = None) -> None:
        if self._current_task is not None:
            self._progress.advance(self._current_task, n)
        if detail:
            self._progress.console.print(f"  [dim]{detail}[/dim]")

    def stage_done(self, detail: str | None = None) -> None:
        if self._current_task is not None:
            self._progress.update(self._current_task, visible=False)
            self._current_task = None
        self._progress.stop()
        if detail:
            console.print(f"  [green]ok[/green] {detail}")

    def warning(self, msg: str) -> None:
        self._progress.console.print(f"  [yellow]warning:[/yellow] {msg}")

    def info(self, msg: str) -> None:
        console.print(f"  [blue]info:[/blue] {msg}")


def print_summary(summary: RunSummary) -> None:
    """The final summary panel (SPEC 11)."""
    lines: list[str] = [f"[bold]book:[/bold] {summary.book_name}"]
    if summary.order_strategy:
        ok = summary.pages_total - len(summary.pages_failed)
        lines.append(
            f"[bold]order:[/bold] {summary.pages_total} pages by "
            f"{summary.order_strategy}, {summary.duplicates_skipped} duplicate(s) skipped"
        )
        lines.append(
            f"[bold]ocr:[/bold] {ok} ok, {len(summary.pages_failed)} failed, "
            f"{summary.pages_empty} empty, {summary.pages_cached} from cache "
            f"(+{summary.pages_ocr_now} fresh)"
        )
    if summary.repair_chunks:
        lines.append(
            f"[bold]repair:[/bold] {summary.repair_chunks} chunk(s), "
            f"{summary.repair_fell_back} fell back to original"
        )
    if summary.format_chunks:
        lines.append(
            f"[bold]format:[/bold] {summary.format_chunks} chunk(s), "
            f"{summary.format_fell_back} fell back to deterministic form"
        )
    for name, path in summary.outputs.items():
        lines.append(f"[bold]{name}:[/bold] {path}")
    if summary.pages_failed:
        lines.append("[yellow]failed pages:[/yellow]")
        lines += [f"  [yellow]- {failure}[/yellow]" for failure in summary.pages_failed]
    console.print(Panel("\n".join(lines), title=f"book2md {__version__} summary"))


def _execute(coro: Any) -> RunSummary:
    """Run a pipeline coroutine, converting expected failures into clear messages."""
    try:
        return asyncio.run(coro)
    except QuotaExceededError as exc:
        console.print(f"[bold red]quota exhausted:[/bold red] {exc}")
        raise typer.Exit(code=3) from None
    except ConfigError as exc:
        console.print(f"[bold red]configuration problem:[/bold red] {exc}")
        raise typer.Exit(code=2) from None
    except (FileNotFoundError, NotADirectoryError) as exc:
        console.print(f"[bold red]input problem:[/bold red] {exc}")
        raise typer.Exit(code=2) from None
    except ValueError as exc:
        console.print(f"[bold red]bad value:[/bold red] {exc}")
        raise typer.Exit(code=2) from None


def _validate_flags(
    order_by: str | None, stop_after: str | None, ocr_provider: str
) -> None:
    if order_by is not None and order_by not in ("exif", "mtime", "name"):
        console.print("[red]--order-by must be one of: exif, mtime, name[/red]")
        raise typer.Exit(code=2)
    if stop_after is not None and stop_after not in ("order", "ocr", "repair", "format"):
        console.print("[red]--stop-after must be one of: order, ocr, repair, format[/red]")
        raise typer.Exit(code=2)
    if ocr_provider not in AVAILABLE_OCR_PROVIDERS:
        console.print(
            f"[red]--ocr-provider must be one of: "
            f"{', '.join(AVAILABLE_OCR_PROVIDERS)}[/red]"
        )
        raise typer.Exit(code=2)


@app.command()
def order(
    photos: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder of page photos."),
    order_by: str | None = _ORDER_BY,
    reverse: bool = _REVERSE,
    out: Path | None = _OUT,
    keep_duplicates: bool = _KEEP_DUPES,
) -> None:
    """Stage 1 only: detect the page order, print it and write order.json."""
    if order_by is not None and order_by not in ("exif", "mtime", "name"):
        console.print("[red]--order-by must be one of: exif, mtime, name[/red]")
        raise typer.Exit(code=2)

    from book2md.ordering import order_folder, write_order_json

    def warn(msg: str) -> None:
        console.print(f"  [yellow]warning:[/yellow] {msg}")

    pages, strategy, duplicates = order_folder(
        photos, order_by=order_by, reverse=reverse,
        keep_duplicates=keep_duplicates, on_warning=warn,
    )
    workdir = (Path(out) if out else Path(pipeline.DEFAULT_OUTPUT_DIR)) / (
        pipeline.sanitize_book_name(photos.name)
    )
    write_order_json(workdir / "order.json", pages, strategy=strategy, reverse=reverse)
    console.print(f"[bold]Detected order ({strategy}):[/bold]")
    for page in pages:
        console.print(f"  page {page.page:>4}  {page.timestamp}  {page.filename}")
    print_summary(
        RunSummary(
            book_name=photos.name,
            pages_total=len(pages),
            order_strategy=strategy,
            duplicates_skipped=duplicates,
            outputs={"order.json": str(workdir / "order.json")},
        )
    )


@app.command()
def run(
    photos: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder of page photos."),
    title: str | None = typer.Option(None, "--title", help="Book title (default: folder name)."),
    out: Path | None = _OUT,
    order_by: str | None = _ORDER_BY,
    reverse: bool = _REVERSE,
    ocr_provider: str = typer.Option("ocrspace", "--ocr-provider", help="OCR provider."),
    llm_model: str | None = typer.Option(None, "--llm-model", help="LLM model override."),
    stop_after: str | None = typer.Option(
        None, "--stop-after", help="Stop after this stage: order|ocr|repair|format."
    ),
    resume: bool = _RESUME,
    keep_duplicates: bool = _KEEP_DUPES,
) -> None:
    """Full pipeline: order -> OCR -> merge -> repair -> format -> export."""
    _validate_flags(order_by, stop_after, ocr_provider)
    summary = _execute(
        pipeline.run(
            photos,
            out_dir=out,
            title=title,
            order_by=order_by,
            reverse=reverse,
            keep_duplicates=keep_duplicates,
            stop_after=stop_after,
            resume=resume,
            ocr_provider_name=ocr_provider,
            llm_model=llm_model,
            reporter=RichReporter(),
        )
    )
    print_summary(summary)


@app.command(name="ocr-only")
def ocr_only(
    photos: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder of page photos."),
    out: Path | None = _OUT,
    order_by: str | None = _ORDER_BY,
    reverse: bool = _REVERSE,
    ocr_provider: str = typer.Option("ocrspace", "--ocr-provider", help="OCR provider."),
    resume: bool = _RESUME,
    keep_duplicates: bool = _KEEP_DUPES,
) -> None:
    """Stages 1-3: order, OCR (cached per image) and merge into 01_raw.txt."""
    _validate_flags(order_by, None, ocr_provider)
    summary = _execute(
        pipeline.ocr_only(
            photos,
            out_dir=out,
            order_by=order_by,
            reverse=reverse,
            keep_duplicates=keep_duplicates,
            resume=resume,
            ocr_provider_name=ocr_provider,
            reporter=RichReporter(),
        )
    )
    print_summary(summary)


@app.command()
def repair(
    raw: Path = typer.Argument(..., exists=True, dir_okay=False, help="Raw OCR text file."),
    out: Path | None = _OUT,
    llm_model: str | None = typer.Option(None, "--llm-model", help="LLM model override."),
) -> None:
    """Stage 4 on an existing raw file -> 02_repaired.txt (+ diff for review)."""
    summary = _execute(
        pipeline.repair_file(raw, out_dir=out, llm_model=llm_model, reporter=RichReporter())
    )
    print_summary(summary)


@app.command()
def format(
    txt: Path = typer.Argument(..., exists=True, dir_okay=False, help="Repaired text file."),
    out: Path | None = _OUT,
    llm_model: str | None = typer.Option(None, "--llm-model", help="LLM model override."),
) -> None:
    """Stage 5 on an existing text file -> 03_formatted.md (Markdown)."""
    summary = _execute(
        pipeline.format_file(txt, out_dir=out, llm_model=llm_model, reporter=RichReporter())
    )
    print_summary(summary)


@app.command()
def export(
    md: Path = typer.Argument(..., exists=True, dir_okay=False, help="Markdown source file."),
    title: str | None = typer.Option(None, "--title", help="Book title (default: folder name)."),
    language: str = typer.Option("pl", "--language", help="Book language code (e.g. pl)."),
    out: Path | None = _OUT,
    reference_docx: Path | None = typer.Option(
        None, "--reference-docx", help="DOCX style template for pandoc."
    ),
) -> None:
    """Stage 6 only: write book.md (front matter) and book.docx from one .md."""
    summary = _execute(
        pipeline.export_file(
            md,
            title=title,
            language=language,
            out_dir=out,
            reference_docx=reference_docx,
            reporter=RichReporter(),
        )
    )
    print_summary(summary)


def main() -> None:  # console_scripts entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
