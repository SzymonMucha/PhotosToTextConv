"""Stage 6: export Markdown -> ``book.md`` + ``book.docx`` (SPEC 10).

Both files come from the SAME Markdown source of truth:

* ``book.md`` = ``03_formatted.md`` + YAML front matter (title, lang, sources);
* ``book.docx`` = pandoc conversion of exactly that ``book.md``.

Details that matter (SPEC 10):

* ``markdown-smart`` - smart typography is required so pandoc does not rewrite
  the author's Polish quotes/dashes; ``lang=pl-PL`` makes Word spell-check in
  Polish; ``--standalone`` embeds the front-matter metadata.
* After export the DOCX is opened again with ``python-docx`` to prove it is
  readable, and paragraph/heading counts are reported for both outputs.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

#: Language code -> BCP-47 locale used for Word spell-check language.
_LANGUAGE_LOCALES = {
    "pl": "pl-PL",
    "en": "en-GB",
    "de": "de-DE",
    "fr": "fr-FR",
    "cs": "cs-CZ",
}


def locale_for_language(language: str) -> str:
    """``pl`` -> ``pl-PL``; passthrough for codes that already contain ``-``.

    The user-provided casing is preserved (BCP-47 style ``en-GB`` stays ``en-GB``);
    the lookup key is lowercased separately.
    """
    key = language.strip().lower()
    if not key:
        return "pl-PL"
    if "-" in key:
        return language.strip()
    return _LANGUAGE_LOCALES.get(key, f"{key}-{key.upper()}")


def yaml_title(title: str) -> str:
    """Quote a title for the YAML front matter (double-quoted YAML scalar)."""
    # json.dumps produces a valid YAML double-quoted string for normal titles.
    import json

    return json.dumps(title, ensure_ascii=False)


def add_front_matter(
    markdown: str,
    *,
    title: str,
    language: str = "pl",
    source_pages: int,
) -> str:
    """Prepend the YAML front matter block (SPEC 10) and return the document."""
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    front = (
        "---\n"
        f"title: {yaml_title(title)}\n"
        f"lang: {locale_for_language(language)}\n"
        f"source_pages: {int(source_pages)}\n"
        f"generated: {generated}\n"
        "---\n\n"
    )
    return front + markdown.lstrip("\n")


@dataclass
class ExportStats:
    """Counts printed in the final summary (SPEC 11)."""

    md_paragraphs: int
    md_headings: int
    docx_paragraphs: int
    docx_headings: int


def export_book(
    markdown_text: str,
    *,
    out_dir: Path,
    title: str,
    language: str = "pl",
    source_pages: int,
    reference_docx: Path | None = None,
) -> tuple[Path, Path, ExportStats]:
    """Write ``book.md`` and ``book.docx`` from ``markdown_text`` (SPEC 10).

    Raises :class:`PandocMissingError` with setup instructions when pandoc is
    not installed; never converts from anything but the just-written ``book.md``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    book_md = out_dir / "book.md"
    final = add_front_matter(
        markdown_text, title=title, language=language, source_pages=source_pages
    )
    # book.md is regenerated whenever run (cheap, no API calls).
    book_md.write_text(final, encoding="utf-8")

    book_docx = out_dir / "book.docx"
    convert_md_to_docx(book_md, book_docx, reference_docx=reference_docx)

    stats = verify_outputs(book_md, book_docx)
    return book_md, book_docx, stats


def convert_md_to_docx(
    source_md: Path,
    out_docx: Path,
    *,
    reference_docx: Path | None = None,
    language: str = "pl-PL",
) -> None:
    """pypandoc conversion exactly as required by SPEC 10."""
    pypandoc = _require_pandoc()
    extra_args = [
        "--standalone",
        "--metadata",
        f"lang={language}",
    ]
    if reference_docx is not None:
        if not reference_docx.is_file():
            raise FileNotFoundError(f"--reference-docx not found: {reference_docx}")
        extra_args += ["--reference-doc", str(reference_docx)]
    try:
        # format="markdown-smart" is REQUIRED (see module docstring / SPEC 10).
        pypandoc.convert_file(
            str(source_md),
            "docx",
            format="markdown-smart",
            outputfile=str(out_docx),
            extra_args=extra_args,
        )
    except Exception as exc:  # pypandoc raises OSError/RuntimeError variants
        if not _pandoc_on_path():
            raise PandocMissingError(_install_hint()) from exc
        raise ExportError(f"Pandoc failed to build the DOCX: {exc}") from exc


def _pandoc_on_path() -> bool:
    return shutil.which("pandoc") is not None


def _require_pandoc():
    try:
        import pypandoc  # noqa: F401 - availability is what we actually check
    except ImportError as exc:
        raise ExportError(
            "pypandoc is not installed in this environment; reinstall the "
            "book2md package (pip install -e .) - see README.md."
        ) from exc
    if not _pandoc_on_path():
        raise PandocMissingError(_install_hint())
    return pypandoc


def _install_hint() -> str:
    return (
        "pandoc is not installed. Install it first, e.g.:\n"
        "  - Debian/Ubuntu:  sudo apt install pandoc\n"
        "  - Arch:           sudo pacman -S pandoc\n"
        "  - Fedora:         sudo dnf install pandoc\n"
        "  - macOS:          brew install pandoc\n"
        "  - Windows:        winget install --id JohnMacFarlane.Pandoc\n"
        "then re-run the export. See README.md."
    )


def verify_outputs(book_md: Path, book_docx: Path) -> ExportStats:
    """Count paragraphs/headings in both artifacts (SPEC 10).

    python-docx verification is load-bearing: it proves the DOCX can really be
    opened (corrupt pandoc output would crash here). If python-docx is missing
    the counts are still returned for the Markdown and the DOCX is left
    unchecked with ``-1`` counts.
    """
    md_text = book_md.read_text(encoding="utf-8")
    md_headings = 0
    in_fence = False
    md_paragraphs = 0
    paragraph_open = False
    for line in md_text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line.startswith(("#",)):
            md_headings += 1
            paragraph_open = False
            continue
        if line.strip():
            if not paragraph_open:
                md_paragraphs += 1
                paragraph_open = True
        else:
            paragraph_open = False

    docx_paragraphs = -1
    docx_headings = -1
    try:
        import docx  # python-docx

        document = docx.Document(str(book_docx))
        docx_paragraphs = len(document.paragraphs)
        docx_headings = sum(
            1
            for paragraph in document.paragraphs
            if paragraph.style and paragraph.style.name.startswith("Heading")
        )
    except ImportError:
        pass  # verification optional; DOCX was still produced by pandoc
    return ExportStats(
        md_paragraphs=md_paragraphs,
        md_headings=md_headings,
        docx_paragraphs=docx_paragraphs,
        docx_headings=docx_headings,
    )


class ExportError(RuntimeError):
    """Export failed (pandoc conversion problems)."""


class PandocMissingError(ExportError):
    """pandoc binary absent; message contains install instructions."""
