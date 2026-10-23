"""Text assembly and non-LLM formatting helpers.

Contains everything that must be deterministic (no LLM involved):

* ``merge_raw_text`` - stage 3: build ``01_raw.txt`` with ``<<<PAGE n>>>`` markers,
  ``[[OCR FAILED]]`` / ``[[EMPTY PAGE]]`` placeholders (SPEC 7, 14);
* ``deterministic_format`` - the fallback when the LLM formatter fails the
  word-sequence check (SPEC 9): join lines into paragraphs by blank-line /
  sentence-end heuristics, remove page markers, no headings;
* ``normalize_heading_levels``, ``collapse_blank_lines`` - post-format cleanup
  (SPEC 9);
* ``write_if_changed`` - resumability: never rewrite unchanged artifacts, so
  user hand-edits survive identical rebuilds (SPEC 4).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from .chunking import strip_page_markers

#: Sentence-final characters that end a "line unit" (Polish conventions,
#: including Polish closing quote „…”). A lone hyphen is never sentence-final.
_SENTENCE_END_CHARS = ".!?…”')"
_DIALOGUE_DASHES = ("—", "–")

#: Bare page-number lines: "42", "- 42 -", "— 42 —" (dropped by the fallback).
_BARE_PAGE_NUMBER_RE = re.compile(r"^\s*[-–—]{0,2}\s*\d{1,4}\s*[-–—]{0,2}\s*$")


def normalize_lines(text: str) -> str:
    """Normalize to NFC, LF line endings, and trim trailing whitespace (SPEC 7).

    This is the ONLY "fixing" allowed at the merge stage: no diacritic folding,
    no ASCII cleaning, no reflow.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def merge_raw_text(pages: list[tuple[int, str]]) -> str:
    """Build the merged raw text (SPEC 7) from ``(page, text)`` pairs.

    Failed pages are passed as ``text == "[[OCR FAILED]]"``, empty OCR pages as
    ``"[[EMPTY PAGE]]"`` (SPEC 14) by the caller, so this stays dumb.
    """
    blocks: list[str] = []
    for number, text in pages:
        body = normalize_lines(text)
        if not body:
            body = "[[EMPTY PAGE]]"
        blocks.append(f"<<<PAGE {number}>>>\n{body}")
    # Single blank line between pages; no trailing newline (writer adds one).
    return "\n\n".join(blocks) + "\n"


def write_if_changed(path: Path, content: str) -> bool:
    """Write ``content`` to ``path`` only when content actually differs.

    Resumability detail (SPEC 2/4): unchanged artifacts keep their original
    mtime and hand-edits when a re-run recomputes the identical text.
    Returns ``True`` when the file was (re)written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    if path.exists() and path.read_bytes() == data:
        return False
    path.write_bytes(data)
    return True


def strip_placeholders(text: str) -> str:
    """Remove ``[[OCR FAILED]]`` / ``[[EMPTY PAGE]]`` markers (format stage)."""
    cleaned = re.sub(
        r"\[\[(?:OCR FAILED|EMPTY PAGE)\]\][ \t]*\n?", "", text
    )
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def _looks_like_dialogue(line: str) -> bool:
    """Lines beginning with a Polish dialogue dash (SPEC 9 conventions)."""
    return line.startswith(_DIALOGUE_DASHES) or line.startswith("- ")


def deterministic_format(text: str) -> str:
    """Fallback formatter used when the LLM formatting output is unsafe.

    Heuristics only, no invented headings (SPEC 9):

    * ``<<<PAGE n>>>`` markers are removed; paragraphs crossing pages merge.
    * blank lines break paragraphs,
    * a line ending in sentence-final punctuation closes a paragraph,
    * a dialogue-dash line starts (usually also closes) a paragraph and keeps
      its dash,
    * bare page numbers ("42", "- 42 -") are dropped - they are layout noise,
      never prose,
    * a hyphen at a line end joins the wrapped word back together.

    Output is plain paragraphs separated by blank lines.
    """
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            paragraphs.append(" ".join(current))
            current.clear()

    first_not_marker = strip_page_markers(text)
    for raw_line in first_not_marker.split("\n"):
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if _BARE_PAGE_NUMBER_RE.match(line):
            continue
        if _looks_like_dialogue(line):
            flush()
        if current and current[-1].endswith("-") and (line[:1].islower()):
            # page-break hyphenation ("poprzed-\nnia" -> "poprzednia")
            current[-1] = current[-1][:-1] + line
        elif current:
            current.append(line)
        else:
            current.append(line)
        if len(line) > 2 and line[-1] in _SENTENCE_END_CHARS and line[-2:] != "..":
            flush()
    flush()
    out = "\n\n".join(p for p in paragraphs if p.strip())
    return collapse_blank_lines(out)


def is_heading_line(line: str) -> tuple[int, str] | None:
    """Parse an ATX heading line; returns ``(level, rest)`` or ``None``.

    Fenced code blocks are a real risk for a book with code snippets, so
    callers scanning line-by-line should track fences; this helper assumes
    plain text (books exported here rarely contain them).
    """
    match = re.match(r"^\s{0,3}(#{1,6})(\s+(.*?))?\s*(#*\s*)$", line)
    if not match:
        return None
    rest = (match.group(3) or "").strip()
    return len(match.group(1)), rest


def normalize_heading_levels(markdown: str) -> str:
    """Close heading-level gaps so no '#'-to-'###' jump remains (SPEC 9).

    Walks headings in order; a heading more than one level deeper than the
    previous one is raised to exactly one level deeper. Fenced code blocks are
    skipped so lines inside them are never rewritten.
    """
    lines: list[str] = []
    in_fence = False
    previous_level = 0
    for line in markdown.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            lines.append(line)
            continue
        if not in_fence:
            heading = is_heading_line(line)
            if heading is not None:
                level, rest = heading
                if previous_level and level > previous_level + 1:
                    level = previous_level + 1
                line = "#" * level + (f" {rest}" if rest else "")
                previous_level = level
        lines.append(line)
    return "\n".join(lines)


def collapse_blank_lines(markdown: str) -> str:
    """Collapse 3+ newlines (2+ blank lines) to exactly one blank line (SPEC 9)."""
    return re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"


def sha1_text(text: str) -> str:
    """SHA-1 of UTF-8 text; cache-file names per chunk (SPEC 8.1)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
