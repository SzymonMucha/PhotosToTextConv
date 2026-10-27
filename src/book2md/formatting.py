"""Text assembly and cleanups. Everything here is deterministic (no LLM):

* ``merge_raw_text`` - stage 3: build ``01_raw.txt`` with ``<<<PAGE n>>>`` markers,
  ``[[OCR FAILED]]`` / ``[[EMPTY PAGE]]`` placeholders;
* ``strip_code_fences`` / ``strip_model_thinking`` - salvage helpers that clean
  up a chat answer before it is stored (some models wrap answers in fences or
  echo thinking blocks; the content itself is kept);
* ``normalize_heading_levels``, ``collapse_blank_lines`` - post-convert cleanup
  of the assembled Markdown;
* ``write_if_changed`` - resumability: never rewrite unchanged artifacts, so
  user hand-edits survive identical rebuilds.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path


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
    ``"[[EMPTY PAGE]]"`` by the caller, so this stays dumb.
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

    Resumability detail: unchanged artifacts keep their original mtime and
    hand-edits when a re-run recomputes the identical text.
    Returns ``True`` when the file was (re)written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    if path.exists() and path.read_bytes() == data:
        return False
    path.write_bytes(data)
    return True


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
    """Close heading-level gaps so no '#'-to-'###' jump remains.

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
    """Collapse 3+ newlines (2+ blank lines) to exactly one blank line."""
    return re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"


def strip_code_fences(text: str) -> str:
    """Remove one leading/trailing ``` fence pair - salvage, not validation.

    Models stubbornly wrap whole documents in fences even when the prompt
    forbids it; the *content* is then perfectly valid, so it is stripped here
    before the answer is stored.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            lines = lines[1:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def strip_model_thinking(text: str) -> str:
    """Remove ``thinking`` blocks some providers echo back.

    Salvage step: only whole wrapping blocks are removed, real text stays.
    """
    return re.sub(
        r"<thinking>.*?</thinking>|<thought>.*?</thought>|<reflection>.*?</reflection>",
        "",
        text,
        flags=re.DOTALL,
    )


def sha1_text(text: str) -> str:
    """SHA-1 of UTF-8 text; cache-file names per chunk."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
