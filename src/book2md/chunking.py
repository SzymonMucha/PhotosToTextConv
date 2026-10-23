"""Chunking for the LLM passes (SPEC 8.1, 9).

``01_raw.txt`` is cut into chunks of roughly 1,500-3,000 words (target from
``CHUNK_WORDS``), *always* on ``<<<PAGE n>>>`` marker boundaries: a marker can
never be separated from its page text. Missing markers are tolerated (the LLM
adds none; the checker asks them to keep the ones that exist).

Each repair request receives the *previous chunk's last ~300 words* and the
*next chunk's first ~300 words* as contextual reference; format requests carry
the previous formatted tail instead (see :mod:`book2md.pipeline`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: ``<<<PAGE 12>>>`` on its own line (with tolerant surrounding whitespace).
PAGE_MARKER_RE = re.compile(r"^[ \t]*<<<PAGE (\d+)>>>[ \t]*$", re.MULTILINE)
#: Same regex for scanning anywhere (e.g., already-repaired text).
ANY_MARKER_RE = re.compile(r"[ \t]*<<<PAGE (\d+)>>>[ \t]*\n?")

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)  # letters/digits, not underscore


def word_count(text: str) -> int:
    """Count words (letters/digits runs). Markers count as 2 words; symmetric
    on input and output so the ±15% check stays meaningful."""
    return len(_WORD_RE.findall(text))


@dataclass
class Block:
    """One marker-led stretch of raw text ending at the next marker."""

    page: int | None   # the ``n`` of ``<<<PAGE n>>>``; None for text before any marker
    text: str          # includes the marker line itself (trailing newline kept clean)


@dataclass
class Chunk:
    """A repair/formatable unit: several blocks, sent to the LLM as one."""

    index: int
    first_page: int | None
    last_page: int | None
    text: str
    words: int


def split_blocks(text: str) -> list[Block]:
    """Split raw text into marker-led blocks, preserving every marker.

    Text before the first marker (possible in article-style RHS text, where a
    file has no leading marker) becomes a single ``Block(page=None)``.
    """
    blocks: list[Block] = []
    matches = list(PAGE_MARKER_RE.finditer(text))
    if not matches:
        return [Block(page=None, text=text)] if text else []
    head = text[: matches[0].start()].strip()
    if head:
        blocks.append(Block(page=None, text=head))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append(
            Block(page=int(match.group(1)), text=text[match.start() : end].strip())
        )
    return blocks


def strip_page_markers(text: str) -> str:
    """Remove ``<<<PAGE n>>>`` lines. Page *text* is never touched."""
    return ANY_MARKER_RE.sub("", text)


def split_chunks(text: str, target_words: int) -> list[Chunk]:
    """Group marker blocks into chunks of ``~target_words`` words.

    Greedy accumulation: as soon as a chunk reaches the target size it is
    closed, so real chunks land in ``[target, target + one page]``. A single
    oversized page stays one chunk - correctness (markers intact) beats size.
    """
    if target_words < 300 or target_words > 3000:
        raise ValueError(
            f"chunk target {target_words} outside sane range [300, 3000]; "
            f"check CHUNK_WORDS in .env"
        )
    chunks: list[Chunk] = []
    current: list[Block] = []
    current_first: int | None = None
    current_last: int | None = None
    current_words = 0

    def flush() -> None:
        nonlocal current, current_first, current_last, current_words
        if not current:
            return
        chunk_text = "\n\n".join(block.text for block in current).strip()
        chunks.append(
            Chunk(
                index=len(chunks),
                first_page=current_first,
                last_page=current_last,
                text=chunk_text + ("\n" if chunk_text else ""),
                words=current_words,
            )
        )
        current, current_first, current_last, current_words = [], None, None, 0

    for block in split_blocks(text):
        block_words = word_count(strip_page_markers(block.text))
        current.append(block)
        current_words += block_words
        if block.page is not None:
            if current_first is None:
                current_first = block.page
            current_last = block.page
        if current_words >= target_words:
            flush()
    flush()
    return chunks


def take_words(text: str, count: int, *, end: bool = False) -> str:
    """Return the first (``end=False``) or last (``end=True``) ``count`` words.

    Used for CONTEXT BEFORE / CONTEXT AFTER. Markers are stripped first, so
    context blocks never confuse the marker-keep rule. Larger ``count`` than
    available words simply returns all words. Text is NFC-normalized by the
    caller when embedded into prompts (see :mod:`book2md.prompts`).
    """
    clean = strip_page_markers(text).strip()
    words = _WORD_RE.findall(clean)
    if count >= len(words):
        picked = words
    elif end:
        picked = words[-count:]
    else:
        picked = words[:count]
    return " ".join(picked)


def context_windows(
    chunks: list[Chunk], *, context_words: int = 300
) -> list[tuple[str, str]]:
    """For each chunk: ``(context_before, context_after)`` texts (SPEC 8.1).

    ``context_before`` is the tail of the previous chunk, ``context_after``
    the head of the next one; the first/last chunk get empty strings.
    """
    windows: list[tuple[str, str]] = []
    for i, _chunk in enumerate(chunks):
        before = take_words(chunks[i - 1].text, context_words, end=True) if i > 0 else ""
        after = (
            take_words(chunks[i + 1].text, context_words, end=False)
            if i + 1 < len(chunks)
            else ""
        )
        windows.append((before, after))
    return windows
