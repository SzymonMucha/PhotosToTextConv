"""Chunking tests: marker-boundary chunks, tails for context, word limits."""

from __future__ import annotations

import pytest

from book2md.chunking import (
    split_blocks,
    split_chunks,
    strip_page_markers,
    take_words,
    word_count,
)


def raw(pages: dict[int, str]) -> str:
    return "\n\n".join(f"<<<PAGE {n}>>>\n{text}" for n, text in pages.items()) + "\n"


def test_word_count_counts_words_not_punctuation():
    assert word_count("ala, ma-kota; 12... ") == 4  # 12 counts as one word
    assert word_count("<<<PAGE 3>>>\ntext") == 3  # PAGE, 3, text


def test_split_blocks_keeps_every_marker_attached_to_its_text():
    text = raw({1: "jeden", 2: "dwa", 3: "trzy"})
    blocks = split_blocks(text)
    assert [(b.page, b.text) for b in blocks] == [
        (1, "<<<PAGE 1>>>\njeden"),
        (2, "<<<PAGE 2>>>\ndwa"),
        (3, "<<<PAGE 3>>>\ntrzy"),
    ]


def test_split_blocks_handles_text_without_markers():
    blocks = split_blocks("plain text with no markers")
    assert len(blocks) == 1
    assert blocks[0].page is None


def test_chunk_sizes_stay_on_marker_boundaries():
    page_text = " ".join(f"slowo{i}" for i in range(400))  # 400 words per page
    text = raw({n: page_text for n in range(1, 11)})
    chunks = split_chunks(text, target_words=1000)  # ~3 pages per chunk
    sizes = [c.words for c in chunks]
    assert sizes == [1200, 1200, 1200, 400]  # last chunk holds the remainder
    # markers never split mid-page: every original marker appears exactly once
    assert text.count("<<<PAGE") == sum(c.text.count("<<<PAGE") for c in chunks)
    assert all("<<<PAGE" in c.text for c in chunks)
    assert [(c.first_page, c.last_page) for c in chunks] == [
        (1, 3),
        (4, 6),
        (7, 9),
        (10, 10),
    ]


def test_single_oversized_page_becomes_one_whole_chunk():
    big = " ".join(["dlugie"] * 5000)
    text = raw({1: big, 2: "mala"})
    chunks = split_chunks(text, target_words=1000)
    assert chunks[0].words >= 5000  # cannot split a page (markers stay intact)
    assert chunks[1].words == 1


def test_chunk_target_bounds_enforced():
    with pytest.raises(ValueError):
        split_chunks("x", target_words=100)
    with pytest.raises(ValueError):
        split_chunks("x", target_words=5000)


def test_strip_page_markers_removes_only_markers():
    text = "<<<PAGE 1>>>\nRzeka\n<<<PAGE 2>>>\nbrzeg"
    assert strip_page_markers(text) == "Rzeka\nbrzeg"


def test_take_words_end_semantics():
    words = " ".join(f"w{i}" for i in range(10))
    assert take_words(words, 3).split() == ["w0", "w1", "w2"]
    assert take_words(words, 3, end=True).split() == ["w7", "w8", "w9"]
    assert take_words("<<<PAGE 1>>>\nnic nie ma", 100) == "nic nie ma"
