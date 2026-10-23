"""LLM repair + format stage tests (SPEC 8/9) with scripted fake LLM answers.

Covers: chunk promptscarry context, per-chunk caching (zero LLM calls on
re-run), mandatory fallbacks on bad answers (missing markers, chatter,
truncation, translation), diff output, and sequential formatting continuity.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from book2md import pipeline
from conftest import FakeLlmProvider, make_settings
import pytest


def pages_text(n_pages: int = 4, words_per_page: int = 300) -> str:
    # words without underscores: the tokenizer counts "slowo1_0" as two words
    pages = []
    for n in range(1, n_pages + 1):
        pages.append(f"<<<PAGE {n}>>>\n" + " ".join(f"slowo{n}{i}" for i in range(words_per_page)))
    return "\n\n".join(pages) + "\n"


def repair_inputs(tmp_path, raw_text: str, reporter=None):
    settings = make_settings(tmp_path, chunk_words=600)  # ~2 pages per chunk
    cache_dir = tmp_path / "cache" / "repair"
    target = tmp_path / "02_repaired.txt"
    diff = tmp_path / "02_repaired.diff.txt"
    return settings, cache_dir, target, diff


def run_repair(tmp_path, raw_text, provider, reporter, **kw):
    settings, cache_dir, target, diff = repair_inputs(tmp_path, raw_text, reporter)
    return asyncio.run(
        pipeline.repair_stage(
            raw_text,
            provider=provider,
            cache_dir=cache_dir,
            target_path=target,
            diff_path=diff,
            settings=settings,
            reporter=reporter or _quiet(),
        )
    ), target, diff, cache_dir


def _quiet():
    from conftest import CollectingReporter

    return CollectingReporter()


class EchoRepair(FakeLlmProvider):
    """Scripted 'perfect repairer': echoes TEXT TO REPAIR verbatim."""

    async def chat(self, *, system: str, user: str, temperature: float = 0.0):
        self.calls.append({"system": system, "user": user})
        return user.split("[TEXT TO REPAIR]\n", 1)[1].split("\n[CONTEXT AFTER]", 1)[0]


# --- repair: prompts, caching, fallback -------------------------------------
def test_repair_sends_context_and_keeps_text(tmp_path, reporter):
    raw = pages_text()
    result, target, diff, _cache_dir = run_repair(tmp_path, raw, EchoRepair(), reporter)
    assert result.chunks == 2  # 4x300 words, target 600 -> 2 chunks of 600
    assert result.fell_back == 0
    # identity repair: the repaired file equals the merged raw text...
    assert target.read_text(encoding="utf-8") == raw
    # ...hence the diff of raw vs repaired is empty
    assert diff.exists() and diff.read_text(encoding="utf-8").strip() == ""


def test_repair_writes_nonempty_diff_when_llm_changed_words(tmp_path, reporter):
    raw = pages_text(1, words_per_page=30)

    class Replacing(FakeLlmProvider):
        async def chat(self, *, system, user, temperature=0.0):
            self.calls.append({"system": system, "user": user})
            chunk = user.split("[TEXT TO REPAIR]\n", 1)[1].split("\n[CONTEXT AFTER]", 1)[0]
            return chunk.replace("slowo10", "czesc", 1)

    result, target, diff, _cache = run_repair(tmp_path, raw, Replacing(), reporter)
    assert result.fell_back == 0
    diff_text = diff.read_text(encoding="utf-8")
    assert "-slowo10" in diff_text and "+czesc" in diff_text
    assert "01_raw.txt" in diff_text


def test_repair_prompt_contains_context_blocks(tmp_path, reporter):
    raw = pages_text(n_pages=2, words_per_page=30)
    provider = EchoRepair()
    run_repair(tmp_path, raw, provider, reporter)
    first_user = provider.calls[0]["user"]
    assert (
        first_user.index("[CONTEXT BEFORE]")
        < first_user.index("[TEXT TO REPAIR]")
        < first_user.index("[CONTEXT AFTER]")
    )
    # the first chunk has nothing before it; the reference says "(beginning..)"
    assert "(beginning of the text)" in first_user


def test_repair_second_run_uses_chunk_cache_no_llm_calls(tmp_path, reporter):
    raw = pages_text()
    echoer = EchoRepair()
    run_repair(tmp_path, raw, echoer, reporter)
    assert len(echoer.calls) == 2
    echoer2 = EchoRepair()
    run_repair(tmp_path, raw, echoer2, reporter)
    assert len(echoer2.calls) == 0  # everything from cache (SPEC 8.1)


def test_repair_rejects_chatty_answer_and_falls_back(tmp_path, reporter):
    raw = pages_text(1, 100)
    provider = FakeLlmProvider(plan=["Here is the repaired text:\n" + raw.strip()])
    result, target, diff, _ = run_repair(tmp_path, raw, provider, reporter)
    assert result.fell_back == 1
    assert target.read_text(encoding="utf-8") == raw  # original chunk kept
    assert any("keeping original" in w for w in reporter.warnings)


def test_repair_rejects_marker_dropping_answer(tmp_path, reporter):
    raw = pages_text(1, 100)
    bad = raw.replace("<<<PAGE 1>>>\n", "")
    provider = FakeLlmProvider(plan=[bad])
    result, target, _, _ = run_repair(tmp_path, raw, provider, reporter)
    assert result.fell_back == 1
    assert "<<<PAGE 1>>>" in target.read_text(encoding="utf-8")


def test_repair_cache_distinguishes_fallback_entries(tmp_path, reporter):
    raw = pages_text(1, 100)
    provider = FakeLlmProvider(plan=["only one marker? no marker here"])
    result, target, _, cache_dir = run_repair(tmp_path, raw, provider, reporter)
    files = list(cache_dir.glob("chunk_*.json"))
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["source"] == "fallback"
    assert data["validated"] is False
    assert result.fell_back == 1


# --- formatting stage --------------------------------------------------------
def sample_repaired_text() -> str:
    """Input for the format stage: repaired text (diacritics already restored)."""
    return (
        "Rozdział pierwszy\n"
        "\n"
        "Tamtej zimy żyli sobie w małej wiosce.\n"
        "Rano chłopaki czekali pod studnią i rozmawiali o żniwach.\n"
        "<<<PAGE 1>>>\n"
        "Zmierzchy nadeszły wolno, wieczór był cichy.\n"
    )


def test_format_produces_markdown_and_removes_markers(tmp_path, reporter):
    text = sample_repaired_text()

    class Formatter(FakeLlmProvider):
        async def chat(self, *, system, user, temperature=0.0):
            self.calls.append({"system": system, "user": user})
            chunk = user.split("[TEXT TO FORMAT]\n", 1)[1]
            return "# Rozdział pierwszy\n\nTamtej zimy żyli sobie w małej wiosce. Rano chłopaki czekali pod studnią i rozmawiali o żniwach. Zmierzchy nadeszły wolno, wieczór był cichy.\n"

    fmt = Formatter()
    settings = make_settings(tmp_path, chunk_words=600)
    result = asyncio.run(
        pipeline.format_stage(
            text,
            provider=fmt,
            cache_dir=tmp_path / "cache" / "format",
            target_path=tmp_path / "03_formatted.md",
            settings=settings,
            reporter=reporter,
        )
    )
    assert result.fell_back == 0
    md = (tmp_path / "03_formatted.md").read_text(encoding="utf-8")
    assert "# Rozdział pierwszy" in md
    assert "<<<PAGE" not in md
    # heading levels were normalized to start at '#'
    assert md.lstrip().startswith("# ")


def test_format_falls_back_to_deterministic_formatter(tmp_path, reporter):
    text = sample_repaired_text()
    provider = FakeLlmProvider(plan=["Completely unrelated English output for the chunk."])
    settings = make_settings(tmp_path, chunk_words=600)
    result = asyncio.run(
        pipeline.format_stage(
            text,
            provider=provider,
            cache_dir=tmp_path / "cache" / "format",
            target_path=tmp_path / "03_formatted.md",
            settings=settings,
            reporter=reporter,
        )
    )
    assert result.fell_back == 1
    md = (tmp_path / "03_formatted.md").read_text(encoding="utf-8")
    assert "<<<PAGE" not in md  # deterministic formatter strips markers
    # deterministic output keeps the words of the input
    assert "Rozdział pierwszy" in md and "żniwach" in md


def test_format_deterministic_fallback_joins_lines_by_sentences():
    raw = "zdanie pierwsze sie\nskonczylo.\nzdanie drugie trwa.\nnadal\ntrwa.\n— mowil.\n42\n"
    out = pipeline.deterministic_format(raw)
    paragraphs = [p for p in out.split("\n\n") if p.strip()]
    assert "zdanie pierwsze sie skonczylo." in paragraphs[0]
    assert any(p.startswith("— ") for p in paragraphs)
    assert "42" not in out  # bare page numbers dropped


def test_format_second_run_uses_cache(tmp_path, reporter):
    text = sample_repaired_text()

    class Formatter(FakeLlmProvider):
        async def chat(self, *, system, user, temperature=0.0):
            self.calls.append({"system": system, "user": user})
            return "Tamtej zimy żyli sobie w małej wiosce. Rano chłopaki czekali pod studnią i rozmawiali o żniwach. Zmierzchy nadeszły wolno, wieczór był cichy.\n"

    fmt = Formatter()
    settings = make_settings(tmp_path, chunk_words=600)
    asyncio.run(
        pipeline.format_stage(
            text,
            provider=fmt,
            cache_dir=tmp_path / "cache" / "format",
            target_path=tmp_path / "03_formatted.md",
            settings=settings,
            reporter=reporter,
        )
    )
    assert len(fmt.calls) == 1
    fmt2 = Formatter()
    asyncio.run(
        pipeline.format_stage(
            text,
            provider=fmt2,
            cache_dir=tmp_path / "cache" / "format",
            target_path=tmp_path / "03_formatted.md",
            settings=settings,
            reporter=reporter,
        )
    )
    assert len(fmt2.calls) == 0


def test_format_carries_previous_chunk_tail(tmp_path, reporter):
    long_text = pages_text(4, words_per_page=250)  # 2 chunks with target 600
    seen: list[str] = []

    class Formatter(FakeLlmProvider):
        async def chat(self, *, system, user, temperature=0.0):
            self.calls.append({"system": system, "user": user})
            context = user.split("[PREVIOUS TEXT END", 1)[-1]
            seen.append(context)
            chunk = user.split("[TEXT TO FORMAT]\n", 1)[1]
            return chunk

    settings = make_settings(tmp_path, chunk_words=600)
    asyncio.run(
        pipeline.format_stage(
            long_text,
            provider=Formatter(),
            cache_dir=tmp_path / "cache" / "format",
            target_path=tmp_path / "03_formatted.md",
            settings=settings,
            reporter=reporter,
        )
    )
    # chunks processed sequentially; after the first, context-before is present
    assert len(seen) >= 2
    assert "(beginning of the text)" in seen[0]
    assert "(beginning of the text)" not in seen[-1]
    # the tail of the previous chunk's answer is carried into the next prompt
    assert "slowo" in seen[-1]
