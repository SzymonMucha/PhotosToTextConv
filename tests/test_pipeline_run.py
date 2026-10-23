"""End-to-end pipeline tests with fake providers (SPEC 16: no network)."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from PIL import Image

from book2md import pipeline
from book2md.retry import QuotaExceededError, RetryConfig
from conftest import FakeOcrProvider, FakeLlmProvider, make_settings, read_fixture, set_mtime


class PhotoBook:
    """Builds a photo folder whose fake OCR text is scripted per filename."""

    def __init__(self, base: Path, page_count: int = 3) -> None:
        self.base = base
        self.page_count = page_count
        for i, suffix_number in enumerate(range(page_count)):
            color = (i * 100, 60, 30)
            Image.new("RGB", (48, 64), color).save(base / f"IMG_{i+1}.jpg", "JPEG")
        now = time.time()
        for i in range(page_count):
            set_mtime(base / f"IMG_{i+1}.jpg", now - 1000 + i * 10)

    def provider(self, page_text=None) -> FakeOcrProvider:
        page_text = page_text or (
            lambda n: f"Strona {n}. Tekst przykladowy zdanie pierwsze zdanie drugie zdanie trzecie."
        )
        return FakeOcrProvider(
            responses={f"IMG_{i+1}.jpg": page_text(i + 1) for i in range(self.page_count)},
        )


def async_run(coro):
    return asyncio.run(coro)


def good_llm_settings(tmp_path, **kw):
    return make_settings(tmp_path, chunk_words=2000, **kw)


class EchoFormat(FakeLlmProvider):
    """Two-mode fake covering both LLM stages in one provider.

    - Repair prompt ([TEXT TO REPAIR]): echoes the chunk back verbatim.
      Identical text trivially passes the similarity checks (SPEC 13).
    - Format prompt ([TEXT TO FORMAT]): regroups lines into paragraphs,
      which keeps every word in order (structure-only rule, SPEC 9).
    """

    async def chat(self, *, system: str, user: str, temperature: float = 0.0):
        self.calls.append({"system": system, "user": user})
        if "[TEXT TO REPAIR]" in user:
            # Extract the repair chunk (between the repair marker and the
            # context section) and return it unchanged, markers included.
            chunk = user.split("[TEXT TO REPAIR]\n", 1)[1]
            chunk = chunk.split("[CONTEXT AFTER]", 1)[0]
            return chunk.rstrip("\n") + "\n"
        if "[TEXT TO FORMAT]" not in user:  # unknown prompt shape: fail loudly
            raise AssertionError("LLM prompt matches neither stage template")
        # Cut off the template's closing "Remember:" boilerplate, otherwise the
        # fake would format it as chunk text and the validator would rightly
        # reject the "added words".
        chunk = user.split("[TEXT TO FORMAT]\n", 1)[1]
        chunk = chunk.split("\nRemember:", 1)[0]
        lines = [ln.strip() for ln in chunk.splitlines()]
        paras: list[str] = []
        current: list[str] = []
        for line in lines:
            if not line or line.startswith("<<<PAGE"):
                continue
            current.append(line)
            if line.endswith("."):
                paras.append(" ".join(current))
                current = []
        if current:
            paras.append(" ".join(current))
        return "\n\n".join(paras) + "\n"


def test_full_run_produces_md_and_docx(tmp_path):
    photos = tmp_path / "Przykladowa Ksiazka"
    photos.mkdir()
    book = PhotoBook(photos, page_count=3)
    ocr = book.provider()
    settings = good_llm_settings(tmp_path)
    summary = async_run(
        pipeline.run(
            photos,
            out_dir=tmp_path / "out",
            title="Przykladowa Ksiega",
            ocr_provider=ocr,
            llm_provider=EchoFormat(),
            settings=settings,
        )
    )
    workdir = tmp_path / "out" / "Przykladowa_Ksiazka"
    for name in (
        "order.json",
        "01_raw.txt",
        "02_repaired.txt",
        "02_repaired.diff.txt",
        "03_formatted.md",
        "book.md",
        "book.docx",
    ):
        assert (workdir / name).exists(), f"missing artifact {name}"
    raw_text = (workdir / "01_raw.txt").read_text(encoding="utf-8")
    for n in (1, 2, 3):
        assert f"<<<PAGE {n}>>>" in raw_text
    book_md = (workdir / "book.md").read_text(encoding="utf-8")
    assert book_md.startswith("---\n")  # YAML front matter (SPEC 10)
    assert '"Przykladowa Ksiega"' in book_md
    assert "lang: pl-PL" in book_md
    assert "<<<PAGE" not in book_md  # markers stripped before export
    assert summary.pages_total == 3
    assert summary.repair_fell_back == 0
    assert summary.format_fell_back == 0


def test_re_run_is_fully_cached_zero_new_calls(tmp_path):
    photos = tmp_path / "pb"
    photos.mkdir()
    book = PhotoBook(photos, page_count=2)
    ocr = book.provider()
    settings = good_llm_settings(tmp_path)
    first = async_run(
        pipeline.run(
            photos, out_dir=tmp_path / "out", ocr_provider=ocr,
            llm_provider=EchoFormat(), settings=settings,
        )
    )
    assert first.outputs
    ocr2 = book.provider()
    summary2 = async_run(
        pipeline.run(
            photos, out_dir=tmp_path / "out", ocr_provider=ocr2,
            llm_provider=EchoFormat(), settings=settings,
        )
    )
    assert ocr2.calls == []  # ZERO new OCR calls (SPEC 16 / definition of done)
    assert summary2.pages_cached == 2
    assert summary2.pages_ocr_now == 0


def test_stop_after_order_and_ocr(tmp_path):
    photos = tmp_path / "pb"
    photos.mkdir()
    book = PhotoBook(photos, page_count=2)
    summary = async_run(
        pipeline.run(
            photos, out_dir=tmp_path / "out", ocr_provider=book.provider(),
            llm_provider=EchoFormat(), settings=good_llm_settings(tmp_path),
            stop_after="order",
        )
    )
    workdir = summary.workdir
    assert (workdir / "order.json").exists()
    assert not (workdir / "01_raw.txt").exists()

    summary_ocr = async_run(
        pipeline.run(
            photos, out_dir=tmp_path / "out", ocr_provider=book.provider(),
            llm_provider=EchoFormat(), settings=good_llm_settings(tmp_path),
            stop_after="ocr",
        )
    )
    assert (summary_ocr.workdir / "01_raw.txt").exists()
    assert not (summary_ocr.workdir / "02_repaired.txt").exists()


def test_quota_error_aborts_and_resume_continues(tmp_path):
    photos = tmp_path / "pb"
    photos.mkdir()
    book = PhotoBook(photos, page_count=3)

    class Limited(FakeOcrProvider):
        calls = 0

        async def ocr_image(self, image_bytes, filename):
            Limited.calls += 1
            if Limited.calls > 1:
                raise QuotaExceededError("Daily limit reached", status_code=429)
            return await super().ocr_image(image_bytes, filename)

    limited = Limited(responses={f"IMG_{i+1}.jpg": f"Strona {i+1}" for i in range(3)})
    limited.retry_config = RetryConfig(max_tries=2, base_delay=0.001)
    with pytest.raises(QuotaExceededError):
        async_run(
            pipeline.run(
                photos, out_dir=tmp_path / "out", ocr_provider=limited,
                llm_provider=EchoFormat(), settings=good_llm_settings(tmp_path),
            )
        )
    # page 1 cached; the rest aborted before burning quota
    workdir = tmp_path / "out" / "pb"
    assert len(list((workdir / "ocr").iterdir())) == 1

    # second run (resume default): remaining pages processed
    summary = async_run(
        pipeline.run(
            photos, out_dir=tmp_path / "out", ocr_provider=book.provider(),
            llm_provider=EchoFormat(), settings=good_llm_settings(tmp_path),
        )
    )
    assert summary.pages_cached == 1
    assert summary.pages_ocr_now == 2


def test_no_resume_clears_caches(tmp_path):
    photos = tmp_path / "pb"
    photos.mkdir()
    book = PhotoBook(photos, page_count=2)
    async_run(
        pipeline.run(
            photos, out_dir=tmp_path / "out", ocr_provider=book.provider(),
            llm_provider=EchoFormat(), settings=good_llm_settings(tmp_path),
        )
    )
    ocr2 = book.provider()
    async_run(
        pipeline.run(
            photos, out_dir=tmp_path / "out", ocr_provider=ocr2,
            llm_provider=EchoFormat(), settings=good_llm_settings(tmp_path),
            resume=False,
        )
    )
    assert len(ocr2.calls) == 2  # --no-resume re-processed everything


def test_llm_quota_aborts_repair_stage_with_resume_hint(tmp_path):
    photos = tmp_path / "pb"
    photos.mkdir()
    book = PhotoBook(photos, page_count=2)

    class QuotaLlm(FakeLlmProvider):
        async def chat(self, *, system, user, temperature=0.0):
            raise QuotaExceededError("rate limit", status_code=429)

    with pytest.raises(QuotaExceededError):
        async_run(
            pipeline.run(
                photos, out_dir=tmp_path / "out", ocr_provider=book.provider(),
                llm_provider=QuotaLlm(), settings=good_llm_settings(tmp_path),
            )
        )


def test_standalone_repair_and_format_commands(tmp_path):
    raw_path = tmp_path / "01_raw.txt"
    raw_path.write_text(
        "<<<PAGE 1>>>\nTak to zylo i dzialalo sie.\n<<<PAGE 2>>>\nkoniec bylu blisko.\n",
        encoding="utf-8",
    )
    summary = async_run(
        pipeline.repair_file(raw_path, llm_provider=EchoFormat(),
                             settings=good_llm_settings(tmp_path))
    )
    assert (tmp_path / "02_repaired.txt").exists()
    assert (tmp_path / "02_repaired.diff.txt").exists()
    assert summary.outputs["02_repaired.txt"].endswith("02_repaired.txt")

    summary_fmt = async_run(
        pipeline.format_file(
            tmp_path / "02_repaired.txt", llm_provider=EchoFormat(),
            settings=good_llm_settings(tmp_path),
        )
    )
    assert (tmp_path / "03_formatted.md").exists()
    assert summary_fmt.outputs["03_formatted.md"].endswith("03_formatted.md")


def test_export_standalone(tmp_path):
    md = tmp_path / "03_formatted.md"
    md.write_text("# Wstęp\n\nTo jest testowa treść książki.\n", encoding="utf-8")
    summary = async_run(
        pipeline.export_file(md, title="Testowa Księga", language="pl")
    )
    book_md = tmp_path / "book.md"
    book_docx = tmp_path / "book.docx"
    assert book_md.exists() and book_docx.exists()
    text = book_md.read_text(encoding="utf-8")
    assert 'title: "Testowa Księga"' in text
    assert "lang: pl-PL" in text
    assert summary.outputs["book.docx"] == str(book_docx)


def test_export_requires_pandoc(monkeypatch, tmp_path):
    import shutil

    md = tmp_path / "03_formatted.md"
    md.write_text("# Nagłówek\n\nŚciąga diakrytyków: ąę ł ś ż.\n", encoding="utf-8")
    monkeypatch.setattr(
        shutil, "which", lambda *_: None
    )
    from book2md.export import PandocMissingError

    with pytest.raises(PandocMissingError):
        asyncio.run(pipeline.export_file(md, title="x"))


def test_polish_fixture_end_to_end(tmp_path):
    """SPEC 15: diacritics survive, quotes/dashes unchanged, word sequence kept."""
    damaged = read_fixture("polish_damaged.txt")
    repaired_fixture = read_fixture("polish_repaired.txt")
    formatted_fixture = read_fixture("polish_formatted.md")

    # stage 4: scripted LLM returns the known-good repaired fixture
    class FixtureRepair(FakeLlmProvider):
        async def chat(self, *, system, user, temperature=0.0):
            self.calls.append({"system": system, "user": user})
            # The marker is part of the input chunk, so a faithful repair
            # keeps it in place (SPEC 8.3 rule 5); without it the validator
            # would reject the answer and fall back to the damaged text.
            return f"<<<PAGE 1>>>\n{repaired_fixture}"

    raw_path = tmp_path / "01_raw.txt"
    raw_path.write_text(f"<<<PAGE 1>>>\n{damaged}", encoding="utf-8")
    asyncio.run(
        pipeline.repair_file(
            raw_path, llm_provider=FixtureRepair(), settings=good_llm_settings(tmp_path)
        )
    )
    repaired_out = (tmp_path / "02_repaired.txt").read_text(encoding="utf-8")
    # diacritics restored
    for letters in ("ą", "ę", "ł", "ż", "ó", "ś", "ń"):
        assert letters in repaired_out
    # Polish quotes and dialogue dashes unchanged (SPEC 15)
    assert "„Może przyjedzie na święta”" in repaired_out
    assert "— Nic się nie zmieniło —" in repaired_out
    # hyphenated word joined
    assert "poprzednia" in repaired_out and "poprzed-\n" not in repaired_out

    # stage 5: scripted formatter returns known-good markdown
    class FixtureFormat(FakeLlmProvider):
        async def chat(self, *, system, user, temperature=0.0):
            self.calls.append({"system": system, "user": user})
            return formatted_fixture

    asyncio.run(
        pipeline.format_file(
            tmp_path / "02_repaired.txt", llm_provider=FixtureFormat(),
            settings=good_llm_settings(tmp_path),
        )
    )
    md = (tmp_path / "03_formatted.md").read_text(encoding="utf-8")
    assert md.strip() == formatted_fixture.strip()
    # diacritics survive the whole pipeline
    assert "się" in md and "żeby" in md and "Rozdział" in md
    # Polish quotes intact: openers and closers pair up 1:1 (SPEC 15)
    assert md.count("„") == md.count("”") > 0
    assert "„Może przyjedzie na święta”" in md
    # word sequence preserved: token equality (casefold, diacritics kept)
    from book2md.validate import tokenize

    stripped_repaired = tokenize(repaired_fixture, drop_markers=True)
    stripped_formatted = tokenize(formatted_fixture, drop_markers=True)
    assert stripped_repaired == stripped_formatted
