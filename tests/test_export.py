"""Stage 6 export tests (SPEC 10): front matter, locale mapping, DOCX content.

The pandoc-dependent test skips automatically on machines without pandoc, so
the suite still passes everywhere without network access (SPEC 16) - pandoc
itself makes no network calls, but it must be installed as a system package.
"""

from __future__ import annotations

import shutil
import zipfile

import pytest

from book2md.export import add_front_matter, export_book, locale_for_language

HAS_PANDOC = shutil.which("pandoc") is not None


def test_locale_mapping():
    """Language codes map to Word locales; already-locale codes pass through."""
    assert locale_for_language("pl") == "pl-PL"
    assert locale_for_language("en-GB") == "en-GB"
    assert locale_for_language("cs") == "cs-CZ"   # mapped Czech locale
    assert locale_for_language("xx") == "xx-XX"   # unknown: passthrough fallback
    assert locale_for_language("") == "pl-PL"     # empty: Polish default


def test_front_matter_fields():
    """book.md carries title/lang/source_pages in YAML front matter (SPEC 10)."""
    final = add_front_matter(
        "# Wstęp\n\nTreść książki z diakrytykami: ą ę ł ś ż.\n",
        title="Moja Książka",
        language="pl",
        source_pages=7,
    )
    assert final.startswith("---\n")          # YAML block first
    assert final.count("---\n") >= 2          # and closed
    assert 'title: "Moja Książka"' in final   # double-quoted even with spaces
    assert "lang: pl-PL" in final
    assert "source_pages: 7" in final
    assert final.endswith("# Wstęp\n\nTreść książki z diakrytykami: ą ę ł ś ż.\n")


@pytest.mark.skipif(not HAS_PANDOC, reason="pandoc not installed")
def test_docx_diacritics_and_word_language(tmp_path):
    """The DOCX really contains the Polish diacritics and lang="pl-PL".

    This is the SPEC 10 acceptance test: pandoc output is opened with
    python-docx and the document styles are inspected for the Word
    spell-check language.
    """
    source = (
        "# Wstęp\n\n"
        "Polskie znaki diakrytyczne: ą ę ć ł ń ó ś ź ż.\n\n"
        "Drugie zdanie ma duże litery: Ą Ę Ć Ł Ń Ó Ś Ź Ż.\n"
    )
    book_md, book_docx, stats = export_book(
        source,
        out_dir=tmp_path,
        title="Diakrytyki",
        language="pl",
        source_pages=1,
    )
    assert book_md.exists() and book_docx.exists()

    import docx  # python-docx

    document = docx.Document(str(book_docx))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    for letter in "ąćęłńóśźż":
        assert letter in text, f"diacritic {letter} lost in DOCX"
    for letter in "ĄĆĘŁŃÓŚŹŻ":
        assert letter in text, f"uppercase diacritic {letter} lost in DOCX"
    assert stats.docx_paragraphs >= 1
    assert stats.docx_headings >= 1  # "# Wstęp" became a Word heading

    # Word UI language comes from the embedded style definitions.
    with zipfile.ZipFile(book_docx) as bundle:
        styles = bundle.read("word/styles.xml").decode("utf-8")
    assert "pl-PL" in styles
