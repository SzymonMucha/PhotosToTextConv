"""book2md: turn a folder of photographed book pages into clean Markdown + DOCX.

Pipeline (see SPEC.md):
1. order photos (EXIF / mtime / filename)
2. OCR each page through a free-tier API (concurrently, cached per image)
3. merge OCR output into ``01_raw.txt`` with ``<<<PAGE n>>>`` markers
4. LLM pass 1: repair OCR errors, restore Polish diacritics, fill gaps
5. LLM pass 2: format into Markdown (paragraphs + headings, no new content)
6. export to ``book.md`` and ``book.docx`` (both from the same Markdown)
"""

__version__ = "0.1.0"
