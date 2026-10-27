"""book2md: turn a folder of photographed book pages into clean Markdown + DOCX.

Pipeline:
1. order photos (EXIF / mtime / filename)
2. OCR each page through a free-tier API (concurrently, cached per image)
3. merge OCR output into ``01_raw.txt`` with ``<<<PAGE n>>>`` markers
4. one LLM pass: repair OCR errors + format into Markdown (convert stage)
5. export to ``book.md`` and ``book.docx`` (both from the same Markdown)
"""

__version__ = "0.2.0"
