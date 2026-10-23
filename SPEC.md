# book2md: Book Photos → OCR → LLM Cleanup → Markdown + DOCX

> This document is the implementation spec for an AI coding assistant. Read it fully before writing code. Build in the phases listed in section 12. Do not skip ahead, and do not add features that are not in this spec.

---

## 1. Goal

Turn a folder of photographed book pages into two clean, identically formatted files:

- `book.md` (for Obsidian)
- `book.docx` (for printing)

Pipeline:

1. **Input**: a folder of page photos. Chronological order of creation = page order.
2. **Order** the photos.
3. **OCR** each photo through an API.
4. **Merge** all OCR output into one raw `.txt` file (with page markers).
5. **LLM pass 1: repair**. Fix OCR errors and fill gaps using context from the rest of the text.
6. **LLM pass 2: format**. Split into paragraphs and headings using only what is already in the text. No new content.
7. **Export** the formatted Markdown to `.md` and `.docx`.

## 2. Hard constraints

- **Free or near-free** to run. No local models, no expensive frontier-model APIs. All providers must be swappable via config.
- **Fast enough** for ~100–400 pages: OCR calls run concurrently, LLM calls run per chunk.
- **Idempotent and resumable**: if the run crashes at page 217, re-running must not re-OCR pages 1–216 (cache per image).
- **No hallucinated content**: the LLM stages may only correct, fill obvious gaps, and format. This is the most important quality rule (see section 8).
- `.md` and `.docx` must come from the **same single source** (the final Markdown), so they never diverge.

## 3. Recommended stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Best library support for all steps |
| Core | A plain Python package (`book2md/`) with **no UI code** | Lets CLI, desktop, and web front-ends reuse it |
| CLI | `typer` | Phase 1 deliverable, easy to test |
| Web UI (Phase 4) | `streamlit` (fast) or `FastAPI` + simple HTML (more control) | Wraps the same core |
| HTTP | `httpx` (async) | Concurrent OCR/LLM calls |
| Image order | `Pillow` (EXIF) + `os.stat` fallback | See section 5 |
| Image prep | `Pillow` | Resize/compress to fit API size limits |
| Config | `pydantic-settings` + `.env` | Keys never in code |
| Cache | `diskcache` or plain JSON files in a `.cache/` dir | Resumability |
| MD → DOCX | `pandoc` via `pypandoc` | Reliable, keeps headings, bold, italics, lists |
| Tests | `pytest` | |

**OCR provider**: implement behind an interface (`OcrProvider`). Start with one free option, add others later. Candidates (verify current free-tier limits yourself before choosing, they change often):

- OCR.space (simple REST, free key, has file-size limits on the free tier, so compress images)
- Google Cloud Vision / Azure Document Intelligence (free monthly quotas, need account setup)
- A vision-capable LLM with a free tier (e.g. Gemini API) used as OCR

**LLM provider**: implement behind an interface (`LlmProvider`) using the **OpenAI-compatible chat completions** protocol (`base_url`, `api_key`, `model` in config). This single interface covers OpenRouter (including its free models), Groq, Gemini's OpenAI-compatible endpoint, and others. Default to a free-tier model; the user will set it in `.env`.

## 4. Project layout

```
book2md/
├── pyproject.toml
├── .env.example
├── README.md
├── SPEC.md                  # this file
├── src/book2md/
│   ├── __init__.py
│   ├── config.py            # settings loaded from env
│   ├── ordering.py          # step 1: sort images
│   ├── imageprep.py         # compress/resize for OCR limits
│   ├── ocr/
│   │   ├── base.py          # OcrProvider ABC
│   │   └── ocrspace.py      # first concrete provider
│   ├── llm/
│   │   ├── base.py          # LlmProvider ABC
│   │   └── openai_compat.py # generic OpenAI-compatible client
│   ├── pipeline.py          # orchestrates all stages
│   ├── chunking.py          # splits raw text for LLM
│   ├── prompts.py           # all LLM prompts as constants
│   ├── validate.py          # anti-hallucination checks
│   ├── export.py            # md → docx via pandoc
│   └── cli.py               # typer app
├── tests/
└── workdir/ (generated)     # per-run output, gitignored
    └── <book_name>/
        ├── ocr/             # one .json per page (cache)
        ├── 01_raw.txt
        ├── 02_repaired.txt
        ├── 03_formatted.md
        ├── book.md
        └── book.docx
```

Every intermediate artifact is written to disk so the user can inspect or hand-edit any stage and resume from there.

## 5. Stage 1: Ordering images

Input: a folder path. Accept `.jpg .jpeg .png .webp .heic` (HEIC only if `pillow-heif` is installed; otherwise warn and skip).

Sort key, in priority order:

1. EXIF `DateTimeOriginal` (plus `SubSecTimeOriginal` if present).
2. File modification time (`st_mtime`).
3. Filename, natural sort (`IMG_2` before `IMG_10`).

Rules:

- Use one strategy for the whole folder: if **any** file lacks EXIF, fall back to mtime for **all** files so the keys are comparable.
- Ties are broken by natural filename sort.
- File *creation* time is not reliable across OS and copy operations, so do not use `st_birthtime`/`st_ctime`.
- Print the detected order (index, filename, timestamp) and write it to `workdir/<book>/order.json`. Add a CLI flag `--order-by [exif|mtime|name]` to override, and `--reverse`.
- Page numbers used internally are the 1-based position in this sorted list, not the printed page numbers in the book.

## 6. Stage 2: OCR

Interface:

```python
class OcrProvider(ABC):
    async def ocr_image(self, image_bytes: bytes, filename: str) -> OcrResult: ...

@dataclass
class OcrResult:
    text: str
    confidence: float | None   # if provider supplies it
    raw: dict                  # raw provider response for debugging
```

Behavior:

- Preprocess with `imageprep`: auto-rotate by EXIF orientation, downscale so the long side is ≤ ~2000 px, re-encode as JPEG quality ~85, and further compress if the provider has a size limit.
- Run OCR calls **concurrently** with a semaphore (default 4, configurable) and retry with exponential backoff on 429/5xx (max 5 tries).
- Cache each page result in `workdir/<book>/ocr/<index>_<sha1-of-image>.json`. If the cache file exists, skip the call.
- If a page fails after retries, record it as failed, continue with the others, and print a summary at the end. Never abort the entire run for one page.
- **The target language is Polish** (`OCR_LANGUAGE=pol` for OCR.space; other providers use `pl`, so map it inside each provider class). Verify that the chosen provider and engine actually support Polish, because some OCR engines handle only a subset of languages and silently drop diacritics (ą ć ę ł ń ó ś ź ż). Do a manual test on 2–3 real pages before building further.
- Always handle text as UTF-8 and normalize it to Unicode NFC right after OCR. Never ASCII-fold, "clean" or strip non-ASCII characters anywhere in the pipeline.
- Keep the language configurable (`BOOK_LANGUAGE=pl`), but Polish is the default and the only language the prompts are tuned for.

## 7. Stage 3: Merge to raw TXT

Write `01_raw.txt` by concatenating pages in order. Insert a page marker line before each page:

```
<<<PAGE 12>>>
...ocr text...
```

- Markers are for traceability and chunking only. They are stripped before the final export.
- Failed pages get `<<<PAGE 12>>>\n[[OCR FAILED]]`.
- Do not "fix" anything at this stage. Only normalize line endings and trim trailing whitespace.

## 8. Stage 4: LLM repair (fill gaps from context)

### 8.1 Chunking

- Split `01_raw.txt` into chunks of roughly 1,500–3,000 words (configurable), always on page-marker boundaries.
- Each LLM request contains: `[CONTEXT BEFORE]` (last ~300 words of the previous chunk, read-only), `[TEXT TO REPAIR]`, and `[CONTEXT AFTER]` (first ~300 words of the next chunk, read-only).
- The model must return **only** the repaired version of `[TEXT TO REPAIR]`, keeping all `<<<PAGE n>>>` markers intact.
- Process chunks concurrently (low concurrency, free tiers have rate limits), with retry/backoff and per-chunk caching.
- Reassemble in order.

### 8.2 Repair prompt (put in `prompts.py`, adjust wording if needed but keep the rules)

```
You are a careful proofreader repairing OCR output of a book written in POLISH.

You receive TEXT TO REPAIR, with CONTEXT BEFORE and CONTEXT AFTER for reference only.
The text is Polish. All of it must stay Polish, with correct Polish diacritics.

Rules:
1. Fix obvious OCR errors: wrong characters (rn/m, l/1/I, 0/O), broken words,
   stray symbols, words split by line-break hyphenation.
1a. Restore Polish diacritics that OCR dropped or confused. Typical confusions:
   ł/l/t, ó/o/6, ą/a, ę/e, ć/c, ń/n, ś/s/ś, ź/ż/z, ż/z, and ci/ć, si/ś, zi/ź
   written as separate letters. Decide from the word and its grammatical context,
   e.g. "sie" next to a verb is "się", "zeby" is "żeby", "ze" meaning "that" is "że".
1b. A word split across lines with a hyphen at the line end ("poprzed-\nnia")
   must be joined ("poprzednia"). Keep legitimate hyphens and dashes: compound words
   ("biało-czerwony"), and the dialogue dash or long dash (— or –) at the start
   of a line or sentence.
1c. Keep Polish quotation marks as the author wrote them („…” or «…» or "…").
   Do not convert them to English style.
1d. Consider Polish inflection: do not "fix" a word into another case form
   unless the OCR error clearly requires it. If the form is grammatical in the
   sentence, leave it alone, even if it is rare or archaic.
   Older books may use archaic spelling or vocabulary; preserve it, do not modernize.
2. Where a word or short phrase is garbled or missing, reconstruct it ONLY if
   the surrounding sentence and the wider context make the correct wording
   clear. If you are not confident, leave the original text and append
   [?] directly after it. Do NOT guess freely.
3. Do NOT add new sentences, ideas, facts, or explanations.
   Do NOT remove or summarize content.
   Do NOT paraphrase or "improve" style. Keep the author's wording.
4. Keep the original language. Do not translate.
5. Keep every <<<PAGE n>>> marker exactly as is, in the same position.
6. Do not repeat CONTEXT BEFORE or CONTEXT AFTER in your answer.
7. Output only the repaired text. No commentary, no code fences.
```

### 8.3 Validation (`validate.py`), mandatory

After each chunk response, check and **fall back to the unrepaired chunk** (and log a warning) if any check fails:

- Word count of output is within ±15% of the input.
- All `<<<PAGE n>>>` markers from the input are present, in order, exactly once.
- The output does not contain the context blocks (check overlap with context text).
- The output is not empty and has no leading chatter ("Here is…", "Sure…").
- Optional: compute a normalized similarity ratio (`difflib`/`rapidfuzz`); if < 0.85, flag it.
- **Polish-specific checks:**
  - Normalize input and output to NFC before comparing.
  - The output must not lose diacritics overall: the share of characters in `ąćęłńóśźżĄĆĘŁŃÓŚŹŻ` among letters must not drop by more than 20% compared to the input, because that indicates the model "simplified" to ASCII. The output must also not be mostly ASCII-folded.
  - Language drift check: if the output contains many common English stopwords (the, and, of, is, to, that) while the input has almost none, the model translated. Reject the chunk.
  - Do not use diacritic-stripping when comparing words anywhere in the tokenizer, because diacritics are exactly what the repair step is meant to restore. Use `casefold()` only.

Also write `02_repaired.diff.txt` (unified diff of raw vs repaired) so the user can review what the LLM changed.

## 9. Stage 5: LLM formatting (no new content)

Operates on `02_repaired.txt` with the same chunking approach (chunk by page markers, and carry a short "previous chunk tail" for continuity of paragraphs and headings across boundaries).

Goal: produce Markdown with paragraphs and headings that **reflect the structure already in the text**.

Prompt rules (put in `prompts.py`):

```
You format plain book text written in POLISH into Markdown. You change structure only, never content.
Keep the text in Polish, with all diacritics exactly as in the input.

Polish conventions to recognize:
- Heading lines often look like: "Rozdział I", "ROZDZIAŁ 3", "Część druga", "Wstęp",
  "Przedmowa", "Zakończenie", "Bibliografia", "Przypisy", "Spis treści", or a short
  line in all caps. Treat these as headings only if they stand on their own line.
- Dialogue lines begin with a dash (— or –). Keep each dialogue line as its own paragraph
  and keep the dash.
- Lists may use "1.", "1)", "a)", "–" or "•". Keep them as Markdown lists.
- Footnote markers and "Przypisy" sections should stay where they are; do not move or delete them.

Rules:
1. Do not add, remove, reorder, rewrite, or summarize any words.
   Every word in the output must appear in the input in the same order.
2. Join lines that were broken by page layout into flowing paragraphs.
   Put a blank line between paragraphs.
3. Mark headings that are already evident in the text (chapter titles,
   section titles, numbered sections, all-caps title lines) with #, ##, ###
   according to hierarchy. Do not invent headings.
4. Preserve evident lists as Markdown lists, and emphasized words as
   *italic* or **bold** only where the original clearly indicates it.
5. A paragraph that continues across a <<<PAGE n>>> marker must be joined
   into one paragraph. Remove all <<<PAGE n>>> markers from your output.
6. Remove running headers/footers and bare page numbers ONLY if they clearly
   repeat on every page (e.g. the book title or "- 42 -" at the page top/bottom).
7. Output only the Markdown. No commentary, no code fences.
```

Validation: the **word-sequence check**. Tokenize input and output (NFC-normalized, `casefold()`ed, punctuation and Markdown symbols stripped, **diacritics kept**). Output tokens must equal input tokens, except for removed running headers/page numbers. If similarity falls below a threshold (default 0.97), fall back to a deterministic non-LLM formatter for that chunk: join lines into paragraphs by blank-line / sentence-end heuristics, no headings.

After all chunks: normalize heading levels globally (e.g. no jumping from `#` to `###`), collapse 3+ blank lines to 1, save as `03_formatted.md`.

## 10. Stage 6: Export

- `book.md`: copy of `03_formatted.md`. Add YAML front matter: `title`, `lang: pl-PL`, `source_pages`, `generated`. The title defaults to the folder name, overridable with `--title`.
- `book.docx`: `pypandoc.convert_file("book.md", "docx", format="markdown-smart", outputfile="book.docx", extra_args=["--standalone", "--metadata", "lang=pl-PL"])`. 
  - **`-smart` is required.** Pandoc's smart typography would otherwise convert the author's quotes and dashes to English-style typographic quotes, and the DOCX would no longer match the MD.
  - The `lang=pl-PL` metadata makes Word spell-check in Polish instead of flagging everything.
  - Optionally accept a `--reference-docx` for custom fonts, margins, and print layout. Any font used must support Polish diacritics (e.g. Calibri, Cambria, Times New Roman, Arial all do).
  - Pandoc must be installed (document this in README, and give a clear error if it is missing).
- Both files come from the same Markdown. Never generate DOCX from a different path.
- Check after export that the DOCX opens (`python-docx` loads it) and print the paragraph and heading counts for both outputs.

## 11. CLI (Phase 1 interface)

```
book2md run  PATH_TO_PHOTOS  [--title TEXT] [--out DIR]
                             [--order-by exif|mtime|name] [--reverse]
                             [--ocr-provider ocrspace] [--llm-model MODEL]
                             [--stop-after order|ocr|repair|format]
                             [--resume/--no-resume]
book2md ocr-only   PATH      # stages 1-3
book2md repair     RAW.txt   # stage 4 on an existing raw file
book2md format     TXT       # stage 5 on an existing text file
book2md export     MD        # stage 6 only
```

`.env.example`:

```
OCR_PROVIDER=ocrspace
OCR_API_KEY=
BOOK_LANGUAGE=pl
OCR_LANGUAGE=pol
OCR_CONCURRENCY=4

LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=
LLM_MODEL=
LLM_CONCURRENCY=2
CHUNK_WORDS=2000
```

Print a progress bar per stage (`rich`) and a final summary: pages OK/failed, chunks repaired/fell back, output paths.

## 12. Build phases (do in this order, one at a time)

1. **Phase 1, Skeleton + ordering**: project setup, config, `ordering.py`, `cli.py` with the `order` command. Tests with temp folders containing files with different mtimes and one fake EXIF.
2. **Phase 2, OCR**: `imageprep`, `OcrProvider`, one concrete provider, caching, concurrency, retries, and merge to `01_raw.txt`. Test with a mocked provider (no real API calls in tests).
3. **Phase 3, LLM stages**: `LlmProvider`, chunking, repair, validation + fallbacks, formatting, diff output. Test chunking and validation thoroughly with fake LLM responses (including bad ones: missing markers, chatty output, truncated output).
4. **Phase 4, Export**: pandoc conversion, front matter, checks. End-to-end run on a sample of 3–5 pages.
5. **Phase 5 (optional), Web UI**: Streamlit (or FastAPI) wrapper around `pipeline.run()`. See section 13.

## 13. Web UI notes (Phase 5, optional)

- The UI must only call `pipeline.run(...)`; no business logic in the UI layer.
- Upload multiple images. Browsers expose `lastModified` per file; send it along and use it as the mtime fallback, since server-side mtime will be the upload time. Show the detected order as a sortable list so the user can fix it before processing.
- **Bring your own key**: users enter their own OCR and LLM keys (kept in session only, never logged or stored). This keeps hosting free for the operator, and free-tier rate limits apply per user instead of to one shared key.
- Run the pipeline as a background job and show progress. Offer both files for download at the end. Delete uploads and results after a short retention period (e.g. 24 h) and say so in the UI. Books may be copyrighted and users' photos are private.

## 14. Edge cases to handle

- Two-page spreads in one photo: out of scope for v1; mention in README.
- Skewed/rotated/dark photos: auto-rotate via EXIF only. No heavy CV in v1.
- Duplicate photos (same SHA1): warn, and keep the first by default (flag `--keep-duplicates`).
- Empty OCR result for a page: insert `[[EMPTY PAGE]]`, not an error.
- LLM rate limits (429): backoff, then reduce concurrency; surface a clear message if the daily quota is exhausted, and tell the user that `--resume` will continue later.
- Very large books: process stages streaming from disk, not all in memory.

## 15. Polish-language testing and model choice

- Pick the free LLM by testing, not by name. Take one real OCR'd chunk (~1,500 words) with known errors and run 2–3 candidate free models on it. Choose the one that restores diacritics and inflected words correctly, does not translate, and does not rewrite sentences. Many small free models are weak in Polish, so expect to try a few. Record the winner in `.env.example` comments.
- Include a Polish fixture in `tests/fixtures/`: a short passage (~200 words) with deliberately damaged diacritics ("sie", "zeby", "Rozdzial"), a hyphenated line break, a dialogue with „ ” quotes and — dashes, and a heading. Unit tests must assert after repair/format fixtures that: diacritics survive the pipeline, quotes and dashes are unchanged, and the word sequence is preserved.
- Add a test that the DOCX contains `ą ę ł ś ż` correctly (`python-docx` read-back) and that its language is Polish.

## 16. Definition of done

- `book2md run ./photos --title "My Book"` produces `book.md` and `book.docx` with the same headings and paragraphs.
- Re-running the same command makes **zero** new OCR/LLM calls (full cache hit).
- Killing the process mid-run and re-running continues where it stopped.
- All tests pass without network access.
- A real Polish test page set (3–5 pages) comes out with correct diacritics, Polish quotation marks preserved, and no English text introduced.
- README explains setup: Python version, `pip install`, installing pandoc, getting free API keys, and an example `.env`.
