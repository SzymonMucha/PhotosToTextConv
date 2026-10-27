# PhotosToTextConv (`book2md`)

Turn a folder of **photographed book pages** into clean **Markdown** and **DOCX**:

```
photos ──▶ page ordering ──▶ OCR ──▶ merge ──▶ one LLM pass (convert) ──▶ book.md + book.docx
```

Built for Polish books: the LLM pass repairs the OCR damage (missing
diacritics like `ą ę ł ś ż`, line-break hyphens, misread characters) and turns
the plain text into Markdown in one step, working directly on the scanned
text. Polish quotes (`„ ”`) and dialogue dashes (`—`) are preserved, and the
prompt instructs the model to stick to the original words - no translating,
no paraphrasing, no inventing.

The stack is free/near-free by design: [OCR.space](https://ocr.space) (free
key) for OCR and any OpenAI-compatible LLM endpoint (e.g. OpenRouter free
models) for the convert pass.

---

## Requirements

| Tool                          | Notes                                                 |
|-------------------------------|-------------------------------------------------------|
| Python **3.11+**              | tested on 3.14                                        |
| **pandoc** (system binary)    | only needed for the final DOCX export (stage 5)       |
| `libheif` *(optional)*        | only if you want HEIC photo support (extra: `[heic]`) |

Install pandoc first (the app prints these exact instructions if it is missing):

```bash
# Debian/Ubuntu     sudo apt install pandoc
# Arch              sudo pacman -S pandoc
# Fedora            sudo dnf install pandoc
# macOS             brew install pandoc
# Windows           winget install --id JohnMacFarlane.Pandoc
```

## Installation

```bash
cd PhotosToTextConv

# 1) virtual environment (system pip is often PEP 668 locked on Linux)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2) the app (+ dev extras for the test suite)
pip install -e ".[dev]"

# optional: HEIC iPhone photos need a system libheif first, then
pip install -e ".[heic]"

# check it works
book2md --version
```

## Configuration

```bash
cp .env.example .env               # then edit .env
```

`.env` fields:

| Key               | Meaning                                                       |
|-------------------|---------------------------------------------------------------|
| `OCR_PROVIDER`    | `ocrspace` (built-in provider)                                |
| `OCR_API_KEY`     | your free OCR.space key (https://ocr.space/ocrapi/freekey)    |
| `BOOK_LANGUAGE`   | `pl` — used for the Word language (`lang=pl-PL`) and prompts  |
| `OCR_LANGUAGE`    | OCR engine language code (`pol`, `eng`, …)                    |
| `OCR_CONCURRENCY` | parallel OCR calls (free tier: keep 4)                        |
| `LLM_BASE_URL`    | OpenAI-compatible endpoint, e.g. `https://openrouter.ai/api/v1` |
| `LLM_API_KEY`     | key for that endpoint                                         |
| `LLM_MODEL`       | model id, e.g. a free OpenRouter model                        |
| `CHUNK_WORDS`     | chunk size in words (300–3000; 2000 recommended)              |

Keys never live in code; the settings are loaded from `.env` by `pydantic-settings`.

## Quick start

```bash
# whole pipeline on a folder of photos:
book2md run "Ścieżka/Do/Zdjęć"

# with an explicit title and output location:
book2md run ~/Fotografie/BookA --title "Tomek w krainie czarów" --out ../wyniki

# only see the detected page order first:
book2md order ~/Fotografie/BookA
```

Open the results in `workdir/<book>/` (see *Output layout*). Every stage
writes its file — nothing is hidden in memory, so you can correct any
intermediate file by hand and re-run the next command.

## Commands

| Command                      | Stages | Produces                                             |
|------------------------------|--------|------------------------------------------------------|
| `book2md run FOLDER`         | 1–5    | everything below (`book.md`, `book.docx`)            |
| `book2md order FOLDER`       | 1      | `order.json` + printed page order                    |
| `book2md ocr-only FOLDER`    | 1–3    | `01_raw.txt` with `<<<PAGE n>>>` markers             |
| `book2md convert 01_raw.txt` | 4      | `03_formatted.md` (one LLM pass: repair + format)    |
| `book2md export 03_formatted.md` | 5 | `book.md` + `book.docx`                            |

Useful options (see `book2md <command> --help`):

- `run --stop-after order|ocr|convert` — run partially, resumable at any point.
- `--resume / --no-resume` (default: resume) — with `--no-resume` the per-page OCR and
  per-chunk LLM caches are deleted, so **all** pages/chunks are processed again.
- `--order-by exif|mtime|name`, `--reverse`, `--keep-duplicates` — page ordering control.
- `--llm-model`, `--ocr-provider` — one-off overrides of `.env`.
- `export --reference-docx template.docx` — belt-and-braces: pandoc takes a style template.
- `export --language pl` — sets the Word spell-check language (`lang=pl-PL` inside
  `book.md`'s YAML front matter).

## Output layout

```
workdir/
└── Przykladowa_Ksiazka/            # sanitized photo-folder name
    ├── order.json                  # detected page order (diagnose wrong order here)
    ├── ocr/                        # per-page OCR cache  <page>_<sha1>.json
    ├── cache/                      # per-chunk LLM cache (convert)
    ├── 01_raw.txt                  # merged OCR text with <<<PAGE n>>> markers
    ├── 03_formatted.md             # Markdown from the single LLM pass
    ├── book.md                     # final Markdown incl. YAML front matter
    └── book.docx                   # Word export from the very same book.md
```

## Resumability

- **Per-page OCR cache** (`ocr/<page>_<sha1>.json`) and **per-chunk LLM cache**
  (`cache/chunk_<idx>_<sha1>.json`): re-running a book costs **zero** API calls
  for what already succeeded. Restarting after an interruption just continues
  where it stopped.
- **Changed caching keys**: the LLM cache is keyed by the chunk's text SHA-1 —
  if you hand-edit `01_raw.txt`, only the edited chunks are recomputed.
- **Quota exhaustion** aborts the run cleanly (exit code **3**) — remaining
  pages/chunks are left untouched so no quota is burned pointlessly. When the
  quota resets, re-run the same command (with `-resume`, the default).
- Both `book.md` and `book.docx` are always built from **one** Markdown source
  (`03_formatted.md`), never from two independent conversions.
- Artifacts are written with "don't rewrite if identical", so hand-edits keep
  their mtime and content across re-runs.

## Troubleshooting

- **“pandoc is not installed”** — run only `order/ocr-only/convert`, install pandoc
  (table above), then re-run. Or finish manually with `book2md export 03_formatted.md`.
- **Wrong page order** — inspect `order.json` (strategy is recorded there), then re-run
  with `--order-by exif|mtime|name` and/or `--reverse`.
- **Scanned two-page spreads** — this pipeline is deliberately the *photographed
  single page* kind: crop your photos to one page each. A spread makes the OCR text of
  two pages merge into the marker block of that one photo.
- **Dash/quote oddities in the DOCX** — stage 5 uses pandoc `markdown-smart`, which
  applies typographic quotes and dashes (e.g. ASCII `"..."` becomes curly `“...”`).
  The Polish `„ ”` quotes and `—` dashes in the source text pass through unchanged.
- **OCR keeps failing with large pages** — the image is downscaled/re-encoded before
  sending (defaults: long side 2000 px, JPEG q85, ≤ ~0.9 MB matching OCR.space free
  limits). You can tune `OCR_MAX_LONG_SIDE`, `OCR_JPEG_QUALITY`, `OCR_MAX_BYTES` in
  `.env` (see `src/book2md/config.py`).
- **Convert step very slow / quota burns fast** — chunks are processed with small
  context windows and one call per chunk; every chunk is cached, so re-runs only pay
  for what failed. Lower `CHUNK_WORDS` to retry smaller pieces.

## Running the tests

The full suite passes without any network access (fake providers only; the two
pandoc-dependent checks skip automatically when pandoc is not installed):

```bash
.venv/bin/python -m pytest tests/
```

## Project layout

```
src/book2md/
├── cli.py            # typer app: order / run / ocr-only / convert / export
├── pipeline.py       # stage orchestration, caches, reporter protocol
├── ordering.py       # page order via EXIF → mtime → name, duplicate detection
├── imageprep.py      # EXIF-fix, downscale, JPEG re-encode to OCR size limits
├── ocr/              # OcrProvider ABC + ocrspace implementation (+ retry/backoff)
├── llm/              # LlmProvider base + OpenAI-compatible client (+ retry/backoff)
├── chunking.py       # chunking + context tails for paragraph flow
├── prompts.py        # the single convert prompt (repair + format, one pass)
├── formatting.py     # merge raw text, salvage, markdown cleanups
├── export.py         # YAML front matter + pandoc DOCX
└── config.py         # pydantic-settings (.env)
```

Additional docs: `SPEC.md` documents the original design this implementation grew from
(stage ideas, error texts, pipeline sketch); the code comments above every module are
the current truth.
