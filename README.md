# PhotosToTextConv (`book2md`)

Turn a folder of **photographed book pages** into clean **Markdown** and **DOCX**:

```
photos ──▶ page ordering ──▶ OCR ──▶ merge ──▶ LLM repair ──▶ LLM format ──▶ book.md + book.docx
```

Built for Polish books: diacritics (`ą ę ł ś ż`), Polish quotes (`„ ”`) and dialogue
dashes (`—`) are preserved, and an **anti-hallucination validator** checks every LLM
answer — rejected answers fall back to the un-repaired / deterministically formatted
text, so the pipeline never invents content on its own.

The stack is free/near-free by design: [OCR.space](https://ocr.space) (free key) for OCR
and any OpenAI-compatible LLM endpoint (e.g. OpenRouter free models) for repair/formatting.

---

## Requirements

| Tool                          | Notes                                                 |
|-------------------------------|-------------------------------------------------------|
| Python **3.11+**              | tested on 3.14                                        |
| **pandoc** (system binary)    | only needed for the final DOCX export (stage 6)       |
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
| `LLM_CONCURRENCY` | parallel repair chunks (free tier: keep 2)                    |
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

Open the results in `workdir/<book>/` (see *Output layout*). Every stage writes its
file — nothing is hidden in memory, so you can correct any intermediate file by hand and
re-run a later command.

## Commands

| Command                      | Stages | Produces                                             |
|------------------------------|--------|------------------------------------------------------|
| `book2md run FOLDER`         | 1–6    | everything below (`book.md`, `book.docx`)            |
| `book2md order FOLDER`       | 1      | `order.json` + printed page order                    |
| `book2md ocr-only FOLDER`    | 1–3    | `01_raw.txt` with `<<<PAGE n>>>` markers             |
| `book2md repair 01_raw.txt`  | 4      | `02_repaired.txt` + `02_repaired.diff.txt`           |
| `book2md format  02_repaired.txt` | 5 | `03_formatted.md`                                  |
| `book2md export  03_formatted.md` | 6 | `book.md` + `book.docx`                            |

Useful options (see `book2md <command> --help`):

- `run --stop-after order|ocr|repair|format` — run partially, resumable at any point.
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
    ├── cache/                      # per-chunk LLM cache repair/ and format/
    ├── 01_raw.txt                  # merged OCR text with <<<PAGE n>>> markers
    ├── 02_repaired.txt             # LLM-repaired text (or fallback per chunk)
    ├── 02_repaired.diff.txt        # unified diff raw vs repaired — review the LLM here
    ├── 03_formatted.md             # LLM-formatted Markdown
    ├── book.md                     # final Markdown incl. YAML front matter
    └── book.docx                   # Word export from the very same book.md
```

## Resumability & safety (what makes this tool trustworthy)

- **Per-page OCR cache** (`ocr/<page>_<sha1>.json`) and **per-chunk LLM caches**:
  re-running a book costs **zero** API calls for what already succeeded. Restarting after
  an interruption just continues where it stopped.
- **Quota exhaustion** aborts the run cleanly (exit code **3**) — remaining pages are left
  untouched so no quota is burned pointlessly. When the quota resets, re-run the same
  command (with `-resume`, the default).
- **Anti-hallucination checks** (see `src/book2md/validate.py`): page-marker integrity,
  ±15 % word count, similarity (repair ≥0.60, warn <0.85; format ≥0.97), diacritic share,
  no repeated context blocks, English-drift detection. Any failed chunk **falls back** to
  the un-repaired resp. deterministically formatted text and logs a warning.
- Both `book.md` and `book.docx` are always built from **one** Markdown source
  (`03_formatted.md`), never from two independent conversions.

## Troubleshooting

- **“pandoc is not installed”** — run only `order/ocr/repair/format`, install pandoc
  (table above), then re-run. Or finish manually with `book2md export 03_formatted.md`.
- **Wrong page order** — inspect `order.json` (strategy is recorded there), then re-run
  with `--order-by exif|mtime|name` and/or `--reverse`.
- **Scanned two-page spreads** — this pipeline is deliberately the *photographed
  single* page kind: crop your photos to one page each. A spread makes the OCR text of
  two pages merge into a marker block of that one photo.
- **Dash/quote oddities in the DOCX** — stage 6 uses pandoc `markdown-smart`, which
  applies typographic quotes and dashes (e.g. ASCII `"..."` becomes curly `“...”`).
  This cannot damage content: the word-sequence validation happens *before* export, and
  the Polish `„ ”` quotes and `—` dashes in the source text pass through unchanged.
- **OCR keeps failing with large pages** — the image is downscaled/re-encoded before
  sending (defaults: long side 2000 px, JPEG q85, ≤ ~0.9 MB matching OCR.space free
  limits). You can tune `OCR_MAX_LONG_SIDE`, `OCR_JPEG_QUALITY`, `OCR_MAX_BYTES` in
  `.env` (see `src/book2md/config.py`).
- **Repair chunks very slow / quota burns fast** — reduce `LLM_CONCURRENCY` to 1 or lower
  `CHUNK_WORDS`; every chunk is cached, so re-runs only pay for what failed.

## Running the tests

The full suite passes without any network access (fake providers only; the two
pandoc-dependent checks skip automatically when pandoc is not installed):

```bash
.venv/bin/python -m pytest tests/
```

## Project layout

```
src/book2md/
├── cli.py            # typer app: order / run / ocr-only / repair / format / export
├── pipeline.py       # stage orchestration, caches, reporter protocol
├── ordering.py       # page order via EXIF → mtime → name, duplicate detection
├── imageprep.py      # EXIF-fix, downscale, JPEG re-encode to OCR size limits
├── ocr/              # OcrProvider ABC + ocrspace implementation (+ retry/backoff)
├── llm/              # LlmProvider base + OpenAI-compatible client (+ retry/backoff)
├── chunking.py       # chunking + context windows
├── prompts.py        # system/user prompts (repair, format)
├── validate.py       # anti-hallucination checks (Polish-aware)
├── formatting.py     # merge raw text, deterministic formatter, write_if_changed
├── export.py         # YAML front matter + pandoc DOCX
└── config.py         # pydantic-settings (.env)
```

Additional docs: `SPEC.md` is the authoritative specification this implementation follows
(stage rules, thresholds, error texts, test plan).
