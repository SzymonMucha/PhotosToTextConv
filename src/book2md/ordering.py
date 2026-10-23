"""Stage 1: order photographed book pages (SPEC 5).

Sort key, in priority order:

1. EXIF ``DateTimeOriginal`` (+ ``SubSecTimeOriginal`` if present).
2. File modification time (``st_mtime``).
3. Filename, natural sort (``IMG_2`` before ``IMG_10``).

Rules:

* One strategy for the whole folder: if *any* file lacks EXIF date, fall back
  to mtime for *all* files so keys stay comparable.
* Ties are broken by natural filename sort.
* Creation time is unreliable across OSes/copies, so ``st_birthtime`` and
  ``st_ctime`` are never used.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

#: Accepted input extensions (SPEC 5).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}

# EXIF tag ids (_EXIF_* aliases exist in Pillow but the numeric tags are stable).
_TAG_DATETIME_ORIGINAL = 36867
_TAG_SUBSEC_TIME_ORIGINAL = 37520
_IFD_EXIF = 0x8769

#: Sort strategies, ordered by preference (SPEC 5).
ORDER_STRATEGIES = ("exif", "mtime", "name")


@dataclass
class PageImage:
    """One input photo, before OCR. ``page`` is the 1-based position once sorted."""

    path: Path
    mtime: float
    sha1: str
    exif_datetime: datetime | None = None
    subsec: str | None = None
    page: int = 0  # assigned after sorting

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def timestamp(self) -> str:
        """Human-readable timestamp for logs and ``order.json``."""
        if self.exif_datetime is not None:
            base = self.exif_datetime.isoformat(timespec="milliseconds")
        else:
            base = datetime.fromtimestamp(self.mtime).isoformat(timespec="milliseconds")
        return base


def compute_sha1(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-1 (used for duplicate detection and caches)."""
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def natural_sort_key(name: str) -> tuple[tuple[int, int, str], ...]:
    """Natural sort key: ``IMG_2`` sorts before ``IMG_10``.

    Digit runs become numeric parts, everything else compares case-folded.
    The inner tuples have a uniform type shape so mixed cases never compare
    ``int`` against ``str`` (which would raise in Python 3).
    """
    parts = re.split(r"(\d+)", name)
    return tuple(
        (1, int(part), "") if part.isdigit() else (0, 0, part.casefold())
        for part in parts
    )


def _parse_exif_datetime(value: str | None) -> datetime | None:
    """EXIF stores ``YYYY:MM:DD HH:MM:SS``; ``None`` when absent/unparseable."""
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def _subsec_to_micros(subsec: str | None) -> timedelta:
    """``SubSecTimeOriginal`` is digits, e.g. ``"0145"`` == 145/10000 s."""
    if not subsec or not subsec.strip().isdigit():
        return timedelta(0)
    digits = subsec.strip()
    micros = int(digits) / (10 ** len(digits)) * 1_000_000
    return timedelta(microseconds=round(micros))


def read_exif_timestamp(path: Path) -> tuple[datetime | None, str | None]:
    """Read EXIF ``DateTimeOriginal`` (+ ``SubSecTimeOriginal``), else ``None``.

    ``DateTimeOriginal`` lives in the Exif sub-IFD, so ``get_ifd`` is required.
    Reading metadata only does not decode the image, this stays fast.
    """
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            exif_ifd = exif.get_ifd(_IFD_EXIF)
            raw_dt = exif_ifd.get(_TAG_DATETIME_ORIGINAL)
            if isinstance(raw_dt, bytes):
                raw_dt = raw_dt.decode("latin-1", "replace")
            raw_subsec = exif_ifd.get(_TAG_SUBSEC_TIME_ORIGINAL)
            if isinstance(raw_subsec, bytes):
                raw_subsec = raw_subsec.decode("latin-1", "replace")
    except (UnidentifiedImageError, OSError, ValueError):
        return None, None
    dt = _parse_exif_datetime(str(raw_dt) if raw_dt else None)
    return dt, (str(raw_subsec) if dt and raw_subsec else None)


def collect_images(
    input_dir: Path,
    *,
    allow_heic: bool = True,
    on_warning: Any = None,
) -> list[PageImage]:
    """Find image files in ``input_dir`` and read their metadata (SPEC 5).

    Satisfaction of the HEIC rule: ``.heic`` files are only kept when
    ``pillow-heif`` is installed; otherwise they warn once and get skipped --
    they could not be OCR'd later either.
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a folder: {input_dir}")

    heic_supported = allow_heic and _register_heif(on_warning)
    skipped_heic = 0

    pages: list[PageImage] = []
    for path in sorted(input_dir.iterdir(), key=lambda p: natural_sort_key(p.name)):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        if ext == ".heic" and not heic_supported:
            skipped_heic += 1
            continue
        dt, subsec = read_exif_timestamp(path)
        pages.append(
            PageImage(
                path=path,
                mtime=path.stat().st_mtime,
                sha1=compute_sha1(path),
                exif_datetime=dt,
                subsec=subsec,
            )
        )

    if skipped_heic and on_warning is not None:
        on_warning(
            f"{skipped_heic} .heic file(s) skipped: install 'book2md[heic]' "
            f"(pillow-heif) to process them."
        )
    if not pages:
        raise FileNotFoundError(
            f"No usable photos (.jpg .jpeg .png .webp .heic) found in {input_dir}"
        )
    return pages


def _register_heif(on_warning: Any = None) -> bool:
    """Best-effort import of pillow-heif; ``True`` when HEIC can be opened."""
    try:
        import pillow_heif  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        pillow_heif.register_heif_opener()
        return True
    except Exception as exc:  # libheif missing/broken
        if on_warning is not None:
            on_warning(f"Heic support broken ({exc}); .heic files will be skipped.")
        return False


def detect_strategy(pages: list[PageImage]) -> str:
    """Choose the whole-folder strategy (SPEC 5): all-EXIF or fall back to mtime."""
    return "exif" if all(p.exif_datetime is not None for p in pages) else "mtime"


def sort_key_factory(strategy: str) -> Any:
    """Return a key function implementing the chosen strategy.

    EXIF keys use ``datetime`` directly (comparable), mtime keys use the float
    ``st_mtime``, name keys use the natural-sort tuple. Every key tuple ends
    with the natural filename, which breaks ties deterministically.
    """
    if strategy == "exif":
        missing_marker = datetime.min + timedelta(days=2)  # stable sentinel

        def key_exif(p: PageImage):
            # EXIF stores no timezone; the naive wall-clock value is comparable
            # across the whole folder. Missing EXIF (shouldn't happen: the
            # strategy is only picked when all pages have it) sorts first.
            if p.exif_datetime is None:
                return (missing_marker, natural_sort_key(p.filename))
            primary = p.exif_datetime + _subsec_to_micros(p.subsec)
            return (primary, natural_sort_key(p.filename))

        return key_exif
    if strategy == "mtime":

        def key_mtime(p: PageImage):
            return (p.mtime, natural_sort_key(p.filename))

        return key_mtime
    if strategy == "name":
        return lambda p: (natural_sort_key(p.filename),)
    raise ValueError(f"Unknown order strategy: {strategy!r}")


def order_images(
    pages: list[PageImage],
    *,
    order_by: str | None = None,
    reverse: bool = False,
) -> tuple[list[PageImage], str]:
    """Return the pages in final page order plus the strategy used.

    ``order_by`` overrides detection. ``--reverse`` reverses the sorted list,
    including tie order (simple, predictable, easy to undo with the flag).
    """
    strategy = order_by or detect_strategy(pages)
    if strategy not in ORDER_STRATEGIES:
        raise ValueError(
            f"Unknown --order-by value {strategy!r}; expected one of {ORDER_STRATEGIES}"
        )
    ordered = sorted(pages, key=sort_key_factory(strategy))
    if reverse:
        ordered.reverse()
    for i, page in enumerate(ordered, start=1):
        page.page = i
    return ordered, strategy


def deduplicate(
    pages: list[PageImage], *, keep_duplicates: bool = False, on_warning: Any = None
) -> tuple[list[PageImage], int]:
    """Drop duplicate photos (same SHA-1), keeping the first (SPEC 14).

    Returns the (possibly unchanged) list and how many duplicates were skipped.
    """
    if keep_duplicates:
        return pages, 0
    seen: set[str] = set()
    unique: list[PageImage] = []
    skipped = 0
    for page in pages:  # already in page order; "keep the first" == lowest page
        if page.sha1 in seen:
            skipped += 1
            if on_warning is not None:
                on_warning(
                    f"duplicate photo skipped: {page.filename} "
                    f"(same SHA-1 as page {len(unique)})"
                )
            continue
        seen.add(page.sha1)
        unique.append(page)
    # Keep page numbers contiguous after dropping duplicates.
    for i, page in enumerate(unique, start=1):
        page.page = i
    return unique, skipped


def write_order_json(
    out_path: Path, pages: list[PageImage], *, strategy: str, reverse: bool
) -> None:
    """Persist the detected order (SPEC 5): ``workdir/<book>/order.json``."""
    payload = {
        "strategy": strategy,
        "reverse": reverse,
        "pages": [
            {
                "page": page.page,
                "filename": page.filename,
                "path": str(page.path),
                "timestamp": page.timestamp,
                "sha1": page.sha1,
                "mtime": page.mtime,
                "exif_datetime": (
                    page.exif_datetime.isoformat() if page.exif_datetime else None
                ),
                "subsec": page.subsec,
            }
            for page in pages
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def order_folder(
    input_dir: Path,
    *,
    order_by: str | None = None,
    reverse: bool = False,
    keep_duplicates: bool = False,
    on_warning: Any = None,
) -> tuple[list[PageImage], str, int]:
    """Convenience: collect, order and de-duplicate a folder in one call."""
    collected = collect_images(input_dir, on_warning=on_warning)
    ordered, strategy = order_images(collected, order_by=order_by, reverse=reverse)
    unique, skipped = deduplicate(
        ordered, keep_duplicates=keep_duplicates, on_warning=on_warning
    )
    return unique, strategy, skipped
