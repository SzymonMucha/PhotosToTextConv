"""Image preparation before OCR (SPEC 6).

Photos taken with a phone are too large and too heavy for free OCR APIs:

* auto-rotate using EXIF orientation (no heavy CV in v1),
* downscale so the long side is at most ~2000 px,
* re-encode as JPEG (quality ~85),
* keep squeezing (lower quality, then smaller size) until the payload fits
  the provider's byte limit (OCR.space free tier: ~1 MB).

Text is handled as UTF-8 end-to-end; nothing here touches characters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

# Reading HEIC pages requires the optional pillow-heif extra; openers are
# registered globally, which is safe (idempotent) to call on import.
try:  # pragma: no cover - depends on the environment
    import pillow_heif  # type: ignore[import-not-found]

    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False


@dataclass
class PreparedImage:
    """Result of preparing one photo for an OCR call."""

    data: bytes          # encoded image bytes ready to upload
    format: str          # "jpeg" always (re-encoded)
    width: int
    height: int
    jpeg_quality: int
    attempts: int        # how many encode attempts it took to satisfy limits


def prepare_bytes(
    raw: bytes,
    *,
    max_long_side: int = 2000,
    jpeg_quality: int = 85,
    max_bytes: int = 900_000,
    min_jpeg_quality: int = 40,
) -> PreparedImage:
    """Auto-rotate, downscale and JPEG-compress photo bytes (SPEC 6).

    Strategy order:

    1. EXIF auto-rotation via ``ImageOps.exif_transpose``.
    2. Downscale the long side to ``max_long_side`` (LANCZOS keeps text legible).
    3. Encode JPEG at ``jpeg_quality``.
    4. If the payload exceeds ``max_bytes``: lower quality first (text survives
       better than resolution), then step the long side down by 10%.
    """
    with Image.open(_controls(raw)) as img:
        img = ImageOps.exif_transpose(img)  # honor EXIF orientation (SPEC 14)
        if img.mode not in ("RGB", "L"):
            # P/RGBA/CMYK modes encode badly as JPEG; RGB is what OCR wants.
            img = img.convert("RGB")

        img = _downscale(img, max_long_side)

        quality = jpeg_quality
        attempts = 0
        while True:
            attempts += 1
            data = _encode_jpeg(img, quality)
            if len(data) <= max_bytes:
                return PreparedImage(
                    data=data,
                    format="jpeg",
                    width=img.width,
                    height=img.height,
                    jpeg_quality=quality,
                    attempts=attempts,
                )
            # Too big: prefer degrading quality down to text-safe minimum.
            if quality > min_jpeg_quality:
                quality = max(min_jpeg_quality, quality - 10)
                continue
            # Otherwise give up resolution by 10% steps and retry quality.
            needs_more = _long_side(img) * 0.9
            if int(needs_more) < 100:  # never shrink into illegibility
                raise ImageTooLargeError(
                    f"Cannot fit image into {max_bytes} bytes "
                    f"(smallest attempt: {len(data)} bytes)."
                )
            img = _downscale(img, int(needs_more))
            quality = jpeg_quality


def prepare_file(path: Path, **kwargs) -> PreparedImage:
    """Read photo ``path`` and prepare it (see :func:`prepare_bytes`)."""
    raw = path.read_bytes()
    return prepare_bytes(raw, **kwargs)


def _controls(raw: bytes):
    """Wrap bytes for ``Image.open`` without touching the disk."""
    from io import BytesIO

    return BytesIO(raw)


def _long_side(img: Image.Image) -> int:
    return max(img.width, img.height)


def _downscale(img: Image.Image, max_long_side: int) -> Image.Image:
    """Halve-free precise downscale; never enlarges an image."""
    long_side = _long_side(img)
    if long_side <= max_long_side:
        return img
    scale = max_long_side / long_side
    new_size = (round(img.width * scale), round(img.height * scale))
    resampler = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    return img.resize(new_size, resampler)


def _encode_jpeg(img: Image.Image, quality: int) -> bytes:
    """JPEG-encode in memory; ``optimize=True`` squeezes a few extra percent."""
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


class ImageTooLargeError(Exception):
    """Raised when even the smallest acceptable encoding exceeds the limit."""
