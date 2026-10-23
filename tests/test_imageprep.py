"""Image preparation tests (SPEC 6): rotate, resize, JPEG, size limits."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from book2md.imageprep import ImageTooLargeError, prepare_bytes, prepare_file
from conftest import create_image
import pytest


def _jpeg_bytes(size=(3000, 4000), color="white") -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def test_downscales_long_side_to_limit():
    prepared = prepare_bytes(_jpeg_bytes((3000, 4000)), max_long_side=2000)
    img = Image.open(BytesIO(prepared.data))
    assert max(img.size) <= 2000
    assert img.format == "JPEG"
    # aspect ratio preserved (2:3 rounded)
    assert abs(img.width / img.height - 3 / 4) < 0.02


def test_never_enlarges_small_images():
    prepared = prepare_bytes(_jpeg_bytes((640, 480)), max_long_side=2000)
    img = Image.open(BytesIO(prepared.data))
    assert img.size == (640, 480)


def test_jpeg_payload_respected_by_dropping_quality_first():
    # a noisy image that is incompressible: quality drops before resolution
    import random

    random.seed(7)
    img = Image.new("RGB", (1600, 1200))
    img.putdata([tuple(random.randrange(256) for _ in range(3)) for _ in range(1600 * 1200)])
    buf = BytesIO()
    img.save(buf, "JPEG", quality=98)
    noisy = buf.getvalue()
    prepared = prepare_bytes(noisy, max_long_side=2000, max_bytes=30_000, jpeg_quality=85)
    assert len(prepared.data) <= 30_000
    assert prepared.jpeg_quality <= 85  # quality was lowered, not only size


def test_extreme_limit_fails_with_clear_error():
    with pytest.raises(ImageTooLargeError):
        prepare_bytes(
            _jpeg_bytes((2000, 3000)), max_long_side=2000, max_bytes=1
        )


def test_image_bytes_decode_back_to_same_pixels_count():
    prepared = prepare_bytes(_jpeg_bytes((1000, 1500)), max_long_side=1000)
    img = Image.open(BytesIO(prepared.data))
    assert (img.width, img.height) == (667, 1000)  # 1000*2/3, 1500*2/3


def test_prepare_file(tmp_path):
    photo = create_image(tmp_path / "p.jpg", size=(1200, 1600))
    prepared = prepare_file(photo, max_long_side=800)
    img = Image.open(BytesIO(prepared.data))
    assert max(img.size) == 800


def test_rgba_converted_to_rgb():
    buf = BytesIO()
    Image.new("RGBA", (300, 400), (255, 0, 0, 120)).save(buf, "PNG")
    prepared = prepare_bytes(buf.getvalue(), max_long_side=2000)
    img = Image.open(BytesIO(prepared.data))
    assert img.format == "JPEG"
    assert img.mode in ("RGB", "L")
