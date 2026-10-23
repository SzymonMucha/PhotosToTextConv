"""Stage 1 tests: ordering (SPEC 5) and dedup (SPEC 14) with temp folders."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from book2md.ordering import (
    collect_images,
    deduplicate,
    detect_strategy,
    natural_sort_key,
    order_folder,
    order_images,
    write_order_json,
)
from conftest import create_image, set_mtime


def test_natural_sort_name_before_ten():
    names = ["IMG_10.jpg", "IMG_2.jpg", "IMG_1.jpg"]
    assert sorted(names, key=natural_sort_key) == ["IMG_1.jpg", "IMG_2.jpg", "IMG_10.jpg"]


def test_collect_reads_exif_and_skips_unsupported(tmp_path: Path):
    base = tmp_path / "p"
    base.mkdir()
    create_image(base / "with_exif.jpg", exif_datetime=datetime(2024, 3, 1, 10, 0, 0))
    create_image(base / "no_exif.jpg")
    (base / "notes.txt").write_text("nope")
    pages = collect_images(base)
    by_name = {p.filename: p for p in pages}
    assert set(by_name) == {"with_exif.jpg", "no_exif.jpg"}
    assert by_name["with_exif.jpg"].exif_datetime == datetime(2024, 3, 1, 10, 0, 0)
    assert by_name["no_exif.jpg"].exif_datetime is None


def test_collect_empty_folder_raises(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        collect_images(empty)


def test_strategy_falls_back_to_mtime_when_any_exif_missing(tmp_path: Path):
    base = tmp_path / "p"
    base.mkdir()
    create_image(base / "a.jpg", exif_datetime=datetime(2024, 1, 2, 8, 0, 0))
    create_image(base / "b.jpg")  # no EXIF -> one missing is enough (SPEC 5)
    pages = collect_images(base)
    assert detect_strategy(pages) == "mtime"


def test_exif_strategy_used_when_all_have_exif(tmp_path: Path):
    base = tmp_path / "p"
    base.mkdir()
    create_image(base / "later.jpg", exif_datetime=datetime(2024, 1, 2, 9, 0, 0))
    create_image(base / "earlier.jpg", exif_datetime=datetime(2024, 1, 2, 8, 0, 0))
    pages, strategy = (lambda r: (r[0], r[1]))(order_images(collect_images(base)))
    assert strategy == "exif"
    assert [p.filename for p in pages] == ["earlier.jpg", "later.jpg"]
    assert pages[0].page == 1 and pages[1].page == 2


def test_mtime_strategy_orders_by_st_mtime(tmp_path: Path):
    base = tmp_path / "p"
    base.mkdir()
    create_image(base / "first.jpg")
    create_image(base / "second.jpg")
    create_image(base / "third.jpg")
    now = 1_700_000_000
    set_mtime(base / "third.jpg", now)
    set_mtime(base / "first.jpg", now - 100)
    set_mtime(base / "second.jpg", now - 50)
    pages, strategy = order_images(collect_images(base))
    assert strategy == "mtime"
    assert [p.filename for p in pages] == ["first.jpg", "second.jpg", "third.jpg"]


def test_name_strategy_ignores_timestamps(tmp_path: Path):
    base = tmp_path / "p"
    base.mkdir()
    create_image(base / "IMG_10.jpg")
    create_image(base / "IMG_2.jpg")
    pages, strategy = order_images(collect_images(base), order_by="name")
    assert strategy == "name"
    assert [p.filename for p in pages] == ["IMG_2.jpg", "IMG_10.jpg"]


def test_reverse_reverses_order(tmp_path: Path):
    base = tmp_path / "p"
    base.mkdir()
    create_image(base / "a.jpg")
    create_image(base / "b.jpg")
    set_mtime(base / "b.jpg", 500)
    set_mtime(base / "a.jpg", 900)
    pages, _ = order_images(collect_images(base), reverse=True)
    assert [p.filename for p in pages] == ["a.jpg", "b.jpg"]  # a has later mtime


def test_exif_trumps_mtime_within_exif_strategy(tmp_path: Path):
    base = tmp_path / "p"
    base.mkdir()
    # mtime says z-last/exif-first; EXIF must win because ALL pages have it
    create_image(base / "z.jpg", exif_datetime=datetime(2020, 1, 1, 8, 0, 0))
    create_image(base / "a.jpg", exif_datetime=datetime(2020, 1, 1, 9, 0, 0))
    now = 1_700_000_000
    set_mtime(base / "z.jpg", now)  # newest file...
    pages, _ = order_images(collect_images(base))
    assert [p.filename for p in pages] == ["z.jpg", "a.jpg"]  # ...but older photo


def test_deduplicate_drops_same_sha1_keeping_first(tmp_path: Path):
    base = tmp_path / "p"
    base.mkdir()
    create_image(base / "orig.jpg", color="red")
    create_image(base / "copy.jpg", color="red")  # identical pixels => same SHA-1
    create_image(base / "other.jpg")  # unique content
    pages, _ = order_images(collect_images(base))
    unique, skipped = deduplicate(pages)
    assert skipped == 1
    assert [p.filename for p in unique] == ["orig.jpg", "other.jpg"]
    # keep_duplicates bypasses the dedup
    keep_all, skipped_all = deduplicate(pages, keep_duplicates=True)
    assert skipped_all == 0
    assert len(keep_all) == 3


def test_duplicates_different_content_are_kept(tmp_path: Path):
    base = tmp_path / "p"
    base.mkdir()
    create_image(base / "one.jpg", color="red")
    create_image(base / "two.jpg", color="blue")
    pages, _ = order_images(collect_images(base))
    unique, skipped = deduplicate(pages)
    assert skipped == 0
    assert len(unique) == 2


def test_order_json_written(tmp_path: Path):
    base = tmp_path / "p"
    base.mkdir()
    create_image(base / "a.jpg")
    create_image(base / "b.jpg")
    set_mtime(base / "b.jpg", 1234567890)
    set_mtime(base / "a.jpg", 1234567800)
    pages, strategy = order_images(collect_images(base))
    out_json = tmp_path / "order.json"
    write_order_json(out_json, pages, strategy=strategy, reverse=False)
    text = out_json.read_text(encoding="utf-8")
    assert '"strategy": "mtime"' in text
    assert '"a.jpg"' in text and '"page": 1' in text


def test_order_folder_convenience(tmp_path: Path):
    base = tmp_path / "p"
    base.mkdir()
    create_image(base / "a.jpg")
    create_image(base / "b.jpg")
    set_mtime(base / "a.jpg", 100)
    set_mtime(base / "b.jpg", 200)
    pages, strategy, duplicates = order_folder(base)
    assert strategy == "mtime"
    assert duplicates == 0
    assert [p.page for p in pages] == [1, 2]


def test_missing_folder_raises(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        order_folder(tmp_path / "definitely_missing")
