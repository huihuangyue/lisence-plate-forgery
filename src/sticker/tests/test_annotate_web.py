from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import sticker.annotate_web as annotate_web
from sticker.annotate_web import (
    AnnotationHandler,
    AnnotationItem,
    AnnotationState,
    discover_annotation_items,
    load_ocr_results,
    parse_annotation_entry,
    parse_problem_entry,
    select_annotation_items,
)
from shape.geometry import PlateDetection, PlateType


def test_parse_annotation_entry_supports_keyboard_multiselect() -> None:
    assert parse_annotation_entry("47", 8) == ("suspicious", (4, 7))
    assert parse_annotation_entry("4, 7", 8) == ("suspicious", (4, 7))
    assert parse_annotation_entry("0", 8) == ("clear", ())
    assert parse_annotation_entry("U", 8) == ("unassessable", ())
    with pytest.raises(ValueError):
        parse_annotation_entry("8", 7)


def test_problem_character_entry_resolves_duplicates_to_slots() -> None:
    assert parse_problem_entry("E 7#2", 8, "京ABE8779") == (
        "suspicious",
        (4, 7),
        "京ABE8779",
        ("E", "7#2"),
    )
    assert parse_problem_entry("S6", 8, "京ABE8779")[1] == (6,)
    with pytest.raises(ValueError, match="出现 2 次"):
        parse_problem_entry("7", 8, "京ABE8779")


def test_zero_is_character_when_plate_text_is_present_and_clear_uses_c() -> None:
    assert parse_problem_entry("0#2", 7, "京A12030") == (
        "suspicious",
        (7,),
        "京A12030",
        ("0#2",),
    )
    assert parse_problem_entry("C", 7, "京A12030")[:2] == ("clear", ())
    assert parse_problem_entry("0", 7, "")[:2] == ("clear", ())


def test_slot_correction_does_not_require_full_plate_text() -> None:
    assert parse_problem_entry("S3 S7", 7, "") == (
        "suspicious",
        (3, 7),
        "",
        ("S3", "S7"),
    )
    assert parse_problem_entry("3 7", 7, "")[:2] == ("suspicious", (3, 7))
    assert parse_problem_entry("S3", 7, "京QJ258") == (
        "suspicious",
        (3,),
        "京QJ258",
        ("S3",),
    )


def test_annotation_selection_is_reproducible() -> None:
    items = [AnnotationItem(str(index), __file__, "", "green_new_energy", 8) for index in range(10)]
    first = select_annotation_items(items, count=4, selection="random", seed=123)
    second = select_annotation_items(items, count=4, selection="random", seed=123)
    assert [item.image_id for item in first] == [item.image_id for item in second]


def test_discovery_and_atomic_csv_save_from_batch(tmp_path) -> None:
    sample_dir = tmp_path / "sample-a"
    sample_dir.mkdir()
    (sample_dir / "03_rectified.jpg").write_bytes(b"jpeg")
    (tmp_path / "batch_report.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "sample-a",
                        "status": "success",
                        "plate_type": "blue_standard",
                        "input_path": "/data/sample-a.jpg",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    items = discover_annotation_items(tmp_path)
    assert len(items) == 1
    assert items[0].slot_count == 7
    output = tmp_path / "labels.csv"
    state = AnnotationState(items, output)
    state.save(0, "B E", "京ABCDEF", "两处可疑")
    assert output.is_file()
    assert not output.with_suffix(".csv.tmp").exists()
    text = output.read_text(encoding="utf-8-sig")
    assert "suspicious" in text
    assert "3;6" in text
    assert "B;E" in text


def test_ocr_prefill_is_kept_separate_from_human_correction(tmp_path) -> None:
    cache_path = tmp_path / "labels.csv.ocr.json"
    cache_path.write_text(
        json.dumps({"results": {"sample-a": {"plate_text": "京QJ258O", "readable": True}}}),
        encoding="utf-8",
    )
    ocr = load_ocr_results(cache_path)
    item = AnnotationItem("sample-a", tmp_path / "03_rectified.jpg", "", "blue_standard", 7)
    output = tmp_path / "labels.csv"
    state = AnnotationState([item], output, ocr)
    assert state.completed == 0
    state.save(0, "S7", "京QJ2580", "OCR 把0识别为O")
    row = state.rows["sample-a"]
    assert row["ocr_plate_text"] == "京QJ258O"
    assert row["plate_text"] == "京QJ2580"
    assert row["plate_text_corrected"] == "true"
    assert row["suspicious_slots"] == "7"


def test_c_and_u_are_click_equivalent_shortcuts_without_auto_save(tmp_path) -> None:
    item = AnnotationItem("sample-a", tmp_path / "03_rectified.jpg", "", "blue_standard", 7)
    state = AnnotationState([item], tmp_path / "labels.csv")
    handler = AnnotationHandler.__new__(AnnotationHandler)
    handler.server = SimpleNamespace(state=state)

    page = handler._annotation_page(0)

    assert "if(key==='c'){setValue('C')" in page
    assert "else if(key==='u'){setValue('U')" in page
    assert "C=整牌正常，U=无法判断，Enter=保存" in page
    assert "setValue('C');document.getElementById('form').requestSubmit()" not in page
    assert "setValue('U');document.getElementById('form').requestSubmit()" not in page
    assert "plate.focus()" not in page


def test_cloud_relocalization_replaces_outputs_and_backs_up_old_crop(tmp_path, monkeypatch) -> None:
    sample_dir = tmp_path / "sample-a"
    sample_dir.mkdir()
    source = tmp_path / "vehicle.jpg"
    source.write_bytes(b"source")
    for filename in ("01_points.jpg", "02_quad.jpg", "03_rectified.jpg", "metadata.json"):
        (sample_dir / filename).write_bytes(b"old")

    image = np.zeros((200, 400, 3), dtype=np.uint8)
    detection = PlateDetection(
        corners=[[20, 20], [380, 20], [380, 180], [20, 180]],
        plate_type=PlateType.GREEN_NEW_ENERGY,
        confidence=0.91,
        method="test-cloud",
    )
    monkeypatch.setattr(annotate_web, "read_image", lambda _: image)
    monkeypatch.setattr(annotate_web, "detect_plate_llm", lambda _: detection)

    state = AnnotationState(
        [AnnotationItem("sample-a", sample_dir / "03_rectified.jpg", str(source), "blue_standard", 7)],
        tmp_path / "labels.csv",
    )
    event = state.relocalize(0)

    assert event["method"] == "test-cloud"
    assert state.items[0].plate_type == "green_new_energy"
    assert state.items[0].slot_count == 8
    assert (sample_dir / "03_rectified.jpg").read_bytes() != b"old"
    backups = list((sample_dir / "relocalization_backups").glob("*/03_rectified.jpg"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"old"
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["relocalized_for_annotation"] is True
    history = json.loads((sample_dir / "relocalization_history.json").read_text(encoding="utf-8"))
    assert history[-1]["method"] == "test-cloud"


def test_failed_cloud_relocalization_keeps_current_crop(tmp_path, monkeypatch) -> None:
    sample_dir = tmp_path / "sample-a"
    sample_dir.mkdir()
    source = tmp_path / "vehicle.jpg"
    source.write_bytes(b"source")
    rectified = sample_dir / "03_rectified.jpg"
    rectified.write_bytes(b"old-crop")
    monkeypatch.setattr(annotate_web, "read_image", lambda _: np.zeros((20, 40, 3), dtype=np.uint8))

    def fail_detection(_: np.ndarray) -> PlateDetection:
        raise RuntimeError("cloud unavailable")

    monkeypatch.setattr(annotate_web, "detect_plate_llm", fail_detection)
    state = AnnotationState(
        [AnnotationItem("sample-a", rectified, str(source), "blue_standard", 7)],
        tmp_path / "labels.csv",
    )

    with pytest.raises(RuntimeError, match="cloud unavailable"):
        state.relocalize(0)

    assert rectified.read_bytes() == b"old-crop"
    assert not (sample_dir / "relocalization_backups").exists()
