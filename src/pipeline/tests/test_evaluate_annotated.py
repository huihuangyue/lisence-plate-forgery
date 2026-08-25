from __future__ import annotations

from pathlib import Path

from pipeline.evaluate_annotated import (
    resolve_annotated_sources,
    select_annotation_rows,
    stage_annotated_sources,
)


def test_select_annotation_rows_random_is_reproducible() -> None:
    rows = [{"image_id": str(index)} for index in range(10)]
    first = select_annotation_rows(rows, count=4, selection="random", seed=17)
    second = select_annotation_rows(rows, count=4, selection="random", seed=17)
    assert first == second
    assert len({row["image_id"] for row in first}) == 4


def test_select_annotation_rows_first_and_full() -> None:
    rows = [{"image_id": str(index)} for index in range(5)]
    assert select_annotation_rows(rows, count=2, selection="first", seed=1) == rows[:2]
    assert select_annotation_rows(rows, count=None, selection="random", seed=1) == rows


def test_resolve_and_stage_annotated_sources(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"image")
    annotations = tmp_path / "labels.csv"
    annotations.write_text("placeholder", encoding="utf-8")
    rows = [{"image_id": "sample-a", "input_path": str(source)}]
    resolved = resolve_annotated_sources(
        rows,
        annotations_path=annotations,
        images_root=None,
    )
    assert resolved == [("sample-a", source.resolve())]
    staging = tmp_path / "staging"
    staging.mkdir()
    stage_annotated_sources(resolved, staging)
    assert (staging / "sample-a.jpg").resolve() == source.resolve()
