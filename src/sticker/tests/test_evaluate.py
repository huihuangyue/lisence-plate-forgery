from __future__ import annotations

import json

from sticker.evaluate import Annotation, evaluate_reports, read_reports


def test_evaluation_computes_plate_and_slot_metrics() -> None:
    annotations = {
        "positive": Annotation("positive", "suspicious", frozenset({2})),
        "negative": Annotation("negative", "clear", frozenset()),
    }
    reports = {
        "positive": {
            "decision": "suspicious",
            "suspected_characters": [{"slot": 2, "confidence_level": "high"}],
        },
        "negative": {"decision": "clear", "suspected_characters": []},
    }
    metrics = evaluate_reports(annotations, reports)
    assert metrics["plate_level"]["f1"] == 1.0
    assert metrics["character_slot_level"]["f1"] == 1.0


def test_uncertain_slot_is_not_counted_as_positive() -> None:
    annotations = {"sample": Annotation("sample", "suspicious", frozenset({3}))}
    reports = {
        "sample": {
            "decision": "clear",
            "suspected_characters": [{"slot": 3, "confidence_level": "medium"}],
        }
    }
    metrics = evaluate_reports(annotations, reports)
    assert metrics["plate_level"]["fn"] == 1
    assert metrics["character_slot_level"]["fn"] == 1


def test_evaluation_reports_accuracy_intervals_and_slot_tn() -> None:
    annotations = {
        "positive": Annotation("positive", "suspicious", frozenset({2}), 8),
        "negative": Annotation("negative", "clear", frozenset(), 7),
    }
    reports = {
        "positive": {
            "decision": "suspicious",
            "suspected_characters": [{"slot": 2, "confidence_level": "high"}],
        },
        "negative": {"decision": "clear", "suspected_characters": []},
    }
    metrics = evaluate_reports(annotations, reports)
    assert metrics["plate_level"]["accuracy"] == 1.0
    assert metrics["plate_level"]["accuracy_ci95_wilson"][0] < 1.0
    assert metrics["plate_level_answered_only"]["coverage"] == 1.0
    assert metrics["character_slot_level"]["total_slots"] == 15
    assert metrics["character_slot_level"]["tn"] == 14
    assert metrics["character_slot_level"]["accuracy"] == 1.0


def test_read_reports_accepts_pipeline_batch_report(tmp_path) -> None:
    batch = {
        "samples": [
            {
                "sample_id": "sample-a",
                "status": "success",
                "decision": "suspicious",
                "tampered_characters": [{"slot": 4}],
            },
            {"sample_id": "failed", "status": "failed"},
        ]
    }
    (tmp_path / "batch_report.json").write_text(json.dumps(batch), encoding="utf-8")
    reports = read_reports(tmp_path)
    assert reports["sample-a"]["decision"] == "suspicious"
    assert reports["sample-a"]["suspected_characters"] == [
        {"slot": 4, "confidence_level": "high"}
    ]
    assert "failed" not in reports
