from __future__ import annotations

import threading
import time
from inspect import signature
from pathlib import Path

import cv2
import numpy as np

from pipeline.batch_test import build_parser
from pipeline.core import SIX_IMAGE_NAMES, _apply_plate_ocr_quality, _cost_report, run_batch, run_random_batch
from shape.geometry import PlateDetection, PlateType, write_image
from sticker.agent import StickerAgentHarness
from sticker.evidence import analyze_plate_evidence


def _fake_detector(_: np.ndarray) -> PlateDetection:
    return PlateDetection(
        corners=[[20, 20], [500, 20], [500, 160], [20, 160]],
        plate_type=PlateType.GREEN_NEW_ENERGY,
        confidence=0.9,
        method="test_detector",
    )


def _write_inputs(root: Path, count: int = 4) -> None:
    for index in range(count):
        image = np.full((180, 520, 3), (80, 190, 80), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (500, 160), (120, 220, 140), -1)
        cv2.putText(image, str(index), (240, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (10, 20, 10), 5)
        write_image(root / f"sample_{index}.jpg", image)


def test_default_agent_call_budget_is_three_everywhere() -> None:
    args = build_parser().parse_args(["input.jpg"])

    assert args.agent_max_calls_per_image == 3
    assert args.workers == 4
    assert args.no_progress is False
    assert signature(run_batch).parameters["agent_max_calls_per_image"].default == 3
    assert signature(run_batch).parameters["workers"].default == 4
    assert signature(run_random_batch).parameters["agent_max_calls_per_image"].default == 3
    assert signature(run_random_batch).parameters["workers"].default == 4
    assert signature(StickerAgentHarness).parameters["max_calls_per_image"].default == 3


def test_full_plate_ocr_is_positive_assessability_evidence() -> None:
    blurred = np.full((290, 992, 3), (125, 220, 160), dtype=np.uint8)
    artifacts = analyze_plate_evidence(blurred, plate_type=PlateType.GREEN_NEW_ENERGY)
    assert not artifacts.bundle.quality.assessable

    gate = _apply_plate_ocr_quality(
        artifacts,
        {
            "plate_text": "京A123456",
            "readable": True,
            "valid_length": True,
            "model": "fake-vlm",
            "error": "",
        },
    )

    assert gate["deterministic_assessable"] is False
    assert gate["ocr_full_plate_readable"] is True
    assert gate["final_assessable"] is True
    assert artifacts.bundle.quality.assessable


def test_failed_or_wrong_length_ocr_cannot_override_quality_gate() -> None:
    blurred = np.full((290, 992, 3), (125, 220, 160), dtype=np.uint8)
    artifacts = analyze_plate_evidence(blurred, plate_type=PlateType.GREEN_NEW_ENERGY)
    gate = _apply_plate_ocr_quality(
        artifacts,
        {"plate_text": "京A12345", "readable": True, "error": ""},
    )
    assert gate["ocr_full_plate_readable"] is False
    assert gate["final_assessable"] is False


def test_four_workers_run_concurrently_report_in_selection_order_and_emit_progress(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_inputs(input_dir, count=4)
    lock = threading.Lock()
    active = 0
    max_active = 0
    progress_events: list[tuple[int, int, str]] = []

    def slow_detector(image: np.ndarray) -> PlateDetection:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.05)
            return _fake_detector(image)
        finally:
            with lock:
                active -= 1

    _, report = run_batch(
        input_dir,
        tmp_path / "outputs",
        count=4,
        selection="first",
        sticker_method="local",
        detector=slow_detector,
        workers=4,
        progress_callback=lambda completed, total, sample: progress_events.append(
            (completed, total, str(sample["sample_id"]))
        ),
    )

    assert max_active >= 2
    assert report["parallelism"] == {
        "executor": "thread_pool",
        "configured_workers": 4,
        "effective_workers": 4,
    }
    assert [sample["selection_index"] for sample in report["samples"]] == [1, 2, 3, 4]
    assert [event[0] for event in progress_events] == [1, 2, 3, 4]
    assert all(event[1] == 4 for event in progress_events)


def test_random_batch_writes_exact_six_images_and_answer_only_results(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_inputs(input_dir)
    run_dir, report = run_random_batch(
        input_dir,
        tmp_path / "outputs",
        count=3,
        seed=123,
        prefix="test",
        detector=_fake_detector,
        sticker_method="local",
    )

    assert report["successful_count"] == 3
    assert report["random_seed"] == 123
    for sample in report["samples"]:
        sample_files = {item.name for item in (run_dir / sample["sample_id"]).iterdir() if item.is_file()}
        assert sample_files == set(SIX_IMAGE_NAMES)
    result_files = list((run_dir / "results").iterdir())
    assert len(result_files) == 3
    assert all(item.is_file() and item.suffix == ".jpg" for item in result_files)


def test_same_seed_selects_same_inputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_inputs(input_dir, count=6)
    _, first = run_random_batch(
        input_dir, tmp_path / "one", count=3, seed=99, detector=_fake_detector, sticker_method="local"
    )
    _, second = run_random_batch(
        input_dir, tmp_path / "two", count=3, seed=99, detector=_fake_detector, sticker_method="local"
    )
    assert [item["input_path"] for item in first["samples"]] == [item["input_path"] for item in second["samples"]]


def test_first_selection_and_timing_cost_metering(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_inputs(input_dir, count=5)
    _, report = run_batch(
        input_dir,
        tmp_path / "outputs",
        count=3,
        selection="first",
        sticker_method="local",
        detector=_fake_detector,
    )

    assert report["schema_version"] == 3
    assert report["selection_method"] == "first"
    assert report["random_seed"] is None
    assert [Path(item["input_path"]).name for item in report["samples"]] == [
        "sample_0.jpg",
        "sample_1.jpg",
        "sample_2.jpg",
    ]
    assert report["timing"]["processing_wall_seconds"] > 0
    assert report["timing"]["batch_stage_seconds"]["image_discovery_and_selection"] >= 0
    assert report["timing"]["batch_stage_seconds"]["sample_processing_loop"] > 0
    assert report["timing"]["successful_image_seconds"]["mean"] > 0
    assert report["timing"]["stage_total_seconds"]["plate_localization"] >= 0
    assert report["cost"]["external_api_calls"] == 0
    assert report["cost"]["input_tokens"] == 0
    assert report["cost"]["output_tokens"] == 0
    assert report["cost"]["external_model_cost_cny"] == 0
    assert all(item["timing_seconds"]["total"] > 0 for item in report["samples"])
    assert all("tampered_characters" in item for item in report["samples"])
    assert all("uncertain_characters" in item for item in report["samples"])


def test_llm_cost_uses_only_billable_non_cached_tokens() -> None:
    samples = [
        {
            "llm_usage": {
                "billable_api_calls": 1,
                "cache_hits": 1,
                "input_tokens": 2_000_000,
                "output_tokens": 500_000,
                "usage_complete": True,
            }
        }
    ]
    samples[0]["llm_usage"].update(
        {
            "model": "manual-model",
            "endpoint_host": "example.test",
            "stages": [
                {
                    "stage": "investigator",
                    "cached": False,
                    "input_tokens": 2_000_000,
                    "output_tokens": 500_000,
                }
            ],
        }
    )
    cost = _cost_report(samples, 2.0, 8.0)
    assert cost["status"] == "calculated_from_tokens"
    assert cost["external_api_calls"] == 1
    assert cost["cache_hits"] == 1
    assert cost["external_model_cost_cny"] == 8.0


def test_llm_cost_is_unknown_without_token_rates() -> None:
    samples = [
        {
            "llm_usage": {
                "billable_api_calls": 1,
                "cache_hits": 0,
                "input_tokens": 100,
                "output_tokens": 20,
                "usage_complete": True,
            }
        }
    ]
    samples[0]["llm_usage"].update(
        {
            "model": "unknown-model",
            "endpoint_host": "example.test",
            "stages": [
                {
                    "stage": "investigator",
                    "cached": False,
                    "input_tokens": 100,
                    "output_tokens": 20,
                }
            ],
        }
    )
    cost = _cost_report(samples, None, None)
    assert cost["status"] == "incomplete_usage_or_unknown_pricing"
    assert cost["external_model_cost_cny"] is None


def test_qwen3_vl_plus_uses_default_official_tiered_price() -> None:
    samples = [
        {
            "llm_usage": {
                "model": "qwen3-vl-plus",
                "endpoint_host": "dashscope.aliyuncs.com",
                "billable_api_calls": 1,
                "cache_hits": 0,
                "input_tokens": 20_000,
                "output_tokens": 1_000,
                "usage_complete": True,
                "stages": [
                    {
                        "stage": "investigator",
                        "cached": False,
                        "input_tokens": 20_000,
                        "output_tokens": 1_000,
                    }
                ],
            }
        }
    ]
    cost = _cost_report(samples, None, None)
    assert cost["status"] == "calculated_from_tokens"
    assert cost["external_model_cost_cny"] == 0.03
    assert cost["call_costs"][0]["input_rate_per_million_cny"] == 1.0
    assert cost["call_costs"][0]["output_rate_per_million_cny"] == 10.0
