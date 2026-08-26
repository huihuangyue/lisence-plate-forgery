from __future__ import annotations

import cv2
import numpy as np

from sticker.agent import StickerAgentHarness
from sticker.evidence import analyze_plate_evidence
from sticker.stroke import stroke_hypotheses, stroke_region_evidence


def _plate() -> np.ndarray:
    image = np.full((290, 992, 3), (125, 220, 160), dtype=np.uint8)
    cv2.rectangle(image, (16, 5), (975, 284), (220, 235, 220), 5)
    for x, character in zip((90, 195, 385, 495, 605, 715, 825, 925), "AZ6E268B", strict=True):
        cv2.putText(
            image,
            character,
            (x - 36, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.35,
            (20, 28, 22),
            10,
            cv2.LINE_AA,
        )
    return image


def test_stroke_watchlist_routes_observed_e_back_to_f() -> None:
    hypotheses = stroke_hypotheses("E")
    match = next(item for item in hypotheses if item["possible_original"] == "F")
    assert match["tamper_type"] == "added_stroke"
    assert match["regions"] == ["bottom"]
    assert match["edit_count"] == 1


def test_stroke_region_evidence_is_structured_and_bounded() -> None:
    artifacts = analyze_plate_evidence(_plate())
    candidate = artifacts.bundle.candidates[3]
    evidence = stroke_region_evidence(artifacts, candidate, stroke_hypotheses("E"))
    assert evidence
    assert all(len(item["bbox"]) == 4 for item in evidence)
    assert all(0.0 <= item["physical_support_score"] <= 1.0 for item in evidence)


def test_high_recall_profile_accepts_cross_stage_stroke_with_physical_support() -> None:
    artifacts = analyze_plate_evidence(_plate())
    # 测试门控而非具体图像增强算法：把既有确定性辅助图设为强局部物理响应。
    artifacts.color_residual[:] = 255
    artifacts.edge_map[:] = 255
    rows = [
        {
            "candidates": [
                {
                    "candidate_id": "C4",
                    "verdict": "tamper_support",
                    "suspected_tamper_types": ["added_stroke"],
                    "possible_originals": ["F"],
                    "stroke_regions": ["bottom"],
                }
            ]
        },
        {
            "candidates": [
                {
                    "candidate_id": "C4",
                    "verdict": "tamper_support",
                    "suspected_tamper_types": ["added_stroke"],
                    "possible_originals": ["F"],
                    "stroke_regions": ["bottom"],
                }
            ]
        },
    ]
    actual = StickerAgentHarness._sanitize_final(
        {
            "decision": "suspicious",
            "selected_candidates": ["C4"],
            "uncertain_candidates": [],
            "recognized_characters": {"C4": "E"},
            "candidate_evidence": {
                "C4": {
                    "geometry": [],
                    "appearance": ["bottom material anomaly"],
                    "counter_evidence": [],
                    "tamper_types": ["added_stroke"],
                    "possible_originals": ["F"],
                    "stroke_regions": ["bottom"],
                }
            },
        },
        {f"C{i}" for i in range(1, 9)},
        artifacts,
        assessments=rows,
        decision_profile="high_recall",
    )
    assert "C4" in actual["selected_candidates"]
    assert actual["candidate_forgery_types"]["C4"] == ["added_stroke"]
    assert actual["candidate_possible_originals"]["C4"] == ["F"]


def test_character_similarity_alone_cannot_select_stroke_candidate() -> None:
    artifacts = analyze_plate_evidence(_plate())
    actual = StickerAgentHarness._sanitize_final(
        {
            "decision": "suspicious",
            "selected_candidates": ["C4"],
            "uncertain_candidates": [],
            "recognized_characters": {"C4": "E"},
            "candidate_evidence": {
                "C4": {
                    "geometry": [],
                    "appearance": [],
                    "counter_evidence": [],
                    "tamper_types": ["added_stroke"],
                    "possible_originals": ["F"],
                    "stroke_regions": ["bottom"],
                }
            },
        },
        {f"C{i}" for i in range(1, 9)},
        artifacts,
        assessments=[],
        decision_profile="high_recall",
    )
    assert "C4" not in actual["selected_candidates"]
