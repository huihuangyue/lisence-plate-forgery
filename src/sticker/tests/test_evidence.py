from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from shape.geometry import PlateType
from sticker.evidence import (
    analyze_plate_evidence,
    candidate_has_multichannel_tamper_support,
    candidate_has_rectangle_material_support,
    candidate_tamper_support_routes,
    character_slots,
    infer_plate_type,
    resolve_adjacent_material_ownership,
)
from sticker.output import character_decision_entries, make_local_decision


def _synthetic_green_plate(with_patch: bool = False) -> np.ndarray:
    image = np.full((290, 992, 3), (125, 220, 160), dtype=np.uint8)
    cv2.rectangle(image, (16, 5), (975, 284), (220, 235, 220), 5)
    for x, character in zip((90, 195, 385, 495, 605, 715, 825, 925), "AZ6Q268B", strict=True):
        cv2.putText(image, character, (x - 36, 220), cv2.FONT_HERSHEY_SIMPLEX, 2.35, (20, 28, 22), 10, cv2.LINE_AA)
    if with_patch:
        cv2.rectangle(image, (330, 35), (440, 255), (113, 204, 149), -1)
        cv2.rectangle(image, (330, 35), (440, 255), (238, 246, 241), 3)
        cv2.line(image, (334, 252), (438, 252), (28, 44, 32), 3)
    return image


def _synthetic_full_height_strip() -> np.ndarray:
    image = _synthetic_green_plate(with_patch=False)
    # 上下边与号牌边框相接，只留下左右两条竖缝和区域材料差异。
    cv2.rectangle(image, (330, 0), (440, image.shape[0] - 1), (95, 185, 125), -1)
    cv2.putText(image, "6", (350, 220), cv2.FONT_HERSHEY_SIMPLEX, 2.35, (20, 28, 22), 10, cv2.LINE_AA)
    for y in range(20, 260, 45):
        cv2.line(image, (330, y), (330, min(y + 27, 270)), (238, 246, 241), 2)
        cv2.line(image, (440, y), (440, min(y + 27, 270)), (238, 246, 241), 2)
    return image


def _synthetic_vertical_gradient_plate(with_flat_patch: bool = False) -> np.ndarray:
    image = np.empty((290, 992, 3), dtype=np.uint8)
    top_color = np.asarray((218, 245, 242), dtype=np.float32)
    bottom_color = np.asarray((100, 188, 132), dtype=np.float32)
    for y in range(image.shape[0]):
        image[y] = np.clip(
            top_color + (bottom_color - top_color) * y / (image.shape[0] - 1),
            0,
            255,
        )
    cv2.rectangle(image, (16, 5), (975, 284), (225, 238, 226), 5)
    for x, character in zip((90, 195, 385, 495, 605, 715, 825, 925), "AZ6Q2689", strict=True):
        cv2.putText(image, character, (x - 36, 220), cv2.FONT_HERSHEY_SIMPLEX, 2.35, (20, 28, 22), 10, cv2.LINE_AA)
    if with_flat_patch:
        # 末位贴片保留字符，但把天然的上白下绿渐变替换成近似均色。
        cv2.rectangle(image, (866, 45), (974, 270), (175, 225, 192), -1)
        cv2.putText(image, "9", (889, 220), cv2.FONT_HERSHEY_SIMPLEX, 2.35, (20, 28, 22), 10, cv2.LINE_AA)
    return image


def test_plate_type_and_slot_count_from_fixed_canvas() -> None:
    green = np.zeros((290, 992, 3), dtype=np.uint8)
    blue = np.zeros((290, 908, 3), dtype=np.uint8)
    assert infer_plate_type(green) is PlateType.GREEN_NEW_ENERGY
    assert infer_plate_type(blue) is PlateType.BLUE_STANDARD
    assert len(character_slots(PlateType.GREEN_NEW_ENERGY, green.shape)) == 8
    assert len(character_slots(PlateType.BLUE_STANDARD, blue.shape)) == 7


def test_evidence_pipeline_returns_finite_slot_candidates() -> None:
    artifacts = analyze_plate_evidence(_synthetic_green_plate(with_patch=True))
    assert artifacts.bundle.plate_type is PlateType.GREEN_NEW_ENERGY
    assert len(artifacts.bundle.candidates) == 8
    assert {item.candidate_id for item in artifacts.bundle.candidates} == {f"C{i}" for i in range(1, 9)}
    assert artifacts.evidence_sheet.shape == (1740, 992, 3)
    assert artifacts.slot_map_view.shape == (290, 992, 3)
    assert artifacts.contrast_view.shape == (290, 992, 3)
    assert artifacts.vertical_profile_view.shape == (290, 992, 3)
    assert artifacts.diagnostic_panel.shape == (870, 1984, 3)
    for candidate in artifacts.bundle.candidates:
        assert 0.0 <= candidate.geometry_score <= 1.0
        assert 0.0 <= candidate.appearance_score <= 1.0
        assert 0.0 <= candidate.paired_edge_score <= 1.0
        assert 0.0 <= candidate.combined_score <= 1.0


def test_rectangle_closure_gate_separates_patch_from_character_like_strokes() -> None:
    normal = analyze_plate_evidence(_synthetic_green_plate(with_patch=False))
    patched = analyze_plate_evidence(_synthetic_green_plate(with_patch=True))
    normal_by_slot = {candidate.slot: candidate for candidate in normal.bundle.candidates}
    patched_by_slot = {candidate.slot: candidate for candidate in patched.bundle.candidates}

    assert not any(candidate_has_rectangle_material_support(candidate, strict=False) for candidate in normal_by_slot.values())
    assert candidate_has_rectangle_material_support(patched_by_slot[3], strict=True)
    assert patched_by_slot[3].features["rectangle_side_count"] >= 3
    assert patched_by_slot[3].features["rectangle_closure_score"] > 0.60
    assert not any(
        candidate_has_rectangle_material_support(candidate, strict=False)
        for slot, candidate in patched_by_slot.items()
        if slot != 3
    )


def test_full_height_strip_gate_accepts_isolated_opposite_vertical_seams() -> None:
    artifacts = analyze_plate_evidence(_synthetic_full_height_strip())
    candidate = next(item for item in artifacts.bundle.candidates if item.candidate_id == "C3")
    assert candidate.features["rectangle_closure_score"] < 0.24
    assert candidate.features["vertical_strip_pair_score"] >= 0.25
    assert candidate.features["vertical_strip_isolation"] >= 1.60
    assert candidate_has_rectangle_material_support(candidate, strict=True)


def test_green_vertical_profile_gate_detects_flat_patch_against_natural_gradient() -> None:
    normal = analyze_plate_evidence(_synthetic_vertical_gradient_plate(with_flat_patch=False))
    patched = analyze_plate_evidence(_synthetic_vertical_gradient_plate(with_flat_patch=True))
    normal_c8 = next(item for item in normal.bundle.candidates if item.slot == 8)
    patched_c8 = next(item for item in patched.bundle.candidates if item.slot == 8)
    features = patched_c8.features

    assert features["green_vertical_gradient_applicable"] == 1.0
    assert features["vertical_expected_gradient_magnitude"] >= 15.0
    assert features["vertical_actual_gradient_magnitude"] <= 0.55 * features["vertical_expected_gradient_magnitude"]
    assert features["vertical_gradient_anomaly_score"] >= 0.60
    assert features["material_side_count"] >= 2.0
    assert features["plate_reference_delta_e_median"] >= 4.0
    assert features["edge_slot_one_sided_control"] == 1.0
    assert features["matched_control_right_pixels"] == 0.0
    assert candidate_has_rectangle_material_support(patched_c8, strict=True)
    assert not candidate_has_rectangle_material_support(normal_c8, strict=False)


def test_partial_axis_seam_accepts_open_l_but_rejects_shared_single_seam() -> None:
    artifacts = analyze_plate_evidence(_synthetic_green_plate())
    base = artifacts.bundle.candidates[0]
    open_l = replace(
        base,
        appearance_score=0.55,
        features={
            **base.features,
            "rectangle_side_count": 1.0,
            "rectangle_closure_score": 0.0,
            "partial_axis_material_side_count": 1.0,
            "partial_axis_material_side_score": 0.72,
            "partial_axis_orthogonal_side_score": 0.68,
            "partial_axis_seam_score": 0.68,
            "plate_reference_delta_e_median": 5.0,
            "plate_reference_delta_e_p75": 7.0,
        },
    )
    shared_single_seam = replace(
        open_l,
        features={
            **open_l.features,
            "partial_axis_orthogonal_side_score": 0.0,
            "partial_axis_seam_score": 0.0,
        },
    )

    assert candidate_has_rectangle_material_support(open_l, strict=True)
    assert not candidate_has_rectangle_material_support(shared_single_seam, strict=False)


def test_single_axis_material_seam_requires_strong_region_difference() -> None:
    artifacts = analyze_plate_evidence(_synthetic_green_plate())
    base = artifacts.bundle.candidates[0]
    strong_region = replace(
        base,
        appearance_score=0.66,
        features={
            **base.features,
            "rectangle_side_count": 1.0,
            "rectangle_closure_score": 0.0,
            "partial_axis_material_side_count": 1.0,
            "partial_axis_seam_score": 0.0,
            "single_axis_material_seam_score": 0.82,
            "delta_e_00": 18.0,
            "plate_reference_delta_e_median": 5.0,
            "plate_reference_delta_e_p75": 7.0,
        },
    )
    adjacent_normal_region = replace(
        strong_region,
        features={**strong_region.features, "delta_e_00": 3.0},
    )

    assert candidate_has_rectangle_material_support(strong_region, strict=True)
    assert not candidate_has_rectangle_material_support(adjacent_normal_region, strict=False)


def test_character_json_maps_selected_candidate_to_slot_and_visible_character() -> None:
    artifacts = analyze_plate_evidence(_synthetic_green_plate(with_patch=True))
    c3 = next(candidate for candidate in artifacts.bundle.candidates if candidate.candidate_id == "C3")
    entries = character_decision_entries(
        artifacts,
        {
            "recognized_characters": {"C3": "6"},
            "character_recognition_status": {"C3": "cross_review_consensus"},
        },
        ["C3"],
        confidence_level="high",
    )
    assert entries == [
        {
            "candidate_id": "C3",
            "slot": 3,
            "character": "6",
            "recognition_status": "cross_review_consensus",
            "bbox": list(c3.bbox),
            "confidence_level": "high",
        }
    ]


def test_local_decision_respects_unassessable_quality() -> None:
    blurred = cv2.GaussianBlur(_synthetic_green_plate(), (0, 0), 35)
    artifacts = analyze_plate_evidence(blurred)
    decision = make_local_decision(artifacts)
    assert decision["decision"] == "unassessable"
    assert decision["selected_candidates"] == []


def test_local_decision_promotes_supported_open_rectangle() -> None:
    artifacts = analyze_plate_evidence(_synthetic_green_plate())
    candidates = []
    for index, candidate in enumerate(artifacts.bundle.candidates):
        if index == 0:
            candidates.append(
                replace(
                    candidate,
                    geometry_score=0.50,
                    appearance_score=0.40,
                    paired_edge_score=0.20,
                    combined_score=0.45,
                    features={
                            **candidate.features,
                            "rectangle_geometry_score": 0.50,
                            "rectangle_side_count": 2.0,
                        "rectangle_closure_score": 0.18,
                        "material_side_count": 2.0,
                        "boundary_material_score": 0.50,
                        "plate_reference_delta_e_median": 3.7,
                        "plate_reference_delta_e_p75": 5.8,
                    },
                )
            )
        else:
            candidates.append(
                replace(
                    candidate,
                    geometry_score=0.0,
                    appearance_score=0.0,
                    paired_edge_score=0.0,
                    combined_score=0.0,
                )
            )
    artifacts.bundle = replace(artifacts.bundle, candidates=candidates)
    decision = make_local_decision(artifacts)
    assert decision["decision"] == "suspicious"
    assert len(decision["selected_candidates"]) == 1
    assert decision["uncertain_candidates"] == []


def test_multichannel_material_path_accepts_owned_low_geometry_candidate() -> None:
    artifacts = analyze_plate_evidence(_synthetic_green_plate())
    base = artifacts.bundle.candidates[0]
    candidate = replace(
        base,
        geometry_score=0.14,
        appearance_score=0.63,
        features={
            **base.features,
            "rectangle_side_count": 0.0,
            "rectangle_closure_score": 0.0,
            "material_ownership_score": 0.71,
            "plate_reference_delta_e_median": 4.8,
            "matched_control_bilateral_min_delta_e": 4.3,
            "single_axis_material_seam_score": 0.48,
            "material_side_count": 2.0,
            "boundary_material_score": 0.75,
        },
    )

    assert not candidate_has_rectangle_material_support(candidate, strict=False)
    assert candidate_has_multichannel_tamper_support(candidate)
    assert "contextual_material_outlier" in candidate_tamper_support_routes(candidate)


@pytest.mark.parametrize(
    ("appearance", "feature_updates", "expected_route"),
    [
        (
            0.74,
            {
                "material_ownership_score": 0.96,
                "plate_reference_delta_e_median": 7.0,
                "matched_control_bilateral_min_delta_e": 4.0,
            },
            "strong_contextual_material_without_complete_edge",
        ),
        (
            0.48,
            {
                "material_ownership_score": 0.42,
                "plate_reference_delta_e_median": 2.9,
                "plate_reference_delta_e_p75": 3.8,
                "matched_control_bilateral_min_delta_e": 2.0,
                "material_side_count": 2.0,
                "single_axis_material_seam_score": 0.70,
            },
            "moderate_contextual_material_with_boundary",
        ),
        (
            0.59,
            {
                "material_side_count": 3.0,
                "boundary_material_score": 1.0,
                "single_axis_material_seam_score": 0.90,
                "matched_control_bilateral_min_delta_e": 2.1,
            },
            "isolated_axis_seam_with_multiside_material",
        ),
    ],
)
def test_multichannel_paths_do_not_require_closed_rectangle(
    appearance: float,
    feature_updates: dict[str, float],
    expected_route: str,
) -> None:
    artifacts = analyze_plate_evidence(_synthetic_green_plate())
    base = artifacts.bundle.candidates[0]
    candidate = replace(
        base,
        geometry_score=0.0,
        appearance_score=appearance,
        features={
            **base.features,
            "rectangle_side_count": 0.0,
            "rectangle_closure_score": 0.0,
            "partial_axis_seam_score": 0.0,
            **feature_updates,
        },
    )

    assert expected_route in candidate_tamper_support_routes(candidate)


def test_adjacent_ownership_keeps_bilateral_owner_and_suppresses_spillover() -> None:
    artifacts = analyze_plate_evidence(_synthetic_green_plate())
    base = artifacts.bundle.candidates[0]
    owner = replace(
        base,
        candidate_id="C5",
        slot=5,
        features={
            **base.features,
            "matched_control_bilateral_min_delta_e": 4.2,
            "plate_reference_delta_e_median": 6.5,
        },
    )
    spillover = replace(
        base,
        candidate_id="C4",
        slot=4,
        features={
            **base.features,
            "matched_control_bilateral_min_delta_e": 1.2,
            "plate_reference_delta_e_median": 2.0,
        },
    )

    kept, suppressed = resolve_adjacent_material_ownership([spillover, owner])
    assert [candidate.candidate_id for candidate in kept] == ["C5"]
    assert suppressed == {"C4": "C5"}
