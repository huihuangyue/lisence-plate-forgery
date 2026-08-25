"""贴片证据与最终裁决的图像、掩膜和 JSON 输出。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from shape.geometry import write_image
except ModuleNotFoundError:  # pragma: no cover
    from src.shape.geometry import write_image

from .types import CandidateRegion, EvidenceArtifacts
from .evidence import (
    candidate_has_multichannel_tamper_support,
    candidate_has_rectangle_material_support,
    candidate_tamper_support_routes,
    resolve_adjacent_material_ownership,
)


def _candidate_map(artifacts: EvidenceArtifacts) -> dict[str, CandidateRegion]:
    return {candidate.candidate_id: candidate for candidate in artifacts.bundle.candidates}


def character_decision_entries(
    artifacts: EvidenceArtifacts,
    decision: dict[str, Any],
    candidate_ids: list[str],
    *,
    confidence_level: str,
) -> list[dict[str, Any]]:
    """把候选编号解析为可直接消费的字符级 JSON 结果。"""

    candidates = _candidate_map(artifacts)
    recognized = decision.get("recognized_characters", {})
    recognition_status = decision.get("character_recognition_status", {})
    local_without_ocr = "recognized_characters" not in decision
    entries = []
    for candidate_id in candidate_ids:
        candidate = candidates.get(str(candidate_id))
        if candidate is None:
            continue
        character = recognized.get(candidate.candidate_id) if isinstance(recognized, dict) else None
        status = recognition_status.get(candidate.candidate_id) if isinstance(recognition_status, dict) else None
        if not status:
            status = "not_available_for_local_method" if local_without_ocr else "unreadable_or_not_returned"
        entries.append(
            {
                "candidate_id": candidate.candidate_id,
                "slot": candidate.slot,
                "character": character,
                "recognition_status": status,
                "bbox": list(candidate.bbox),
                "confidence_level": confidence_level,
            }
        )
    return entries


def make_local_decision(artifacts: EvidenceArtifacts) -> dict[str, Any]:
    """未校准的本地筛查结论；用于基线而非执法式最终判断。"""

    if not artifacts.bundle.quality.assessable:
        return {
            "decision": "unassessable",
            "selected_candidates": [],
            "uncertain_candidates": [],
            "unassessable_reason": "；".join(artifacts.bundle.quality.reasons),
            "decision_note": "确定性候选基线，阈值未经贴片真值校准",
        }
    candidate_order = lambda value: int(value.removeprefix("C"))
    suspicious_candidates, suppressed_spillover = resolve_adjacent_material_ownership(
        [
            candidate
            for candidate in artifacts.bundle.candidates
            if candidate_has_multichannel_tamper_support(candidate)
        ]
    )
    suspicious = sorted(
        [candidate.candidate_id for candidate in suspicious_candidates], key=candidate_order
    )
    uncertain = sorted(
        [
            candidate.candidate_id
            for candidate in artifacts.bundle.candidates
            if candidate.candidate_id not in suspicious
            and candidate.candidate_id not in suppressed_spillover
            and candidate_has_rectangle_material_support(candidate, strict=False)
        ],
        key=candidate_order,
    )
    if suspicious:
        final_decision = "suspicious"
        unassessable_reason = None
    else:
        final_decision = "clear"
        unassessable_reason = None
    return {
        "decision": final_decision,
        "selected_candidates": suspicious,
        "uncertain_candidates": uncertain,
        "suppressed_adjacent_spillover": suppressed_spillover,
        "unassessable_reason": unassessable_reason,
        "decision_note": "多通路确定性候选基线；开发集校准阈值仍需独立测试集验证",
    }


def render_final(
    image: np.ndarray,
    artifacts: EvidenceArtifacts,
    selected: list[str],
    uncertain: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """渲染最终高风险/待复核方框，并返回仅含高风险框的二值掩膜。"""

    result = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    candidates = _candidate_map(artifacts)
    for candidate_id in uncertain:
        candidate = candidates.get(candidate_id)
        if candidate is None:
            continue
        x1, y1, x2, y2 = candidate.bbox
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 165, 255), 3, cv2.LINE_AA)
        cv2.putText(result, f"REVIEW {candidate_id}/S{candidate.slot}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 80, 255), 2, cv2.LINE_AA)
    for candidate_id in selected:
        candidate = candidates.get(candidate_id)
        if candidate is None:
            continue
        x1, y1, x2, y2 = candidate.bbox
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 4, cv2.LINE_AA)
        cv2.putText(result, f"SUSPECT {candidate_id}/S{candidate.slot}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return result, mask


def write_analysis_outputs(
    image: np.ndarray,
    artifacts: EvidenceArtifacts,
    decision: dict[str, Any],
    output_root: str | Path,
    stem: str,
    trajectory: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_root) / stem
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    selected = [str(value) for value in decision.get("selected_candidates", [])]
    uncertain = [str(value) for value in decision.get("uncertain_candidates", [])]
    final_marked, candidate_mask = render_final(image, artifacts, selected, uncertain)
    paths = {
        "input": output_dir / "01_input_rectified.jpg",
        "candidates": output_dir / "02_candidate_overlay.jpg",
        "final": output_dir / "03_final_marked.jpg",
        "mask": output_dir / "candidate_mask.png",
        "report": output_dir / "report.json",
        "trajectory": output_dir / "trajectory.json",
        "edge_map": evidence_dir / "edge_map.png",
        "line_overlay": evidence_dir / "line_overlay.jpg",
        "color_residual": evidence_dir / "color_residual.jpg",
        "bright_dark": evidence_dir / "bright_dark.jpg",
        "contrast": evidence_dir / "clahe_and_inverted.jpg",
        "slot_map": evidence_dir / "slot_map.jpg",
        "vertical_profiles": evidence_dir / "vertical_profiles.jpg",
        "plate_reference": evidence_dir / "plate_row_reference_residual.jpg",
        "diagnostic_panel": evidence_dir / "diagnostic_panel.jpg",
        "ink_mask": evidence_dir / "ink_mask.png",
        "screw_mask": evidence_dir / "screw_mask.png",
        "sheet": evidence_dir / "evidence_sheet.jpg",
    }
    write_image(paths["input"], image)
    write_image(paths["candidates"], artifacts.candidate_overlay)
    write_image(paths["final"], final_marked)
    write_image(paths["mask"], candidate_mask)
    write_image(paths["edge_map"], artifacts.edge_map)
    write_image(paths["line_overlay"], artifacts.line_overlay)
    write_image(paths["color_residual"], artifacts.color_residual)
    write_image(paths["bright_dark"], artifacts.bright_dark_view)
    write_image(paths["contrast"], artifacts.contrast_view)
    write_image(paths["slot_map"], artifacts.slot_map_view)
    write_image(paths["vertical_profiles"], artifacts.vertical_profile_view)
    write_image(paths["plate_reference"], artifacts.plate_reference_view)
    write_image(paths["diagnostic_panel"], artifacts.diagnostic_panel)
    write_image(paths["ink_mask"], artifacts.ink_mask)
    write_image(paths["screw_mask"], artifacts.screw_mask)
    write_image(paths["sheet"], artifacts.evidence_sheet)
    tampered_characters = character_decision_entries(
        artifacts, decision, selected, confidence_level="high"
    )
    uncertain_characters = character_decision_entries(
        artifacts, decision, uncertain, confidence_level="medium"
    )
    report = {
        "schema_version": 2,
        "method": "sticker_agent_v8_multichannel_ownership" if trajectory else "sticker_local_v8_multichannel_ownership",
        "decision": decision.get("decision", "unassessable"),
        "selected_candidates": selected,
        "uncertain_candidates": uncertain,
        "tampered_characters": tampered_characters,
        "uncertain_characters": uncertain_characters,
        "suspected_characters": [
            {
                "candidate_id": candidate.candidate_id,
                "slot": candidate.slot,
                "recognized_text": decision.get("recognized_characters", {}).get(candidate.candidate_id),
                "bbox": list(candidate.bbox),
                "region_ids": [candidate.candidate_id],
                "confidence_level": "high" if candidate.candidate_id in selected else "medium",
            }
            for candidate in artifacts.bundle.candidates
            if candidate.candidate_id in set(selected + uncertain)
        ],
        "regions": [
            {
                "id": candidate.candidate_id,
                "slot": candidate.slot,
                "bbox": list(candidate.bbox),
                "geometry_score": candidate.geometry_score,
                "appearance_score": candidate.appearance_score,
                "paired_edge_score": candidate.paired_edge_score,
                "combined_score": candidate.combined_score,
                "features": candidate.features,
                "deterministic_support_routes": candidate_tamper_support_routes(candidate),
                "agent_evidence": decision.get("candidate_evidence", {}).get(candidate.candidate_id, {}),
            }
            for candidate in artifacts.bundle.candidates
            if candidate.candidate_id in set(selected + uncertain)
        ],
        "evidence": artifacts.bundle.to_dict(),
        "agent_summary": decision.get("reasoning_summary"),
        "unassessable_reason": decision.get("unassessable_reason"),
        "decision_note": decision.get("decision_note", "大模型裁决仍需用人工真值校准，不能解释为司法结论"),
        "suppressed_adjacent_spillover": decision.get("suppressed_adjacent_spillover", {}),
    }
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if trajectory is not None:
        paths["trajectory"].write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        paths.pop("trajectory")
    return paths
