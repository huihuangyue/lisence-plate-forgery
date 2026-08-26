"""增加/消除笔画变造的候选映射与局部物理证据。

字符转换表只负责提出待核验假设；任何字符相似关系都不能单独构成变造结论。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from .types import CandidateRegion, EvidenceArtifacts


STROKE_TAMPER_TYPES = {"added_stroke", "removed_stroke", "mixed_stroke_edit"}


@dataclass(frozen=True)
class StrokeHypothesis:
    observed_character: str
    possible_original: str
    tamper_type: str
    edit_count: int
    operation: str
    regions: tuple[str, ...]
    priority: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["regions"] = list(self.regions)
        return payload


# 数字使用七区域抽象建立完整的一段/二段差异。它用于检查区域路由，不冒充
# GA 36 号牌字模。真实字模兼容性仍由视觉模型和物理证据共同复核。
_DIGIT_SEGMENTS: dict[str, set[str]] = {
    "0": {"top", "upper_left", "upper_right", "lower_left", "lower_right", "bottom"},
    "1": {"upper_right", "lower_right"},
    "2": {"top", "upper_right", "center", "lower_left", "bottom"},
    "3": {"top", "upper_right", "center", "lower_right", "bottom"},
    "4": {"upper_left", "upper_right", "center", "lower_right"},
    "5": {"top", "upper_left", "center", "lower_right", "bottom"},
    "6": {"top", "upper_left", "center", "lower_left", "lower_right", "bottom"},
    "7": {"top", "upper_right", "lower_right"},
    "8": {"top", "upper_left", "upper_right", "center", "lower_left", "lower_right", "bottom"},
    "9": {"top", "upper_left", "upper_right", "center", "lower_right", "bottom"},
}


def _digit_hypotheses() -> list[StrokeHypothesis]:
    output: list[StrokeHypothesis] = []
    for original, original_segments in _DIGIT_SEGMENTS.items():
        for observed, observed_segments in _DIGIT_SEGMENTS.items():
            if original == observed:
                continue
            added = sorted(observed_segments - original_segments)
            removed = sorted(original_segments - observed_segments)
            edit_count = len(added) + len(removed)
            if edit_count not in {1, 2}:
                continue
            if added and removed:
                tamper_type = "mixed_stroke_edit"
                operation = "add_" + "_and_".join(added) + "__remove_" + "_and_".join(removed)
            elif added:
                tamper_type = "added_stroke"
                operation = "add_" + "_and_".join(added)
            else:
                tamper_type = "removed_stroke"
                operation = "remove_" + "_and_".join(removed)
            output.append(
                StrokeHypothesis(
                    observed,
                    original,
                    tamper_type,
                    edit_count,
                    operation,
                    tuple(added + removed),
                    "tier1" if edit_count == 1 else "tier2",
                )
            )
    return output


def _pair(
    left: str,
    right: str,
    *,
    left_to_right: tuple[str, str],
    right_to_left: tuple[str, str],
    region: str,
    priority: str = "tier1",
) -> list[StrokeHypothesis]:
    edit_count = 2 if priority == "tier2" else 1
    return [
        StrokeHypothesis(right, left, left_to_right[0], edit_count, left_to_right[1], (region,), priority),
        StrokeHypothesis(left, right, right_to_left[0], edit_count, right_to_left[1], (region,), priority),
    ]


_LETTER_HYPOTHESES: list[StrokeHypothesis] = []
for args in (
    ("F", "E", ("added_stroke", "add_bottom"), ("removed_stroke", "remove_bottom"), "bottom", "tier1"),
    ("C", "G", ("added_stroke", "add_right_middle"), ("removed_stroke", "remove_right_middle"), "right_middle", "tier1"),
    ("P", "R", ("added_stroke", "add_lower_right_tail"), ("removed_stroke", "remove_lower_right_tail"), "lower_right", "tier1"),
    ("0", "Q", ("added_stroke", "add_lower_right_tail"), ("removed_stroke", "remove_lower_right_tail"), "lower_right", "tier1"),
    ("C", "0", ("added_stroke", "close_right_opening"), ("removed_stroke", "open_right_side"), "right_middle", "tier1"),
    ("F", "P", ("added_stroke", "close_upper_right"), ("removed_stroke", "open_upper_right"), "upper_right", "tier1"),
    ("P", "B", ("added_stroke", "add_lower_loop"), ("removed_stroke", "remove_lower_loop"), "lower_half", "tier1"),
    ("J", "U", ("added_stroke", "add_left_side"), ("removed_stroke", "remove_left_side"), "left_side", "tier1"),
    ("7", "Z", ("added_stroke", "add_bottom"), ("removed_stroke", "remove_bottom"), "bottom", "tier1"),
    ("U", "0", ("added_stroke", "close_top"), ("removed_stroke", "open_top"), "top", "conditional"),
    ("1", "T", ("added_stroke", "add_top"), ("removed_stroke", "remove_top"), "top", "conditional"),
    ("V", "Y", ("added_stroke", "add_lower_tail"), ("removed_stroke", "remove_lower_tail"), "bottom", "conditional"),
    ("X", "Y", ("removed_stroke", "remove_lower_arm"), ("added_stroke", "add_lower_arm"), "lower_half", "conditional"),
    ("L", "U", ("added_stroke", "add_right_side"), ("removed_stroke", "remove_right_side"), "right_side", "conditional"),
    ("M", "N", ("removed_stroke", "remove_internal_diagonal"), ("added_stroke", "add_internal_diagonal"), "internal", "conditional"),
):
    left, right, ltr, rtl, region, priority = args
    _LETTER_HYPOTHESES.extend(
        _pair(left, right, left_to_right=ltr, right_to_left=rtl, region=region, priority=priority)
    )


_ALL_HYPOTHESES = _digit_hypotheses() + _LETTER_HYPOTHESES
_BY_OBSERVED: dict[str, list[StrokeHypothesis]] = {}
for _hypothesis in _ALL_HYPOTHESES:
    _BY_OBSERVED.setdefault(_hypothesis.observed_character, []).append(_hypothesis)


def stroke_hypotheses(observed_character: str | None) -> list[dict[str, Any]]:
    """按当前可见字符反查一段/二段修改假设。"""

    if not observed_character:
        return []
    hypotheses = _BY_OBSERVED.get(str(observed_character).strip().upper(), [])
    return [item.to_dict() for item in sorted(hypotheses, key=lambda item: (item.edit_count, item.possible_original))]


def compact_watchlist() -> dict[str, list[dict[str, Any]]]:
    """供模型提示使用的紧凑反查表。"""

    return {character: stroke_hypotheses(character) for character in sorted(_BY_OBSERVED)}


def _region_bbox(candidate: CandidateRegion, region: str) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = candidate.bbox
    width, height = max(1, x2 - x1), max(1, y2 - y1)
    regions = {
        "top": (0.12, 0.00, 0.88, 0.30),
        "center": (0.08, 0.35, 0.92, 0.65),
        "bottom": (0.12, 0.70, 0.88, 1.00),
        "upper_left": (0.00, 0.05, 0.45, 0.52),
        "upper_right": (0.55, 0.05, 1.00, 0.52),
        "lower_left": (0.00, 0.48, 0.45, 0.95),
        "lower_right": (0.55, 0.48, 1.00, 0.95),
        "right_middle": (0.50, 0.25, 1.00, 0.75),
        "left_side": (0.00, 0.05, 0.42, 0.95),
        "right_side": (0.58, 0.05, 1.00, 0.95),
        "upper_half": (0.05, 0.00, 0.95, 0.55),
        "lower_half": (0.05, 0.45, 0.95, 1.00),
        "internal": (0.20, 0.15, 0.80, 0.85),
    }
    xa, ya, xb, yb = regions.get(region, (0.05, 0.05, 0.95, 0.95))
    return (
        int(round(x1 + xa * width)),
        int(round(y1 + ya * height)),
        int(round(x1 + xb * width)),
        int(round(y1 + yb * height)),
    )


def _crop_mean(array: np.ndarray, bbox: tuple[int, int, int, int], *, binary: bool = False) -> float:
    x1, y1, x2, y2 = bbox
    crop = array[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return 0.0
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(np.mean(crop > 0)) if binary else float(np.mean(crop) / 255.0)


def stroke_region_evidence(
    artifacts: EvidenceArtifacts,
    candidate: CandidateRegion,
    hypotheses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """对假设涉及的局部区域计算可复核的定位分数。

    分数来自已有固定尺度辅助图，只用于物理证据门控和排序；原图仍是最终核验依据。
    """

    region_names = sorted({region for item in hypotheses for region in item.get("regions", [])})
    output = []
    for region in region_names:
        bbox = _region_bbox(candidate, region)
        material = _crop_mean(artifacts.color_residual, bbox)
        edge = _crop_mean(artifacts.edge_map, bbox, binary=True)
        ink = _crop_mean(artifacts.ink_mask, bbox, binary=True)
        support = float(np.clip(0.55 * material + 0.30 * min(1.0, edge * 4.0) + 0.15 * min(1.0, ink * 2.0), 0.0, 1.0))
        output.append(
            {
                "region": region,
                "bbox": list(bbox),
                "material_anomaly_score": round(material, 4),
                "edge_density_score": round(edge, 4),
                "ink_occupancy_score": round(ink, 4),
                "physical_support_score": round(support, 4),
            }
        )
    return output


def maximum_stroke_physical_support(evidence: list[dict[str, Any]]) -> float:
    return max((float(item.get("physical_support_score", 0.0)) for item in evidence), default=0.0)
