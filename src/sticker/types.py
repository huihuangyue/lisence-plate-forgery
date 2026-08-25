"""整体贴片检测的结构化类型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

try:  # 同时支持 python -m src.sticker.run 与 PYTHONPATH=src python -m sticker.run
    from shape.geometry import PlateType
except ModuleNotFoundError:  # pragma: no cover - 取决于启动方式
    from src.shape.geometry import PlateType


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class QualityAssessment:
    assessable: bool
    reasons: list[str]
    laplacian_variance: float
    clipped_fraction: float
    dimensions: tuple[int, int]


@dataclass(frozen=True)
class CharacterSlot:
    slot: int
    bbox: BBox


@dataclass(frozen=True)
class LineEvidence:
    endpoints: tuple[float, float, float, float]
    orientation: str
    length: float
    angle_degrees: float
    ink_overlap: float
    paired_edge_score: float
    bright_edge_score: float
    dark_edge_score: float


@dataclass(frozen=True)
class CandidateRegion:
    candidate_id: str
    slot: int
    bbox: BBox
    geometry_score: float
    appearance_score: float
    paired_edge_score: float
    combined_score: float
    features: dict[str, float]
    lines: list[LineEvidence] = field(default_factory=list)


@dataclass(frozen=True)
class EvidenceBundle:
    plate_type: PlateType
    quality: QualityAssessment
    slots: list[CharacterSlot]
    candidates: list[CandidateRegion]
    method: str = "deterministic_sticker_evidence_v6_plate_reference_controls"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plate_type"] = self.plate_type.value
        payload["quality"]["dimensions"] = list(self.quality.dimensions)
        for slot in payload["slots"]:
            slot["bbox"] = list(slot["bbox"])
        for candidate in payload["candidates"]:
            candidate["bbox"] = list(candidate["bbox"])
            for line in candidate["lines"]:
                line["endpoints"] = list(line["endpoints"])
        return payload


@dataclass
class EvidenceArtifacts:
    bundle: EvidenceBundle
    ink_mask: Any
    screw_mask: Any
    edge_map: Any
    line_overlay: Any
    color_residual: Any
    bright_dark_view: Any
    contrast_view: Any
    candidate_overlay: Any
    slot_map_view: Any
    vertical_profile_view: Any
    plate_reference_view: Any
    diagnostic_panel: Any
    evidence_sheet: Any
