"""车牌字符整体贴片检测：确定性证据与受控多轮视觉模型复核。"""

from .evidence import analyze_plate_evidence, infer_plate_type
from .types import CandidateRegion, EvidenceBundle, QualityAssessment

__all__ = [
    "CandidateRegion",
    "EvidenceBundle",
    "QualityAssessment",
    "analyze_plate_evidence",
    "infer_plate_type",
]
