"""车牌外边框四角检测与物理尺度归一化。"""

from .geometry import PlateDetection, PlateType, rectify_plate, write_outputs

__all__ = ["PlateDetection", "PlateType", "rectify_plate", "write_outputs"]
