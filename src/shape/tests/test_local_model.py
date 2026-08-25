from __future__ import annotations

import numpy as np

from shape.local_model import _nms, _plate_type_from_pixels
from shape.geometry import PlateType


def test_nms_suppresses_overlapping_lower_score_box() -> None:
    boxes = np.array(
        [[0, 0, 100, 40], [2, 1, 102, 41], [200, 20, 300, 60]],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert _nms(boxes, scores, 0.45) == [0, 2]


def test_plate_type_uses_colour_only_after_localization() -> None:
    quad = np.array([[10, 10], [190, 10], [190, 70], [10, 70]], dtype=np.float32)
    blue = np.zeros((80, 200, 3), dtype=np.uint8)
    blue[10:71, 10:191] = (220, 80, 20)
    green = np.zeros((80, 200, 3), dtype=np.uint8)
    green[10:71, 10:191] = (80, 190, 80)
    assert _plate_type_from_pixels(blue, quad) is PlateType.BLUE_STANDARD
    assert _plate_type_from_pixels(green, quad) is PlateType.GREEN_NEW_ENERGY
