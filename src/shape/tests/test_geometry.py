from __future__ import annotations

import cv2
import numpy as np
import pytest

from shape.geometry import (
    MARGIN_PX,
    PLATE_BODY_PX,
    PlateDetection,
    PlateType,
    order_corners,
    output_geometry,
    rectify_plate,
)


@pytest.mark.parametrize("plate_type", list(PlateType))
def test_nominal_occupancy_over_90_percent(plate_type: PlateType) -> None:
    _, _, _, occupancy = output_geometry(plate_type)
    assert occupancy > 0.90


def test_order_corners() -> None:
    shuffled = [[80, 70], [10, 10], [12, 75], [85, 8]]
    actual = order_corners(shuffled)
    assert actual.tolist() == [[10.0, 10.0], [85.0, 8.0], [80.0, 70.0], [12.0, 75.0]]


def test_rectification_has_fixed_physical_scale() -> None:
    image = np.zeros((300, 700, 3), dtype=np.uint8)
    corners = np.array([[100, 90], [610, 60], [600, 220], [90, 235]], dtype=np.float32)
    cv2.fillConvexPoly(image, corners.astype(np.int32), (80, 210, 130))
    detection = PlateDetection(corners.tolist(), PlateType.GREEN_NEW_ENERGY, 1.0, "test")
    result, _ = rectify_plate(image, detection)
    body_w, body_h = PLATE_BODY_PX[PlateType.GREEN_NEW_ENERGY]
    assert result.shape[:2] == (body_h + 10, body_w + 32)


@pytest.mark.parametrize("plate_type", list(PlateType))
def test_rectification_size_for_each_physical_template(plate_type: PlateType) -> None:
    image = np.zeros((180, 520, 3), dtype=np.uint8)
    corners = [[20, 20], [500, 20], [500, 160], [20, 160]]
    detection = PlateDetection(corners, plate_type, 1.0, "test")
    result, homography = rectify_plate(image, detection)
    body_w, body_h = PLATE_BODY_PX[plate_type]
    margin_x, margin_y = MARGIN_PX[plate_type]
    assert result.shape == (body_h + 2 * margin_y, body_w + 2 * margin_x, 3)
    np.testing.assert_allclose(homography @ np.linalg.inv(homography), np.eye(3), atol=1e-6)
