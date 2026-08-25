from __future__ import annotations

import cv2
import numpy as np

from shape.llm import _normalized_corners, _rectify_for_refinement


def test_normalized_corners_map_to_image_pixels() -> None:
    payload = {"corners": [[100, 200], [900, 200], [900, 800], [100, 800]]}
    actual = _normalized_corners(payload, width=2000, height=1000)
    np.testing.assert_allclose(actual, [[200, 200], [1800, 200], [1800, 800], [200, 800]])


def test_refinement_rectification_can_map_points_back() -> None:
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    corners = np.array([[200, 150], [600, 120], [620, 260], [180, 250]], dtype=np.float32)
    patch, inverse = _rectify_for_refinement(image, corners)
    assert patch.shape == (400, 1200, 3)
    patch_corners = np.array([[[0, 0], [1199, 0], [1199, 399], [0, 399]]], dtype=np.float32)
    restored = cv2.perspectiveTransform(patch_corners, inverse)[0]
    np.testing.assert_allclose(restored, corners, atol=1e-3)
