"""不依赖神经网络或云 API 的车牌外框检测器。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from .geometry import PlateDetection, PlateType, expand_quad, order_corners, polygon_area


@dataclass
class _Candidate:
    score: float
    contour: np.ndarray
    plate_type: PlateType
    rectangularity: float
    color_fill: float


def _resize_for_detection(image: np.ndarray, max_side: int = 1600) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / float(max(height, width)))
    if scale == 1.0:
        return image.copy(), 1.0
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _color_masks(image: np.ndarray) -> dict[PlateType, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 号牌并非纯色；阈值刻意覆盖曝光后的浅绿/青绿和偏暗蓝。
    # OpenCV 的 H 范围为 0..179。项目样本中绿牌主体集中在 53..80，
    # 蓝牌样本集中在 96..105；在 90 处分界，避免同一块青蓝色号牌同时进入
    # 两类候选后仅凭很接近的长宽比分错牌型。
    green = cv2.inRange(hsv, np.array([32, 18, 65]), np.array([89, 255, 255]))
    # 蓝牌采用更高饱和度下限，避免车漆/镀铬反光把蓝色区域连成巨大轮廓；
    # 白色字符造成的孔洞由后续横向闭运算补齐。
    blue = cv2.inRange(hsv, np.array([90, 110, 35]), np.array([142, 255, 255]))

    result: dict[PlateType, np.ndarray] = {}
    for plate_type, mask in (
        (PlateType.GREEN_NEW_ENERGY, green),
        (PlateType.BLUE_STANDARD, blue),
    ):
        # 横向闭运算跨过字符、螺钉和局部反光，纵向核保持较小，避免并入保险杠。
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7)))
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
        result[plate_type] = closed
    return result


def _edge_density(gray: np.ndarray, box: np.ndarray) -> float:
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(box).astype(np.int32), 255)
    edges = cv2.Canny(gray, 60, 170)
    count = cv2.countNonZero(mask)
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
    return float(cv2.countNonZero(masked_edges)) / max(count, 1)


def _candidate_score(
    contour: np.ndarray,
    mask: np.ndarray,
    gray: np.ndarray,
    image_area: float,
    plate_type: PlateType,
) -> _Candidate | None:
    contour_area = float(cv2.contourArea(contour))
    if contour_area < image_area * 0.00008 or contour_area > image_area * 0.12:
        return None
    rect = cv2.minAreaRect(contour)
    side_a, side_b = rect[1]
    if min(side_a, side_b) < 5:
        return None
    long_side, short_side = max(side_a, side_b), min(side_a, side_b)
    ratio = long_side / short_side
    if not 1.65 <= ratio <= 7.2:
        return None
    rect_area = side_a * side_b
    rectangularity = contour_area / max(rect_area, 1.0)
    if rectangularity < 0.20:
        return None

    box = order_corners(cv2.boxPoints(rect))
    region = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(region, np.rint(box).astype(np.int32), 255)
    region_area = cv2.countNonZero(region)
    color_fill = cv2.countNonZero(cv2.bitwise_and(mask, region)) / max(region_area, 1)
    edge_density = _edge_density(gray, box)
    center_y = float(box[:, 1].mean()) / gray.shape[0]
    target_ratio = 480 / 140 if plate_type is PlateType.GREEN_NEW_ENERGY else 440 / 140
    ratio_score = float(np.exp(-abs(np.log(ratio / target_ratio)) / 0.62))
    rectangle_score = min(1.0, rectangularity / 0.72)
    fill_score = min(1.0, color_fill / 0.58)
    # 字符和边框带来适量边缘；纯色车身通常 edge_density 过低。
    texture_score = float(np.exp(-abs(edge_density - 0.10) / 0.11))
    lower_bonus = 0.84 + 0.16 * np.clip((center_y - 0.25) / 0.65, 0.0, 1.0)
    area_ratio = rect_area / image_area
    size_score = float(np.clip((area_ratio / 0.0012) ** 0.28, 0.45, 1.0))
    base_score = (
        0.31 * ratio_score
        + 0.24 * rectangle_score
        + 0.25 * fill_score
        + 0.14 * texture_score
        + 0.06 * size_score
    ) * lower_bonus
    # 小型同色装饰条常比真实号牌更“纯”、更矩形。面积作为乘法先验抑制它们，
    # 但用四次方根保持远景小号牌仍有机会进入候选。
    size_prior = float(np.clip((area_ratio / 0.004) ** 0.25, 0.25, 1.0))
    score = base_score * size_prior
    return _Candidate(float(score), contour, plate_type, rectangularity, float(color_fill))


def _contour_quad(contour: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(contour)
    hull_points = hull.reshape(-1, 2).astype(np.float32)
    if len(hull_points) > 40:
        step = len(hull_points) / 40.0
        hull_points = hull_points[[int(index * step) for index in range(40)]]
    hull_area = max(float(cv2.contourArea(hull)), 1.0)

    # 在有序凸包上找最大面积内接四边形。它可将 5~8 个点的锯齿/高光轮廓
    # 还原成真实梯形，而不会像 minAreaRect 那样强迫两组对边平行。
    best_quad: np.ndarray | None = None
    best_area = -1.0
    for indices in combinations(range(len(hull_points)), 4):
        quad = hull_points[list(indices)]
        area = polygon_area(quad)
        if area > best_area:
            best_area = area
            best_quad = quad
    if best_quad is not None:
        best_quad = order_corners(best_quad)
        coverage = best_area / hull_area
        if cv2.isContourConvex(best_quad.astype(np.int32)) and 0.62 <= coverage <= 1.08:
            return best_quad

    sums = hull_points[:, 0] + hull_points[:, 1]
    diffs = hull_points[:, 0] - hull_points[:, 1]
    extreme = np.array(
        [
            hull_points[np.argmin(sums)],
            hull_points[np.argmax(diffs)],
            hull_points[np.argmax(sums)],
            hull_points[np.argmin(diffs)],
        ],
        dtype=np.float32,
    )
    if len(np.unique(extreme, axis=0)) == 4:
        extreme = order_corners(extreme)
        coverage = polygon_area(extreme) / hull_area
        if cv2.isContourConvex(extreme.astype(np.int32)) and 0.65 <= coverage <= 1.08:
            return extreme

    perimeter = cv2.arcLength(hull, True)
    options: list[tuple[float, np.ndarray]] = []
    for epsilon_fraction in np.linspace(0.008, 0.085, 40):
        approx = cv2.approxPolyDP(hull, float(epsilon_fraction * perimeter), True).reshape(-1, 2)
        if len(approx) == 4 and cv2.isContourConvex(approx.astype(np.int32)):
            quad = order_corners(approx)
            coverage = polygon_area(quad) / max(float(cv2.contourArea(hull)), 1.0)
            if 0.86 <= coverage <= 1.35:
                options.append((abs(1.0 - coverage), quad))
    if options:
        return min(options, key=lambda item: item[0])[1]
    return order_corners(cv2.boxPoints(cv2.minAreaRect(contour)))


def _classify_plate_color(image: np.ndarray, quad: np.ndarray, fallback: PlateType) -> PlateType:
    """在最终候选四边形内用高饱和像素的主色相判定蓝牌/绿牌。"""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    region = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(region, np.rint(order_corners(quad)).astype(np.int32), 255)
    chromatic = (region > 0) & (hsv[:, :, 1] >= 50) & (hsv[:, :, 2] >= 40)
    hues = hsv[:, :, 0][chromatic]
    if hues.size < 40:
        return fallback
    # 字符、压边与高光会稀释颜色，但高饱和像素的中位色相对曝光较稳定。
    return PlateType.BLUE_STANDARD if float(np.median(hues)) >= 90.0 else PlateType.GREEN_NEW_ENERGY


def detect_plate_classical(image: np.ndarray) -> PlateDetection:
    """检测最可信的一块绿/蓝单层号牌，并返回外框四角。"""

    working, scale = _resize_for_detection(image)
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    masks = _color_masks(working)
    candidates: list[_Candidate] = []
    image_area = float(working.shape[0] * working.shape[1])
    for plate_type, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            candidate = _candidate_score(contour, mask, gray, image_area, plate_type)
            if candidate is not None:
                candidates.append(candidate)

    if not candidates:
        raise RuntimeError("没有找到满足颜色、长宽比和矩形度约束的蓝/绿车牌候选")
    best = max(candidates, key=lambda candidate: candidate.score)
    working_quad = _contour_quad(best.contour)
    plate_type = _classify_plate_color(working, working_quad, best.plate_type)
    quad = working_quad / scale
    quad = expand_quad(quad, fraction=0.018)
    height, width = image.shape[:2]
    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
    confidence = float(np.clip(best.score, 0.0, 0.99))
    return PlateDetection(
        corners=order_corners(quad).tolist(),
        plate_type=plate_type,
        confidence=confidence,
        method="classical_color_structure_v2",
    )
