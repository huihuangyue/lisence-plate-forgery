"""从固定尺寸平面车牌中生成可复核的贴片候选证据。"""

from __future__ import annotations

import math
import cv2
import numpy as np

try:  # 同时支持两种模块启动方式
    from shape.geometry import MARGIN_PX, PLATE_BODY_PX, PlateType
except ModuleNotFoundError:  # pragma: no cover
    from src.shape.geometry import MARGIN_PX, PLATE_BODY_PX, PlateType

from .color import delta_e_ciede2000, opencv_lab_to_cie
from .types import (
    CandidateRegion,
    CharacterSlot,
    EvidenceArtifacts,
    EvidenceBundle,
    LineEvidence,
    QualityAssessment,
)


SLOT_CENTER_FRACTIONS: dict[PlateType, tuple[float, ...]] = {
    PlateType.GREEN_NEW_ENERGY: (0.075, 0.185, 0.385, 0.495, 0.605, 0.715, 0.825, 0.930),
    PlateType.BLUE_STANDARD: (0.075, 0.195, 0.395, 0.515, 0.635, 0.755, 0.875),
}


def infer_plate_type(image: np.ndarray) -> PlateType:
    height, width = image.shape[:2]
    if (width, height) == (992, 290):
        return PlateType.GREEN_NEW_ENERGY
    if (width, height) == (908, 290):
        return PlateType.BLUE_STANDARD
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturated = hsv[..., 1] > 45
    median_hue = float(np.median(hsv[..., 0][saturated])) if saturated.any() else 0.0
    if 35.0 <= median_hue <= 95.0:
        return PlateType.GREEN_NEW_ENERGY
    if 90.0 <= median_hue <= 140.0:
        return PlateType.BLUE_STANDARD
    raise ValueError(f"无法从尺寸 {width}x{height} 或颜色可靠判断蓝/绿牌型")


def character_slots(plate_type: PlateType, image_shape: tuple[int, ...]) -> list[CharacterSlot]:
    height, width = image_shape[:2]
    nominal_w, nominal_h = PLATE_BODY_PX[plate_type]
    nominal_margin_x, nominal_margin_y = MARGIN_PX[plate_type]
    scale_x = width / float(nominal_w + 2 * nominal_margin_x)
    scale_y = height / float(nominal_h + 2 * nominal_margin_y)
    body_x1 = nominal_margin_x * scale_x
    body_y1 = nominal_margin_y * scale_y
    body_w = nominal_w * scale_x
    body_h = nominal_h * scale_y
    half_width = body_w * (0.054 if plate_type is PlateType.GREEN_NEW_ENERGY else 0.060)
    top = int(round(body_y1 + body_h * 0.055))
    bottom = int(round(body_y1 + body_h * 0.955))
    slots = []
    for index, fraction in enumerate(SLOT_CENTER_FRACTIONS[plate_type], start=1):
        center = body_x1 + fraction * body_w
        x1 = max(0, int(round(center - half_width)))
        x2 = min(width - 1, int(round(center + half_width)))
        slots.append(CharacterSlot(index, (x1, top, x2, bottom)))
    return slots


def assess_quality(image: np.ndarray) -> QualityAssessment:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    clipped_fraction = float(np.mean((gray <= 3) | (gray >= 252)))
    height, width = gray.shape
    reasons: list[str] = []
    if min(height, width) < 100:
        reasons.append("有效尺寸过小")
    if laplacian_variance < 12.0:
        reasons.append("图像严重模糊")
    if clipped_fraction > 0.42:
        reasons.append("大面积过曝或欠曝")
    return QualityAssessment(
        assessable=not reasons,
        reasons=reasons,
        laplacian_variance=round(laplacian_variance, 3),
        clipped_fraction=round(clipped_fraction, 6),
        dimensions=(width, height),
    )


def _body_mask(image_shape: tuple[int, ...], plate_type: PlateType, inset: int = 12) -> np.ndarray:
    height, width = image_shape[:2]
    body_w, body_h = PLATE_BODY_PX[plate_type]
    margin_x, margin_y = MARGIN_PX[plate_type]
    scale_x = width / float(body_w + 2 * margin_x)
    scale_y = height / float(body_h + 2 * margin_y)
    x1 = int(round(margin_x * scale_x)) + inset
    y1 = int(round(margin_y * scale_y)) + inset
    x2 = int(round((margin_x + body_w) * scale_x)) - inset
    y2 = int(round((margin_y + body_h) * scale_y)) - inset
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def _ink_mask(image: np.ndarray, plate_type: PlateType, body_mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[..., 0]
    pixels = lightness[body_mask > 0]
    if not pixels.size:
        return np.zeros_like(lightness)
    threshold, _ = cv2.threshold(pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if plate_type is PlateType.GREEN_NEW_ENERGY:
        mask = (lightness < min(float(threshold), float(np.percentile(pixels, 48)))).astype(np.uint8) * 255
    else:
        mask = (lightness > max(float(threshold), float(np.percentile(pixels, 70)))).astype(np.uint8) * 255
    mask[body_mask == 0] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _screw_mask(image: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.25,
        minDist=max(30, image.shape[1] // 7),
        param1=100,
        param2=24,
        minRadius=max(5, image.shape[0] // 45),
        maxRadius=max(13, image.shape[0] // 10),
    )
    mask = np.zeros(gray.shape, dtype=np.uint8)
    if circles is None:
        return mask
    height, width = gray.shape
    for x, y, radius in np.rint(circles[0]).astype(int):
        normalized_x, normalized_y = x / width, y / height
        likely_fastener = 0.12 < normalized_x < 0.88 and (normalized_y < 0.34 or normalized_y > 0.66)
        if likely_fastener and body_mask[min(max(y, 0), height - 1), min(max(x, 0), width - 1)]:
            cv2.circle(mask, (x, y), int(radius * 1.35), 255, -1)
    return mask


def _normalize_u8(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    low, high = np.percentile(values, (1.0, 99.0))
    if high <= low + 1e-6:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip((values - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def _edge_views(image: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(12, 6)).apply(gray)
    denoised = cv2.GaussianBlur(clahe, (5, 5), 0)
    median = float(np.median(denoised))
    lower = int(max(0, 0.58 * median))
    upper = int(min(255, max(lower + 20, 1.45 * median)))
    canny = cv2.Canny(denoised, lower, upper, L2gradient=True)
    scharr_x = cv2.Scharr(denoised, cv2.CV_32F, 1, 0)
    scharr_y = cv2.Scharr(denoised, cv2.CV_32F, 0, 1)
    gradient = _normalize_u8(cv2.magnitude(scharr_x, scharr_y))
    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 21))
    top_hat = np.maximum(
        cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, horizontal),
        cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, vertical),
    )
    black_hat = np.maximum(
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, horizontal),
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, vertical),
    )
    smooth_lab = cv2.GaussianBlur(lab.astype(np.float32), (0, 0), sigmaX=25.0, sigmaY=13.0)
    color_residual = _normalize_u8(np.linalg.norm(lab.astype(np.float32) - smooth_lab, axis=2))
    cie_delta = lab.astype(np.float32) - smooth_lab
    cie_delta[..., 0] /= 2.55
    delta_e76_residual = np.linalg.norm(cie_delta, axis=2)
    sobel_x = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    high_frequency = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0, sigmaY=2.0))
    return {
        "gray": gray,
        "clahe": clahe,
        "lab": lab,
        "canny": canny,
        "gradient": gradient,
        "top_hat": top_hat,
        "black_hat": black_hat,
        "color_residual": color_residual,
        "delta_e76_residual": delta_e76_residual,
        "sobel_magnitude": sobel_magnitude,
        "high_frequency": high_frequency,
    }


def _sample_mask_fraction(mask: np.ndarray, endpoints: tuple[float, float, float, float], radius: int = 3) -> float:
    x1, y1, x2, y2 = endpoints
    count = max(12, int(round(math.hypot(x2 - x1, y2 - y1))))
    xs = np.linspace(x1, x2, count).round().astype(int)
    ys = np.linspace(y1, y2, count).round().astype(int)
    values = []
    height, width = mask.shape
    for x, y in zip(xs, ys, strict=True):
        xa, xb = max(0, x - radius), min(width, x + radius + 1)
        ya, yb = max(0, y - radius), min(height, y + radius + 1)
        values.append(float(np.mean(mask[ya:yb, xa:xb] > 0)))
    return float(np.mean(values)) if values else 0.0


def _sample_map_mean(values: np.ndarray, endpoints: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = endpoints
    count = max(12, int(round(math.hypot(x2 - x1, y2 - y1))))
    xs = np.clip(np.linspace(x1, x2, count).round().astype(int), 0, values.shape[1] - 1)
    ys = np.clip(np.linspace(y1, y2, count).round().astype(int), 0, values.shape[0] - 1)
    return float(np.mean(values[ys, xs]) / 255.0)


def _paired_edge_score(gray: np.ndarray, endpoints: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = endpoints
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 8.0:
        return 0.0
    nx, ny = -dy / length, dx / length
    positions = np.linspace(0.1, 0.9, 11)
    offsets = np.arange(-7, 8, dtype=np.float32)
    profiles = []
    for position in positions:
        cx, cy = x1 + dx * position, y1 + dy * position
        xs = np.clip(np.rint(cx + offsets * nx).astype(int), 0, gray.shape[1] - 1)
        ys = np.clip(np.rint(cy + offsets * ny).astype(int), 0, gray.shape[0] - 1)
        profiles.append(gray[ys, xs].astype(np.float32))
    profile = np.median(np.stack(profiles), axis=0)
    gradient = np.diff(profile)
    positive = float(max(0.0, gradient.max(initial=0.0)))
    negative = float(max(0.0, -gradient.min(initial=0.0)))
    if positive < 3.0 or negative < 3.0:
        return 0.0
    pos_index, neg_index = int(np.argmax(gradient)), int(np.argmin(gradient))
    separation = abs(pos_index - neg_index)
    if not 1 <= separation <= 9:
        return 0.0
    return float(np.clip(min(positive, negative) / 35.0, 0.0, 1.0))


def _detect_lines(
    views: dict[str, np.ndarray],
    body_mask: np.ndarray,
    ink_mask: np.ndarray,
    screw_mask: np.ndarray,
) -> list[LineEvidence]:
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    # CLAHE 只用于提高微弱接缝的可见度；候选仍需通过墨迹排除和矩形闭合约束。
    source = cv2.addWeighted(views["clahe"], 0.72, views["gradient"], 0.28, 0)
    detected = detector.detect(source)[0]
    if detected is None:
        return []
    height, width = views["gray"].shape
    exclusion = cv2.bitwise_or(cv2.dilate(ink_mask, np.ones((9, 9), np.uint8)), screw_mask)
    lines: list[LineEvidence] = []
    for raw in detected.reshape(-1, 4):
        x1, y1, x2, y2 = (float(value) for value in raw)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < max(22.0, width * 0.023):
            continue
        mx, my = int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))
        if not (0 <= mx < width and 0 <= my < height and body_mask[my, mx]):
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        horizontal_error = min(angle, 180.0 - angle)
        vertical_error = abs(angle - 90.0)
        if horizontal_error <= 14.0:
            orientation = "horizontal"
        elif vertical_error <= 14.0:
            orientation = "vertical"
        else:
            continue
        endpoints = (x1, y1, x2, y2)
        ink_overlap = _sample_mask_fraction(exclusion, endpoints, radius=4)
        # 字符直线通常在线段一侧紧邻大块油墨；贴片边缘应主要穿过背景。
        if ink_overlap > 0.12:
            continue
        lines.append(
            LineEvidence(
                endpoints=endpoints,
                orientation=orientation,
                length=round(length, 3),
                angle_degrees=round(angle, 3),
                ink_overlap=round(ink_overlap, 4),
                paired_edge_score=round(_paired_edge_score(views["gray"], endpoints), 4),
                bright_edge_score=round(_sample_map_mean(views["top_hat"], endpoints), 4),
                dark_edge_score=round(_sample_map_mean(views["black_hat"], endpoints), 4),
            )
        )
    lines.sort(key=lambda item: item.length * (1.0 - item.ink_overlap), reverse=True)
    return lines[:180]


def _line_midpoint(line: LineEvidence) -> tuple[float, float]:
    x1, y1, x2, y2 = line.endpoints
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _in_bbox(point: tuple[float, float], bbox: tuple[int, int, int, int], pad: int = 8) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 - pad <= x <= x2 + pad and y1 - pad <= y <= y2 + pad


def _endpoint_join_score(horizontal: list[LineEvidence], vertical: list[LineEvidence], scale: float) -> float:
    best = 0.0
    for h_line in horizontal:
        hx1, hy1, hx2, hy2 = h_line.endpoints
        h_points = np.array([[hx1, hy1], [hx2, hy2]], dtype=np.float32)
        for v_line in vertical:
            vx1, vy1, vx2, vy2 = v_line.endpoints
            v_points = np.array([[vx1, vy1], [vx2, vy2]], dtype=np.float32)
            distance = float(np.linalg.norm(h_points[:, None, :] - v_points[None, :, :], axis=2).min())
            best = max(best, 1.0 - distance / max(scale, 1.0))
    return float(np.clip(best, 0.0, 1.0))


def _parallel_pair_score(lines: list[LineEvidence], axis: str, span: float) -> float:
    if len(lines) < 2:
        return 0.0
    coordinates = []
    for line in lines:
        midpoint = _line_midpoint(line)
        coordinates.append(midpoint[1] if axis == "horizontal" else midpoint[0])
    separation = max(coordinates) - min(coordinates)
    return float(np.clip(separation / max(span * 0.55, 1.0), 0.0, 1.0))


def _line_strength(line: LineEvidence, reference_span: float) -> float:
    edge_signal = max(line.paired_edge_score, line.bright_edge_score, line.dark_edge_score)
    length_support = min(line.length / max(reference_span * 0.58, 1.0), 1.0)
    return float(np.clip(length_support * (1.0 - line.ink_overlap) * (0.72 + 0.28 * edge_signal), 0, 1))


def _axis_alignment_score(line: LineEvidence) -> float:
    """线段相对已矫正号牌横/纵轴的一致性；8度外不再视为贴片直边。"""

    angle = float(line.angle_degrees) % 180.0
    error = min(angle, 180.0 - angle) if line.orientation == "horizontal" else abs(angle - 90.0)
    return float(np.clip(1.0 - error / 8.0, 0.0, 1.0))


def _collinear_locality_score(
    line: LineEvidence,
    all_lines: list[LineEvidence],
    image_shape: tuple[int, ...],
) -> float:
    """抑制贯穿号牌多槽的框线；局部贴边的同一轴坐标覆盖应较短。"""

    midpoint = _line_midpoint(line)
    coordinate = midpoint[1] if line.orientation == "horizontal" else midpoint[0]
    intervals: list[tuple[float, float]] = []
    for other in all_lines:
        if other.orientation != line.orientation:
            continue
        other_midpoint = _line_midpoint(other)
        other_coordinate = other_midpoint[1] if line.orientation == "horizontal" else other_midpoint[0]
        if abs(other_coordinate - coordinate) > 6.0:
            continue
        x1, y1, x2, y2 = other.endpoints
        interval = (min(x1, x2), max(x1, x2)) if line.orientation == "horizontal" else (min(y1, y2), max(y1, y2))
        intervals.append(interval)
    intervals.sort()
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + 5.0:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    covered = sum(end - start for start, end in merged)
    span = float(image_shape[1] if line.orientation == "horizontal" else image_shape[0])
    coverage = covered / max(span, 1.0)
    return float(np.clip(1.0 - max(0.0, coverage - 0.28) / 0.35, 0.0, 1.0))


def _slot_overlap_score(line: LineEvidence, slot_bbox: tuple[int, int, int, int]) -> float:
    """线段沿自身方向必须实际进入本字符槽，避免把邻槽残边重复分配过来。"""

    sx1, sy1, sx2, sy2 = slot_bbox
    x1, y1, x2, y2 = line.endpoints
    if line.orientation == "horizontal":
        overlap = max(0.0, min(max(x1, x2), sx2) - max(min(x1, x2), sx1))
    else:
        overlap = max(0.0, min(max(y1, y2), sy2) - max(min(y1, y2), sy1))
    return float(np.clip(overlap / max(min(12.0, line.length * 0.35), 1.0), 0.0, 1.0))


def _separation_plausibility(first: LineEvidence | None, second: LineEvidence | None, axis: str, span: float) -> float:
    if first is None or second is None:
        return 0.0
    first_midpoint, second_midpoint = _line_midpoint(first), _line_midpoint(second)
    first_coordinate = first_midpoint[1] if axis == "horizontal" else first_midpoint[0]
    second_coordinate = second_midpoint[1] if axis == "horizontal" else second_midpoint[0]
    ratio = abs(first_coordinate - second_coordinate) / max(span, 1.0)
    if ratio < 0.34 or ratio > 1.35:
        return 0.0
    if 0.58 <= ratio <= 1.12:
        return 1.0
    if ratio < 0.58:
        return float((ratio - 0.34) / 0.24)
    return float((1.35 - ratio) / 0.23)


def _corner_join(first: LineEvidence | None, second: LineEvidence | None, tolerance: float) -> float:
    if first is None or second is None:
        return 0.0
    first_points = np.asarray(first.endpoints, dtype=np.float32).reshape(2, 2)
    second_points = np.asarray(second.endpoints, dtype=np.float32).reshape(2, 2)
    distance = float(np.linalg.norm(first_points[:, None, :] - second_points[None, :, :], axis=2).min())
    return float(np.clip(1.0 - distance / max(tolerance, 1.0), 0, 1))


def _rectangle_closure_features(
    lines: list[LineEvidence],
    slot_bbox: tuple[int, int, int, int],
    all_lines: list[LineEvidence],
    image_shape: tuple[int, ...],
) -> dict[str, float]:
    """估计背景线段是否属于同一矩形，而不只是若干互不相关的平行笔画。"""

    x1, y1, x2, y2 = slot_bbox
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    slot_w, slot_h = float(x2 - x1), float(y2 - y1)
    side_lines: dict[str, list[LineEvidence]] = {"top": [], "right": [], "bottom": [], "left": []}
    for line in lines:
        midpoint_x, midpoint_y = _line_midpoint(line)
        if line.orientation == "horizontal":
            side_lines["top" if midpoint_y < center_y else "bottom"].append(line)
        else:
            side_lines["left" if midpoint_x < center_x else "right"].append(line)

    selected: dict[str, LineEvidence | None] = {}
    strengths: dict[str, float] = {}
    alignments: dict[str, float] = {}
    localities: dict[str, float] = {}
    slot_overlaps: dict[str, float] = {}
    for side, candidates in side_lines.items():
        reference_span = slot_w if side in {"top", "bottom"} else slot_h
        if candidates:
            best = max(candidates, key=lambda line: _line_strength(line, reference_span))
            selected[side] = best
            strengths[side] = _line_strength(best, reference_span)
            alignments[side] = _axis_alignment_score(best)
            localities[side] = _collinear_locality_score(best, all_lines, image_shape)
            slot_overlaps[side] = _slot_overlap_score(best, slot_bbox)
        else:
            selected[side] = None
            strengths[side] = 0.0
            alignments[side] = 0.0
            localities[side] = 0.0
            slot_overlaps[side] = 0.0

    horizontal_pair = (
        min(strengths["top"], strengths["bottom"])
        * _separation_plausibility(selected["top"], selected["bottom"], "horizontal", slot_h)
    )
    vertical_pair = (
        min(strengths["left"], strengths["right"])
        * _separation_plausibility(selected["left"], selected["right"], "vertical", slot_w)
    )
    tolerance = max(18.0, min(slot_w, slot_h) * 0.24)
    corner_scores = [
        _corner_join(selected["top"], selected["left"], tolerance),
        _corner_join(selected["top"], selected["right"], tolerance),
        _corner_join(selected["bottom"], selected["left"], tolerance),
        _corner_join(selected["bottom"], selected["right"], tolerance),
    ]
    strong_sides = sum(value >= 0.32 for value in strengths.values())
    strong_corners = sum(value >= 0.35 for value in corner_scores)
    side_coverage = sum(strengths.values()) / 4.0
    opposite_pair_support = max(horizontal_pair, vertical_pair)
    corner_support = float(np.mean(sorted(corner_scores, reverse=True)[:2]))
    if strong_sides < 2:
        closure = 0.0
    else:
        side_count_support = (strong_sides - 1) / 3.0
        closure = float(
            np.clip(
                0.34 * side_coverage
                + 0.26 * side_count_support
                + 0.24 * opposite_pair_support
                + 0.16 * corner_support,
                0,
                1,
            )
        )
    return {
        "top_side_support": round(strengths["top"], 4),
        "right_side_support": round(strengths["right"], 4),
        "bottom_side_support": round(strengths["bottom"], 4),
        "left_side_support": round(strengths["left"], 4),
        "top_axis_alignment": round(alignments["top"], 4),
        "right_axis_alignment": round(alignments["right"], 4),
        "bottom_axis_alignment": round(alignments["bottom"], 4),
        "left_axis_alignment": round(alignments["left"], 4),
        "top_line_locality": round(localities["top"], 4),
        "right_line_locality": round(localities["right"], 4),
        "bottom_line_locality": round(localities["bottom"], 4),
        "left_line_locality": round(localities["left"], 4),
        "top_slot_overlap": round(slot_overlaps["top"], 4),
        "right_slot_overlap": round(slot_overlaps["right"], 4),
        "bottom_slot_overlap": round(slot_overlaps["bottom"], 4),
        "left_slot_overlap": round(slot_overlaps["left"], 4),
        "rectangle_side_count": float(strong_sides),
        "rectangle_corner_count": float(strong_corners),
        "opposite_pair_support": round(opposite_pair_support, 4),
        "rectangle_closure_score": round(closure, 4),
    }


def _partial_axis_seam_features(
    rectangle: dict[str, float],
    material: dict[str, float],
) -> dict[str, float]:
    """检测受偏光影响而不闭合的贴片边：一条材料缝加一条正交残边即可。"""

    orthogonal = {
        "top": ("left", "right"),
        "bottom": ("left", "right"),
        "left": ("top", "bottom"),
        "right": ("top", "bottom"),
    }
    best_score = 0.0
    best_material_side = 0.0
    best_orthogonal_side = 0.0
    strongest_qualified_material_side = 0.0
    qualified_material_sides = 0
    for side, orthogonal_sides in orthogonal.items():
        support = float(rectangle.get(f"{side}_side_support", 0.0))
        alignment = float(rectangle.get(f"{side}_axis_alignment", 0.0))
        delta_e = float(material.get(f"{side}_boundary_delta_e", 0.0))
        material_side = min(support / 0.24, alignment, delta_e / 7.0, 1.0)
        material_side_qualified = support >= 0.16 and alignment >= 0.45 and delta_e >= 4.5
        if material_side_qualified:
            qualified_material_sides += 1
            strongest_qualified_material_side = max(strongest_qualified_material_side, material_side)
        orthogonal_side = max(
            min(
                float(rectangle.get(f"{other}_side_support", 0.0)) / 0.34,
                float(rectangle.get(f"{other}_axis_alignment", 0.0)),
                float(rectangle.get(f"{other}_line_locality", 0.0)),
                float(rectangle.get(f"{other}_slot_overlap", 0.0)),
                1.0,
            )
            for other in orthogonal_sides
        )
        # 未达到同侧材料突变下限的线仍保留在基础几何证据中，但不抬高本路径分数。
        score = min(material_side, orthogonal_side) if material_side_qualified else 0.0
        if score > best_score:
            best_score = score
            best_material_side = material_side
            best_orthogonal_side = orthogonal_side
    return {
        "partial_axis_material_side_score": round(best_material_side, 4),
        "partial_axis_orthogonal_side_score": round(best_orthogonal_side, 4),
        "partial_axis_material_side_count": float(qualified_material_sides),
        "partial_axis_seam_score": round(best_score, 4),
        "single_axis_material_seam_score": round(strongest_qualified_material_side, 4),
    }


def _boundary_material_features(
    views: dict[str, np.ndarray],
    bbox: tuple[int, int, int, int],
    valid_background: np.ndarray,
) -> dict[str, float]:
    """比较矩形边界两侧的邻近背景，抑制整幅绿牌的缓慢颜色渐变。"""

    x1, y1, x2, y2 = bbox
    height, width = valid_background.shape
    inset, strip, trim = 3, 5, 9
    side_slices = {
        "top": ((slice(max(0, y1 + inset), min(height, y1 + inset + strip)), slice(max(0, x1 + trim), min(width, x2 - trim))),
                (slice(max(0, y1 - inset - strip), max(0, y1 - inset)), slice(max(0, x1 + trim), min(width, x2 - trim)))),
        "bottom": ((slice(max(0, y2 - inset - strip), max(0, y2 - inset)), slice(max(0, x1 + trim), min(width, x2 - trim))),
                   (slice(min(height, y2 + inset), min(height, y2 + inset + strip)), slice(max(0, x1 + trim), min(width, x2 - trim)))),
        "left": ((slice(max(0, y1 + trim), min(height, y2 - trim)), slice(max(0, x1 + inset), min(width, x1 + inset + strip))),
                 (slice(max(0, y1 + trim), min(height, y2 - trim)), slice(max(0, x1 - inset - strip), max(0, x1 - inset)))),
        "right": ((slice(max(0, y1 + trim), min(height, y2 - trim)), slice(max(0, x2 - inset - strip), max(0, x2 - inset))),
                  (slice(max(0, y1 + trim), min(height, y2 - trim)), slice(min(width, x2 + inset), min(width, x2 + inset + strip)))),
    }
    lab = views["lab"].astype(np.float32)
    deltas: dict[str, float] = {}
    for side, (inner_slice, outer_slice) in side_slices.items():
        inner_valid = valid_background[inner_slice] > 0
        outer_valid = valid_background[outer_slice] > 0
        inner_values, outer_values = lab[inner_slice][inner_valid], lab[outer_slice][outer_valid]
        if inner_values.shape[0] < 16 or outer_values.shape[0] < 16:
            deltas[side] = 0.0
            continue
        inner_lab = opencv_lab_to_cie(np.median(inner_values, axis=0))
        outer_lab = opencv_lab_to_cie(np.median(outer_values, axis=0))
        deltas[side] = float(delta_e_ciede2000(inner_lab, outer_lab))
    material_side_count = sum(value >= 2.2 for value in deltas.values())
    return {
        "top_boundary_delta_e": round(deltas["top"], 4),
        "right_boundary_delta_e": round(deltas["right"], 4),
        "bottom_boundary_delta_e": round(deltas["bottom"], 4),
        "left_boundary_delta_e": round(deltas["left"], 4),
        "material_side_count": float(material_side_count),
        "boundary_material_score": round(float(np.clip(np.mean(list(deltas.values())) / 5.0, 0, 1)), 4),
    }


def _vertical_strip_projection_features(
    slots: list[CharacterSlot],
    views: dict[str, np.ndarray],
    valid_background: np.ndarray,
) -> dict[int, dict[str, float]]:
    """检测上下边贴近号牌边框时仍可见的、成对整高竖缝。

    单个字符间隙也可能形成强竖边，因此同时计算候选相对相邻槽的孤立性；
    最终阳性还必须通过 Hough 平行边和材料差异门槛。
    """

    projection_valid = cv2.erode(valid_background, np.ones((7, 7), np.uint8)) > 0
    scharr_x = np.abs(cv2.Scharr(views["clahe"], cv2.CV_32F, 1, 0))
    background_values = scharr_x[projection_valid]
    if background_values.size < 100:
        return {
            slot.slot: {
                "vertical_strip_left_continuity": 0.0,
                "vertical_strip_right_continuity": 0.0,
                "vertical_strip_pair_score": 0.0,
                "vertical_strip_isolation": 0.0,
                "vertical_strip_left_x": float(slot.bbox[0]),
                "vertical_strip_right_x": float(slot.bbox[2]),
            }
            for slot in slots
        }
    adaptive_threshold = float(np.percentile(background_values, 90.0))
    raw: dict[int, tuple[float, float, int, int]] = {}
    height, width = projection_valid.shape
    for slot in slots:
        x1, y1, x2, y2 = slot.bbox
        row_start, row_end = max(0, y1 + 8), min(height, y2 - 8)
        side_results: list[tuple[float, int]] = []
        for boundary_x in (x1, x2):
            best_score, best_x = 0.0, boundary_x
            for x in range(max(0, boundary_x - 10), min(width, boundary_x + 11)):
                valid_rows = projection_valid[row_start:row_end, x]
                if int(valid_rows.sum()) < 25:
                    continue
                values = scharr_x[row_start:row_end, x][valid_rows]
                score = float(np.mean(values > adaptive_threshold))
                if score > best_score:
                    best_score, best_x = score, x
            side_results.append((best_score, best_x))
        raw[slot.slot] = (
            side_results[0][0],
            side_results[1][0],
            side_results[0][1],
            side_results[1][1],
        )
    pair_scores = {slot: min(values[0], values[1]) for slot, values in raw.items()}
    result: dict[int, dict[str, float]] = {}
    for index, slot in enumerate(slots):
        left_neighbor = pair_scores.get(slots[index - 1].slot, 0.0) if index else 0.0
        right_neighbor = pair_scores.get(slots[index + 1].slot, 0.0) if index + 1 < len(slots) else 0.0
        neighbor = max(left_neighbor, right_neighbor, 0.01)
        left, right, left_x, right_x = raw[slot.slot]
        pair = pair_scores[slot.slot]
        result[slot.slot] = {
            "vertical_strip_left_continuity": round(left, 4),
            "vertical_strip_right_continuity": round(right, 4),
            "vertical_strip_pair_score": round(pair, 4),
            "vertical_strip_isolation": round(float(min(pair / neighbor, 4.0)), 4),
            "vertical_strip_left_x": float(left_x),
            "vertical_strip_right_x": float(right_x),
        }
    return result


def _vertical_gradient_profile_features(
    slots: list[CharacterSlot],
    views: dict[str, np.ndarray],
    valid_background: np.ndarray,
    plate_type: PlateType,
) -> tuple[dict[int, dict[str, float]], np.ndarray]:
    """保留候选内部与邻域期望颜色随纵向位置变化的方向和形状。"""

    height, width = valid_background.shape
    lab = views["lab"].astype(np.float32)
    profile_view = np.full((height, width, 3), 24, dtype=np.uint8)
    cell_width = width // len(slots)
    results: dict[int, dict[str, float]] = {}
    for index, slot in enumerate(slots):
        x1, y1, x2, y2 = slot.bbox
        rows: list[int] = []
        actual_rows: list[np.ndarray] = []
        expected_rows: list[np.ndarray] = []
        for y in range(max(0, y1 + 8), min(height, y2 - 8)):
            inner_x1, inner_x2 = max(0, x1 + 7), min(width, x2 - 7)
            inner_mask = valid_background[y, inner_x1:inner_x2] > 0
            inner_values = lab[y, inner_x1:inner_x2][inner_mask]
            outer_parts = []
            for outer_x1, outer_x2 in ((x1 - 42, x1 - 8), (x2 + 8, x2 + 42)):
                outer_x1, outer_x2 = max(0, outer_x1), min(width, outer_x2)
                if outer_x2 <= outer_x1:
                    continue
                outer_mask = valid_background[y, outer_x1:outer_x2] > 0
                outer_values = lab[y, outer_x1:outer_x2][outer_mask]
                if outer_values.size:
                    outer_parts.append(outer_values)
            if inner_values.shape[0] < 5 or not outer_parts or sum(part.shape[0] for part in outer_parts) < 5:
                continue
            rows.append(y)
            actual_rows.append(np.median(inner_values, axis=0))
            expected_rows.append(np.median(np.concatenate(outer_parts, axis=0), axis=0))
        default = {
            "vertical_expected_gradient_magnitude": 0.0,
            "vertical_actual_gradient_magnitude": 0.0,
            "vertical_gradient_direction_cosine": 1.0,
            "vertical_gradient_slope_mismatch": 0.0,
            "vertical_gradient_profile_residual": 0.0,
            "vertical_gradient_reversal_score": 0.0,
            "vertical_gradient_uniformity_score": 0.0,
            "vertical_gradient_anomaly_score": 0.0,
            "vertical_gradient_valid_rows": float(len(rows)),
            "green_vertical_gradient_applicable": float(plate_type is PlateType.GREEN_NEW_ENERGY),
        }
        if len(rows) < 24:
            results[slot.slot] = default
            continue
        actual = np.asarray(actual_rows, dtype=np.float32)
        expected = np.asarray(expected_rows, dtype=np.float32)
        # 平滑逐行中位色，只去除压缩噪声，保留宏观纵向渐变。
        actual = cv2.GaussianBlur(actual.reshape(-1, 1, 3), (1, 9), 0).reshape(-1, 3)
        expected = cv2.GaussianBlur(expected.reshape(-1, 1, 3), (1, 9), 0).reshape(-1, 3)
        row_array = np.asarray(rows, dtype=np.float32)
        normalized_y = (row_array - row_array.mean()) / max(float(row_array.max() - row_array.min()), 1.0)
        design = np.column_stack((normalized_y, np.ones_like(normalized_y)))
        actual_slope = np.asarray(
            [np.linalg.lstsq(design, actual[:, channel], rcond=None)[0][0] for channel in range(3)]
        )
        expected_slope = np.asarray(
            [np.linalg.lstsq(design, expected[:, channel], rcond=None)[0][0] for channel in range(3)]
        )
        actual_slope[0] /= 2.55
        expected_slope[0] /= 2.55
        actual_magnitude = float(np.linalg.norm(actual_slope))
        expected_magnitude = float(np.linalg.norm(expected_slope))
        cosine = float(
            np.dot(actual_slope, expected_slope)
            / max(actual_magnitude * expected_magnitude, 1e-6)
        )
        slope_mismatch = float(np.linalg.norm(actual_slope - expected_slope))
        difference = actual - expected
        difference[:, 0] /= 2.55
        centered_difference = difference - np.median(difference, axis=0)
        profile_residual = float(np.percentile(np.linalg.norm(centered_difference, axis=1), 75))
        reversal = float(
            max(0.0, -cosine)
            * min(expected_magnitude / 10.0, 1.0)
            * min(actual_magnitude / 10.0, 1.0)
        )
        uniformity = float(
            max(0.0, 1.0 - actual_magnitude / max(expected_magnitude, 1e-6))
            * min(expected_magnitude / 8.0, 1.0)
        )
        anomaly = float(
            np.clip(
                0.35 * uniformity
                + 0.25 * reversal
                + 0.25 * min(slope_mismatch / 20.0, 1.0)
                + 0.15 * min(profile_residual / 12.0, 1.0),
                0,
                1,
            )
        )
        if plate_type is not PlateType.GREEN_NEW_ENERGY:
            anomaly = reversal = uniformity = 0.0
        results[slot.slot] = {
            "vertical_expected_gradient_magnitude": round(expected_magnitude, 4),
            "vertical_actual_gradient_magnitude": round(actual_magnitude, 4),
            "vertical_gradient_direction_cosine": round(cosine, 4),
            "vertical_gradient_slope_mismatch": round(slope_mismatch, 4),
            "vertical_gradient_profile_residual": round(profile_residual, 4),
            "vertical_gradient_reversal_score": round(reversal, 4),
            "vertical_gradient_uniformity_score": round(uniformity, 4),
            "vertical_gradient_anomaly_score": round(anomaly, 4),
            "vertical_gradient_valid_rows": float(len(rows)),
            "green_vertical_gradient_applicable": float(plate_type is PlateType.GREEN_NEW_ENERGY),
        }
        cell_x1 = index * cell_width
        cell_x2 = width if index == len(slots) - 1 else (index + 1) * cell_width
        bar_top, bar_bottom = 31, height - 8
        target_rows = np.linspace(0, len(rows) - 1, bar_bottom - bar_top + 1).round().astype(int)
        expected_u8 = np.clip(expected[target_rows], 0, 255).astype(np.uint8)
        actual_u8 = np.clip(actual[target_rows], 0, 255).astype(np.uint8)
        expected_bgr = cv2.cvtColor(expected_u8.reshape(-1, 1, 3), cv2.COLOR_LAB2BGR).reshape(-1, 3)
        actual_bgr = cv2.cvtColor(actual_u8.reshape(-1, 1, 3), cv2.COLOR_LAB2BGR).reshape(-1, 3)
        middle = (cell_x1 + cell_x2) // 2
        left_start, left_end = cell_x1 + 7, max(cell_x1 + 8, middle - 2)
        right_start, right_end = min(cell_x2 - 8, middle + 2), cell_x2 - 7
        for row_offset, y in enumerate(range(bar_top, bar_bottom + 1)):
            profile_view[y, left_start:left_end] = expected_bgr[row_offset]
            profile_view[y, right_start:right_end] = actual_bgr[row_offset]
        cv2.putText(
            profile_view,
            f"S{slot.slot} E|A {anomaly:.2f}",
            (cell_x1 + 3, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.line(profile_view, (cell_x2 - 1, 0), (cell_x2 - 1, height - 1), (70, 70, 70), 1)
    return results, profile_view


def _region_color_features(
    views: dict[str, np.ndarray],
    bbox: tuple[int, int, int, int],
    valid_background: np.ndarray,
) -> tuple[float, float, float]:
    x1, y1, x2, y2 = bbox
    height, width = valid_background.shape
    shrink = 9
    inner = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(inner, (x1 + shrink, y1 + shrink), (x2 - shrink, y2 - shrink), 255, -1)
    outer = np.zeros((height, width), dtype=np.uint8)
    pad_x, pad_y = 18, 13
    cv2.rectangle(
        outer,
        (max(0, x1 - pad_x), max(0, y1 - pad_y)),
        (min(width - 1, x2 + pad_x), min(height - 1, y2 + pad_y)),
        255,
        -1,
    )
    cv2.rectangle(outer, (x1, y1), (x2, y2), 0, -1)
    inner_valid = (inner > 0) & (valid_background > 0)
    outer_valid = (outer > 0) & (valid_background > 0)
    if inner_valid.sum() < 40 or outer_valid.sum() < 30:
        return 0.0, 0.0, 0.0
    # 用外围环带拟合局部二维颜色平面，再看内部相对该平面的残差。
    # 这会抑制绿牌渐变和缓慢照明变化，不再直接比较两个区域的平均 RGB/Lab。
    lab_cv = views["lab"].astype(np.float32)
    outer_y, outer_x = np.nonzero(outer_valid)
    inner_y, inner_x = np.nonzero(inner_valid)
    design_outer = np.column_stack((outer_x, outer_y, np.ones_like(outer_x))).astype(np.float32)
    design_inner = np.column_stack((inner_x, inner_y, np.ones_like(inner_x))).astype(np.float32)
    predicted_inner = np.empty((inner_x.size, 3), dtype=np.float32)
    for channel in range(3):
        coefficients, *_ = np.linalg.lstsq(design_outer, lab_cv[outer_valid, channel], rcond=None)
        predicted_inner[:, channel] = design_inner @ coefficients
    actual_inner = lab_cv[inner_valid]
    actual_lab = opencv_lab_to_cie(np.median(actual_inner, axis=0))
    predicted_lab = opencv_lab_to_cie(np.median(predicted_inner, axis=0))
    delta_e = delta_e_ciede2000(actual_lab, predicted_lab)
    residual = float(np.percentile(views["color_residual"][inner_valid], 75) / 255.0)
    boundary = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(boundary, (x1, y1), (x2, y2), 255, 5)
    boundary_valid = (boundary > 0) & (valid_background > 0)
    boundary_residual = (
        float(np.mean(views["color_residual"][boundary_valid]) / 255.0) if boundary_valid.any() else 0.0
    )
    return float(delta_e), residual, boundary_residual


def _cie_delta_between_opencv_lab(first: np.ndarray, second: np.ndarray) -> float:
    """计算两个 OpenCV Lab 三元组的 CIEDE2000 色差。"""

    return float(
        delta_e_ciede2000(
            opencv_lab_to_cie(np.asarray(first, dtype=np.float32)),
            opencv_lab_to_cie(np.asarray(second, dtype=np.float32)),
        )
    )


def _plate_row_reference_features(
    slots: list[CharacterSlot],
    views: dict[str, np.ndarray],
    valid_background: np.ndarray,
) -> tuple[dict[int, dict[str, float]], np.ndarray]:
    """用号牌同一高度的多数背景建立参考，保留整牌固有的纵向颜色分布。

    绿牌常见的上浅下绿和不均匀照明都随行进入参考值，因此不会被错误地
    当成贴片；相反，覆盖某个字符的独立材料会表现为相对同高度多数槽位的
    稳定离群。该参考也解决了相邻槽共享同一条贴片边时的归属问题。
    """

    lab = views["lab"]
    high_frequency = views["high_frequency"].astype(np.float32)
    height, width = valid_background.shape
    per_slot_delta: dict[int, list[float]] = {slot.slot: [] for slot in slots}
    per_slot_hf_ratio: dict[int, list[float]] = {slot.slot: [] for slot in slots}
    residual_map = np.zeros((height, width), dtype=np.float32)
    top = max(0, min(slot.bbox[1] for slot in slots) + 8)
    bottom = min(height, max(slot.bbox[3] for slot in slots) - 8)
    for y in range(top, bottom):
        row_lab: dict[int, np.ndarray] = {}
        row_hf: dict[int, float] = {}
        for slot in slots:
            x1, _, x2, _ = slot.bbox
            x1, x2 = max(0, x1 + 7), min(width, x2 - 7)
            if x2 <= x1:
                continue
            mask = valid_background[y, x1:x2] > 0
            if int(mask.sum()) < 4:
                continue
            row_lab[slot.slot] = np.median(lab[y, x1:x2][mask], axis=0)
            row_hf[slot.slot] = float(np.median(high_frequency[y, x1:x2][mask]))
        if len(row_lab) < max(4, len(slots) // 2):
            continue
        reference_lab = np.median(np.stack(list(row_lab.values())), axis=0)
        reference_hf = float(np.median(list(row_hf.values())))
        for slot_id, value in row_lab.items():
            per_slot_delta[slot_id].append(_cie_delta_between_opencv_lab(value, reference_lab))
            per_slot_hf_ratio[slot_id].append((row_hf[slot_id] + 1.0) / (reference_hf + 1.0))
        reference_cie = opencv_lab_to_cie(reference_lab)
        for slot in slots:
            x1, _, x2, _ = slot.bbox
            x1, x2 = max(0, x1), min(width, x2)
            mask = valid_background[y, x1:x2] > 0
            if not mask.any():
                continue
            # 热图只负责定位，使用可向量化的 ΔE76；结构化门控仍使用上面的 ΔE00。
            residual_map[y, x1:x2][mask] = np.linalg.norm(
                opencv_lab_to_cie(lab[y, x1:x2][mask]) - reference_cie,
                axis=1,
            )

    result: dict[int, dict[str, float]] = {}
    for slot in slots:
        deltas = np.asarray(per_slot_delta[slot.slot], dtype=np.float32)
        hf_ratios = np.asarray(per_slot_hf_ratio[slot.slot], dtype=np.float32)
        if deltas.size < 24:
            result[slot.slot] = {
                "plate_reference_delta_e_median": 0.0,
                "plate_reference_delta_e_p75": 0.0,
                "plate_reference_hf_ratio": 1.0,
                "plate_reference_valid_rows": float(deltas.size),
                "plate_reference_material_score": 0.0,
            }
            continue
        median_delta = float(np.median(deltas))
        p75_delta = float(np.percentile(deltas, 75))
        hf_ratio = float(np.median(hf_ratios)) if hf_ratios.size else 1.0
        material_score = float(
            np.clip(max(median_delta / 7.0, p75_delta / 10.0), 0.0, 1.0)
        )
        result[slot.slot] = {
            "plate_reference_delta_e_median": round(median_delta, 4),
            "plate_reference_delta_e_p75": round(p75_delta, 4),
            "plate_reference_hf_ratio": round(hf_ratio, 4),
            "plate_reference_valid_rows": float(deltas.size),
            "plate_reference_material_score": round(material_score, 4),
        }
    residual_u8 = np.clip(residual_map * (255.0 / 12.0), 0, 255).astype(np.uint8)
    residual_view = cv2.applyColorMap(residual_u8, cv2.COLORMAP_INFERNO)
    residual_view[valid_background == 0] = (18, 18, 18)
    return result, residual_view


def _matched_lateral_control_features(
    views: dict[str, np.ndarray],
    bbox: tuple[int, int, int, int],
    valid_background: np.ndarray,
    *,
    first_slot: bool,
    last_slot: bool,
) -> dict[str, float]:
    """把候选与同高度、紧邻左右的车牌背景逐行配对比较。

    首末字符只使用朝向号牌内部的一侧，避免把号牌外部、压边或车身颜色
    引入对照区。字符和螺钉像素在进入统计前已经由 valid_background 排除。
    """

    x1, y1, x2, y2 = bbox
    height, width = valid_background.shape
    lab = views["lab"]
    high_frequency = views["high_frequency"].astype(np.float32)
    control_width = int(np.clip(round((x2 - x1) * 0.26), 18, 32))
    gap = 7
    combined_deltas: list[float] = []
    left_deltas: list[float] = []
    right_deltas: list[float] = []
    hf_ratios: list[float] = []
    candidate_pixels = left_pixels = right_pixels = 0

    for y in range(max(0, y1 + 8), min(height, y2 - 8)):
        def row_summary(start: int, end: int) -> tuple[np.ndarray, float, int] | None:
            start, end = max(0, start), min(width, end)
            if end <= start:
                return None
            mask = valid_background[y, start:end] > 0
            count = int(mask.sum())
            if count < 3:
                return None
            return (
                np.median(lab[y, start:end][mask], axis=0),
                float(np.median(high_frequency[y, start:end][mask])),
                count,
            )

        candidate = row_summary(x1 + 7, x2 - 7)
        left = None if first_slot else row_summary(x1 - gap - control_width, x1 - gap)
        right = None if last_slot else row_summary(x2 + gap, x2 + gap + control_width)
        if candidate is None or (left is None and right is None):
            continue
        candidate_lab, candidate_hf, candidate_count = candidate
        controls = [item for item in (left, right) if item is not None]
        reference_lab = np.median(np.stack([item[0] for item in controls]), axis=0)
        reference_hf = float(np.median([item[1] for item in controls]))
        combined_deltas.append(_cie_delta_between_opencv_lab(candidate_lab, reference_lab))
        hf_ratios.append((candidate_hf + 1.0) / (reference_hf + 1.0))
        candidate_pixels += candidate_count
        if left is not None:
            left_deltas.append(_cie_delta_between_opencv_lab(candidate_lab, left[0]))
            left_pixels += left[2]
        if right is not None:
            right_deltas.append(_cie_delta_between_opencv_lab(candidate_lab, right[0]))
            right_pixels += right[2]

    def robust_median(values: list[float]) -> float:
        return float(np.median(values)) if values else 0.0

    combined = robust_median(combined_deltas)
    left_delta = robust_median(left_deltas)
    right_delta = robust_median(right_deltas)
    bilateral_valid = bool(left_deltas and right_deltas)
    bilateral_min = min(left_delta, right_delta) if bilateral_valid else 0.0
    bilateral_max = max(left_delta, right_delta) if bilateral_valid else 0.0
    balance = (
        float(np.clip(bilateral_min / max(bilateral_max, 1e-6), 0.0, 1.0))
        if bilateral_valid
        else 0.0
    )
    valid = len(combined_deltas) >= 24
    return {
        "matched_control_delta_e_median": round(combined, 4),
        "matched_control_delta_e_p75": round(
            float(np.percentile(combined_deltas, 75)) if combined_deltas else 0.0, 4
        ),
        "matched_control_left_delta_e": round(left_delta, 4),
        "matched_control_right_delta_e": round(right_delta, 4),
        "matched_control_bilateral_min_delta_e": round(bilateral_min, 4),
        "matched_control_bilateral_balance": round(balance, 4),
        "matched_control_hf_ratio": round(robust_median(hf_ratios) if hf_ratios else 1.0, 4),
        "matched_control_valid_rows": float(len(combined_deltas)),
        "matched_control_candidate_pixels": float(candidate_pixels),
        "matched_control_left_pixels": float(left_pixels),
        "matched_control_right_pixels": float(right_pixels),
        "matched_control_valid": float(valid),
        "matched_control_bilateral_valid": float(bilateral_valid),
        "edge_slot_one_sided_control": float(first_slot or last_slot),
    }


def _refine_bbox_from_lines(
    slot_bbox: tuple[int, int, int, int],
    lines: list[LineEvidence],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = slot_bbox
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    vertical_x = [_line_midpoint(line)[0] for line in lines if line.orientation == "vertical"]
    horizontal_y = [_line_midpoint(line)[1] for line in lines if line.orientation == "horizontal"]
    left = [value for value in vertical_x if x1 - 18 <= value < center_x]
    right = [value for value in vertical_x if center_x < value <= x2 + 18]
    top = [value for value in horizontal_y if y1 - 16 <= value < center_y]
    bottom = [value for value in horizontal_y if center_y < value <= y2 + 16]
    if left:
        x1 = int(round(min(left, key=lambda value: abs(value - x1))))
    if right:
        x2 = int(round(min(right, key=lambda value: abs(value - x2))))
    if top:
        y1 = int(round(min(top, key=lambda value: abs(value - y1))))
    if bottom:
        y2 = int(round(min(bottom, key=lambda value: abs(value - y2))))
    height, width = image_shape[:2]
    x1, x2 = sorted((max(0, x1), min(width - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(height - 1, y2)))
    if x2 - x1 < 30 or y2 - y1 < 60:
        return slot_bbox
    return x1, y1, x2, y2


def _candidate_for_slot(
    slot: CharacterSlot,
    all_lines: list[LineEvidence],
    views: dict[str, np.ndarray],
    valid_background: np.ndarray,
    strip_projection: dict[str, float],
    gradient_profile: dict[str, float],
    plate_reference: dict[str, float],
    *,
    first_slot: bool,
    last_slot: bool,
) -> CandidateRegion:
    # 贴片可宽于印刷字符；扩大候选搜索只负责找边，后续必须由材料归属门控。
    nearby = [line for line in all_lines if _in_bbox(_line_midpoint(line), slot.bbox, pad=42)]
    sx1, sy1, sx2, sy2 = slot.bbox
    slot_w, slot_h = float(sx2 - sx1), float(sy2 - sy1)
    lines = []
    for line in nearby:
        midpoint_x, midpoint_y = _line_midpoint(line)
        if line.orientation == "vertical":
            near_candidate_border = min(abs(midpoint_x - sx1), abs(midpoint_x - sx2)) <= slot_w * 0.42
        else:
            near_candidate_border = min(abs(midpoint_y - sy1), abs(midpoint_y - sy2)) <= slot_h * 0.30
        if near_candidate_border:
            lines.append(line)
    lines = lines[:16]
    horizontal = [line for line in lines if line.orientation == "horizontal"]
    vertical = [line for line in lines if line.orientation == "vertical"]
    x1, y1, x2, y2 = slot.bbox
    h_support = float(np.clip(sum(line.length * (1.0 - line.ink_overlap) for line in horizontal) / (2.0 * slot_w), 0, 1))
    v_support = float(np.clip(sum(line.length * (1.0 - line.ink_overlap) for line in vertical) / (2.0 * slot_h), 0, 1))
    parallel = max(
        _parallel_pair_score(horizontal, "horizontal", slot_h),
        _parallel_pair_score(vertical, "vertical", slot_w),
    )
    join = _endpoint_join_score(horizontal, vertical, max(15.0, min(slot_w, slot_h) * 0.18))
    orientation_diversity = 1.0 if horizontal and vertical else 0.0
    paired = max((line.paired_edge_score for line in lines), default=0.0)
    bright_dark = max((max(line.bright_edge_score, line.dark_edge_score) for line in lines), default=0.0)
    rectangle = _rectangle_closure_features(lines, slot.bbox, all_lines, views["gray"].shape)
    closure = rectangle["rectangle_closure_score"]
    rectangle_geometry = float(
        np.clip(
            0.14 * h_support
            + 0.14 * v_support
            + 0.10 * parallel
            + 0.12 * join
            + 0.08 * orientation_diversity
            + 0.42 * closure,
            0,
            1,
        )
    )
    strip_pair = float(strip_projection.get("vertical_strip_pair_score", 0.0))
    strip_isolation = float(strip_projection.get("vertical_strip_isolation", 0.0))
    strip_side_support = min(rectangle["left_side_support"], rectangle["right_side_support"])
    if strip_pair >= 0.18 and strip_isolation >= 1.25 and parallel >= 0.65 and strip_side_support >= 0.10:
        strip_geometry = float(
            np.clip(
                0.34 * strip_pair
                + 0.22 * min(strip_isolation / 4.0, 1.0)
                + 0.22 * parallel
                + 0.22 * min(strip_side_support / 0.35, 1.0),
                0,
                1,
            )
        )
    else:
        strip_geometry = 0.0
    refined_bbox = _refine_bbox_from_lines(slot.bbox, lines, views["gray"].shape)
    delta_e, residual, boundary_residual = _region_color_features(views, refined_bbox, valid_background)
    material_boundary = _boundary_material_features(views, refined_bbox, valid_background)
    matched_control = _matched_lateral_control_features(
        views,
        refined_bbox,
        valid_background,
        first_slot=first_slot,
        last_slot=last_slot,
    )
    partial_axis = _partial_axis_seam_features(rectangle, material_boundary)
    partial_axis_geometry = float(partial_axis["partial_axis_seam_score"])
    geometry = max(rectangle_geometry, strip_geometry, partial_axis_geometry)
    boundary_material_score = material_boundary["boundary_material_score"]
    plate_material_score = float(plate_reference.get("plate_reference_material_score", 0.0))
    matched_material_score = min(
        float(matched_control.get("matched_control_delta_e_median", 0.0)) / 7.0, 1.0
    )
    contextual_material_score = max(plate_material_score, matched_material_score)
    material_ownership_score = (
        plate_material_score
        if first_slot or last_slot
        else max(plate_material_score, matched_material_score)
    )
    base_appearance = float(
        np.clip(
            0.20 * min(delta_e / 6.0, 1.0)
            + 0.14 * min(residual / 0.42, 1.0)
            + 0.14 * min(boundary_residual / 0.36, 1.0)
            + 0.10 * min(bright_dark / 0.25, 1.0)
            + 0.14 * boundary_material_score
            + 0.28 * contextual_material_score,
            0,
            1,
        )
    )
    gradient_anomaly = float(gradient_profile.get("vertical_gradient_anomaly_score", 0.0))
    appearance = max(base_appearance, 0.68 * base_appearance + 0.32 * gradient_anomaly)
    combined = float(np.clip(0.55 * geometry + 0.35 * appearance + 0.10 * paired, 0, 1))
    features = {
        "horizontal_support": round(h_support, 4),
        "vertical_support": round(v_support, 4),
        "parallel_pair_support": round(parallel, 4),
        "orthogonal_join_support": round(join, 4),
        "orientation_diversity": round(orientation_diversity, 4),
        "delta_e_00": round(delta_e, 4),
        "interior_color_residual": round(residual, 4),
        "boundary_color_residual": round(boundary_residual, 4),
        "bright_dark_support": round(bright_dark, 4),
        "contextual_material_score": round(contextual_material_score, 4),
        "material_ownership_score": round(material_ownership_score, 4),
        "base_appearance_score": round(base_appearance, 4),
        "rectangle_geometry_score": round(rectangle_geometry, 4),
        "vertical_strip_geometry_score": round(strip_geometry, 4),
        "partial_axis_geometry_score": round(partial_axis_geometry, 4),
        **rectangle,
        **partial_axis,
        **strip_projection,
        **gradient_profile,
        **material_boundary,
        **plate_reference,
        **matched_control,
    }
    return CandidateRegion(
        candidate_id=f"C{slot.slot}",
        slot=slot.slot,
        bbox=refined_bbox,
        geometry_score=round(geometry, 4),
        appearance_score=round(appearance, 4),
        paired_edge_score=round(paired, 4),
        combined_score=round(combined, 4),
        features=features,
        lines=lines,
    )


def candidate_has_rectangle_material_support(candidate: CandidateRegion, *, strict: bool) -> bool:
    """确定性物理门槛：几何/剖面异常与材料边界形成独立证据组合。"""

    features = candidate.features
    side_count = float(features.get("rectangle_side_count", 0.0))
    closure = float(features.get("rectangle_closure_score", 0.0))
    material_side_count = float(features.get("material_side_count", 0.0))
    boundary_material = float(features.get("boundary_material_score", 0.0))
    delta_e = float(features.get("delta_e_00", 0.0))
    context_available = (
        float(features.get("matched_control_valid", 0.0)) >= 1.0
        and float(features.get("plate_reference_valid_rows", 0.0)) >= 24.0
    )
    reference_median = float(features.get("plate_reference_delta_e_median", 0.0))
    reference_p75 = float(features.get("plate_reference_delta_e_p75", 0.0))
    matched_delta = float(features.get("matched_control_delta_e_median", 0.0))
    bilateral_min = float(features.get("matched_control_bilateral_min_delta_e", 0.0))
    edge_slot = float(features.get("edge_slot_one_sided_control", 0.0)) >= 1.0
    if edge_slot:
        plate_owned_material = (
            reference_median >= (4.0 if strict else 3.6)
            or (
                reference_median >= (3.3 if strict else 3.0)
                and reference_p75 >= (6.0 if strict else 5.5)
            )
        )
    else:
        plate_owned_material = (
            reference_median >= (3.8 if strict else 3.0)
            or (
                reference_median >= (3.1 if strict else 2.5)
                and reference_p75 >= (5.6 if strict else 4.6)
            )
        )
    matched_material = matched_delta >= (4.2 if strict else 4.0)
    bilateral_material = bilateral_min >= (4.5 if strict else 3.5)
    # 首末槽只准由整牌参考确认归属，防止把号牌外部或相邻贴片当成自身材料差异。
    owned_material = (
        not context_available
        or plate_owned_material
        or (not edge_slot and matched_material)
    )
    material_supported = material_side_count >= 2 or boundary_material >= 0.42 or delta_e >= 4.0
    rectangle_supported = (
        side_count >= 2
        and closure >= (0.24 if strict else 0.16)
        and float(features.get("rectangle_geometry_score", candidate.geometry_score))
        >= (0.28 if strict else 0.20)
        and candidate.appearance_score >= (0.42 if strict else 0.30)
        and material_supported
        and owned_material
    )
    # 打光偏斜时贴片边不会总是闭合；两条低闭合度直边仍可由候选内部材料归属确认。
    open_rectangle_material_supported = (
        side_count >= 2
        and closure >= (0.18 if strict else 0.14)
        and float(features.get("rectangle_geometry_score", candidate.geometry_score))
        >= (0.34 if strict else 0.27)
        and candidate.appearance_score >= (0.50 if strict else 0.40)
        and material_supported
        and owned_material
    )
    strip_pair = float(features.get("vertical_strip_pair_score", 0.0))
    strip_isolation = float(features.get("vertical_strip_isolation", 0.0))
    parallel_pair = float(features.get("parallel_pair_support", 0.0))
    strip_side_support = min(
        float(features.get("left_side_support", 0.0)),
        float(features.get("right_side_support", 0.0)),
    )
    side_delta = max(
        float(features.get("left_boundary_delta_e", 0.0)),
        float(features.get("right_boundary_delta_e", 0.0)),
    )
    strip_material_supported = material_side_count >= 2 and side_delta >= 5.0
    strip_supported = (
        strip_pair >= 0.25
        and strip_isolation >= 1.60
        and parallel_pair >= 0.70
        and strip_side_support >= 0.12
        and candidate.appearance_score >= (0.42 if strict else 0.35)
        and strip_material_supported
        and (not context_available or plate_owned_material or bilateral_material)
    )
    partial_axis = float(features.get("partial_axis_seam_score", 0.0))
    partial_material_side = float(features.get("partial_axis_material_side_score", 0.0))
    partial_orthogonal_side = float(features.get("partial_axis_orthogonal_side_score", 0.0))
    partial_axis_supported = (
        partial_axis >= (0.54 if strict else 0.44)
        and partial_material_side >= (0.65 if strict else 0.52)
        and partial_orthogonal_side >= (0.54 if strict else 0.44)
        and float(features.get("partial_axis_material_side_count", 0.0)) >= 1.0
        and candidate.appearance_score >= (0.48 if strict else 0.40)
        and (not context_available or plate_owned_material or bilateral_material)
    )
    single_axis_material_supported = (
        float(features.get("single_axis_material_seam_score", 0.0))
        >= (0.62 if strict else 0.52)
        and float(features.get("partial_axis_material_side_count", 0.0)) >= 1.0
        and delta_e >= (12.0 if strict else 9.0)
        and candidate.appearance_score >= (0.58 if strict else 0.50)
        and (not context_available or plate_owned_material)
    )
    expected_gradient = float(features.get("vertical_expected_gradient_magnitude", 0.0))
    actual_gradient = float(features.get("vertical_actual_gradient_magnitude", 0.0))
    gradient_cosine = float(features.get("vertical_gradient_direction_cosine", 1.0))
    gradient_anomaly = float(features.get("vertical_gradient_anomaly_score", 0.0))
    edge_side_support = max(
        float(features.get("top_side_support", 0.0)),
        float(features.get("right_side_support", 0.0)),
        float(features.get("bottom_side_support", 0.0)),
        float(features.get("left_side_support", 0.0)),
    )
    boundary_delta = max(
        float(features.get("top_boundary_delta_e", 0.0)),
        float(features.get("right_boundary_delta_e", 0.0)),
        float(features.get("bottom_boundary_delta_e", 0.0)),
        float(features.get("left_boundary_delta_e", 0.0)),
    )
    gradient_supported = (
        float(features.get("green_vertical_gradient_applicable", 0.0)) >= 1.0
        and gradient_anomaly >= 0.60
        and expected_gradient >= 15.0
        and (
            actual_gradient / max(expected_gradient, 1e-6) <= 0.55
            or gradient_cosine <= 0.0
        )
        and material_side_count >= 2
        and boundary_delta >= 5.0
        and edge_side_support >= 0.30
        and candidate.appearance_score >= (0.45 if strict else 0.40)
        and (not context_available or plate_owned_material or bilateral_material)
    )
    return (
        rectangle_supported
        or open_rectangle_material_supported
        or strip_supported
        or partial_axis_supported
        or single_axis_material_supported
        or gradient_supported
    )


def candidate_tamper_support_routes(candidate: CandidateRegion) -> list[str]:
    """返回候选成立的可解释证据路径。

    基础物理门覆盖闭合/开放矩形、整高条带和纵向渐变异常。补充路径
    针对人工假阴性中反复出现、但基础门会漏掉的情形：材料离群但边线
    不完整、三边矩形但局部颜色对照较弱、孤立单缝，以及末槽只能取得
    单侧对照。阈值只使用图像证据，不依赖文件名、OCR 文本或人工标签。
    """

    routes: list[str] = []
    if candidate_has_rectangle_material_support(candidate, strict=True):
        routes.append("strict_rectangle_material")
    elif candidate_has_rectangle_material_support(candidate, strict=False):
        routes.append("open_rectangle_material")

    features = candidate.features
    appearance = float(candidate.appearance_score)
    ownership = float(features.get("material_ownership_score", 0.0))
    reference_median = float(features.get("plate_reference_delta_e_median", 0.0))
    reference_p75 = float(features.get("plate_reference_delta_e_p75", 0.0))
    bilateral_min = float(features.get("matched_control_bilateral_min_delta_e", 0.0))
    material_sides = float(features.get("material_side_count", 0.0))
    boundary_material = float(features.get("boundary_material_score", 0.0))
    single_axis_seam = float(features.get("single_axis_material_seam_score", 0.0))
    side_count = float(features.get("rectangle_side_count", 0.0))
    closure = float(features.get("rectangle_closure_score", 0.0))

    contextual_material = (
        appearance >= 0.48
        and ownership >= 0.40
        and reference_median >= 3.0
        and bilateral_min >= 2.3
        and (
            single_axis_seam >= 0.45
            or material_sides >= 2.0
            or boundary_material >= 0.62
        )
    )
    if contextual_material:
        routes.append("contextual_material_outlier")

    # 边线会随透视重采样、JPEG 和偏光出现较大波动；当整牌同高度参考与
    # 左右双侧对照同时给出极强材料离群时，不再要求第二条可见边。
    strong_contextual_material = (
        appearance >= 0.70
        and ownership >= 0.90
        and reference_median >= 6.0
        and bilateral_min >= 3.5
    )
    if strong_contextual_material:
        routes.append("strong_contextual_material_without_complete_edge")

    # 中等强度材料离群必须再由槽内残缝或整牌行参考的高分位差异确认，
    # 避免单纯放宽均值色差门槛。
    moderate_contextual_boundary = (
        appearance >= 0.46
        and ownership >= 0.40
        and reference_median >= 2.3
        and bilateral_min >= 1.9
        and material_sides >= 2.0
        and (single_axis_seam >= 0.65 or reference_p75 >= 3.7)
    )
    if moderate_contextual_boundary:
        routes.append("moderate_contextual_material_with_boundary")

    # 偏光只留下单边时，必须同时出现近饱和边界响应、至少三侧材料响应和
    # 双侧对照差异；这条路径不依赖矩形闭合，但仍要求边与材料相互印证。
    isolated_material_seam = (
        appearance >= 0.57
        and material_sides >= 3.0
        and boundary_material >= 0.95
        and single_axis_seam >= 0.88
        and bilateral_min >= 2.0
    )
    if isolated_material_seam:
        routes.append("isolated_axis_seam_with_multiside_material")

    three_side_geometry = (
        candidate.geometry_score >= 0.60
        and side_count >= 3.0
        and closure >= 0.45
        and boundary_material >= 0.50
        and appearance >= 0.45
    )
    if three_side_geometry:
        routes.append("three_side_aligned_patch")

    edge_slot_material = (
        float(features.get("edge_slot_one_sided_control", 0.0)) >= 1.0
        and appearance >= 0.62
        and reference_median >= 3.6
        and reference_p75 >= 4.0
        and boundary_material >= 0.60
        and float(features.get("delta_e_00", 0.0)) >= 3.0
        and side_count >= 1.0
    )
    if edge_slot_material:
        routes.append("edge_slot_one_sided_material")
    return routes


def candidate_has_multichannel_tamper_support(candidate: CandidateRegion) -> bool:
    """任一经过材料归属约束的物理路径成立即进入高召回候选集合。"""

    return bool(candidate_tamper_support_routes(candidate))


def resolve_adjacent_material_ownership(
    candidates: list[CandidateRegion],
) -> tuple[list[CandidateRegion], dict[str, str]]:
    """抑制强贴片边界泄漏到相邻字符槽产生的串槽。

    只有当相邻候选具有明显更强的双侧材料证据和整牌行参考，且当前候选
    的双侧证据不足时才移除当前候选。这样不会压掉两个相邻但各自都有
    独立材料归属的贴片。
    """

    by_slot = {candidate.slot: candidate for candidate in candidates}
    suppressed: dict[str, str] = {}
    for slot, candidate in by_slot.items():
        features = candidate.features
        bilateral = float(features.get("matched_control_bilateral_min_delta_e", 0.0))
        reference = float(features.get("plate_reference_delta_e_median", 0.0))
        for adjacent_slot in (slot - 1, slot + 1):
            owner = by_slot.get(adjacent_slot)
            if owner is None:
                continue
            owner_bilateral = float(
                owner.features.get("matched_control_bilateral_min_delta_e", 0.0)
            )
            owner_reference = float(
                owner.features.get("plate_reference_delta_e_median", 0.0)
            )
            if (
                bilateral < 2.0
                and owner_bilateral >= 3.0
                and owner_reference >= 4.0
                and owner_reference >= reference + 2.0
            ):
                suppressed[candidate.candidate_id] = owner.candidate_id
                break
    kept = [
        candidate
        for candidate in candidates
        if candidate.candidate_id not in suppressed
    ]
    return kept, suppressed


def analyze_plate_evidence(image: np.ndarray, plate_type: PlateType | None = None) -> EvidenceArtifacts:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("输入必须是 BGR 三通道车牌图")
    plate_type = plate_type or infer_plate_type(image)
    quality = assess_quality(image)
    slots = character_slots(plate_type, image.shape)
    body_mask = _body_mask(image.shape, plate_type)
    ink_mask = _ink_mask(image, plate_type, body_mask)
    screw_mask = _screw_mask(image, body_mask)
    views = _edge_views(image)
    lines = _detect_lines(views, body_mask, ink_mask, screw_mask)
    valid_background = body_mask.copy()
    valid_background[cv2.dilate(ink_mask, np.ones((7, 7), np.uint8)) > 0] = 0
    valid_background[screw_mask > 0] = 0
    strip_projections = _vertical_strip_projection_features(slots, views, valid_background)
    gradient_profiles, vertical_profile_view = _vertical_gradient_profile_features(
        slots, views, valid_background, plate_type
    )
    plate_references, plate_reference_view = _plate_row_reference_features(
        slots, views, valid_background
    )
    candidates = [
        _candidate_for_slot(
            slot,
            lines,
            views,
            valid_background,
            strip_projections[slot.slot],
            gradient_profiles[slot.slot],
            plate_references[slot.slot],
            first_slot=index == 0,
            last_slot=index == len(slots) - 1,
        )
        for index, slot in enumerate(slots)
    ]
    candidates = sorted(candidates, key=lambda item: item.combined_score, reverse=True)
    bundle = EvidenceBundle(plate_type=plate_type, quality=quality, slots=slots, candidates=candidates)

    line_overlay = image.copy()
    for line in lines:
        x1, y1, x2, y2 = np.rint(line.endpoints).astype(int)
        color = (0, 220, 255) if line.orientation == "horizontal" else (255, 180, 0)
        cv2.line(line_overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    color_heat = cv2.applyColorMap(views["color_residual"], cv2.COLORMAP_TURBO)
    bright_dark = np.zeros_like(image)
    bright_dark[..., 1] = _normalize_u8(views["top_hat"])
    bright_dark[..., 2] = _normalize_u8(views["black_hat"])
    clahe_bgr = cv2.cvtColor(views["clahe"], cv2.COLOR_GRAY2BGR)
    inverted_bgr = cv2.cvtColor(255 - views["clahe"], cv2.COLOR_GRAY2BGR)
    half_width = image.shape[1] // 2
    contrast_view = np.hstack(
        (
            cv2.resize(clahe_bgr, (half_width, image.shape[0]), interpolation=cv2.INTER_AREA),
            cv2.resize(inverted_bgr, (image.shape[1] - half_width, image.shape[0]), interpolation=cv2.INTER_AREA),
        )
    )
    candidate_overlay = image.copy()
    for candidate in sorted(candidates, key=lambda item: item.slot):
        x1, y1, x2, y2 = candidate.bbox
        if candidate_has_multichannel_tamper_support(candidate):
            box_color = (0, 0, 255)
        elif candidate_has_rectangle_material_support(candidate, strict=False):
            box_color = (0, 165, 255)
        else:
            box_color = (0, 220, 80)
        cv2.rectangle(candidate_overlay, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
        cv2.putText(
            candidate_overlay,
            f"{candidate.candidate_id}/S{candidate.slot} {candidate.combined_score:.2f}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            candidate_overlay,
            f"{candidate.candidate_id}/S{candidate.slot} {candidate.combined_score:.2f}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            box_color,
            1,
            cv2.LINE_AA,
        )
    slot_map = np.zeros_like(image)
    ordered_slots = sorted(slots, key=lambda item: item.slot)
    cell_width = image.shape[1] // len(ordered_slots)
    for index, slot in enumerate(ordered_slots):
        x1, y1, x2, y2 = slot.bbox
        pad_x, pad_y = 8, 4
        crop = image[max(0, y1 - pad_y) : min(image.shape[0], y2 + pad_y), max(0, x1 - pad_x) : min(image.shape[1], x2 + pad_x)]
        cell_x1 = index * cell_width
        cell_x2 = image.shape[1] if index == len(ordered_slots) - 1 else (index + 1) * cell_width
        target_width, target_height = max(1, cell_x2 - cell_x1 - 4), image.shape[0] - 32
        resized = cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        slot_map[28 : 28 + target_height, cell_x1 + 2 : cell_x1 + 2 + target_width] = resized
        cv2.putText(slot_map, f"C{slot.slot}=S{slot.slot}", (cell_x1 + 4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(slot_map, (cell_x2 - 1, 0), (cell_x2 - 1, image.shape[0] - 1), (80, 80, 80), 1)

    panels = []
    for label, panel in (
        ("CANDIDATES", candidate_overlay),
        ("LEFT-TO-RIGHT SLOT MAP: Cn = Sn", slot_map),
        ("AXIS-ALIGNED LINE SEGMENTS", line_overlay),
        ("LOCAL LAB COLOR RESIDUAL", color_heat),
        ("BRIGHT EDGES=GREEN / DARK SEAMS=RED", bright_dark),
        ("CLAHE / INVERTED CLAHE - VISUAL AID ONLY", contrast_view),
    ):
        labeled = panel.copy()
        cv2.rectangle(labeled, (0, 0), (min(520, image.shape[1] - 1), 25), (0, 0, 0), -1)
        cv2.putText(labeled, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(labeled)
    evidence_sheet = np.vstack(panels)

    delta_e_u8 = np.clip(views["delta_e76_residual"] * (255.0 / 25.0), 0, 255).astype(np.uint8)
    delta_e_heat = cv2.applyColorMap(delta_e_u8, cv2.COLORMAP_INFERNO)
    sobel_u8 = np.clip(views["sobel_magnitude"] * (255.0 / 400.0), 0, 255).astype(np.uint8)
    sobel_heat = cv2.applyColorMap(sobel_u8, cv2.COLORMAP_INFERNO)
    high_frequency_u8 = np.clip(views["high_frequency"] * (255.0 / 35.0), 0, 255).astype(np.uint8)
    high_frequency_heat = cv2.applyColorMap(high_frequency_u8, cv2.COLORMAP_VIRIDIS)

    diagnostic_panels = []
    for label, panel in (
        ("SLOT LOCALIZATION: RED=SUSPICIOUS, ORANGE=REVIEW", candidate_overlay),
        ("PLATE ROW REFERENCE RESIDUAL: DELTA E76 0..12", plate_reference_view),
        ("SOBEL GRADIENT MAGNITUDE: 0..400", sobel_heat),
        ("2PX HIGH-FREQUENCY RESIDUAL: 0..35", high_frequency_heat),
        ("VERTICAL LAB PROFILE: EXPECTED | ACTUAL", vertical_profile_view),
        ("AXIS-ALIGNED BACKGROUND LINE SEGMENTS", line_overlay),
    ):
        labeled = panel.copy()
        cv2.rectangle(labeled, (0, 0), (min(720, image.shape[1] - 1), 25), (0, 0, 0), -1)
        cv2.putText(
            labeled,
            label,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        diagnostic_panels.append(labeled)
    diagnostic_panel = np.vstack(
        [
            np.hstack(diagnostic_panels[0:2]),
            np.hstack(diagnostic_panels[2:4]),
            np.hstack(diagnostic_panels[4:6]),
        ]
    )
    return EvidenceArtifacts(
        bundle=bundle,
        ink_mask=ink_mask,
        screw_mask=screw_mask,
        edge_map=views["canny"],
        line_overlay=line_overlay,
        color_residual=color_heat,
        bright_dark_view=bright_dark,
        contrast_view=contrast_view,
        candidate_overlay=candidate_overlay,
        slot_map_view=slot_map,
        vertical_profile_view=vertical_profile_view,
        plate_reference_view=plate_reference_view,
        diagnostic_panel=diagnostic_panel,
        evidence_sheet=evidence_sheet,
    )
