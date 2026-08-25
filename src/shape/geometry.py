"""两种检测器共用的几何、校验和可视化代码。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


class PlateType(str, Enum):
    GREEN_NEW_ENERGY = "green_new_energy"
    BLUE_STANDARD = "blue_standard"


# 2 px/mm。外圈余量约为号牌宽高的 1.7%，号牌理论占图面积均 > 93%。
PLATE_BODY_PX: dict[PlateType, tuple[int, int]] = {
    PlateType.GREEN_NEW_ENERGY: (960, 280),  # 480 mm x 140 mm
    PlateType.BLUE_STANDARD: (880, 280),  # 440 mm x 140 mm
}
MARGIN_PX: dict[PlateType, tuple[int, int]] = {
    PlateType.GREEN_NEW_ENERGY: (16, 5),
    PlateType.BLUE_STANDARD: (14, 5),
}


@dataclass(frozen=True)
class PlateDetection:
    """四点顺序固定为左上、右上、右下、左下。"""

    corners: list[list[float]]
    plate_type: PlateType
    confidence: float
    method: str


def read_image(path: str | Path) -> np.ndarray:
    """支持非 ASCII 路径，并统一处理 EXIF 旋转。"""

    from PIL import Image, ImageOps

    with Image.open(path) as source:
        rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_image(path: str | Path, image: np.ndarray, quality: int = 95) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if suffix in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise ValueError(f"无法编码图像：{path}")
    encoded.tofile(path)


def order_corners(points: Iterable[Iterable[float]]) -> np.ndarray:
    """将凸四边形稳定排序为 TL, TR, BR, BL。"""

    pts = np.asarray(list(points), dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        raise ValueError("corners 必须是 4x2 的有限数值")

    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    cyclic = pts[np.argsort(angles)]
    start = int(np.argmin(cyclic[:, 0] + cyclic[:, 1]))
    cyclic = np.roll(cyclic, -start, axis=0)
    # 图像坐标下 TL->TR->BR->BL 应为顺时针；第二点应比末点更靠右。
    if cyclic[1, 0] < cyclic[-1, 0]:
        cyclic = cyclic[[0, 3, 2, 1]]
    return cyclic.astype(np.float32)


def polygon_area(points: np.ndarray) -> float:
    return float(abs(cv2.contourArea(points.astype(np.float32))))


def validate_corners(points: np.ndarray, image_shape: tuple[int, ...]) -> None:
    points = order_corners(points)
    height, width = image_shape[:2]
    if not cv2.isContourConvex(points.astype(np.int32)):
        raise ValueError("四角不是凸四边形")
    area_ratio = polygon_area(points) / float(width * height)
    if not 0.00005 <= area_ratio <= 0.98:
        raise ValueError(f"四边形面积比例异常：{area_ratio:.6f}")
    tolerance = max(width, height) * 0.03
    if (
        (points[:, 0] < -tolerance).any()
        or (points[:, 0] > width - 1 + tolerance).any()
        or (points[:, 1] < -tolerance).any()
        or (points[:, 1] > height - 1 + tolerance).any()
    ):
        raise ValueError("四角明显超出图像范围")


def expand_quad(points: np.ndarray, fraction: float = 0.018) -> np.ndarray:
    """从色块边缘轻微外扩，以覆盖金属压边/窄框。"""

    points = order_corners(points)
    center = points.mean(axis=0)
    return center + (points - center) * (1.0 + 2.0 * fraction)


def output_geometry(plate_type: PlateType) -> tuple[int, int, np.ndarray, float]:
    body_w, body_h = PLATE_BODY_PX[plate_type]
    margin_x, margin_y = MARGIN_PX[plate_type]
    canvas_w, canvas_h = body_w + 2 * margin_x, body_h + 2 * margin_y
    destination = np.array(
        [
            [margin_x, margin_y],
            [margin_x + body_w - 1, margin_y],
            [margin_x + body_w - 1, margin_y + body_h - 1],
            [margin_x, margin_y + body_h - 1],
        ],
        dtype=np.float32,
    )
    occupancy = body_w * body_h / float(canvas_w * canvas_h)
    return canvas_w, canvas_h, destination, occupancy


def rectify_plate(image: np.ndarray, detection: PlateDetection) -> tuple[np.ndarray, np.ndarray]:
    source = order_corners(detection.corners)
    validate_corners(source, image.shape)
    canvas_w, canvas_h, destination, _ = output_geometry(detection.plate_type)
    homography = cv2.getPerspectiveTransform(source, destination)
    rectified = cv2.warpPerspective(
        image,
        homography,
        (canvas_w, canvas_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rectified, homography


def _annotation_scale(image: np.ndarray) -> tuple[int, float]:
    height, width = image.shape[:2]
    base = max(height, width)
    return max(5, int(round(base / 300))), max(0.7, base / 2200.0)


def draw_points(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    result = image.copy()
    radius, font_scale = _annotation_scale(image)
    labels = ("TL", "TR", "BR", "BL")
    colors = ((0, 0, 255), (0, 200, 0), (255, 80, 0), (0, 180, 255))
    for label, color, point in zip(labels, colors, order_corners(corners), strict=True):
        x, y = np.rint(point).astype(int)
        cv2.circle(result, (x, y), radius + 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(result, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.putText(
            result,
            label,
            (x + radius + 5, y - radius - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(2, radius // 2 + 2),
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            label,
            (x + radius + 5, y - radius - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, radius // 2),
            cv2.LINE_AA,
        )
    return result


def draw_quad(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    result = image.copy()
    radius, _ = _annotation_scale(image)
    poly = np.rint(order_corners(corners)).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(result, [poly], True, (0, 255, 255), max(3, radius // 2), cv2.LINE_AA)
    return result


def write_outputs(
    image: np.ndarray,
    detection: PlateDetection,
    output_dir: str | Path,
    stem: str,
) -> dict[str, Path]:
    output_dir = Path(output_dir) / stem
    output_dir.mkdir(parents=True, exist_ok=True)
    corners = order_corners(detection.corners)
    rectified, homography = rectify_plate(image, detection)
    _, _, _, occupancy = output_geometry(detection.plate_type)

    paths = {
        "points": output_dir / "01_points.jpg",
        "quad": output_dir / "02_quad.jpg",
        "rectified": output_dir / "03_rectified.jpg",
        "metadata": output_dir / "metadata.json",
    }
    write_image(paths["points"], draw_points(image, corners))
    write_image(paths["quad"], draw_quad(image, corners))
    write_image(paths["rectified"], rectified)
    metadata = asdict(detection)
    metadata["plate_type"] = detection.plate_type.value
    metadata["confidence_note"] = "检测器内部得分，未经人工真值校准，不能直接解释为正确概率"
    metadata["corners"] = corners.round(3).tolist()
    metadata["homography"] = homography.round(10).tolist()
    metadata["inverse_homography"] = np.linalg.inv(homography).round(10).tolist()
    metadata["output_size"] = [int(rectified.shape[1]), int(rectified.shape[0])]
    metadata["nominal_plate_occupancy"] = round(occupancy, 6)
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
