"""本地轻量四关键点模型：ONNX Runtime 推理，不调用云端 API。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .geometry import PlateDetection, PlateType, order_corners


DEFAULT_MODEL = Path(__file__).with_name("models") / "yolov7-lite-t-plate-kpt.onnx"
INPUT_SIZE = 640


def _letterbox(image: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_width, new_height = int(width * ratio), int(height * ratio)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        size - new_height - top,
        left,
        size - new_width - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor[None]), ratio, left, top


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    result = boxes.copy()
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[index, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[index, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[index, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[index, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_a = (boxes[index, 2] - boxes[index, 0]) * (boxes[index, 3] - boxes[index, 1])
        area_b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = intersection / np.maximum(area_a + area_b - intersection, 1e-6)
        order = rest[iou <= threshold]
    return keep


def _postprocess(
    prediction: np.ndarray,
    ratio: float,
    left: int,
    top: int,
    confidence: float,
    iou_threshold: float,
) -> list[tuple[float, np.ndarray]]:
    rows = np.asarray(prediction).squeeze(0)
    rows = rows[rows[:, 4] >= confidence]
    if not len(rows):
        return []

    class_scores = rows[:, 5:7] * rows[:, 4:5]
    scores = class_scores.max(axis=1)
    selected = scores >= confidence
    rows, scores = rows[selected], scores[selected]
    if not len(rows):
        return []

    boxes = _xywh_to_xyxy(rows[:, :4])
    keypoint_columns = [7, 8, 10, 11, 13, 14, 16, 17]
    keypoints = rows[:, keypoint_columns].reshape(-1, 4, 2).copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / ratio
    keypoints[:, :, 0] = (keypoints[:, :, 0] - left) / ratio
    keypoints[:, :, 1] = (keypoints[:, :, 1] - top) / ratio

    keep = _nms(boxes, scores, iou_threshold)
    return [(float(scores[index]), order_corners(keypoints[index])) for index in keep]


def _plate_type_from_pixels(image: np.ndarray, quad: np.ndarray) -> PlateType:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    region = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(region, np.rint(order_corners(quad)).astype(np.int32), 255)
    selected = (region > 0) & (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 40)
    hues = hsv[:, :, 0][selected]
    if hues.size >= 40:
        return PlateType.BLUE_STANDARD if float(np.median(hues)) >= 90.0 else PlateType.GREEN_NEW_ENERGY

    ordered = order_corners(quad)
    width = max(np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3]))
    height = max(np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1]))
    return PlateType.GREEN_NEW_ENERGY if width / max(height, 1.0) >= 3.28 else PlateType.BLUE_STANDARD


@lru_cache(maxsize=2)
def _session(model_path: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("本地小模型需要 onnxruntime，请安装 src/shape/requirements.txt") from exc
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])


def detect_plate_local(
    image: np.ndarray,
    model_path: str | Path = DEFAULT_MODEL,
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
) -> PlateDetection:
    """用约 0.25M 参数的本地模型检测最可信单层车牌及四角。"""

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到本地四点模型：{model_path}")
    tensor, ratio, left, top = _letterbox(image)
    session = _session(str(model_path.resolve()))
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    detections = _postprocess(output, ratio, left, top, confidence, iou_threshold)
    if not detections:
        raise RuntimeError(f"本地四点模型未检测到置信度 >= {confidence:.2f} 的车牌")

    score, quad = max(detections, key=lambda item: item[0])
    plate_type = _plate_type_from_pixels(image, quad)
    height, width = image.shape[:2]
    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
    return PlateDetection(
        corners=order_corners(quad).tolist(),
        plate_type=plate_type,
        confidence=float(np.clip(score, 0.0, 0.999)),
        method="local_onnx_yolov7_lite_t_kpt",
    )
