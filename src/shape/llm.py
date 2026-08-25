"""多模态大模型四角检测；几何变换仍由本地确定性代码完成。"""

from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image

from .geometry import PlateDetection, PlateType, order_corners


COARSE_PROMPT = """你是车牌区域粗定位器。只找图中最主要的一张中国单层汽车号牌。
四点只需包围整块号牌，供下一步裁图使用；宁可略偏外，不要裁掉号牌。
按左上、右上、右下、左下顺序输出，横纵坐标均为当前输入图的归一化整数 0..1000。
plate_type 只能是 green_new_energy 或 blue_standard。只返回 JSON，不要 Markdown：
{"plate_type":"green_new_energy","corners":[[x,y],[x,y],[x,y],[x,y]],"confidence":0.0}"""


PRECISE_PROMPT = """你是车牌金属板外轮廓精标器。输入图已经按粗定位透视展开，号牌接近水平且占画面主体。
目标是号牌金属板（包括其最外侧白色/银色压边）的四个真实角。
必须排除车牌外侧的黑色塑料安装架、保险杠、格栅、阴影和普通水平饰条；也不能只标字符区或彩色底面。
如果画面上下存在黑色带，只标黑色带之间的蓝色或绿色金属号牌；不要把整张输入图的四角直接作为答案。
四点应落在金属牌四条外边缘的交点上，不得人为增加余量；螺钉不改变边界位置。
按左上、右上、右下、左下顺序输出，横纵坐标均为当前输入图的归一化整数 0..1000。
plate_type 只能是 green_new_energy 或 blue_standard。只返回 JSON，不要 Markdown：
{"plate_type":"green_new_energy","corners":[[x,y],[x,y],[x,y],[x,y]],"confidence":0.0}"""


def _encode_for_api(image: np.ndarray, max_side: int = 2048) -> str:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    if max(pil.size) > max_side:
        scale = max_side / max(pil.size)
        pil = pil.resize((round(pil.width * scale), round(pil.height * scale)), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    pil.save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _extract_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"模型没有返回 JSON：{text[:180]}")
        text = text[start : end + 1]
    return json.loads(text)


def _call_corner_model(client: OpenAI, model: str, image: np.ndarray, prompt: str) -> dict:
    height, width = image.shape[:2]
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"当前输入图尺寸为 {width}x{height}，请返回四点。"},
                    {"type": "image_url", "image_url": {"url": _encode_for_api(image)}},
                ],
            },
        ],
    )
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise RuntimeError("模型返回内容为空或不是文本 JSON")
    return _extract_json(content)


def _normalized_corners(payload: dict, width: int, height: int) -> np.ndarray:
    normalized = np.asarray(payload["corners"], dtype=np.float32)
    if normalized.shape != (4, 2) or (normalized < -20).any() or (normalized > 1020).any():
        raise ValueError("模型 corners 必须是 0..1000 范围的 4x2 坐标")
    scale = np.array([width / 1000.0, height / 1000.0], dtype=np.float32)
    return order_corners(normalized * scale)


def _rectify_for_refinement(
    image: np.ndarray,
    corners: np.ndarray,
    output_size: tuple[int, int] = (1200, 400),
) -> tuple[np.ndarray, np.ndarray]:
    """把粗四边形展开成大图，并返回展开图到原图的逆单应矩阵。"""

    canvas_width, canvas_height = output_size
    source = order_corners(corners)
    destination = np.array(
        [[0, 0], [canvas_width - 1, 0], [canvas_width - 1, canvas_height - 1], [0, canvas_height - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, destination)
    patch = cv2.warpPerspective(
        image,
        homography,
        output_size,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return patch, np.linalg.inv(homography)


def detect_plate_llm(
    image: np.ndarray,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> PlateDetection:
    """通过 OpenAI 兼容接口定位四角；不主动读取 local_env.sh。"""

    model = model or os.environ.get("SHAPE_VISION_MODEL") or os.environ.get("OPENAI_MODEL")
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    if not model:
        raise RuntimeError("缺少模型名：设置 SHAPE_VISION_MODEL 或传入 --model")
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url=base_url)
    height, width = image.shape[:2]
    coarse_payload = _call_corner_model(client, model, image, COARSE_PROMPT)
    coarse_corners = _normalized_corners(coarse_payload, width, height)
    refinement_image, inverse_homography = _rectify_for_refinement(image, coarse_corners)
    precise_payload = _call_corner_model(client, model, refinement_image, PRECISE_PROMPT)
    refine_height, refine_width = refinement_image.shape[:2]
    refine_corners = _normalized_corners(precise_payload, refine_width, refine_height)
    corners = cv2.perspectiveTransform(refine_corners.reshape(1, 4, 2), inverse_homography)[0]
    plate_type = PlateType(precise_payload["plate_type"])
    coarse_confidence = float(coarse_payload.get("confidence", 0.5))
    precise_confidence = float(precise_payload.get("confidence", 0.5))
    confidence = float(np.clip(min(coarse_confidence, precise_confidence), 0.0, 1.0))
    return PlateDetection(
        corners=order_corners(corners).tolist(),
        plate_type=plate_type,
        confidence=confidence,
        method=f"vision_llm_two_stage:{model}",
    )
