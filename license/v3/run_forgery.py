"""v3 车牌伪造筛查。

流程：原图 -> 本地多候选定位/结构校验 -> 裁剪车牌或整图回退 -> SWE 默认 Agent 流程 -> JSON 结果。

用法：
    python -m license.v3.run_forgery data/raw/images/example.jpg
    python -m license.v3.run_forgery image.jpg --plate-box 120,340,680,160
"""

import argparse
import asyncio
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWEVISSION_ROOT = PROJECT_ROOT / "SWEVISSION"
if not (SWEVISSION_ROOT / "swe_vision").is_dir():
    raise ImportError(f"找不到项目内的 SWEVISSION/swe_vision：{SWEVISSION_ROOT}")
if str(SWEVISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SWEVISSION_ROOT))

from swe_vision import VLMToolCallAgent


ROOT = Path(__file__).resolve().parent
PROMPT_PATH = ROOT / "forgery_prompt.md"
SWE_TRAJECTORY_ROOT = PROJECT_ROOT / "runtime" / "license-v3" / "trajectories" / "run"
CODE_VERSION = "v3.0.3"
SWE_MAX_ITERATIONS = 10
QWEN3_VL_PLUS_PRICING_URL = "https://help.aliyun.com/zh/model-studio/qwen3-vl-plus"


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def project_path(value: str | Path) -> Path:
    """把相对路径固定到项目根，避免 /mnt/g 瞬时 getcwd 失败。"""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_prompt(plate_width: int, plate_height: int, *, input_is_plate_crop: bool = True) -> str:
    """将本次模型输入的实际尺寸和坐标口径加入提示词。"""
    if input_is_plate_crop:
        input_scope = (
            f"本次输入车牌裁剪图的实际尺寸为 {plate_width} × {plate_height} 像素。"
            "所有 `bbox` 必须以该图左上角为原点、以该实际像素尺寸为坐标系；"
            "不得假定其他图像尺寸。"
        )
    else:
        input_scope = (
            "【原图模式覆盖规则，优先于前文关于“裁剪图”的表述】"
            f"本次输入是未裁剪原图，实际尺寸为 {plate_width} × {plate_height} 像素。"
            "先在原图中定位车牌，再检查该车牌。此模式的 crop_status=complete 表示原图内存在一块完整可见的车牌；"
            "原图本身未裁剪不是 uncertain 或 incomplete 的理由。仅在无法定位车牌，或定位到的车牌确有边框/字符缺失时，才返回 uncertain 或 incomplete。"
            "优先用 execute_code 做定位、放大和局部分析；不得把未验证的固定坐标当作定位证据。"
            "所有 bbox 必须以原图左上角为原点，并覆盖原图中对应的异常区域。"
        )
    return f"{load_prompt()}\n\n{input_scope}"


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_box(value: str) -> Tuple[int, int, int, int]:
    try:
        x, y, width, height = (int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("格式应为 x,y,width,height") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width 和 height 必须大于 0")
    return x, y, width, height


def _plate_colored(hue: int, saturation: int, value: int) -> bool:
    """复现 v1 的 HSV 掩码思路；Pillow 的色相范围是 0..255。"""
    if saturation < 80 or value < 50:
        return False
    # v1 绿色阈值为 OpenCV H=40..80，换算为 Pillow H≈57..113。
    green = 57 <= hue <= 113
    blue = 135 <= hue <= 190
    return blue or green


def _padded_box(left: int, top: int, right: int, bottom: int, work_w: int, work_h: int) -> tuple[int, int, int, int]:
    """给颜色连通域加少量边距，保留车牌边框和字符。"""
    # 横向多留边：颜色掩码可能被字符、反光或固定螺丝切断，10% 容易截掉首尾字符。
    padding_x = max(2, round((right - left + 1) * 0.18))
    padding_y = max(2, round((bottom - top + 1) * 0.12))
    return (max(0, left - padding_x), max(0, top - padding_y), min(work_w, right + padding_x + 1), min(work_h, bottom + padding_y + 1))


def _structure_score(work: Image.Image, box: tuple[int, int, int, int]) -> tuple[float, dict[str, float]]:
    """仅按候选的几何、底色、边界、字符状笔画评分；不调用模型或 OCR。"""
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    if width < 2 or height < 2:
        return 0.0, {"total": 0.0}
    ratio = width / height
    area_fraction = (width * height) / (work.width * work.height)
    crop = work.crop(box).resize((min(360, max(40, width)), min(120, max(20, height))), Image.Resampling.BILINEAR)
    hsv = crop.convert("HSV")
    color_coverage = sum(_plate_colored(*pixel) for pixel in hsv.getdata()) / (crop.width * crop.height)

    # 用高对比竖向笔画统计字符结构，不假定字符必为黑色或白色。
    gray = crop.convert("L")
    pixels = gray.load()
    inner_top, inner_bottom = max(1, round(crop.height * 0.15)), min(crop.height - 1, round(crop.height * 0.85))
    active_columns = []
    for x in range(1, crop.width):
        changes = sum(abs(pixels[x, y] - pixels[x - 1, y]) >= 35 for y in range(inner_top, inner_bottom))
        active_columns.append(changes / max(1, inner_bottom - inner_top) >= 0.16)
    stroke_groups, in_group = 0, False
    for active in active_columns:
        if active and not in_group:
            stroke_groups += 1
        in_group = active
    # 一个汉字/数字本身可贡献多条竖笔画，不能把较多笔画误判为非车牌。
    glyph_score = min(1.0, stroke_groups / 5.0) if stroke_groups >= 3 else 0.15

    # 数据集中的绿牌为深色字符。与“边缘很多”的玻璃反光不同，真实字符会在
    # 牌照中部形成约 5--10 个横向分离、且有足够高度的深色字符带。
    dark_columns = []
    dark_pixels = 0
    inner_height = max(1, inner_bottom - inner_top)
    for x in range(crop.width):
        dark_count = sum(pixels[x, y] <= 95 for y in range(inner_top, inner_bottom))
        dark_pixels += dark_count
        dark_columns.append(dark_count / inner_height >= 0.22)
    dark_groups = 0
    last_active = -10
    for x, active in enumerate(dark_columns):
        if active:
            if x - last_active > 7:
                dark_groups += 1
            last_active = x
    dark_coverage = dark_pixels / max(1, crop.width * inner_height)
    char_band_score = 1.0 if 5 <= dark_groups <= 10 and 0.025 <= dark_coverage <= 0.42 else 0.35 if 3 <= dark_groups <= 14 else 0.0

    # 上下长边的亮度跃迁是矩形牌照边框的弱证据。
    horizontal_diffs = []
    for y in (max(1, round(crop.height * 0.10)), min(crop.height - 1, round(crop.height * 0.90))):
        horizontal_diffs.extend(abs(pixels[x, y] - pixels[x, y - 1]) for x in range(1, crop.width))
    edge_score = min(1.0, (sum(value >= 25 for value in horizontal_diffs) / max(1, len(horizontal_diffs))) / 0.20)

    ratio_score = math.exp(-abs(math.log(ratio / 3.1))) if ratio > 0 else 0.0
    coverage_score = max(0.0, 1.0 - abs(color_coverage - 0.62) / 0.62)
    area_score = 1.0 if 0.00015 <= area_fraction <= 0.045 else 0.45 if area_fraction <= 0.10 else 0.0
    touches_top = top <= max(2, round(work.height * 0.02))
    touches_side = left <= 1 or right >= work.width - 1
    border_score = 0.15 if touches_top else 0.55 if touches_side else 1.0
    total = 0.23 * ratio_score + 0.14 * coverage_score + 0.12 * glyph_score + 0.22 * char_band_score + 0.11 * edge_score + 0.11 * area_score + 0.07 * border_score
    return total, {
        "total": round(total, 4), "ratio": round(ratio, 3), "ratio_score": round(ratio_score, 4),
        "color_coverage": round(color_coverage, 4), "coverage_score": round(coverage_score, 4),
        "stroke_groups": float(stroke_groups), "glyph_score": round(glyph_score, 4),
        "dark_character_groups": float(dark_groups), "dark_character_coverage": round(dark_coverage, 4), "character_band_score": round(char_band_score, 4),
        "edge_score": round(edge_score, 4), "area_fraction": round(area_fraction, 6),
        "area_score": round(area_score, 4), "border_score": round(border_score, 4),
    }


def find_plate_localization(image: Image.Image, max_side: int = 1920) -> tuple[Optional[Tuple[int, int, int, int]], dict[str, Any]]:
    """选择通过结构校验的 HSV 车牌候选；无可信候选时返回 None 以回退整图。"""
    source_w, source_h = image.size
    scale = min(1.0, max_side / max(source_w, source_h))
    work_w = max(1, round(source_w * scale))
    work_h = max(1, round(source_h * scale))
    work = image.convert("RGB").resize((work_w, work_h), Image.Resampling.BILINEAR)
    hsv = work.convert("HSV")
    pixels = hsv.load()
    raw_mask = bytearray(work_w * work_h)
    for y in range(work_h):
        offset = y * work_w
        for x in range(work_w):
            if _plate_colored(*pixels[x, y]):
                raw_mask[offset + x] = 255

    # 复现 v1 的 5×5 闭运算后开运算：连接牌照底色并去除孤立噪点。
    mask_image = Image.frombytes("L", (work_w, work_h), bytes(raw_mask))
    mask_image = mask_image.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    mask_image = mask_image.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    mask = mask_image.tobytes()

    visited = bytearray(work_w * work_h)
    candidates: list[dict[str, Any]] = []
    for start in range(work_w * work_h):
        if not mask[start] or visited[start]:
            continue
        queue = deque([start])
        visited[start] = 1
        count = 0
        min_x = max_x = start % work_w
        min_y = max_y = start // work_w
        while queue:
            point = queue.popleft()
            x, y = point % work_w, point // work_w
            count += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < work_w and 0 <= ny < work_h:
                    neighbor = ny * work_w + nx
                    if mask[neighbor] and not visited[neighbor]:
                        visited[neighbor] = 1
                        queue.append(neighbor)

        width, height = max_x - min_x + 1, max_y - min_y + 1
        area = width * height
        ratio = width / height
        coverage = count / area
        if area < 250 or not 1.7 <= ratio <= 8.5 or coverage < 0.18:
            continue
        candidate_box = _padded_box(min_x, min_y, max_x, max_y, work_w, work_h)
        score, details = _structure_score(work, candidate_box)
        candidates.append({"work_box": candidate_box, "score": score, "details": details})

    if not candidates:
        return None, {"source": "fallback_full_image", "reason": "no_hsv_candidate", "candidate_count": 0}
    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)

    # 宁可回退整图，也不把“有一点蓝绿和边缘”的区域当成车牌。阈值是对
    # 本数据集绿牌的保守完整性检查：候选必须有合理比例/尺度，以及足够的
    # 深色字符带；任一条件不足时不裁剪。
    def accepted(candidate: dict[str, Any]) -> bool:
        details = candidate["details"]
        return (
            candidate["score"] >= 0.62
            and 2.1 <= details["ratio"] <= 5.0
            and 0.001 <= details["area_fraction"] <= 0.035
            and details["dark_character_groups"] >= 4
            and 0.08 <= details["dark_character_coverage"] <= 0.65
        )

    best = candidates[0]
    best_accepted = next((candidate for candidate in candidates if accepted(candidate)), None)
    diagnostics = {
        "source": "local_hsv_structural", "candidate_count": len(candidates), "acceptance_threshold": 0.62,
        "structural_requirements": {
            "ratio": [2.1, 5.0], "area_fraction": [0.001, 0.035],
            "dark_character_groups_min": 4, "dark_character_coverage": [0.08, 0.65],
        },
        "best_score": round(best["score"], 4),
        "top_candidates": [
            {"score": round(candidate["score"], 4), "work_box": list(candidate["work_box"]), **candidate["details"]}
            for candidate in candidates[:5]
        ],
    }
    if best_accepted is None:
        diagnostics.update({"source": "fallback_full_image", "reason": "no_candidate_passed_structural_validation"})
        return None, diagnostics
    diagnostics["selected_score"] = round(best_accepted["score"], 4)
    left, top, right, bottom = best_accepted["work_box"]
    return (
        round(left / scale),
        round(top / scale),
        round((right - left) / scale),
        round((bottom - top) / scale),
    ), diagnostics


def find_plate_box(image: Image.Image, max_side: int = 1920) -> Optional[Tuple[int, int, int, int]]:
    """兼容旧调用：仅返回经 v3 结构校验后的车牌框。"""
    box, _ = find_plate_localization(image, max_side=max_side)
    return box


def crop_plate(image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    x, y, width, height = box
    image_w, image_h = image.size
    left, top = max(0, x), max(0, y)
    right, bottom = min(image_w, x + width), min(image_h, y + height)
    if left >= right or top >= bottom:
        raise ValueError("车牌框不在图片范围内")
    return image.crop((left, top, right, bottom))


class OutputValidationError(ValueError):
    """模型最终输出不符合 v2 的机器可解析结果协议。"""


def _validate_evidence(value: object, path: str, errors: list[str], *, required: bool = True, max_length: int = 30) -> None:
    if not isinstance(value, list) or len(value) > 1 or (required and len(value) != 1) or any(not isinstance(item, str) or not item.strip() or len(item) > max_length for item in value):
        requirement = "恰好一条" if required else "至多一条"
        errors.append(f"{path} 必须是{requirement}、每条不超过{max_length}字符的非空字符串列表")


def extract_json_object(text: str) -> dict:
    """从模型回复或 Markdown 代码块中提取一个完整的 JSON 根对象。"""
    cleaned = text.strip()
    decoder = json.JSONDecoder()
    candidates = [cleaned]
    candidates.extend(match.group(1).strip() for match in re.finditer(
        r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL,
    ))

    decoded_objects: list[dict] = []
    for candidate in candidates:
        starts = [0] if candidate.startswith("{") else []
        starts.extend(match.start() for match in re.finditer(r"\{", candidate))
        for start in dict.fromkeys(starts):
            try:
                value, _ = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                # 优先选择 v2 根对象，避免回复中的示例或嵌套 JSON 被误取。
                if {"crop_check", "characters", "anomaly_regions", "overall"}.issubset(value):
                    return value
                decoded_objects.append(value)
    if decoded_objects:
        return decoded_objects[0]
    raise OutputValidationError("未能从模型答复中提取可解析的 JSON 对象")


def _normalize_evidence(value: object, max_length: int) -> object:
    """修复不改变判断语义的证据格式瑕疵。

    模型偶尔把单条证据写成字符串、写出多条，或略超过协议长度。只保留
    第一条已给出的证据并截断；绝不凭程序补造证据。
    """
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = [item for item in value if isinstance(item, str) and item.strip()]
    else:
        return value
    return [values[0].strip()[:max_length]] if values else []


def normalize_protocol_format(result: dict) -> dict:
    """在严格校验前规范化无歧义的 JSON 表示，保留模型的检测结论。"""
    crop_check = result.get("crop_check")
    if isinstance(crop_check, dict):
        crop_check["visual_evidence"] = _normalize_evidence(crop_check.get("visual_evidence"), 80)

    characters = result.get("characters")
    if isinstance(characters, list):
        for character in characters:
            if not isinstance(character, dict):
                continue
            if character.get("status") == "normal":
                # 协议规定正常字符不写理由；丢弃多余说明不改变 normal 判断。
                character["visual_evidence"] = []
            elif character.get("status") in {"suspected", "unreadable"}:
                character["visual_evidence"] = _normalize_evidence(character.get("visual_evidence"), 30)

    regions = result.get("anomaly_regions")
    if isinstance(regions, list):
        for region in regions:
            if isinstance(region, dict):
                region["visual_evidence"] = _normalize_evidence(region.get("visual_evidence"), 30)
    return result


def parse_and_validate_result(text: str, plate_width: int, plate_height: int) -> dict:
    """提取并严格验证最终 JSON、字段约束及异常框的裁剪图像素坐标。"""
    errors: list[str] = []
    result = normalize_protocol_format(extract_json_object(text))

    required_root = {"crop_check", "characters", "anomaly_regions", "overall"}
    if set(result) != required_root:
        errors.append("根节点字段必须且只能为 crop_check、characters、anomaly_regions、overall")
    crop_check = result.get("crop_check")
    if not isinstance(crop_check, dict) or set(crop_check) != {"crop_status", "visual_evidence"}:
        errors.append("crop_check 字段不符合规范")
        crop_status = None
    else:
        crop_status = crop_check.get("crop_status")
        if crop_status not in {"complete", "incomplete", "uncertain"}:
            errors.append("crop_status 取值无效")
        _validate_evidence(crop_check.get("visual_evidence"), "crop_check.visual_evidence", errors, max_length=80)

    characters = result.get("characters")
    if not isinstance(characters, list):
        errors.append("characters 必须是列表")
        characters = []
    positions: list[int] = []
    suspected_positions: set[int] = set()
    character_fields = {"position", "character", "status", "forgery_method", "visual_evidence", "confidence"}
    for index, character in enumerate(characters):
        path = f"characters[{index}]"
        if not isinstance(character, dict) or set(character) != character_fields:
            errors.append(f"{path} 字段不符合规范")
            continue
        position = character.get("position")
        status = character.get("status")
        method = character.get("forgery_method")
        if not isinstance(position, int) or position < 1:
            errors.append(f"{path}.position 必须为正整数")
        else:
            positions.append(position)
        if not isinstance(character.get("character"), str) or not character["character"].strip():
            errors.append(f"{path}.character 必须为非空字符串")
        if status not in {"normal", "suspected", "unreadable"}:
            errors.append(f"{path}.status 取值无效")
        if method not in {"none", "character_sticker", "stroke_sticker", "overpaint", "abrasion_reengraving", "printed_overlay", "font_or_manufacturing_anomaly", "other", "unknown"}:
            errors.append(f"{path}.forgery_method 取值无效")
        evidence = character.get("visual_evidence")
        if status == "normal":
            if method != "none":
                errors.append(f"{path} normal 状态必须使用 forgery_method=none")
            if evidence != []:
                errors.append(f"{path} normal 状态的 visual_evidence 必须为空列表")
        if status == "suspected":
            if method in {None, "none"}:
                errors.append(f"{path} suspected 状态必须给出伪造方式")
            elif isinstance(position, int):
                suspected_positions.add(position)
        if character.get("confidence") not in {"high", "medium", "low"}:
            errors.append(f"{path}.confidence 取值无效")
        if status != "normal":
            _validate_evidence(evidence, f"{path}.visual_evidence", errors)
    if positions and sorted(positions) != list(range(1, len(positions) + 1)):
        errors.append("characters.position 必须从1连续编号且不重复")

    regions = result.get("anomaly_regions")
    if not isinstance(regions, list):
        errors.append("anomaly_regions 必须是列表")
        regions = []
    region_positions: set[int] = set()
    region_fields = {"character_positions", "bbox", "label", "visual_evidence"}
    for index, region in enumerate(regions):
        path = f"anomaly_regions[{index}]"
        if not isinstance(region, dict) or set(region) != region_fields:
            errors.append(f"{path} 字段不符合规范")
            continue
        linked = region.get("character_positions")
        if not isinstance(linked, list) or not linked or any(not isinstance(position, int) or position < 1 for position in linked):
            errors.append(f"{path}.character_positions 必须是正整数列表")
        else:
            region_positions.update(linked)
        bbox = region.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or any(not isinstance(value, (int, float)) for value in bbox):
            errors.append(f"{path}.bbox 必须是四个数值")
        else:
            x, y, width, height = bbox
            if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > plate_width or y + height > plate_height:
                errors.append(f"{path}.bbox 超出 {plate_width}×{plate_height} 裁剪图范围")
        if not isinstance(region.get("label"), str) or not region["label"].strip():
            errors.append(f"{path}.label 必须为非空字符串")
        _validate_evidence(region.get("visual_evidence"), f"{path}.visual_evidence", errors)

    overall = result.get("overall")
    if crop_status in {"incomplete", "uncertain"}:
        if characters or regions or overall != "unreadable":
            errors.append("不完整或不确定裁剪必须返回空列表和 overall=unreadable")
    elif crop_status == "complete":
        if not characters:
            errors.append("完整裁剪必须逐字符返回 characters")
        if overall == "normal" and (suspected_positions or regions):
            errors.append("存在可疑字符或异常框时 overall 必须为 suspected_forgery")
        if overall == "suspected_forgery" and (not suspected_positions or not regions):
            errors.append("suspected_forgery 必须同时含可疑字符和异常框")
        if overall not in {"normal", "suspected_forgery"}:
            errors.append("完整裁剪的 overall 取值无效")
    if suspected_positions != region_positions:
        errors.append("可疑字符位置必须与异常框 character_positions 一致")
    if errors:
        raise OutputValidationError("；".join(errors))
    return result


def merge_usage(*usages: dict) -> dict:
    return {
        "input_tokens": sum(usage.get("input_tokens") or 0 for usage in usages),
        "output_tokens": sum(usage.get("output_tokens") or 0 for usage in usages),
        "total_tokens": sum(usage.get("total_tokens") or 0 for usage in usages),
        "api_calls": sum(usage.get("api_calls") or 0 for usage in usages),
        "calls": [call for usage in usages for call in usage.get("calls", [])],
    }


def usage_snapshot(usage: dict) -> dict:
    """复制累计 usage，供复用 Agent 时计算当前图片的增量。"""
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "api_calls": usage.get("api_calls", 0),
        "calls_count": len(usage.get("calls", [])),
    }


def usage_delta(usage: dict, before: dict) -> dict:
    calls = usage.get("calls", [])[before["calls_count"]:]
    return {
        "input_tokens": usage.get("input_tokens", 0) - before["input_tokens"],
        "output_tokens": usage.get("output_tokens", 0) - before["output_tokens"],
        "total_tokens": usage.get("total_tokens", 0) - before["total_tokens"],
        "api_calls": usage.get("api_calls", 0) - before["api_calls"],
        "calls": calls,
    }


async def call_model_async(
    prompt: str,
    plate_path: Path,
    model: str,
    agent: Optional[VLMToolCallAgent] = None,
) -> tuple[str, dict]:
    """运行一张图；传入 agent 时复用其 Docker 内核，但每图重置消息对话。"""
    own_agent = agent is None
    if own_agent:
        agent = create_agent(model, prompt)
    assert agent is not None
    # 同一个批次复用 Agent 时，run() 会重新读取该值建立本图的 system message。
    agent.system_prompt = prompt
    before = usage_snapshot(agent.token_usage)
    try:
        answer = await agent.run(
            "请按系统提示词规定的 JSON 格式，检查这张已裁剪的完整车牌图片。",
            [str(plate_path)],
        )
        return answer, usage_delta(agent.token_usage, before)
    finally:
        if own_agent:
            await agent.cleanup()


def create_agent(model: str, prompt: str) -> VLMToolCallAgent:
    """创建 v3 统一配置的 SWE Agent。"""
    return VLMToolCallAgent(
            system_prompt=prompt,
            model=model,
            max_iterations=SWE_MAX_ITERATIONS,
            reasoning=False,
            finish_only_final_iteration=True,
            finalization_grace_rounds=1,
            include_budget_feedback=True,
            save_trajectory=str(SWE_TRAJECTORY_ROOT),
            verbose=True,
    )


def call_model(prompt: str, plate_path: Path, model: str) -> tuple[str, dict]:
    return asyncio.run(call_model_async(prompt, plate_path, model))


def calculate_cost(model: str, usage: dict) -> dict:
    """按 API 实际返回的 token usage 估算成本，费率单位为元/百万 token。"""
    call_usages = usage.get("calls") or [usage]
    if any(call.get("input_tokens") is None or call.get("output_tokens") is None for call in call_usages):
        return {"currency": "CNY", "cost": None, "reason": "API 未返回 token usage"}
    if not model.startswith("qwen3-vl-plus"):
        return {"currency": "CNY", "cost": None, "reason": "未配置该模型的费率"}
    cost = 0.0
    rates = []
    # 多轮时按每一次 API 调用分别计价，不能用累计 token 跨档判断。
    for call in call_usages:
        input_tokens = call["input_tokens"]
        output_tokens = call["output_tokens"]
        if input_tokens <= 32_000:
            input_rate, output_rate = 1.0, 10.0
        elif input_tokens <= 128_000:
            input_rate, output_rate = 1.5, 15.0
        else:
            input_rate, output_rate = 3.0, 30.0
        cost += input_tokens * input_rate / 1_000_000 + output_tokens * output_rate / 1_000_000
        rates.append({"input_price_per_million_tokens": input_rate, "output_price_per_million_tokens": output_rate})
    return {
        "currency": "CNY",
        "cost": round(cost, 8),
        "per_api_call_rates": rates,
        "price_source": QWEN3_VL_PLUS_PRICING_URL,
    }


def draw_anomaly_boxes(plate: Image.Image, result: dict) -> Image.Image:
    """在车牌裁剪图上绘制模型返回的异常矩形；非法坐标将被忽略。"""
    annotated = plate.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    # 仅画边框，不填充；线宽随车牌大小变化并限制为 2–5 像素。
    line_width = max(2, min(5, round(min(plate.size) / 180)))
    for region in result.get("anomaly_regions", []):
        if not isinstance(region, dict):
            continue
        bbox = region.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x, y, width, height = (round(float(value)) for value in bbox)
        except (TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        left, top = max(0, x), max(0, y)
        right, bottom = min(plate.width - 1, x + width), min(plate.height - 1, y + height)
        if left < right and top < bottom:
            draw.rectangle((left, top, right, bottom), outline="red", width=line_width)
    return annotated


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def process_image_async(
    source_path: Path,
    model: str,
    output_dir: Path,
    answers_dir: Optional[Path] = None,
    plate_box: Optional[Tuple[int, int, int, int]] = None,
    use_plate_crop: bool = True,
    batch_scope: Optional[dict] = None,
    run_dir_override: Optional[Path] = None,
    run_timestamp: Optional[str] = None,
    agent: Optional[VLMToolCallAgent] = None,
) -> dict:
    """处理一张原图，并返回写入 answer.json 的内容及输出目录。"""
    started = time.perf_counter()
    source_path = project_path(source_path)
    output_dir = project_path(output_dir)
    answers_dir = project_path(answers_dir) if answers_dir else None
    if not source_path.is_file():
        raise FileNotFoundError(f"图片不存在：{source_path}")
    with Image.open(source_path) as source:
        image = source.convert("RGB")

    model_uses_plate_crop = use_plate_crop
    if not use_plate_crop:
        box = (0, 0, image.width, image.height)
        localization = {"source": "disabled_full_image"}
        plate = image.copy()
    elif plate_box:
        box = plate_box
        localization = {"source": "manual_plate_box"}
    else:
        box, localization = find_plate_localization(image)
        if box is None:
            box = (0, 0, image.width, image.height)
            plate = image.copy()
            model_uses_plate_crop = False
        else:
            plate = crop_plate(image, box)
    if model_uses_plate_crop and plate_box:
        plate = crop_plate(image, box)

    timestamp = run_timestamp or datetime.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    stem = source_path.stem
    run_dir = run_dir_override or output_dir / stem / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    final_dir = answers_dir / f"{stem}_{timestamp}" if answers_dir else None
    if final_dir:
        final_dir.mkdir(parents=True, exist_ok=False)
    plate_path = run_dir / "plate.jpg"
    annotated_path = run_dir / "plate_annotated.jpg"
    result_path = run_dir / "result.json"
    plate.save(plate_path, format="JPEG", quality=95)

    prompt = build_prompt(plate.width, plate.height, input_is_plate_crop=model_uses_plate_crop)
    own_agent = agent is None
    active_agent = agent or create_agent(model, prompt)
    validation = {"initial_valid": False, "retry_performed": False, "final_valid": False, "errors": []}
    try:
        try:
            answer, first_usage = await call_model_async(prompt, plate_path, model, agent=active_agent)
            parsed = parse_and_validate_result(answer, plate.width, plate.height)
            validation["initial_valid"] = True
            validation["final_valid"] = True
            judgment_usage = first_usage
        except OutputValidationError as exc:
            validation["retry_performed"] = True
            validation["errors"].append(str(exc))
            retry_prompt = (
                f"{prompt}\n\n"
                "上一份输出被程序拒绝，原因如下："
                f"{str(exc)[:1500]}。请重新检查同一张图片并修正；最终只能调用 finish 返回合规 JSON。"
            )
            remaining_calls = SWE_MAX_ITERATIONS - (first_usage.get("api_calls") or 0)
            if remaining_calls <= 0:
                parsed = {
                    "raw_model_output": answer,
                    "output_validation_errors": validation["errors"] + ["已用尽本图10次调用预算，未执行重试"],
                }
                judgment_usage = first_usage
            else:
                original_limit = active_agent.max_iterations
                active_agent.max_iterations = remaining_calls
                try:
                    retry_answer, retry_usage = await call_model_async(retry_prompt, plate_path, model, agent=active_agent)
                finally:
                    active_agent.max_iterations = original_limit
                judgment_usage = merge_usage(first_usage, retry_usage)
                try:
                    parsed = parse_and_validate_result(retry_answer, plate.width, plate.height)
                    validation["final_valid"] = True
                except OutputValidationError as retry_exc:
                    validation["errors"].append(str(retry_exc))
                    parsed = {
                        "raw_model_output": retry_answer,
                        "output_validation_errors": validation["errors"],
                    }
    finally:
        if own_agent:
            await active_agent.cleanup()
    annotated = draw_anomaly_boxes(plate, parsed)
    annotated.save(annotated_path, format="JPEG", quality=95)
    token_usage = {
        "judgment": judgment_usage,
        "total": judgment_usage,
    }
    judgment_cost = calculate_cost(model, judgment_usage)
    data_scope = {
        "source_image": str(source_path),
        "source_image_size": {"width": image.width, "height": image.height},
        "plate_box_in_source": {"x": box[0], "y": box[1], "width": box[2], "height": box[3]},
        "model_input": "one complete cropped plate image" if model_uses_plate_crop else "one uncropped original image",
        "model_input_size": {"width": plate.width, "height": plate.height},
        "model_calls": judgment_usage.get("api_calls"),
        "localization": localization["source"],
        "original_image_sent_to_model_for_localization": not model_uses_plate_crop,
        "original_image_sent_to_model_for_judgment": not model_uses_plate_crop,
    }
    if batch_scope:
        data_scope["batch"] = batch_scope
    answer_document = {
        "metadata": {
            "timestamp": timestamp,
            "model": model,
            "code_version": CODE_VERSION,
            "code_sha256": file_sha256(Path(__file__)),
            "prompt_template_sha256": file_sha256(PROMPT_PATH),
            "prompt_sha256": text_sha256(prompt),
            "plate_localization": {"model_output": localization, "plate_box_in_source": box},
            "token_usage": token_usage,
            "cost": {"judgment": judgment_cost, "total_cny": judgment_cost.get("cost")},
            "output_validation": validation,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "data_scope": data_scope,
        },
        "result": parsed,
    }
    result_path.write_text(json.dumps(answer_document, ensure_ascii=False, indent=2), encoding="utf-8")
    if final_dir:
        shutil.copy2(annotated_path, final_dir / "plate_annotated.jpg")
        shutil.copy2(result_path, final_dir / "answer.json")
    return {
        "run_dir": str(run_dir),
        "final_answer_dir": str(final_dir) if final_dir else None,
        "annotated_plate": str(annotated_path),
        **answer_document,
    }


def process_image(
    source_path: Path,
    model: str,
    output_dir: Path,
    answers_dir: Optional[Path] = None,
    plate_box: Optional[Tuple[int, int, int, int]] = None,
    use_plate_crop: bool = True,
    batch_scope: Optional[dict] = None,
    run_dir_override: Optional[Path] = None,
    run_timestamp: Optional[str] = None,
) -> dict:
    """单图同步入口；每次运行创建并关闭自己的 SWE Agent。"""
    return asyncio.run(process_image_async(
        source_path, model, output_dir, answers_dir, plate_box, use_plate_crop, batch_scope,
        run_dir_override, run_timestamp,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="v3 车牌伪造筛查（结构校验裁剪 + SWE 默认 Agent 流程）")
    parser.add_argument("image", help="原始图片路径")
    parser.add_argument("--plate-box", type=parse_box, help="人工车牌框：x,y,width,height")
    parser.add_argument("--no-crop-plate", dest="use_plate_crop", action="store_false", help="不做本地车牌裁剪，直接将原图传给模型")
    parser.set_defaults(use_plate_crop=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "qwen3-vl-plus"))
    parser.add_argument("--output-dir", default="runtime/license-v3/single/results", help="裁剪图和结果保存目录")
    parser.add_argument("--answers-dir", default="runtime/license-v3/single/answers", help="最终答案保存目录")
    args = parser.parse_args()

    source_path = project_path(args.image)
    if not source_path.is_file():
        parser.error(f"图片不存在：{source_path}")
    if not args.use_plate_crop and args.plate_box:
        parser.error("--no-crop-plate 不能与 --plate-box 同时使用")
    try:
        document = process_image(
            source_path,
            args.model,
            project_path(args.output_dir),
            project_path(args.answers_dir),
            args.plate_box,
            args.use_plate_crop,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(document, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
