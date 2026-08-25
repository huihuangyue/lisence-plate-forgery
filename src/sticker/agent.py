"""受控多轮视觉模型复核：模型只能选择确定性候选编号。"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image

from .evidence import (
    candidate_has_multichannel_tamper_support,
    candidate_has_rectangle_material_support,
    candidate_tamper_support_routes,
    resolve_adjacent_material_ownership,
)
from .types import CandidateRegion, EvidenceArtifacts


INVESTIGATOR_SYSTEM = """你是车牌物理贴片证据调查员，任务是判断矩形字符贴、矩形底片或磁贴是否整体覆盖某个字符。
输入依次为原始平面车牌、从左到右的字符槽映射图和六联确定性诊断图。候选 Cn 与从左到右第 n 个字符槽 Sn 一一绑定，候选框表示待核查区域。
按五条证据路径观察：闭合矩形；因偏光只显出一条材料缝与一条正交残边、但两者仍与矫正后号牌边框横平竖直的局部贴片；只显出一条轴对齐材料缝、但槽内区域相对局部车牌背景具有很强材料差异；上下边贴近号牌边框时由孤立成对竖缝形成的整高条带；绿牌候选内部纵向 Lab 剖面相对左右邻域期望剖面的均色、反向或形状异常。每条路径都结合白边、暗缝、亮暗成对边缘以及候选边界两侧的颜色、反光和纹理连续性。
同时记录最强的正常解释，包括字符结构、号牌压边、螺钉高光、污渍划痕、局部照明、归一化形变和压缩伪影，并比较其解释力。
所有 anomaly_score 越高表示越异常。green_vertical_gradient_applicable=1 时，expected/actual gradient、direction cosine、uniformity、reversal 与 anomaly 共同描述纵向底色关系；伪彩色图负责定位，数值和原图负责核验。
坐标和候选集合沿用输入证据表。verdict 采用固定语义：tamper_support=支持整体贴片；normal_support=更支持正常结构或成像效应；uncertain=现有证据仍有多种解释。
输出采用给定 JSON 协议。"""


REVIEWER_SYSTEM = """你是独立证据审查员，在未知调查员结论的条件下逐个复核候选。输入顺序与 Cn=Sn 槽位绑定和调查员相同。
先为每个候选建立正常解释：字符结构、号牌外压边、螺钉与高光、污渍划痕、局部反光、透视归一化误差、模糊和 JPEG 块效应；再与贴片解释比较。
贴片解释由两类独立证据共同支撑：几何边界加材料差异；或绿牌纵向期望/实际底色剖面异常加可见边界与材料差异。几何边界允许不闭合：一条与号牌边框平行或垂直的材料缝，可由另一条正交残边确认。整高条带的证据组合包含成对竖缝、相邻槽孤立性和双侧材料差异。
所有 anomaly_score 越高表示越异常。伪彩色用于定位，原图、固定量程和结构化数值用于核验。坐标和候选集合沿用输入证据表。
verdict 语义固定：tamper_support=支持整体贴片；normal_support=更支持正常结构或成像效应；uncertain=现有证据仍有多种解释。输出采用给定 JSON 协议。"""


RECHECK_SYSTEM = """你是冲突证据复查员，复查清单中的候选并沿用 Cn=Sn 的从左到右槽位绑定。
逐项核验矩形边数与闭合关系、未闭合局部边是否与号牌边框横平竖直且有正交残边、整高条带的成对竖缝与相邻槽孤立性、绿牌纵向期望/实际 Lab 剖面的方向与幅度、内外材料差异、亮暗成对边缘，以及最强正常解释。
verdict 采用 tamper_support、normal_support、uncertain 的固定语义。输出采用给定 JSON 协议。"""


ADJUDICATOR_SYSTEM = """你是车牌物理贴片裁决员。输入包含确定性视觉证据表、独立调查、独立审查和可选冲突复查。
对同一候选汇总五类成立路径：闭合矩形与材料差异；一条材料缝和一条与其正交、且均与号牌边框横平竖直的残边；单条轴对齐材料缝与槽内强区域材料差异；孤立成对整高竖缝与双侧材料差异；绿牌纵向底色剖面的均色/反向异常与可见边界及材料差异。随后比较字符结构、号牌压边、螺钉高光、污渍、照明和压缩效应等正常解释。
Cn 与从左到右的 Sn 一一对应，候选编号来自输入证据表。decision 采用 suspicious、clear、unassessable；仍有竞争解释的位置写入 uncertain_candidates。unassessable 只用于号牌不可见、严重模糊、严重曝光损坏或车牌提取失败；证据不足、证据冲突或存在 uncertain_candidates 时仍必须在 suspicious 与 clear 之间裁决，不能用 unassessable 回避。
tamper_support 表示支持贴片，normal_support 表示支持正常解释。recognized_characters 按候选编号记录槽位图中可读出的单个字符，视觉不足时记录 null；字符识别与贴片成立条件相互独立。
输出采用给定 JSON 协议。"""


_ASSESSMENT_STAGES = {"investigator", "reviewer", "recheck"}
_VERDICTS = {"tamper_support", "normal_support", "uncertain"}
_DECISIONS = {"suspicious", "clear", "unassessable"}


def _json_object_text(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    start = stripped.find("{")
    if start < 0:
        raise ValueError(f"模型没有返回 JSON：{stripped[:220]}")
    end = stripped.rfind("}")
    return stripped[start : end + 1] if end > start else stripped[start:]


def _strip_trailing_commas(text: str) -> str:
    """只在字符串外删除对象/数组末尾的逗号。"""

    output: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue
        output.append(char)
        index += 1
    return "".join(output)


def _close_truncated_json(text: str) -> str:
    """补齐被截断响应的引号和右括号，不改动已有内容。"""

    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char == "]" and stack and stack[-1] == "[":
            stack.pop()
        elif char == "}" and stack and stack[-1] == "{":
            stack.pop()
    if in_string:
        text += '"'
    return text + "".join("}" if char == "{" else "]" for char in reversed(stack))


def _repair_json(text: str) -> str:
    """保守修复视觉模型常见 JSON 语法损坏。

    只修复 Unicode 引号、尾逗号、截断闭合符，以及解析器明确指出位置的
    缺失逗号/控制字符；业务字段的增删交由后续校验拒绝，而不是猜测。
    """

    repaired = text.translate(str.maketrans({"“": '"', "”": '"', "＂": '"'}))
    repaired = _strip_trailing_commas(_close_truncated_json(repaired))
    for _ in range(24):
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError as exc:
            position = exc.pos
            if exc.msg == "Expecting ',' delimiter":
                repaired = repaired[:position] + "," + repaired[position:]
                continue
            if exc.msg.startswith("Invalid control character") and position < len(repaired):
                replacement = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(repaired[position])
                if replacement is not None:
                    repaired = repaired[:position] + replacement + repaired[position + 1 :]
                    continue
            if exc.msg == "Expecting property name enclosed in double quotes":
                previous = position - 1
                while previous >= 0 and repaired[previous].isspace():
                    previous -= 1
                if previous >= 0 and repaired[previous] == ",":
                    repaired = repaired[:previous] + repaired[previous + 1 :]
                    continue
            break
    return repaired


def _extract_json_with_metadata(text: str) -> tuple[dict[str, Any], bool]:
    object_text = _json_object_text(text)
    try:
        value = json.loads(object_text)
        repaired = False
    except json.JSONDecodeError as original_error:
        repaired_text = _repair_json(object_text)
        try:
            value = json.loads(repaired_text)
        except json.JSONDecodeError as repaired_error:
            raise ValueError(
                f"模型 JSON 无法修复：原始错误={original_error}; 修复后错误={repaired_error}"
            ) from repaired_error
        repaired = repaired_text != object_text
    if not isinstance(value, dict):
        raise ValueError("模型 JSON 顶层必须是对象")
    return value, repaired


def _extract_json(text: str) -> dict[str, Any]:
    value, _ = _extract_json_with_metadata(text)
    return value


def _normalize_character(value: Any) -> str | None:
    """只接受单个车牌字符，拒绝把解释性文字写成识别结果。"""

    if value is None:
        return None
    text = str(value).strip().upper()
    if re.fullmatch(r"[0-9A-Z\u3400-\u9FFF]", text):
        return text
    return None


def _candidate_sort_key(candidate_id: str) -> tuple[int, str]:
    suffix = candidate_id.removeprefix("C")
    return (int(suffix), candidate_id) if suffix.isdigit() else (10**9, candidate_id)


def _string_list_schema() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "maxItems": 8}


def _assessment_schema(
    allowed_ids: set[str],
    slot_by_candidate: dict[str, int],
) -> dict[str, Any]:
    candidate_ids = sorted(allowed_ids, key=_candidate_sort_key)
    slot_ids = [f"S{slot_by_candidate[candidate_id]}" for candidate_id in candidate_ids]
    item_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidate_id": {"type": "string", "enum": candidate_ids},
            "slot_id": {"type": "string", "enum": slot_ids},
            "observed_character": {
                "anyOf": [
                    {"type": "string", "minLength": 1, "maxLength": 1},
                    {"type": "null"},
                ]
            },
            "verdict": {"type": "string", "enum": sorted(_VERDICTS)},
            "geometry_observations": _string_list_schema(),
            "appearance_observations": _string_list_schema(),
            "normal_explanations": _string_list_schema(),
            "needs_recheck": {"type": "boolean"},
        },
        "required": [
            "candidate_id",
            "slot_id",
            "observed_character",
            "verdict",
            "geometry_observations",
            "appearance_observations",
            "normal_explanations",
            "needs_recheck",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "plate_quality": {"type": "string", "enum": ["assessable", "unassessable"]},
            "candidates": {
                "type": "array",
                "items": item_schema,
                "minItems": len(candidate_ids),
                "maxItems": len(candidate_ids),
            },
            "summary": {"type": "string"},
        },
        "required": ["plate_quality", "candidates", "summary"],
    }


def _adjudicator_schema(allowed_ids: set[str]) -> dict[str, Any]:
    candidate_ids = sorted(allowed_ids, key=_candidate_sort_key)
    character_schema = {
        "anyOf": [
            {"type": "string", "minLength": 1, "maxLength": 1},
            {"type": "null"},
        ]
    }
    evidence_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "geometry": _string_list_schema(),
            "appearance": _string_list_schema(),
            "counter_evidence": _string_list_schema(),
        },
        "required": ["geometry", "appearance", "counter_evidence"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": sorted(_DECISIONS)},
            "selected_candidates": {
                "type": "array",
                "items": {"type": "string", "enum": candidate_ids},
            },
            "uncertain_candidates": {
                "type": "array",
                "items": {"type": "string", "enum": candidate_ids},
            },
            "recognized_characters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {candidate_id: character_schema for candidate_id in candidate_ids},
            },
            "candidate_evidence": {
                "type": "object",
                "additionalProperties": False,
                "properties": {candidate_id: evidence_schema for candidate_id in candidate_ids},
            },
            "reasoning_summary": {"type": "string"},
            "unassessable_reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": [
            "decision",
            "selected_candidates",
            "uncertain_candidates",
            "recognized_characters",
            "candidate_evidence",
            "reasoning_summary",
            "unassessable_reason",
        ],
    }


def _require_exact_fields(value: dict[str, Any], required: set[str], location: str) -> None:
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing or extra:
        raise ValueError(f"{location} 字段不符合协议：missing={missing}, extra={extra}")


def _validate_string_list(value: Any, location: str) -> None:
    if not isinstance(value, list) or len(value) > 8 or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{location} 必须是不超过 8 项的字符串数组")


def _validate_assessment_payload(
    payload: dict[str, Any],
    allowed_ids: set[str],
    slot_by_candidate: dict[str, int],
) -> dict[str, Any]:
    _require_exact_fields(payload, {"plate_quality", "candidates", "summary"}, "assessment")
    if payload["plate_quality"] not in {"assessable", "unassessable"}:
        raise ValueError("assessment.plate_quality 枚举非法")
    if not isinstance(payload["summary"], str):
        raise ValueError("assessment.summary 必须是字符串")
    candidates = payload["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("assessment.candidates 必须是数组")
    candidate_fields = {
        "candidate_id",
        "slot_id",
        "observed_character",
        "verdict",
        "geometry_observations",
        "appearance_observations",
        "normal_explanations",
        "needs_recheck",
    }
    seen: set[str] = set()
    for index, item in enumerate(candidates):
        location = f"assessment.candidates[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{location} 必须是对象")
        _require_exact_fields(item, candidate_fields, location)
        candidate_id = item["candidate_id"]
        if candidate_id not in allowed_ids:
            raise ValueError(f"{location}.candidate_id 不在候选集合：{candidate_id!r}")
        if candidate_id in seen:
            raise ValueError(f"{location}.candidate_id 重复：{candidate_id}")
        seen.add(candidate_id)
        expected_slot = f"S{slot_by_candidate[candidate_id]}"
        if item["slot_id"] != expected_slot:
            raise ValueError(f"{location}.slot_id 应为 {expected_slot}")
        if item["observed_character"] is not None and _normalize_character(item["observed_character"]) is None:
            raise ValueError(f"{location}.observed_character 必须是单个车牌字符或 null")
        if item["verdict"] not in _VERDICTS:
            raise ValueError(f"{location}.verdict 枚举非法")
        for field in ("geometry_observations", "appearance_observations", "normal_explanations"):
            _validate_string_list(item[field], f"{location}.{field}")
        if not isinstance(item["needs_recheck"], bool):
            raise ValueError(f"{location}.needs_recheck 必须是布尔值")
    if seen != allowed_ids:
        raise ValueError(f"assessment.candidates 必须完整覆盖候选：missing={sorted(allowed_ids - seen)}")
    return payload


def _validate_adjudicator_payload(payload: dict[str, Any], allowed_ids: set[str]) -> dict[str, Any]:
    required = {
        "decision",
        "selected_candidates",
        "uncertain_candidates",
        "recognized_characters",
        "candidate_evidence",
        "reasoning_summary",
        "unassessable_reason",
    }
    _require_exact_fields(payload, required, "adjudicator")
    if payload["decision"] not in _DECISIONS:
        raise ValueError("adjudicator.decision 枚举非法")
    selected = payload["selected_candidates"]
    uncertain = payload["uncertain_candidates"]
    for field, values in (("selected_candidates", selected), ("uncertain_candidates", uncertain)):
        if not isinstance(values, list) or not all(isinstance(item, str) and item in allowed_ids for item in values):
            raise ValueError(f"adjudicator.{field} 包含非法候选")
        if len(values) != len(set(values)):
            raise ValueError(f"adjudicator.{field} 包含重复候选")
    overlap = sorted(set(selected) & set(uncertain))
    if overlap:
        raise ValueError(f"selected_candidates 与 uncertain_candidates 冲突：{overlap}")
    characters = payload["recognized_characters"]
    if not isinstance(characters, dict) or not set(characters).issubset(allowed_ids):
        raise ValueError("adjudicator.recognized_characters 包含非法候选键")
    for candidate_id, value in characters.items():
        if value is not None and _normalize_character(value) is None:
            raise ValueError(f"recognized_characters.{candidate_id} 必须是单个车牌字符或 null")
    evidence = payload["candidate_evidence"]
    if not isinstance(evidence, dict) or not set(evidence).issubset(allowed_ids):
        raise ValueError("adjudicator.candidate_evidence 包含非法候选键")
    evidence_fields = {"geometry", "appearance", "counter_evidence"}
    for candidate_id, item in evidence.items():
        if not isinstance(item, dict):
            raise ValueError(f"candidate_evidence.{candidate_id} 必须是对象")
        _require_exact_fields(item, evidence_fields, f"candidate_evidence.{candidate_id}")
        for field in evidence_fields:
            _validate_string_list(item[field], f"candidate_evidence.{candidate_id}.{field}")
    if not isinstance(payload["reasoning_summary"], str):
        raise ValueError("adjudicator.reasoning_summary 必须是字符串")
    if payload["unassessable_reason"] is not None and not isinstance(payload["unassessable_reason"], str):
        raise ValueError("adjudicator.unassessable_reason 必须是字符串或 null")
    return payload


def _stage_schema(
    stage: str,
    allowed_ids: set[str] | None,
    slot_by_candidate: dict[str, int] | None,
) -> dict[str, Any] | None:
    if not allowed_ids:
        return None
    if stage in _ASSESSMENT_STAGES:
        if slot_by_candidate is None:
            raise ValueError(f"{stage} 缺少候选到槽位映射")
        return _assessment_schema(allowed_ids, slot_by_candidate)
    if stage == "adjudicator":
        return _adjudicator_schema(allowed_ids)
    return None


def _validate_stage_payload(
    stage: str,
    payload: dict[str, Any],
    allowed_ids: set[str] | None,
    slot_by_candidate: dict[str, int] | None,
) -> dict[str, Any]:
    if not allowed_ids:
        return payload
    if stage in _ASSESSMENT_STAGES:
        if slot_by_candidate is None:
            raise ValueError(f"{stage} 缺少候选到槽位映射")
        return _validate_assessment_payload(payload, allowed_ids, slot_by_candidate)
    if stage == "adjudicator":
        return _validate_adjudicator_payload(payload, allowed_ids)
    return payload


def _is_local_endpoint(base_url: str | None) -> bool:
    host = urlparse(base_url or "").hostname
    return host in {"127.0.0.1", "localhost", "::1"}


def _encode_image(image: np.ndarray, max_side: int = 1800) -> str:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    if max(pil.size) > max_side:
        ratio = max_side / max(pil.size)
        pil = pil.resize((round(pil.width * ratio), round(pil.height * ratio)), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    pil.save(buffer, format="JPEG", quality=93)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _candidate_table(candidates: list[CandidateRegion]) -> list[dict[str, Any]]:
    table = []
    for item in sorted(candidates, key=lambda candidate: candidate.slot):
        features = item.features
        table.append({
            "candidate_id": item.candidate_id,
            "slot_id": f"S{item.slot}",
            "left_to_right_index": item.slot,
            "bbox": list(item.bbox),
            "anomaly_scores": {
                "rectangle_geometry_anomaly_score": item.geometry_score,
                "material_appearance_anomaly_score": item.appearance_score,
                "paired_bright_dark_edge_anomaly_score": item.paired_edge_score,
                "combined_screening_anomaly_score": item.combined_score,
            },
            "rectangle_evidence": {
                key: features.get(key, 0.0)
                for key in (
                    "top_side_support",
                    "right_side_support",
                    "bottom_side_support",
                    "left_side_support",
                    "top_axis_alignment",
                    "right_axis_alignment",
                    "bottom_axis_alignment",
                    "left_axis_alignment",
                    "top_line_locality",
                    "right_line_locality",
                    "bottom_line_locality",
                    "left_line_locality",
                    "top_slot_overlap",
                    "right_slot_overlap",
                    "bottom_slot_overlap",
                    "left_slot_overlap",
                    "rectangle_side_count",
                    "rectangle_corner_count",
                    "opposite_pair_support",
                    "rectangle_closure_score",
                    "orthogonal_join_support",
                    "rectangle_geometry_score",
                    "vertical_strip_geometry_score",
                    "vertical_strip_left_continuity",
                    "vertical_strip_right_continuity",
                    "vertical_strip_pair_score",
                    "vertical_strip_isolation",
                    "partial_axis_material_side_score",
                    "partial_axis_orthogonal_side_score",
                    "partial_axis_material_side_count",
                    "partial_axis_seam_score",
                    "partial_axis_geometry_score",
                    "single_axis_material_seam_score",
                )
            },
            "material_evidence": {
                key: features.get(key, 0.0)
                for key in (
                    "delta_e_00",
                    "interior_color_residual",
                    "boundary_color_residual",
                    "boundary_material_score",
                    "material_side_count",
                    "bright_dark_support",
                    "vertical_expected_gradient_magnitude",
                    "vertical_actual_gradient_magnitude",
                    "vertical_gradient_direction_cosine",
                    "vertical_gradient_slope_mismatch",
                    "vertical_gradient_profile_residual",
                    "vertical_gradient_reversal_score",
                    "vertical_gradient_uniformity_score",
                    "vertical_gradient_anomaly_score",
                    "green_vertical_gradient_applicable",
                    "contextual_material_score",
                    "material_ownership_score",
                    "plate_reference_delta_e_median",
                    "plate_reference_delta_e_p75",
                    "plate_reference_hf_ratio",
                    "plate_reference_valid_rows",
                    "matched_control_delta_e_median",
                    "matched_control_delta_e_p75",
                    "matched_control_left_delta_e",
                    "matched_control_right_delta_e",
                    "matched_control_bilateral_min_delta_e",
                    "matched_control_bilateral_balance",
                    "matched_control_hf_ratio",
                    "matched_control_valid_rows",
                    "edge_slot_one_sided_control",
                )
            },
        })
    return table


def _assessment_prompt(artifacts: EvidenceArtifacts) -> str:
    table = _candidate_table(artifacts.bundle.candidates)
    slot_bindings = ", ".join(
        f"C{slot.slot}=S{slot.slot}=从左第{slot.slot}位" for slot in sorted(artifacts.bundle.slots, key=lambda item: item.slot)
    )
    return (
        "槽位绑定：" + slot_bindings + "。\n"
        "候选证据表如下。所有 anomaly_score 越高表示越异常；数值是相对证据强度。"
        "rectangle_side_count 表示背景边界中可归入矩形四侧的边数。"
        "partial_axis_seam_score 表示不要求闭合的局部直角边证据，由同侧材料突变边和正交残边共同形成；"
        "single_axis_material_seam_score 表示只有一条轴对齐材料缝时的证据，需结合强区域材料异常归属槽位。"
        "绿牌纵向字段比较槽内实际底色剖面与左右邻域推断的期望剖面。\n"
        + json.dumps(table, ensure_ascii=False, separators=(",", ":"))
        + "\n返回协议："
        '{"plate_quality":"assessable|unassessable","candidates":['
        '{"candidate_id":"C1","slot_id":"S1","observed_character":null,'
        '"verdict":"tamper_support|normal_support|uncertain",'
        '"geometry_observations":["..."],"appearance_observations":["..."],'
        '"normal_explanations":["..."],"needs_recheck":false}],"summary":"..."}'
    )


def _sanitize_assessment(
    payload: dict[str, Any],
    allowed_ids: set[str],
    slot_by_candidate: dict[str, int] | None = None,
) -> dict[str, Any]:
    candidates = []
    for raw in payload.get("candidates", []):
        if not isinstance(raw, dict):
            continue
        candidate_id = str(raw.get("candidate_id", ""))
        if candidate_id not in allowed_ids:
            continue
        verdict = str(raw.get("verdict", "uncertain"))
        if verdict not in {"tamper_support", "normal_support", "uncertain"}:
            verdict = "uncertain"
        slot = (slot_by_candidate or {}).get(candidate_id)
        if slot is None and candidate_id.removeprefix("C").isdigit():
            slot = int(candidate_id.removeprefix("C"))
        observed = _normalize_character(raw.get("observed_character"))
        candidates.append(
            {
                "candidate_id": candidate_id,
                "slot_id": f"S{slot}" if slot is not None else None,
                "observed_character": observed,
                "verdict": verdict,
                "geometry_observations": [str(value) for value in raw.get("geometry_observations", [])][:8],
                "appearance_observations": [str(value) for value in raw.get("appearance_observations", [])][:8],
                "normal_explanations": [str(value) for value in raw.get("normal_explanations", [])][:8],
                "needs_recheck": bool(raw.get("needs_recheck", False)),
            }
        )
    return {
        "plate_quality": payload.get("plate_quality", "assessable"),
        "candidates": candidates,
        "summary": str(payload.get("summary", "")),
    }


def _verdict_map(payload: dict[str, Any]) -> dict[str, str]:
    return {str(item["candidate_id"]): str(item["verdict"]) for item in payload.get("candidates", [])}


def _merge_character_recognition(
    decision: dict[str, Any],
    assessments: list[dict[str, Any] | None],
) -> None:
    """将裁决员字符与独立观察合并，并显式记录字符来源/不确定性。"""

    candidate_ids = list(decision.get("selected_candidates", [])) + list(
        decision.get("uncertain_candidates", [])
    )
    recognized = {
        candidate_id: _normalize_character(decision.get("recognized_characters", {}).get(candidate_id))
        for candidate_id in candidate_ids
    }
    statuses: dict[str, str] = {}
    for candidate_id in candidate_ids:
        if recognized[candidate_id] is not None:
            statuses[candidate_id] = "adjudicator_reported"
            continue
        observations = [
            _normalize_character(item.get("observed_character"))
            for assessment in assessments
            if assessment is not None
            for item in assessment.get("candidates", [])
            if item.get("candidate_id") == candidate_id
        ]
        observations = [value for value in observations if value is not None]
        counts = {value: observations.count(value) for value in set(observations)}
        if not counts:
            statuses[candidate_id] = "unreadable_or_not_returned"
            continue
        best_count = max(counts.values())
        winners = sorted(value for value, count in counts.items() if count == best_count)
        if len(winners) != 1:
            statuses[candidate_id] = "conflicting_model_observations"
            continue
        recognized[candidate_id] = winners[0]
        statuses[candidate_id] = (
            "cross_review_consensus" if best_count >= 2 else "single_review_observation"
        )
    decision["recognized_characters"] = recognized
    decision["character_recognition_status"] = statuses


@dataclass
class AgentCall:
    stage: str
    cache_key: str
    cached: bool
    prompt: str
    response: dict[str, Any]
    usage: dict[str, int | None]
    structured_output: bool = False
    json_repaired: bool = False


class StickerAgentHarness:
    """按 2/3/4 次调用预算运行的多轮视觉模型复核 harness。"""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        max_calls_per_image: int = 3,
    ) -> None:
        self.model = (
            model
            or os.environ.get("STICKER_AGENT_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("SHAPE_VISION_MODEL")
        )
        self.api_key = api_key or os.environ.get("STICKER_AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("STICKER_AGENT_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        structured_output = os.environ.get("STICKER_AGENT_STRUCTURED_OUTPUT", "auto").strip().lower()
        if structured_output not in {"", "auto", "0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise RuntimeError("STICKER_AGENT_STRUCTURED_OUTPUT 必须是 auto 或布尔值")
        self.structured_output = (
            _is_local_endpoint(self.base_url)
            if structured_output in {"", "auto"}
            else structured_output in {"1", "true", "yes", "on"}
        )
        disable_thinking = os.environ.get("STICKER_AGENT_DISABLE_THINKING", "").strip().lower()
        if disable_thinking not in {"", "0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise RuntimeError("STICKER_AGENT_DISABLE_THINKING 必须是 0/1、false/true、no/yes 或 off/on")
        self.disable_thinking = disable_thinking in {"1", "true", "yes", "on"}
        max_output_tokens = os.environ.get("STICKER_AGENT_MAX_OUTPUT_TOKENS", "").strip()
        self.max_output_tokens = int(max_output_tokens) if max_output_tokens else None
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise RuntimeError("STICKER_AGENT_MAX_OUTPUT_TOKENS 必须是正整数")
        adjudicator_max_tokens = os.environ.get(
            "STICKER_AGENT_ADJUDICATOR_MAX_OUTPUT_TOKENS", ""
        ).strip()
        self.adjudicator_max_output_tokens = (
            int(adjudicator_max_tokens) if adjudicator_max_tokens else self.max_output_tokens
        )
        if self.adjudicator_max_output_tokens is not None and self.adjudicator_max_output_tokens <= 0:
            raise RuntimeError("STICKER_AGENT_ADJUDICATOR_MAX_OUTPUT_TOKENS 必须是正整数")
        if not self.model:
            raise RuntimeError("缺少模型名：设置 OPENAI_MODEL 或传入 --model")
        if not self.api_key:
            raise RuntimeError("缺少 OPENAI_API_KEY；请由操作者显式 source local_env.sh")
        if max_calls_per_image not in {2, 3, 4}:
            raise ValueError("max_calls_per_image 必须是 2、3 或 4")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.use_cache = use_cache
        self.max_calls_per_image = max_calls_per_image
        self.calls: list[AgentCall] = []

    def _cache_key(
        self,
        stage: str,
        system: str,
        prompt: str,
        images: list[np.ndarray],
        schema: dict[str, Any] | None,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(b"sticker-agent-json-protocol-v2")
        digest.update(self.model.encode())
        digest.update((self.base_url or "").encode())
        digest.update(str(self.structured_output).encode())
        digest.update(str(self.disable_thinking).encode())
        digest.update(str(self._max_tokens_for_stage(stage)).encode())
        digest.update(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode() if schema else b"none")
        digest.update(stage.encode())
        digest.update(system.encode())
        digest.update(prompt.encode())
        for image in images:
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 93])
            if not ok:
                raise ValueError("无法为 API 缓存编码图像")
            digest.update(encoded.tobytes())
        return digest.hexdigest()

    def _max_tokens_for_stage(self, stage: str) -> int | None:
        if stage == "adjudicator":
            return self.adjudicator_max_output_tokens
        return self.max_output_tokens

    def _call(
        self,
        stage: str,
        system: str,
        prompt: str,
        images: list[np.ndarray],
        allowed_ids: set[str] | None = None,
        slot_by_candidate: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        schema = _stage_schema(stage, allowed_ids, slot_by_candidate)
        key = self._cache_key(stage, system, prompt, images, schema)
        cache_path = self.cache_dir / f"{stage}_{key}.json" if self.cache_dir is not None else None
        if self.use_cache and cache_path is not None and cache_path.is_file():
            cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            try:
                response_payload = _validate_stage_payload(
                    stage, cached_payload["response"], allowed_ids, slot_by_candidate
                )
            except (KeyError, TypeError, ValueError):
                response_payload = None
            if response_payload is not None:
                self.calls.append(
                    AgentCall(
                        stage,
                        key,
                        True,
                        prompt,
                        response_payload,
                        cached_payload.get("usage", {}),
                        bool(cached_payload.get("structured_output", False)),
                        bool(cached_payload.get("json_repaired", False)),
                    )
                )
                return response_payload
        encoded_images = [_encode_image(image) for image in images]
        last_error: Exception | None = None
        json_repaired = False
        stage_max_tokens = self._max_tokens_for_stage(stage)
        for attempt in range(3):
            try:
                retry_note = (
                    ""
                    if attempt == 0
                    else f"\n上一响应未通过 JSON 协议校验：{last_error}。请重新生成完整对象，不要省略字段。"
                )
                content: list[dict[str, Any]] = [{"type": "text", "text": prompt + retry_note}]
                content.extend(
                    {"type": "image_url", "image_url": {"url": encoded}}
                    for encoded in encoded_images
                )
                request: dict[str, Any] = {
                    "model": self.model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": content},
                    ],
                }
                if stage_max_tokens is not None:
                    request["max_tokens"] = stage_max_tokens
                if self.structured_output and schema is not None:
                    request["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": f"sticker_{stage}_response",
                            "description": "车牌整体贴片检测的受控阶段输出",
                            "strict": True,
                            "schema": schema,
                        },
                    }
                if self.disable_thinking:
                    # vLLM/SGLang 上的 Qwen3.5 使用 chat_template_kwargs 控制思考模式。
                    # DashScope 的参数协议不同，因此只在操作者显式设置时发送。
                    request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                response = self.client.chat.completions.create(
                    **request,
                )
                raw_content = response.choices[0].message.content
                if not isinstance(raw_content, str):
                    raise RuntimeError(f"{stage} 模型返回为空或不是文本 JSON")
                response_payload, json_repaired = _extract_json_with_metadata(raw_content)
                response_payload = _validate_stage_payload(
                    stage, response_payload, allowed_ids, slot_by_candidate
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt == 2:
                    raise RuntimeError(f"{stage} 模型接口、JSON 修复或字段校验连续失败 3 次：{exc}") from exc
                time.sleep(1.0 + attempt)
        else:  # pragma: no cover - 循环只会 break 或 raise
            raise RuntimeError(f"{stage} 模型接口调用失败：{last_error}")
        usage_obj = getattr(response, "usage", None)
        usage = {
            "input_tokens": getattr(usage_obj, "prompt_tokens", None),
            "output_tokens": getattr(usage_obj, "completion_tokens", None),
            "total_tokens": getattr(usage_obj, "total_tokens", None),
        }
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "stage": stage,
                        "model": self.model,
                        "response": response_payload,
                        "usage": usage,
                        "max_output_tokens": stage_max_tokens,
                        "structured_output": bool(self.structured_output and schema is not None),
                        "json_repaired": json_repaired,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        self.calls.append(
            AgentCall(
                stage,
                key,
                False,
                prompt,
                response_payload,
                usage,
                bool(self.structured_output and schema is not None),
                json_repaired,
            )
        )
        return response_payload

    def run(self, image: np.ndarray, artifacts: EvidenceArtifacts) -> tuple[dict[str, Any], dict[str, Any]]:
        # 一个 harness 可被批处理复用，但每张图片的轨迹必须相互独立。
        self.calls = []
        allowed_ids = {candidate.candidate_id for candidate in artifacts.bundle.candidates}
        slot_by_candidate = {candidate.candidate_id: candidate.slot for candidate in artifacts.bundle.candidates}
        common_prompt = _assessment_prompt(artifacts)
        views = [image, artifacts.slot_map_view, artifacts.diagnostic_panel]
        investigator = _sanitize_assessment(
            self._call(
                "investigator",
                INVESTIGATOR_SYSTEM,
                common_prompt,
                views,
                allowed_ids,
                slot_by_candidate,
            ),
            allowed_ids,
            slot_by_candidate,
        )
        reviewer: dict[str, Any] | None = None
        disputed: list[str] = []
        if self.max_calls_per_image >= 3:
            reviewer = _sanitize_assessment(
                self._call(
                    "reviewer",
                    REVIEWER_SYSTEM,
                    common_prompt,
                    views,
                    allowed_ids,
                    slot_by_candidate,
                ),
                allowed_ids,
                slot_by_candidate,
            )
            investigator_map, reviewer_map = _verdict_map(investigator), _verdict_map(reviewer)
            disputed = sorted(
                candidate_id
                for candidate_id in allowed_ids
                if {investigator_map.get(candidate_id), reviewer_map.get(candidate_id)}
                == {"tamper_support", "normal_support"}
            )
        recheck: dict[str, Any] | None = None
        if disputed and self.max_calls_per_image >= 4:
            recheck_prompt = (
                "冲突候选复查清单："
                + json.dumps(disputed, ensure_ascii=False)
                + "\n"
                + common_prompt
            )
            recheck = _sanitize_assessment(
                self._call(
                    "recheck",
                    RECHECK_SYSTEM,
                    recheck_prompt,
                    views,
                    set(disputed),
                    slot_by_candidate,
                ),
                set(disputed),
                slot_by_candidate,
            )
        adjudication_prompt = (
            "确定性证据表：\n"
            + json.dumps(_candidate_table(artifacts.bundle.candidates), ensure_ascii=False)
            + "\n独立调查：\n"
            + json.dumps(investigator, ensure_ascii=False)
            + "\n独立反证：\n"
            + json.dumps(reviewer, ensure_ascii=False)
            + "\n冲突复查：\n"
            + json.dumps(recheck, ensure_ascii=False)
            + "\n返回协议："
            '{"decision":"suspicious|clear|unassessable","selected_candidates":["C1"],'
            '"uncertain_candidates":["C2"],"recognized_characters":{"C1":"F","C2":null},'
            '"candidate_evidence":{"C1":{"geometry":["..."],'
            '"appearance":["..."],"counter_evidence":["..."]}},"reasoning_summary":"...",'
            '"unassessable_reason":null}'
        )
        raw_final = self._call(
            "adjudicator",
            ADJUDICATOR_SYSTEM,
            adjudication_prompt,
            views,
            allowed_ids,
            slot_by_candidate,
        )
        decision = self._sanitize_final(raw_final, allowed_ids, artifacts)
        _merge_character_recognition(decision, [investigator, reviewer, recheck])
        trajectory = {
            "schema_version": 2,
            "agent_protocol_version": "sticker_agent_v8_structured_json_v1",
            "max_calls_per_image": self.max_calls_per_image,
            "model": self.model,
            "base_url": self.base_url,
            "structured_output": getattr(self, "structured_output", False),
            "disable_thinking": getattr(self, "disable_thinking", False),
            "max_output_tokens": getattr(self, "max_output_tokens", None),
            "adjudicator_max_output_tokens": getattr(
                self, "adjudicator_max_output_tokens", None
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_shape": list(image.shape),
            "calls": [
                {
                    "stage": call.stage,
                    "cache_key": call.cache_key,
                    "cached": call.cached,
                    "prompt": call.prompt,
                    "response": call.response,
                    "usage": call.usage,
                    "structured_output": call.structured_output,
                    "json_repaired": call.json_repaired,
                }
                for call in self.calls
            ],
            "investigator": investigator,
            "reviewer": reviewer,
            "recheck": recheck,
            "final": decision,
        }
        return decision, trajectory

    @staticmethod
    def _sanitize_final(
        payload: dict[str, Any],
        allowed_ids: set[str],
        artifacts: EvidenceArtifacts,
    ) -> dict[str, Any]:
        raw_selected = [str(value) for value in payload.get("selected_candidates", []) if str(value) in allowed_ids]
        raw_uncertain = [
            str(value)
            for value in payload.get("uncertain_candidates", [])
            if str(value) in allowed_ids and str(value) not in raw_selected
        ]
        candidate_by_id = {candidate.candidate_id: candidate for candidate in artifacts.bundle.candidates}
        # 模型负责复核、字符读取和反证说明，但不能否决已经通过确定性物理
        # 路径的候选。此前只过滤模型主动选择的编号，导致证据已经成立却被
        # 裁决遗漏。v8 将所有多通路物理候选并入，再处理相邻槽材料归属。
        deterministic_candidates = [
            candidate
            for candidate in artifacts.bundle.candidates
            if candidate.candidate_id in allowed_ids
            and candidate_has_multichannel_tamper_support(candidate)
        ]
        selected_candidates, suppressed_spillover = resolve_adjacent_material_ownership(
            deterministic_candidates
        )
        selected = [candidate.candidate_id for candidate in selected_candidates]
        uncertain = [
            value
            for value in raw_uncertain + [item for item in raw_selected if item not in selected]
            if value in candidate_by_id
            and candidate_has_rectangle_material_support(candidate_by_id[value], strict=False)
            and value not in suppressed_spillover
        ]
        selected = sorted(set(selected), key=lambda value: candidate_by_id[value].slot)
        uncertain = sorted(set(uncertain) - set(selected), key=lambda value: candidate_by_id[value].slot)
        selected_slots = {candidate_by_id[value].slot for value in selected if value in candidate_by_id}
        uncertain = [
            value
            for value in uncertain
            if not (
                value in candidate_by_id
                and any(abs(candidate_by_id[value].slot - slot) == 1 for slot in selected_slots)
                and candidate_by_id[value].geometry_score < 0.45
                and candidate_by_id[value].combined_score < 0.55
            )
        ]
        # 最终三态由确定性质量门和物理候选门控制，模型不能把证据不确定性
        # 升级为整牌不可评估。只要输入质量合格，有多通路候选即 suspicious，
        # 否则为 clear；中等候选继续保留供人工复核。
        if artifacts.bundle.quality.assessable:
            decision = "suspicious" if selected else "clear"
        else:
            decision = "unassessable"
            selected = []
            uncertain = []
        raw_evidence = payload.get("candidate_evidence", {})
        candidate_evidence = {}
        for candidate_id in selected + uncertain:
            model_evidence = (
                raw_evidence.get(candidate_id, {})
                if isinstance(raw_evidence, dict)
                else {}
            )
            if not isinstance(model_evidence, dict):
                model_evidence = {}
            selection_sources = ["deterministic_multichannel"]
            if candidate_id in raw_selected:
                selection_sources.append("cloud_adjudicator")
            candidate_evidence[candidate_id] = {
                **model_evidence,
                "deterministic_support_routes": candidate_tamper_support_routes(
                    candidate_by_id[candidate_id]
                ),
                "selection_sources": sorted(selection_sources),
            }
        raw_characters = payload.get("recognized_characters", {})
        recognized_characters = {}
        if isinstance(raw_characters, dict):
            for candidate_id in selected + uncertain:
                value = raw_characters.get(candidate_id)
                if value is None:
                    recognized_characters[candidate_id] = None
                else:
                    recognized_characters[candidate_id] = _normalize_character(value)
        return {
            "decision": decision,
            "selected_candidates": selected,
            "uncertain_candidates": uncertain,
            "candidate_evidence": candidate_evidence,
            "recognized_characters": recognized_characters,
            "suppressed_adjacent_spillover": suppressed_spillover,
            "reasoning_summary": str(payload.get("reasoning_summary", "")),
            "unassessable_reason": (
                "；".join(artifacts.bundle.quality.reasons)
                if not artifacts.bundle.quality.assessable
                else None
            ),
            "decision_note": "质量失败才允许拒判；质量合格时由多通路物理候选并入云端复核结果，并以双侧材料归属抑制相邻串槽；阈值基于开发标注集校准，仍需独立测试集验证",
        }
