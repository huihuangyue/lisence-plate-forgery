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
from .stroke import (
    STROKE_TAMPER_TYPES,
    compact_watchlist,
    maximum_stroke_physical_support,
    stroke_hypotheses,
    stroke_region_evidence,
)
from .types import CandidateRegion, EvidenceArtifacts


INVESTIGATOR_SYSTEM = """你是车牌物理变造证据调查员，核查 whole_character_overlay（整体矩形贴片）、added_stroke（增加笔画）、removed_stroke（遮除/擦除笔画）和 mixed_stroke_edit（一增一消）。
输入依次为原始平面车牌、字符槽映射图和确定性诊断图。Cn 与从左到右第 n 个字符槽 Sn 一一绑定。
整体贴片检查闭合或不闭合的轴对齐材料缝、亮暗成对边缘、材料差异、整高条带及绿牌纵向底色异常。增加/消除笔画先按当前可见字符提出可能原字符，再检查指定局部区域的端点、连接、材料、反光、纹理和冲压亮暗边是否异常。
OCR和字符转换关系只负责提出假设，不能单独构成变造证据。必须同时比较字符天然结构、压边、螺钉、污渍划痕、照明、归一化形变、模糊及压缩伪影等正常解释。
verdict：tamper_support=至少支持一种物理变造；normal_support=更支持正常结构或成像效应；uncertain=仍有竞争解释。suspected_tamper_types 记录具体类型，无类型时为空数组。输出采用给定 JSON 协议。"""


REVIEWER_SYSTEM = """你是独立物理变造证据审查员，在未知调查员结论时逐槽复核 whole_character_overlay、added_stroke、removed_stroke 和 mixed_stroke_edit。
先建立正常解释，再和变造解释比较。整体贴片要求几何/边界与材料证据组合，边界不要求闭合。加/消笔画要求可行字符转换与目标区域内至少一种物理异常组合，重点看端点、接合、局部材料离群、覆盖残槽和冲压边中断。字符相似或 OCR 混淆本身不算证据。
伪彩色只用于定位，原图和结构化数值用于核验。verdict 与 suspected_tamper_types 采用给定协议。"""


RECHECK_SYSTEM = """你是冲突证据复查员。逐项复核整体贴片的边界/材料证据，以及增加或消除笔画在指定区域的端点、连接、材料、纹理与冲压残余；明确区分物理异常与字符相似、照明、模糊和压缩伪影。沿用 Cn=Sn 绑定和给定 JSON 协议。"""


ADJUDICATOR_SYSTEM = """你是车牌物理变造裁决员。汇总整体贴片、增加笔画、消除笔画和一增一消四类解释，并与正常结构和成像效应比较。
整体贴片可由闭合矩形、不完整轴对齐材料缝、整高条带或绿牌纵向底色异常等组合成立。增加/消除笔画必须同时具备可行字符转换和目标局部区域的物理异常；只有字符相似不得判伪。
Cn 与 Sn 一一对应。decision 使用 suspicious、clear、unassessable；竞争解释写入 uncertain_candidates。unassessable 只用于车牌不可见、严重模糊、严重曝光损坏或提取失败。
recognized_characters 记录可见字符。candidate_evidence 可记录 tamper_types、possible_originals 和 stroke_regions，且应写明物理依据。输出采用给定 JSON 协议。"""


_ASSESSMENT_STAGES = {"investigator", "reviewer", "recheck"}
_VERDICTS = {"tamper_support", "normal_support", "uncertain"}
_DECISIONS = {"suspicious", "clear", "unassessable"}
_TAMPER_TYPES = {"whole_character_overlay", *STROKE_TAMPER_TYPES}


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
            "suspected_tamper_types": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(_TAMPER_TYPES)},
                "uniqueItems": True,
                "maxItems": 4,
            },
            "possible_originals": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 1},
                "uniqueItems": True,
                "maxItems": 8,
            },
            "stroke_regions": _string_list_schema(),
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
            "suspected_tamper_types",
            "possible_originals",
            "stroke_regions",
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
            "tamper_types": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(_TAMPER_TYPES)},
                "uniqueItems": True,
                "maxItems": 4,
            },
            "possible_originals": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 1},
                "uniqueItems": True,
                "maxItems": 8,
            },
            "stroke_regions": _string_list_schema(),
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
        "suspected_tamper_types",
        "possible_originals",
        "stroke_regions",
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
        tamper_types = item["suspected_tamper_types"]
        if (
            not isinstance(tamper_types, list)
            or len(tamper_types) != len(set(tamper_types))
            or not set(tamper_types).issubset(_TAMPER_TYPES)
        ):
            raise ValueError(f"{location}.suspected_tamper_types 枚举非法")
        originals = item["possible_originals"]
        if (
            not isinstance(originals, list)
            or len(originals) > 8
            or any(_normalize_character(value) is None for value in originals)
        ):
            raise ValueError(f"{location}.possible_originals 必须是车牌字符数组")
        _validate_string_list(item["stroke_regions"], f"{location}.stroke_regions")
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
    evidence_fields = {
        "geometry",
        "appearance",
        "counter_evidence",
        "tamper_types",
        "possible_originals",
        "stroke_regions",
    }
    for candidate_id, item in evidence.items():
        if not isinstance(item, dict):
            raise ValueError(f"candidate_evidence.{candidate_id} 必须是对象")
        required_evidence_fields = {"geometry", "appearance", "counter_evidence"}
        missing = required_evidence_fields - set(item)
        extra = set(item) - evidence_fields
        if missing or extra:
            raise ValueError(
                f"candidate_evidence.{candidate_id} 字段不符合协议："
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for field in required_evidence_fields:
            _validate_string_list(item[field], f"candidate_evidence.{candidate_id}.{field}")
        if "tamper_types" in item and (
            not isinstance(item["tamper_types"], list)
            or not set(item["tamper_types"]).issubset(_TAMPER_TYPES)
        ):
            raise ValueError(f"candidate_evidence.{candidate_id}.tamper_types 枚举非法")
        if "possible_originals" in item and (
            not isinstance(item["possible_originals"], list)
            or any(_normalize_character(value) is None for value in item["possible_originals"])
        ):
            raise ValueError(f"candidate_evidence.{candidate_id}.possible_originals 非法")
        if "stroke_regions" in item:
            _validate_string_list(item["stroke_regions"], f"candidate_evidence.{candidate_id}.stroke_regions")
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
        "增加/消除笔画反查表如下；它只生成待核验假设，不能单独判伪：\n"
        + json.dumps(compact_watchlist(), ensure_ascii=False, separators=(",", ":"))
        + "\n"
        + json.dumps(table, ensure_ascii=False, separators=(",", ":"))
        + "\n返回协议："
        '{"plate_quality":"assessable|unassessable","candidates":['
        '{"candidate_id":"C1","slot_id":"S1","observed_character":null,'
        '"verdict":"tamper_support|normal_support|uncertain",'
        '"suspected_tamper_types":["whole_character_overlay|added_stroke|removed_stroke|mixed_stroke_edit"],'
        '"possible_originals":["F"],"stroke_regions":["bottom"],'
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
        tamper_types = sorted(
            {
                str(value)
                for value in raw.get("suspected_tamper_types", [])
                if str(value) in _TAMPER_TYPES
            }
        )
        possible_originals = sorted(
            {
                normalized
                for value in raw.get("possible_originals", [])
                if (normalized := _normalize_character(value)) is not None
            }
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "slot_id": f"S{slot}" if slot is not None else None,
                "observed_character": observed,
                "verdict": verdict,
                "suspected_tamper_types": tamper_types,
                "possible_originals": possible_originals,
                "stroke_regions": [str(value) for value in raw.get("stroke_regions", [])][:8],
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
        self.decision_profile = os.environ.get(
            "STICKER_AGENT_DECISION_PROFILE", "high_recall"
        ).strip().lower()
        if self.decision_profile not in {"balanced", "high_recall"}:
            raise RuntimeError("STICKER_AGENT_DECISION_PROFILE 必须是 balanced 或 high_recall")
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
        digest.update(b"physical-tamper-agent-json-protocol-v3")
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
                            "description": "车牌整片、加笔和消笔物理变造检测的受控阶段输出",
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
            + "\n增加/消除笔画反查表（只用于生成假设）：\n"
            + json.dumps(compact_watchlist(), ensure_ascii=False)
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
            '"appearance":["..."],"counter_evidence":["..."],'
            '"tamper_types":["added_stroke"],"possible_originals":["F"],'
            '"stroke_regions":["bottom"]}},"reasoning_summary":"...",'
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
        _merge_character_recognition(raw_final, [investigator, reviewer, recheck])
        decision = self._sanitize_final(
            raw_final,
            allowed_ids,
            artifacts,
            assessments=[investigator, reviewer, recheck],
            decision_profile=self.decision_profile,
        )
        trajectory = {
            "schema_version": 3,
            "agent_protocol_version": "physical_tamper_agent_v9_stroke_structured_json_v1",
            "decision_profile": self.decision_profile,
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
        assessments: list[dict[str, Any] | None] | None = None,
        decision_profile: str = "balanced",
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
        overlay_uncertain = [
            value
            for value in raw_uncertain + [item for item in raw_selected if item not in selected]
            if value in candidate_by_id
            and candidate_has_rectangle_material_support(candidate_by_id[value], strict=False)
            and value not in suppressed_spillover
        ]
        raw_evidence = payload.get("candidate_evidence", {})
        raw_characters = payload.get("recognized_characters", {})
        assessment_rows: dict[str, list[dict[str, Any]]] = {candidate_id: [] for candidate_id in allowed_ids}
        for assessment in assessments or []:
            if not isinstance(assessment, dict):
                continue
            for row in assessment.get("candidates", []):
                if isinstance(row, dict) and row.get("candidate_id") in assessment_rows:
                    assessment_rows[str(row["candidate_id"])].append(row)

        candidate_forgery_types: dict[str, list[str]] = {}
        candidate_possible_originals: dict[str, list[str]] = {}
        candidate_stroke_regions: dict[str, list[str]] = {}
        candidate_stroke_hypotheses: dict[str, list[dict[str, Any]]] = {}
        candidate_stroke_evidence: dict[str, list[dict[str, Any]]] = {}
        stroke_selection_sources: dict[str, list[str]] = {}
        stroke_selected: list[str] = []
        stroke_uncertain: list[str] = []

        for candidate_id, candidate in candidate_by_id.items():
            if candidate_id not in allowed_ids:
                continue
            model_evidence = raw_evidence.get(candidate_id, {}) if isinstance(raw_evidence, dict) else {}
            if not isinstance(model_evidence, dict):
                model_evidence = {}
            reported_types = {
                str(value)
                for value in model_evidence.get("tamper_types", [])
                if str(value) in _TAMPER_TYPES
            }
            reported_originals = {
                normalized
                for value in model_evidence.get("possible_originals", [])
                if (normalized := _normalize_character(value)) is not None
            }
            reported_regions = {
                str(value) for value in model_evidence.get("stroke_regions", []) if str(value)
            }
            type_votes = {tamper_type: 0 for tamper_type in STROKE_TAMPER_TYPES}
            observed_votes: list[str] = []
            for row in assessment_rows[candidate_id]:
                row_types = {
                    str(value)
                    for value in row.get("suspected_tamper_types", [])
                    if str(value) in _TAMPER_TYPES
                }
                reported_types.update(row_types)
                reported_originals.update(
                    normalized
                    for value in row.get("possible_originals", [])
                    if (normalized := _normalize_character(value)) is not None
                )
                reported_regions.update(str(value) for value in row.get("stroke_regions", []) if str(value))
                observed_value = _normalize_character(row.get("observed_character"))
                if observed_value is not None:
                    observed_votes.append(observed_value)
                if row.get("verdict") == "tamper_support":
                    for tamper_type in row_types & STROKE_TAMPER_TYPES:
                        type_votes[tamper_type] += 1

            stroke_types = reported_types & STROKE_TAMPER_TYPES
            observed = (
                _normalize_character(raw_characters.get(candidate_id))
                if isinstance(raw_characters, dict)
                else None
            )
            if observed is None and observed_votes:
                counts = {value: observed_votes.count(value) for value in set(observed_votes)}
                best = max(counts.values())
                winners = sorted(value for value, count in counts.items() if count == best)
                if len(winners) == 1:
                    observed = winners[0]
            hypotheses = stroke_hypotheses(observed)
            if stroke_types:
                hypotheses = [item for item in hypotheses if item["tamper_type"] in stroke_types]
            if reported_originals:
                hypotheses = [item for item in hypotheses if item["possible_original"] in reported_originals]
            if reported_regions and hypotheses:
                region_matched = [
                    item for item in hypotheses if set(item.get("regions", [])) & reported_regions
                ]
                if region_matched:
                    hypotheses = region_matched
            region_evidence = stroke_region_evidence(artifacts, candidate, hypotheses)
            physical_support = maximum_stroke_physical_support(region_evidence)
            if candidate_id in raw_selected:
                for tamper_type in stroke_types:
                    type_votes[tamper_type] += 1
            support_count = max((type_votes[value] for value in stroke_types), default=0)
            tier1 = any(item.get("priority") == "tier1" for item in hypotheses)

            if hypotheses and stroke_types:
                sources = [
                    "model_type_votes:"
                    + ",".join(f"{key}={type_votes[key]}" for key in sorted(stroke_types))
                ]
                if candidate_id in raw_selected:
                    sources.append("adjudicator_stroke_selection")
                sources.append(f"local_stroke_physical:{physical_support:.4f}")
                stroke_selection_sources[candidate_id] = sources
                candidate_forgery_types[candidate_id] = sorted(stroke_types)
                candidate_possible_originals[candidate_id] = sorted(
                    {str(item["possible_original"]) for item in hypotheses}
                )
                candidate_stroke_regions[candidate_id] = sorted(
                    {str(region) for item in hypotheses for region in item.get("regions", [])}
                )
                candidate_stroke_hypotheses[candidate_id] = hypotheses
                candidate_stroke_evidence[candidate_id] = region_evidence

                if decision_profile == "high_recall":
                    selected_threshold = 0.14 if tier1 else 0.20
                    review_threshold = 0.07
                    enough_votes = support_count >= 2
                else:
                    selected_threshold = 0.22 if tier1 else 0.28
                    review_threshold = 0.12
                    enough_votes = support_count >= 3
                if enough_votes and physical_support >= selected_threshold:
                    stroke_selected.append(candidate_id)
                elif support_count >= 1 and physical_support >= review_threshold:
                    stroke_uncertain.append(candidate_id)

        selected.extend(stroke_selected)
        uncertain = overlay_uncertain + stroke_uncertain
        selected = sorted(set(selected), key=lambda value: candidate_by_id[value].slot)
        uncertain = sorted(set(uncertain) - set(selected), key=lambda value: candidate_by_id[value].slot)
        selected_slots = {candidate_by_id[value].slot for value in selected if value in candidate_by_id}
        uncertain = [
            value
            for value in uncertain
            if not (
                value in candidate_by_id
                and value not in candidate_forgery_types
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
        candidate_evidence = {}
        for candidate_id in selected + uncertain:
            model_evidence = (
                raw_evidence.get(candidate_id, {})
                if isinstance(raw_evidence, dict)
                else {}
            )
            if not isinstance(model_evidence, dict):
                model_evidence = {}
            selection_sources = []
            if candidate_has_multichannel_tamper_support(candidate_by_id[candidate_id]):
                selection_sources.append("deterministic_multichannel")
            if candidate_id in raw_selected:
                selection_sources.append("cloud_adjudicator")
            selection_sources.extend(stroke_selection_sources.get(candidate_id, []))
            tamper_types = set(candidate_forgery_types.get(candidate_id, []))
            if candidate_has_multichannel_tamper_support(candidate_by_id[candidate_id]):
                tamper_types.add("whole_character_overlay")
            candidate_forgery_types[candidate_id] = sorted(tamper_types)
            candidate_evidence[candidate_id] = {
                **model_evidence,
                "deterministic_support_routes": candidate_tamper_support_routes(
                    candidate_by_id[candidate_id]
                ),
                "selection_sources": sorted(selection_sources),
                "stroke_hypotheses": candidate_stroke_hypotheses.get(candidate_id, []),
                "stroke_region_evidence": candidate_stroke_evidence.get(candidate_id, []),
            }
        recognized_characters = {}
        if isinstance(raw_characters, dict):
            for candidate_id in selected + uncertain:
                value = raw_characters.get(candidate_id)
                if value is None:
                    observations = [
                        _normalize_character(row.get("observed_character"))
                        for row in assessment_rows.get(candidate_id, [])
                    ]
                    observations = [item for item in observations if item is not None]
                    counts = {item: observations.count(item) for item in set(observations)}
                    winners = (
                        sorted(item for item, count in counts.items() if count == max(counts.values()))
                        if counts
                        else []
                    )
                    recognized_characters[candidate_id] = winners[0] if len(winners) == 1 else None
                else:
                    recognized_characters[candidate_id] = _normalize_character(value)
        recognition_status = {
            candidate_id: str(status)
            for candidate_id, status in payload.get("character_recognition_status", {}).items()
            if candidate_id in selected + uncertain
        } if isinstance(payload.get("character_recognition_status"), dict) else {}
        for candidate_id in selected + uncertain:
            if candidate_id in recognition_status:
                continue
            observations = [
                _normalize_character(row.get("observed_character"))
                for row in assessment_rows.get(candidate_id, [])
            ]
            matches = sum(
                value == recognized_characters.get(candidate_id)
                for value in observations
                if value is not None
            )
            recognition_status[candidate_id] = (
                "cross_review_consensus"
                if recognized_characters.get(candidate_id) is not None and matches >= 2
                else "single_review_observation"
                if recognized_characters.get(candidate_id) is not None
                else "unreadable_or_not_returned"
            )
        return {
            "decision": decision,
            "selected_candidates": selected,
            "uncertain_candidates": uncertain,
            "candidate_evidence": candidate_evidence,
            "recognized_characters": recognized_characters,
            "character_recognition_status": recognition_status,
            "candidate_forgery_types": {
                candidate_id: candidate_forgery_types.get(candidate_id, [])
                for candidate_id in selected + uncertain
            },
            "candidate_possible_originals": {
                candidate_id: candidate_possible_originals.get(candidate_id, [])
                for candidate_id in selected + uncertain
            },
            "candidate_stroke_regions": {
                candidate_id: candidate_stroke_regions.get(candidate_id, [])
                for candidate_id in selected + uncertain
            },
            "suppressed_adjacent_spillover": suppressed_spillover,
            "reasoning_summary": str(payload.get("reasoning_summary", "")),
            "unassessable_reason": (
                "；".join(artifacts.bundle.quality.reasons)
                if not artifacts.bundle.quality.assessable
                else None
            ),
            "decision_profile": decision_profile,
            "decision_note": "质量失败才允许拒判；整体贴片由多通路物理候选控制；加/消笔画须由字符转换假设、局部物理异常和多阶段模型支持共同成立。high_recall 为待校准工作点，召回率与查准率必须在独立人工标注集验证",
        }
