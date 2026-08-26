"""将号牌四点定位、透视归一化和整体贴片筛查串成可复现流水线。"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from queue import SimpleQueue
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np

try:
    from shape.geometry import (
        PlateDetection,
        draw_points,
        draw_quad,
        order_corners,
        read_image,
        rectify_plate,
        write_image,
    )
    from shape.local_model import DEFAULT_MODEL, detect_plate_local
    from sticker.agent import StickerAgentHarness
    from sticker.evidence import analyze_plate_evidence, candidate_tamper_support_routes
    from sticker.output import character_decision_entries, make_local_decision, render_final
except ModuleNotFoundError:  # pragma: no cover - 支持 python -m src.pipeline...
    from src.shape.geometry import (
        PlateDetection,
        draw_points,
        draw_quad,
        order_corners,
        read_image,
        rectify_plate,
        write_image,
    )
    from src.shape.local_model import DEFAULT_MODEL, detect_plate_local
    from src.sticker.agent import StickerAgentHarness
    from src.sticker.evidence import analyze_plate_evidence, candidate_tamper_support_routes
    from src.sticker.output import character_decision_entries, make_local_decision, render_final


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SIX_IMAGE_NAMES = (
    "01_points.jpg",
    "02_quad.jpg",
    "03_rectified.jpg",
    "04_candidate_overlay.jpg",
    "05_final_marked.jpg",
    "06_candidate_mask.png",
)

Detector = Callable[[np.ndarray], PlateDetection]
ProgressCallback = Callable[[int, int, dict[str, Any]], None]
QWEN3_VL_PLUS_PRICING_SOURCE = "https://help.aliyun.com/zh/model-studio/qwen3-vl-plus"
QWEN3_VL_PLUS_PRICE_TIERS_CNY = (
    (32_768, 1.0, 10.0),
    (131_072, 1.5, 15.0),
    (262_144, 3.0, 30.0),
)


def read_plate_ocr_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    """读取独立 OCR 预识别缓存；它是质量证据，不是人工真值。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    results = payload.get("results", {})
    if not isinstance(results, dict):
        raise ValueError(f"车牌 OCR 缓存 results 必须是 object：{path}")
    return {
        str(image_id): result
        for image_id, result in results.items()
        if isinstance(result, dict)
    }


def _apply_plate_ocr_quality(
    artifacts: Any,
    ocr_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """完整 OCR 可读是充分条件；OCR 失败不能否定本地质量门。"""

    deterministic_quality = artifacts.bundle.quality
    raw_plate = str((ocr_result or {}).get("plate_text", ""))
    plate_text = re.sub(r"[\s·•.・\-]+", "", raw_plate.strip()).upper()
    slot_count = len(artifacts.bundle.slots)
    ocr_full_plate_readable = bool(
        ocr_result
        and ocr_result.get("readable")
        and not ocr_result.get("error")
        and len(plate_text) == slot_count
    )
    if ocr_full_plate_readable and not deterministic_quality.assessable:
        artifacts.bundle = replace(
            artifacts.bundle,
            quality=replace(deterministic_quality, assessable=True, reasons=[]),
        )
    return {
        "rule": "deterministic_quality_or_full_plate_ocr",
        "deterministic_assessable": deterministic_quality.assessable,
        "deterministic_reasons": list(deterministic_quality.reasons),
        "ocr_supplied": ocr_result is not None,
        "ocr_plate_text": plate_text,
        "ocr_model": str((ocr_result or {}).get("model", "")),
        "ocr_full_plate_readable": ocr_full_plate_readable,
        "final_assessable": artifacts.bundle.quality.assessable,
    }


def _agent_usage_summary(harness: StickerAgentHarness) -> dict[str, Any]:
    calls = list(harness.calls)
    billable = [call for call in calls if not call.cached]

    def token_sum(name: str) -> int:
        return sum(int(call.usage.get(name) or 0) for call in billable)

    usage_complete = all(
        call.usage.get("input_tokens") is not None and call.usage.get("output_tokens") is not None
        for call in billable
    )
    return {
        "model": harness.model,
        "endpoint_host": urlparse(harness.base_url or "").hostname,
        "recorded_calls": len(calls),
        "billable_api_calls": len(billable),
        "cache_hits": sum(call.cached for call in calls),
        "input_tokens": token_sum("input_tokens"),
        "output_tokens": token_sum("output_tokens"),
        "total_tokens": token_sum("total_tokens"),
        "usage_complete": usage_complete,
        "stages": [
            {
                "stage": call.stage,
                "cached": call.cached,
                "input_tokens": call.usage.get("input_tokens"),
                "output_tokens": call.usage.get("output_tokens"),
                "total_tokens": call.usage.get("total_tokens"),
            }
            for call in calls
        ],
    }


def discover_images(input_path: str | Path) -> list[Path]:
    """递归发现输入图片；单文件输入保持为一项。"""

    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"输入文件不是支持的图片格式：{path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"输入路径不存在：{path}")
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def _safe_sample_id(path: Path, used: set[str]) -> str:
    base = re.sub(r"[^0-9A-Za-z._-]+", "_", path.stem).strip("._-") or "image"
    candidate = base
    if candidate in used:
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8]
        candidate = f"{base}_{digest}"
    counter = 2
    while candidate in used:
        candidate = f"{base}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def _candidate_summary(artifacts: Any, decision: dict[str, Any]) -> list[dict[str, Any]]:
    selected = {str(value) for value in decision.get("selected_candidates", [])}
    uncertain = {str(value) for value in decision.get("uncertain_candidates", [])}
    wanted = selected | uncertain
    recognized = decision.get("recognized_characters", {})
    recognition_status = decision.get("character_recognition_status", {})
    return sorted([
        {
            "candidate_id": candidate.candidate_id,
            "slot": candidate.slot,
            "character": recognized.get(candidate.candidate_id) if isinstance(recognized, dict) else None,
            "recognition_status": (
                recognition_status.get(candidate.candidate_id)
                if isinstance(recognition_status, dict)
                else None
            ),
            "bbox": list(candidate.bbox),
            "confidence_level": "high" if candidate.candidate_id in selected else "medium",
            "tamper_types": (
                decision.get("candidate_forgery_types", {}).get(candidate.candidate_id, [])
                if isinstance(decision.get("candidate_forgery_types"), dict)
                else []
            ),
            "possible_original_characters": (
                decision.get("candidate_possible_originals", {}).get(candidate.candidate_id, [])
                if isinstance(decision.get("candidate_possible_originals"), dict)
                else []
            ),
            "stroke_regions": (
                decision.get("candidate_stroke_regions", {}).get(candidate.candidate_id, [])
                if isinstance(decision.get("candidate_stroke_regions"), dict)
                else []
            ),
            "geometry_score": candidate.geometry_score,
            "appearance_score": candidate.appearance_score,
            "paired_edge_score": candidate.paired_edge_score,
            "combined_score": candidate.combined_score,
        }
        for candidate in artifacts.bundle.candidates
        if candidate.candidate_id in wanted
    ], key=lambda item: int(item["slot"]))


def analyze_vehicle_image(
    input_path: str | Path,
    sample_dir: str | Path,
    *,
    detector_model: str | Path = DEFAULT_MODEL,
    detector: Detector | None = None,
    sticker_method: str = "agent",
    agent_harness: StickerAgentHarness | None = None,
    plate_ocr_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """处理一张整车图，并在样本目录中严格写出六张阶段图。"""

    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    input_path = Path(input_path)
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=False)

    stage_started = time.perf_counter()
    image = read_image(input_path)
    timings["input_read"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    if detector is None:
        detection = detect_plate_local(image, model_path=detector_model)
    else:
        detection = detector(image)
    timings["plate_localization"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    corners = order_corners(detection.corners)
    rectified, homography = rectify_plate(image, detection)
    timings["perspective_rectification"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    artifacts = analyze_plate_evidence(rectified, plate_type=detection.plate_type)
    quality_gate = _apply_plate_ocr_quality(artifacts, plate_ocr_result)
    timings["sticker_evidence"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    llm_usage = None
    if sticker_method == "agent":
        if agent_harness is None:
            raise RuntimeError("agent 贴牌判断需要 StickerAgentHarness")
        decision, _ = agent_harness.run(rectified, artifacts)
        llm_usage = _agent_usage_summary(agent_harness)
    elif sticker_method == "local":
        decision = make_local_decision(artifacts)
    else:
        raise ValueError("sticker_method 必须是 local 或 agent")
    timings["sticker_decision"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    selected = [str(value) for value in decision.get("selected_candidates", [])]
    uncertain = [str(value) for value in decision.get("uncertain_candidates", [])]
    final_marked, candidate_mask = render_final(
        rectified, artifacts, selected, uncertain, decision
    )
    timings["final_render"] = time.perf_counter() - stage_started

    images = {
        "01_points.jpg": draw_points(image, corners),
        "02_quad.jpg": draw_quad(image, corners),
        "03_rectified.jpg": rectified,
        "04_candidate_overlay.jpg": artifacts.candidate_overlay,
        "05_final_marked.jpg": final_marked,
        "06_candidate_mask.png": candidate_mask,
    }
    stage_started = time.perf_counter()
    for filename in SIX_IMAGE_NAMES:
        write_image(sample_dir / filename, images[filename])
    timings["six_image_write"] = time.perf_counter() - stage_started

    actual_files = {item.name for item in sample_dir.iterdir() if item.is_file()}
    if actual_files != set(SIX_IMAGE_NAMES):
        raise RuntimeError(f"六图输出契约失败：实际文件={sorted(actual_files)}")

    stage_started = time.perf_counter()
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    timings["input_hash"] = time.perf_counter() - stage_started
    timings["total"] = time.perf_counter() - total_started
    rounded_timings = {name: round(value, 6) for name, value in timings.items()}

    result = {
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "plate_type": detection.plate_type.value,
        "localization_method": detection.method,
        "localization_confidence": round(float(detection.confidence), 6),
        "corners": corners.round(3).tolist(),
        "homography": homography.round(10).tolist(),
        "rectified_size": [int(rectified.shape[1]), int(rectified.shape[0])],
        "quality": {
            "assessable": artifacts.bundle.quality.assessable,
            "reasons": artifacts.bundle.quality.reasons,
            "laplacian_variance": artifacts.bundle.quality.laplacian_variance,
            "clipped_fraction": artifacts.bundle.quality.clipped_fraction,
            **quality_gate,
        },
        "decision": decision.get("decision", "unassessable"),
        "tamper_scope": (
            ["whole_character_overlay", "added_stroke", "removed_stroke", "mixed_stroke_edit"]
            if sticker_method == "agent"
            else ["whole_character_overlay"]
        ),
        "decision_profile": decision.get("decision_profile"),
        "selected_candidates": selected,
        "uncertain_candidates": uncertain,
        "tampered_characters": character_decision_entries(
            artifacts, decision, selected, confidence_level="high"
        ),
        "uncertain_characters": character_decision_entries(
            artifacts, decision, uncertain, confidence_level="medium"
        ),
        "suspected_regions": _candidate_summary(artifacts, decision),
        "candidate_diagnostics": [
            {
                "candidate_id": candidate.candidate_id,
                "slot": candidate.slot,
                "geometry_score": candidate.geometry_score,
                "appearance_score": candidate.appearance_score,
                "paired_edge_score": candidate.paired_edge_score,
                "combined_score": candidate.combined_score,
                "support_routes": candidate_tamper_support_routes(candidate),
                "features": candidate.features,
            }
            for candidate in sorted(artifacts.bundle.candidates, key=lambda item: item.slot)
        ],
        "suppressed_adjacent_spillover": decision.get(
            "suppressed_adjacent_spillover", {}
        ),
        "decision_note": decision.get("decision_note"),
        "six_images": list(SIX_IMAGE_NAMES),
        "timing_seconds": rounded_timings,
    }
    if llm_usage is not None:
        result["llm_usage"] = llm_usage
    return result


def _unique_run_dir(output_root: Path, prefix: str, timestamp: str) -> Path:
    run_dir = output_root / f"{prefix}_{timestamp}"
    counter = 2
    while run_dir.exists():
        run_dir = output_root / f"{prefix}_{timestamp}_{counter:02d}"
        counter += 1
    run_dir.mkdir(parents=True)
    return run_dir


def _answer_filename(sample_id: str, result: dict[str, Any]) -> str:
    decision = str(result["decision"])
    high_slots = [
        str(region["slot"])
        for region in result["suspected_regions"]
        if region["confidence_level"] == "high"
    ]
    review_slots = [
        str(region["slot"])
        for region in result["suspected_regions"]
        if region["confidence_level"] == "medium"
    ]
    if high_slots:
        suffix = f"suspect-slots-{'-'.join(high_slots)}"
    elif review_slots:
        suffix = f"review-slots-{'-'.join(review_slots)}"
    else:
        suffix = "slots-none"
    return f"{sample_id}__{decision}__{suffix}.jpg"


def _write_manifest_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "input_path",
                "status",
                "decision",
                "plate_type",
                "selected_candidates",
                "uncertain_candidates",
                "total_seconds",
                "error",
            ),
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample["sample_id"],
                    "input_path": sample["input_path"],
                    "status": sample["status"],
                    "decision": sample.get("decision", ""),
                    "plate_type": sample.get("plate_type", ""),
                    "selected_candidates": ";".join(sample.get("selected_candidates", [])),
                    "uncertain_candidates": ";".join(sample.get("uncertain_candidates", [])),
                    "total_seconds": sample.get("timing_seconds", {}).get("total", ""),
                    "error": sample.get("error", ""),
                }
            )


def _duration_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(array)), 6),
        "p50": round(float(np.percentile(array, 50)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "max": round(float(np.max(array)), 6),
    }


def _timing_report(
    samples: list[dict[str, Any]],
    processing_wall_seconds: float,
    batch_stage_seconds: dict[str, float],
) -> dict[str, Any]:
    successful = [sample for sample in samples if sample["status"] == "success"]
    totals = [float(sample["timing_seconds"]["total"]) for sample in successful]
    stage_names = (
        "input_read",
        "plate_localization",
        "perspective_rectification",
        "sticker_evidence",
        "sticker_decision",
        "final_render",
        "six_image_write",
        "input_hash",
    )
    stage_totals = {
        name: round(sum(float(sample["timing_seconds"][name]) for sample in successful), 6)
        for name in stage_names
    }
    return {
        "clock": "time.perf_counter_monotonic_wall_clock",
        "processing_wall_seconds": round(processing_wall_seconds, 6),
        "batch_stage_seconds": {
            name: round(value, 6) for name, value in batch_stage_seconds.items()
        },
        "successful_image_seconds": _duration_stats(totals),
        "stage_total_seconds": stage_totals,
        "throughput_successful_images_per_second": (
            round(len(successful) / processing_wall_seconds, 6) if processing_wall_seconds > 0 else None
        ),
        "scope_note": "批次处理时间包含图片发现、选择、推理、六图写出和答案图复制；不包含最终 JSON/CSV 报告写出。",
    }


def _cost_report(
    samples: list[dict[str, Any]],
    input_price_per_million_cny: float | None,
    output_price_per_million_cny: float | None,
) -> dict[str, Any]:
    usages = [sample["llm_usage"] for sample in samples if "llm_usage" in sample]
    api_calls = sum(int(usage["billable_api_calls"]) for usage in usages)
    cache_hits = sum(int(usage["cache_hits"]) for usage in usages)
    input_tokens = sum(int(usage["input_tokens"]) for usage in usages)
    output_tokens = sum(int(usage["output_tokens"]) for usage in usages)
    usage_complete = all(bool(usage["usage_complete"]) for usage in usages)
    manual_rates = input_price_per_million_cny is not None and output_price_per_million_cny is not None
    call_costs: list[dict[str, Any]] = []
    pricing_complete = usage_complete
    total_cost = 0.0
    for usage in usages:
        model = str(usage.get("model", ""))
        endpoint_host = str(usage.get("endpoint_host") or "")
        for stage in usage.get("stages", []):
            if stage.get("cached"):
                continue
            stage_input = stage.get("input_tokens")
            stage_output = stage.get("output_tokens")
            if stage_input is None or stage_output is None:
                pricing_complete = False
                continue
            if manual_rates:
                input_rate = float(input_price_per_million_cny)
                output_rate = float(output_price_per_million_cny)
                rate_source = "manual_flat_override"
            elif model.startswith("qwen3-vl-plus") and endpoint_host.endswith("dashscope.aliyuncs.com"):
                matching_tier = next(
                    (tier for tier in QWEN3_VL_PLUS_PRICE_TIERS_CNY if int(stage_input) <= tier[0]),
                    None,
                )
                if matching_tier is None:
                    pricing_complete = False
                    continue
                _, input_rate, output_rate = matching_tier
                rate_source = "aliyun_official_tiered_2026-08-21"
            else:
                pricing_complete = False
                continue
            stage_cost = (int(stage_input) * input_rate + int(stage_output) * output_rate) / 1_000_000.0
            total_cost += stage_cost
            call_costs.append(
                {
                    "model": model,
                    "stage": stage.get("stage"),
                    "input_tokens": int(stage_input),
                    "output_tokens": int(stage_output),
                    "input_rate_per_million_cny": input_rate,
                    "output_rate_per_million_cny": output_rate,
                    "cost_cny": round(stage_cost, 8),
                    "rate_source": rate_source,
                }
            )
    if api_calls == 0:
        cost_cny: float | None = 0.0
        status = "no_billable_api_calls"
    elif not pricing_complete or len(call_costs) != api_calls:
        cost_cny = None
        status = "incomplete_usage_or_unknown_pricing"
    else:
        cost_cny = total_cost
        status = "calculated_from_tokens"
    return {
        "scope": "external_large_model_api_only",
        "status": status,
        "external_api_calls": api_calls,
        "cache_hits": cache_hits,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_price_per_million_tokens_cny": input_price_per_million_cny,
        "output_price_per_million_tokens_cny": output_price_per_million_cny,
        "default_pricing_model": "qwen3-vl-plus",
        "default_pricing_source": QWEN3_VL_PLUS_PRICING_SOURCE,
        "default_pricing_checked_at": "2026-08-21",
        "default_price_tiers_cny": [
            {
                "max_input_tokens_per_request": max_tokens,
                "input_per_million": input_rate,
                "output_per_million": output_rate,
            }
            for max_tokens, input_rate, output_rate in QWEN3_VL_PLUS_PRICE_TIERS_CNY
        ],
        "call_costs": call_costs,
        "external_model_cost_cny": round(cost_cny, 8) if cost_cny is not None else None,
        "basis_note": "只对非缓存且有 Token 用量记录的成功响应计费；DashScope qwen3-vl-plus 默认按官方单次请求输入长度阶梯价，手工单价参数可覆盖默认值。",
        "limitations": [
            "API 重试中未产生可用 usage 的失败响应可能无法计入。",
            "默认单价不包含免费额度、活动折扣、上下文缓存优惠或其他账单调整，最终以服务商账单为准。",
        ],
    }


def run_batch(
    input_path: str | Path,
    output_root: str | Path,
    *,
    count: int,
    selection: str = "random",
    seed: int | None = None,
    prefix: str = "pipeline_batch",
    detector_model: str | Path = DEFAULT_MODEL,
    detector: Detector | None = None,
    sticker_method: str = "agent",
    agent_model: str | None = None,
    agent_base_url: str | None = None,
    agent_use_cache: bool = True,
    agent_max_calls_per_image: int = 3,
    workers: int = 4,
    progress_callback: ProgressCallback | None = None,
    input_price_per_million_cny: float | None = None,
    output_price_per_million_cny: float | None = None,
    plate_ocr_results: dict[str, dict[str, Any]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """按随机或文件名排序前 N 张运行流水线，并记录耗时与估算成本。"""

    if count <= 0:
        raise ValueError("count 必须大于 0")
    if selection not in {"random", "first"}:
        raise ValueError("selection 必须是 random 或 first")
    if sticker_method not in {"local", "agent"}:
        raise ValueError("sticker_method 必须是 local 或 agent")
    if agent_max_calls_per_image not in {2, 3, 4}:
        raise ValueError("agent_max_calls_per_image 必须是 2、3 或 4")
    if workers <= 0:
        raise ValueError("workers 必须大于 0")
    for price in (input_price_per_million_cny, output_price_per_million_cny):
        if price is not None and price < 0:
            raise ValueError("大模型 Token 单价不能为负数")
    processing_started = time.perf_counter()

    discovery_started = time.perf_counter()
    images = discover_images(input_path)
    if count > len(images):
        raise ValueError(f"请求处理 {count} 张，但只发现 {len(images)} 张图片")

    if selection == "first":
        selected_paths = images[:count]
        effective_seed = None
    else:
        if seed is None:
            raise ValueError("random 选择方式必须提供 seed")
        selected_paths = random.Random(seed).sample(images, count)
        effective_seed = seed
    discovery_and_selection_seconds = time.perf_counter() - discovery_started

    setup_started = time.perf_counter()
    now = datetime.now().astimezone()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    run_dir = _unique_run_dir(Path(output_root), prefix, timestamp)
    results_dir = run_dir / "results"
    results_dir.mkdir()
    used_ids: set[str] = set()
    jobs: list[tuple[int, Path, str, Path]] = []
    for index, path in enumerate(selected_paths, start=1):
        sample_id = _safe_sample_id(path, used_ids)
        jobs.append((index, path, sample_id, run_dir / sample_id))

    effective_workers = min(workers, count)
    harness_pool: SimpleQueue[StickerAgentHarness] | None = None
    if sticker_method == "agent":
        harness_pool = SimpleQueue()
        for _ in range(effective_workers):
            harness_pool.put(
                StickerAgentHarness(
                    model=agent_model,
                    base_url=agent_base_url,
                    cache_dir=Path(output_root) / ".pipeline_api_cache",
                    use_cache=agent_use_cache,
                    max_calls_per_image=agent_max_calls_per_image,
                )
            )
    run_setup_seconds = time.perf_counter() - setup_started

    def process_job(job: tuple[int, Path, str, Path]) -> dict[str, Any]:
        index, path, sample_id, sample_dir = job
        sample_started = time.perf_counter()
        agent_harness = harness_pool.get() if harness_pool is not None else None
        if agent_harness is not None:
            agent_harness.calls = []
        try:
            result = analyze_vehicle_image(
                path,
                sample_dir,
                detector_model=detector_model,
                detector=detector,
                sticker_method=sticker_method,
                agent_harness=agent_harness,
                plate_ocr_result=(plate_ocr_results or {}).get(path.stem),
            )
            result.update({"sample_id": sample_id, "status": "success", "selection_index": index})
            answer_path = results_dir / _answer_filename(sample_id, result)
            shutil.copy2(sample_dir / "05_final_marked.jpg", answer_path)
            result["answer_image"] = str(answer_path.relative_to(run_dir))
        except Exception as exc:  # 单张失败应保留批次其余结果和错误证据。
            cleanup_error = None
            try:
                if sample_dir.exists():
                    shutil.rmtree(sample_dir)
            except OSError as cleanup_exc:
                cleanup_error = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            result = {
                "sample_id": sample_id,
                "input_path": str(path),
                "status": "failed",
                "selection_index": index,
                "error": f"{type(exc).__name__}: {exc}",
                "timing_seconds": {"total": round(time.perf_counter() - sample_started, 6)},
            }
            if agent_harness is not None and agent_harness.calls:
                result["llm_usage"] = _agent_usage_summary(agent_harness)
            if cleanup_error is not None:
                result["cleanup_error"] = cleanup_error
        finally:
            if harness_pool is not None and agent_harness is not None:
                harness_pool.put(agent_harness)
        return result

    sample_loop_started = time.perf_counter()
    samples_by_index: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="plate-pipeline") as executor:
        futures = {executor.submit(process_job, job): job[0] for job in jobs}
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            samples_by_index[int(result["selection_index"])] = result
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, count, result)
    samples = [samples_by_index[index] for index in range(1, count + 1)]
    sample_loop_seconds = time.perf_counter() - sample_loop_started

    successful = sum(sample["status"] == "success" for sample in samples)
    processing_wall_seconds = time.perf_counter() - processing_started
    batch_stage_seconds = {
        "image_discovery_and_selection": discovery_and_selection_seconds,
        "run_directory_setup": run_setup_seconds,
        "sample_processing_loop": sample_loop_seconds,
    }
    report = {
        "schema_version": 3,
        "run_id": run_dir.name,
        "created_at": now.isoformat(timespec="seconds"),
        "input_root": str(Path(input_path)),
        "selection_method": selection,
        "random_seed": effective_seed,
        "discovered_images": len(images),
        "requested_count": count,
        "successful_count": successful,
        "failed_count": count - successful,
        "parallelism": {
            "executor": "thread_pool",
            "configured_workers": workers,
            "effective_workers": effective_workers,
        },
        "localization_method": "local_onnx_yolov7_lite_t_kpt",
        "sticker_method": (
            "controlled_multiround_cloud_agent_v7_quality_gated_abstention"
            if sticker_method == "agent"
            else "deterministic_sticker_v7_quality_gated_abstention"
        ),
        "plate_ocr_quality_gate": {
            "enabled": plate_ocr_results is not None,
            "cached_results": len(plate_ocr_results or {}),
            "rule": "deterministic_quality_or_full_plate_ocr",
        },
        "agent_max_calls_per_image": agent_max_calls_per_image if sticker_method == "agent" else 0,
        "calibration_warning": "贴牌阈值尚未使用真实贴片真值校准，结果是筛查答案而非准确率或司法结论。",
        "sample_image_contract": list(SIX_IMAGE_NAMES),
        "timing": _timing_report(samples, processing_wall_seconds, batch_stage_seconds),
        "cost": _cost_report(
            samples,
            input_price_per_million_cny,
            output_price_per_million_cny,
        ),
        "samples": samples,
    }
    (run_dir / "batch_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_manifest_csv(run_dir / "selection_manifest.csv", samples)

    answer_files = [item for item in results_dir.iterdir() if item.is_file()]
    if len(answer_files) != successful or any(item.suffix.lower() != ".jpg" for item in answer_files):
        raise RuntimeError("results 目录契约失败：必须只包含每个成功样本的一张最终 JPG 答案图")
    return run_dir, report


def run_random_batch(
    input_path: str | Path,
    output_root: str | Path,
    *,
    count: int,
    seed: int,
    prefix: str = "pipeline_batch",
    detector_model: str | Path = DEFAULT_MODEL,
    detector: Detector | None = None,
    sticker_method: str = "agent",
    agent_max_calls_per_image: int = 3,
    workers: int = 4,
    progress_callback: ProgressCallback | None = None,
    plate_ocr_results: dict[str, dict[str, Any]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """兼容原有调用：随机抽样并运行统一流水线。"""

    return run_batch(
        input_path,
        output_root,
        count=count,
        selection="random",
        seed=seed,
        prefix=prefix,
        detector_model=detector_model,
        detector=detector,
        sticker_method=sticker_method,
        agent_max_calls_per_image=agent_max_calls_per_image,
        workers=workers,
        progress_callback=progress_callback,
        plate_ocr_results=plate_ocr_results,
    )
