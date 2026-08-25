"""为人工贴牌标注集合预填云端车牌 OCR；结果独立缓存，不计作人工标签。"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from .annotate_web import AnnotationItem, load_or_create_selection, normalize_plate_text


OCR_SYSTEM = """你负责读取已经透视校正的中国汽车号牌文字。只进行文字识别。"""

OCR_PROMPT = """识别图中完整车牌号并输出一个 JSON object：
{"plate":"京A12345","readable":true,"note":"清楚"}
plate 保留中文省份简称，去掉圆点、空格和连字符。新能源号牌通常8个字符，普通蓝牌通常7个字符。
无法完整读出时 readable=false，plate 填最可能的可见结果或空字符串，并在 note 简述原因。只输出 JSON。"""


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"模型没有返回 JSON：{stripped[:200]}")
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("OCR 模型 JSON 顶层必须是对象")
    return payload


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _usage(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def recognize_item(
    client: OpenAI,
    item: AnnotationItem,
    *,
    model: str,
    retries: int = 3,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=160,
                messages=[
                    {"role": "system", "content": OCR_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": _image_data_url(item.rectified_path)}},
                            {"type": "text", "text": OCR_PROMPT},
                        ],
                    },
                ],
            )
            raw_content = response.choices[0].message.content
            if not isinstance(raw_content, str):
                raise ValueError("OCR 模型返回内容为空或不是文本")
            parsed = _extract_json(raw_content)
            plate_text = normalize_plate_text(str(parsed.get("plate", "")))
            return {
                "image_id": item.image_id,
                "plate_text": plate_text,
                "readable": bool(parsed.get("readable", False)),
                "note": str(parsed.get("note", ""))[:300],
                "slot_count": item.slot_count,
                "valid_length": len(plate_text) == item.slot_count,
                "model": model,
                "usage": _usage(response),
                "error": "",
            }
        except Exception as exc:  # 单图失败不应丢掉其他已完成 OCR
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.0 + attempt)
    return {
        "image_id": item.image_id,
        "plate_text": "",
        "readable": False,
        "note": "识别调用失败",
        "slot_count": item.slot_count,
        "valid_length": False,
        "model": model,
        "usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None},
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def _new_cache(batch: Path, annotation_output: Path, model: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "updated_at": "",
        "batch_root": str(batch.resolve()),
        "annotation_output": str(annotation_output.resolve()),
        "model": model,
        "results": {},
        "summary": {},
    }


def _update_summary(
    cache: dict[str, Any],
    *,
    input_rate: float,
    output_rate: float,
) -> None:
    results = list(cache.get("results", {}).values())
    input_tokens = sum(int(item.get("usage", {}).get("input_tokens") or 0) for item in results)
    output_tokens = sum(int(item.get("usage", {}).get("output_tokens") or 0) for item in results)
    cache["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    cache["summary"] = {
        "completed": len(results),
        "successful": sum(not item.get("error") for item in results),
        "valid_length": sum(bool(item.get("valid_length")) for item in results),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_price_per_million_cny": input_rate,
        "output_price_per_million_cny": output_rate,
        "estimated_api_cost_cny": round(
            input_tokens * input_rate / 1_000_000 + output_tokens * output_rate / 1_000_000,
            6,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path, help="含 batch_report.json 与 03_rectified.jpg 的批次")
    parser.add_argument("--annotation-output", type=Path, required=True, help="人工标注 CSV 路径；用于锁定同一 selection 清单")
    parser.add_argument("--output", type=Path, default=None, help="OCR JSON；默认是标注 CSV 后加 .ocr.json")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--selection", choices=("random", "first"), default="random")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--force", action="store_true", help="忽略已有 OCR 缓存并重新调用")
    parser.add_argument("--input-price-per-million-cny", type=float, default=1.0)
    parser.add_argument("--output-price-per-million-cny", type=float, default=10.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = args.model or os.environ.get("OPENAI_MODEL") or os.environ.get("SHAPE_VISION_MODEL")
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    if not model or not api_key:
        raise SystemExit("缺少 OPENAI_MODEL/SHAPE_VISION_MODEL 或 OPENAI_API_KEY；请先 source local_env.sh")
    output = args.output or args.annotation_output.with_suffix(args.annotation_output.suffix + ".ocr.json")
    items = load_or_create_selection(
        args.batch,
        args.annotation_output,
        count=args.count,
        selection=args.selection,
        seed=args.seed,
    )
    cache = _new_cache(args.batch, args.annotation_output, model)
    if output.is_file() and not args.force:
        loaded = json.loads(output.read_text(encoding="utf-8"))
        if loaded.get("model") == model and isinstance(loaded.get("results"), dict):
            cache = loaded
    pending = [item for item in items if item.image_id not in cache["results"]]
    print(f"车牌 OCR：已缓存 {len(items) - len(pending)}/{len(items)}，待调用 {len(pending)}", flush=True)
    client = OpenAI(api_key=api_key, base_url=base_url)
    completed = len(items) - len(pending)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(recognize_item, client, item, model=model): item
            for item in pending
        }
        for future in as_completed(futures):
            item = futures[future]
            result = future.result()
            cache["results"][item.image_id] = result
            completed += 1
            _update_summary(
                cache,
                input_rate=args.input_price_per_million_cny,
                output_rate=args.output_price_per_million_cny,
            )
            _atomic_write_json(output, cache)
            status = result["plate_text"] or result["error"] or "未读出"
            print(f"[{completed}/{len(items)}] {item.image_id}: {status}", flush=True)
    _update_summary(
        cache,
        input_rate=args.input_price_per_million_cny,
        output_rate=args.output_price_per_million_cny,
    )
    _atomic_write_json(output, cache)
    print(json.dumps(cache["summary"], ensure_ascii=False), flush=True)
    print(f"OCR 缓存：{output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
