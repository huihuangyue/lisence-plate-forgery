"""v2 批处理入口：默认处理 data/manifest.csv 的前 100 张图片。"""

import argparse
import asyncio
import csv
import datetime
import json
import logging
import os
import shutil
import time
from pathlib import Path

from .run_forgery import (
    CODE_VERSION, PROMPT_PATH, SWE_MAX_ITERATIONS, SWE_TRAJECTORY_ROOT,
    VLMToolCallAgent, file_sha256, load_prompt, project_path, process_image_async,
)


def print_progress(position: int, total: int, filename: str, status: str) -> None:
    """不依赖第三方库的单行终端进度条。"""
    width = 30
    filled = round(width * position / total) if total else 0
    bar = "#" * filled + "-" * (width - filled)
    percent = position * 100 / total if total else 100
    text = f"\r[{bar}] {position}/{total} ({percent:5.1f}%) {status}: {filename}"
    print(text[:180], end="", flush=True)
    if position == total:
        print()


async def run_batch(args: argparse.Namespace) -> None:
    started_at = datetime.datetime.now().astimezone()
    started = time.perf_counter()
    manifest_path = project_path(args.manifest)
    with manifest_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))[:args.limit]

    batch_timestamp = started_at.strftime("%Y%m%dT%H%M%S%z")
    batch_scope = {"manifest": str(manifest_path), "requested_limit": args.limit, "selected_count": len(rows)}
    batch_dir = project_path(args.runs_dir) / batch_timestamp
    answer_dir = batch_dir / "answer"
    result_dir = batch_dir / "result"
    answer_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)
    report = {"metadata": {}, "items": []}
    if not rows:
        raise ValueError("manifest 没有可处理的图片")

    # 批处理终端只显示总进度；每图详细信息仍写入结果与轨迹文件。
    logging.getLogger("vlm_agent").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    completed_count = 0
    progress_lock = asyncio.Lock()

    async def advance_progress(image_path: Path, status: str) -> None:
        nonlocal completed_count
        async with progress_lock:
            completed_count += 1
            print_progress(completed_count, len(rows), image_path.name, status)

    async def run_worker(worker_id: int, work_items: list[tuple[int, dict]]) -> tuple[list[dict], float]:
        """One isolated Agent/kernel handles one group sequentially."""
        agent = VLMToolCallAgent(
            model=args.model,
            system_prompt=load_prompt(),
            max_iterations=SWE_MAX_ITERATIONS,
            reasoning=False,
            finish_only_final_iteration=True,
            include_budget_feedback=True,
            save_trajectory=str(SWE_TRAJECTORY_ROOT / f"worker_{worker_id}"),
            kernel_slot=worker_id,
            verbose=False,
        )
        worker_items: list[dict] = []
        startup_started = time.perf_counter()
        try:
            await agent._ensure_kernel()
            startup_seconds = round(time.perf_counter() - startup_started, 3)
            for position, row in work_items:
                image_path = project_path(row["image_path"])
                item_scope = {**batch_scope, "position": position, "worker": worker_id + 1}
                try:
                    document = await process_image_async(
                        image_path, args.model, result_dir, None, batch_scope=item_scope,
                        run_dir_override=result_dir / image_path.stem,
                        run_timestamp=batch_timestamp, agent=agent, use_plate_crop=args.use_plate_crop,
                    )
                    usage = document["metadata"]["token_usage"]["total"]
                    cost = document["metadata"]["cost"]
                    annotated_name = f"{image_path.stem}.jpg"
                    shutil.copy2(document["annotated_plate"], answer_dir / annotated_name)
                    worker_items.append({
                        "position": position, "image_path": str(image_path), "status": "ok",
                        "annotated_plate": f"answer/{annotated_name}",
                        "result_folder": f"result/{image_path.stem}", "token_usage": usage,
                        "cost": cost, "elapsed_seconds": document["metadata"]["elapsed_seconds"],
                        "output_validation": document["metadata"]["output_validation"],
                    })
                    await advance_progress(image_path, "完成")
                except Exception as exc:
                    worker_items.append({"position": position, "image_path": str(image_path), "status": "error", "error": str(exc)})
                    await advance_progress(image_path, "失败")
        except Exception as exc:
            startup_seconds = round(time.perf_counter() - startup_started, 3)
            for position, row in work_items:
                image_path = project_path(row["image_path"])
                worker_items.append({"position": position, "image_path": str(image_path), "status": "error", "error": f"worker 初始化失败：{exc}"})
                await advance_progress(image_path, "失败")
        finally:
            await agent.cleanup()
        return worker_items, startup_seconds

    worker_count = min(args.workers, len(rows))
    groups = [list(enumerate(rows, start=1))[worker_id::worker_count] for worker_id in range(worker_count)]
    print_progress(0, len(rows), "", f"启动 {worker_count} 个并行 worker")
    worker_results = await asyncio.gather(*(run_worker(worker_id, group) for worker_id, group in enumerate(groups)))
    report["items"] = sorted((item for items, _ in worker_results for item in items), key=lambda item: item["position"])
    docker_startup_seconds = [startup for _, startup in worker_results]

    total_input_tokens = sum((item.get("token_usage", {}).get("input_tokens") or 0) for item in report["items"])
    total_output_tokens = sum((item.get("token_usage", {}).get("output_tokens") or 0) for item in report["items"])
    total_tokens = sum((item.get("token_usage", {}).get("total_tokens") or 0) for item in report["items"])
    known_costs = [item["cost"]["total_cny"] for item in report["items"] if item["status"] == "ok" and item["cost"].get("total_cny") is not None]
    total_cost = sum(known_costs)
    known_cost_count = len(known_costs)

    completed = sum(item["status"] == "ok" for item in report["items"])
    failed = len(rows) - completed
    report["metadata"] = {
        "timestamp": batch_timestamp,
        "model": args.model,
        "code_version": CODE_VERSION,
        "run_forgery_sha256": file_sha256(Path(__file__).with_name("run_forgery.py")),
        "run_batch_sha256": file_sha256(Path(__file__)),
        "prompt_template_sha256": file_sha256(PROMPT_PATH),
        "prompt_is_dynamic_per_image": True,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "started_at": started_at.isoformat(),
        "finished_at": datetime.datetime.now().astimezone().isoformat(),
        "docker": {"kernels_started": worker_count, "startup_seconds_per_worker": docker_startup_seconds},
        "token_usage_total": {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens, "total_tokens": total_tokens},
        "cost_total": {"currency": "CNY", "cost": round(total_cost, 8) if known_cost_count == completed else None, "priced_items": known_cost_count, "completed_items": completed},
        "data_scope": {**batch_scope, "completed_count": completed, "failed_count": failed},
    }
    report_path = batch_dir / "run.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"批处理完成：{completed}/{len(rows)} 成功，{failed} 失败；"
        f"结果：{report_path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 车牌伪造筛查批处理")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--limit", type=int, default=100, help="默认处理 manifest 前 100 条")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "qwen3-vl-plus"))
    parser.add_argument("--runs-dir", default="runtime/license-v2/runs", help="每次批处理结果的根目录")
    parser.add_argument("--workers", type=int, default=4, help="并行 Agent/Docker 组数（默认 4）")
    parser.add_argument("--no-crop-plate", dest="use_plate_crop", action="store_false", help="不做本地车牌裁剪，直接将原图传给模型")
    parser.set_defaults(use_plate_crop=True)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit 必须大于 0")
    if args.workers <= 0:
        parser.error("--workers 必须大于 0")
    if args.workers > 4:
        parser.error("--workers 最大为 4（Docker 内核端口范围限制）")
    if not project_path(args.manifest).is_file():
        parser.error(f"manifest 不存在：{args.manifest}")
    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
