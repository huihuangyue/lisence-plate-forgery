"""按随机抽样或文件名顺序批量运行“整车到贴牌筛查答案”流水线。"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.core import read_plate_ocr_cache, run_batch
    from shape.local_model import DEFAULT_MODEL
else:
    from .core import read_plate_ocr_cache, run_batch
    try:
        from shape.local_model import DEFAULT_MODEL
    except ModuleNotFoundError:  # pragma: no cover
        from src.shape.local_model import DEFAULT_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="整车图片或图片目录")
    parser.add_argument("--count", type=int, default=20, help="处理数量，默认 20")
    parser.add_argument("--selection", choices=("random", "first"), default="random", help="随机抽样，或按文件名排序取前 N 张")
    parser.add_argument("--seed", type=int, help="random 模式的随机种子；省略时自动生成")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="带时间戳批次目录的父目录")
    parser.add_argument("--prefix", default="pipeline_batch", help="时间戳目录名前缀")
    parser.add_argument("--detector-model", type=Path, default=DEFAULT_MODEL, help="四关键点 ONNX 模型")
    parser.add_argument("--sticker-method", choices=("agent", "local"), default="agent", help="贴牌判断方法；默认云端多轮 agent")
    parser.add_argument("--model", help="云端视觉模型；默认读取 OPENAI_MODEL/SHAPE_VISION_MODEL")
    parser.add_argument("--base-url", help="OpenAI 兼容端点；默认读取 OPENAI_BASE_URL")
    parser.add_argument("--no-cache", action="store_true", help="agent 模式禁用 API 响应缓存")
    parser.add_argument("--workers", type=int, default=4, help="并行处理的图片数；默认4")
    parser.add_argument("--no-progress", action="store_true", help="禁用终端进度条")
    parser.add_argument(
        "--agent-max-calls-per-image",
        type=int,
        choices=(2, 3, 4),
        default=3,
        help="每张图最多调用大模型2/3/4次；默认3次（调查、反证、裁决）",
    )
    parser.add_argument("--input-price-per-million-cny", type=float, help="每百万输入 Token 的人民币价格")
    parser.add_argument("--output-price-per-million-cny", type=float, help="每百万输出 Token 的人民币价格")
    parser.add_argument("--plate-ocr-cache", type=Path, help="可选独立模型 OCR JSON，用作可评估质量的正向证据")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seed = None
    if args.selection == "random":
        seed = args.seed if args.seed is not None else secrets.randbits(63)
    try:
        with tqdm(
            total=args.count,
            desc="车牌批处理",
            unit="张",
            dynamic_ncols=True,
            disable=args.no_progress,
        ) as progress:
            def update_progress(_: int, __: int, sample: dict[str, object]) -> None:
                progress.update(1)
                progress.set_postfix_str(f"{sample['sample_id']}:{sample['status']}")

            run_dir, report = run_batch(
                args.input,
                args.output,
                count=args.count,
                selection=args.selection,
                seed=seed,
                prefix=args.prefix,
                detector_model=args.detector_model,
                sticker_method=args.sticker_method,
                agent_model=args.model,
                agent_base_url=args.base_url,
                agent_use_cache=not args.no_cache,
                agent_max_calls_per_image=args.agent_max_calls_per_image,
                workers=args.workers,
                progress_callback=update_progress,
                input_price_per_million_cny=args.input_price_per_million_cny,
                output_price_per_million_cny=args.output_price_per_million_cny,
                plate_ocr_results=read_plate_ocr_cache(args.plate_ocr_cache) if args.plate_ocr_cache else None,
            )
    except Exception as exc:
        print(f"批处理启动失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    for sample in report["samples"]:
        if sample["status"] == "success":
            print(
                f"[{sample['selection_index']}/{report['requested_count']}] {sample['sample_id']}: "
                f"{sample['decision']}, selected={sample['selected_candidates']}"
            )
        else:
            print(f"[{sample['selection_index']}/{report['requested_count']}] {sample['sample_id']}: FAILED: {sample['error']}")
    print(f"选择方式：{args.selection}")
    if seed is not None:
        print(f"随机种子：{seed}")
    print(f"成功：{report['successful_count']}/{report['requested_count']}")
    print(f"并行工作槽：{report['parallelism']['effective_workers']}")
    print(f"处理耗时：{report['timing']['processing_wall_seconds']:.3f} 秒")
    print(f"图片扫描与选择：{report['timing']['batch_stage_seconds']['image_discovery_and_selection']:.3f} 秒")
    print(f"逐图处理循环：{report['timing']['batch_stage_seconds']['sample_processing_loop']:.3f} 秒")
    mean_seconds = report["timing"]["successful_image_seconds"]["mean"]
    throughput = report["timing"]["throughput_successful_images_per_second"]
    print(f"平均每张：{mean_seconds:.3f} 秒" if mean_seconds is not None else "平均每张：N/A")
    print(f"吞吐量：{throughput:.3f} 张/秒" if throughput is not None else "吞吐量：N/A")
    print(f"大模型 API 调用：{report['cost']['external_api_calls']} 次")
    print(f"每张最大调用：{report['agent_max_calls_per_image']} 次")
    print(f"输入 Token：{report['cost']['input_tokens']}")
    print(f"输出 Token：{report['cost']['output_tokens']}")
    model_cost = report["cost"]["external_model_cost_cny"]
    print(f"大模型 API 费用：{model_cost:.6f} 元" if model_cost is not None else f"大模型 API 费用：N/A（{report['cost']['status']}）")
    print(f"批次目录：{run_dir}")
    print(f"最终答案：{run_dir / 'results'}")
    return 0 if report["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
