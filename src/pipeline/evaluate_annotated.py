"""对人工标注的固定样本重新运行流水线，并立即计算检测指标。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import tempfile
from pathlib import Path

from .core import IMAGE_SUFFIXES, read_plate_ocr_cache, run_batch

try:
    from sticker.evaluate import evaluate_reports, read_annotations, read_reports
    from shape.local_model import DEFAULT_MODEL
except ModuleNotFoundError:  # pragma: no cover
    from src.sticker.evaluate import evaluate_reports, read_annotations, read_reports
    from src.shape.local_model import DEFAULT_MODEL


def _read_source_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "decision", "input_path"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"标注 CSV 必须包含列：{sorted(required)}")
        rows = []
        for row in reader:
            image_id = str(row.get("image_id", "")).strip()
            decision = str(row.get("decision", "")).strip()
            if image_id and decision != "unassessable":
                rows.append({key: str(value or "") for key, value in row.items()})
    if not rows:
        raise ValueError("标注集中没有可评估的 suspicious/clear 样本")
    return rows


def select_annotation_rows(
    rows: list[dict[str, str]],
    *,
    count: int | None,
    selection: str,
    seed: int,
) -> list[dict[str, str]]:
    """从人工标注集中选择可复现的评估子集。"""

    if count is None or count >= len(rows):
        return list(rows)
    if count <= 0:
        raise ValueError("--count 必须大于 0")
    if selection == "first":
        return list(rows[:count])
    if selection == "random":
        return random.Random(seed).sample(rows, count)
    raise ValueError(f"不支持的选择方式：{selection}")


def _image_index(images_root: Path | None) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    if images_root is None:
        return index
    for path in images_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.stem, []).append(path.resolve())
    return index


def resolve_annotated_sources(
    rows: list[dict[str, str]],
    *,
    annotations_path: Path,
    images_root: Path | None,
) -> list[tuple[str, Path]]:
    index = _image_index(images_root)
    resolved: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for row in rows:
        image_id = row["image_id"].strip()
        if image_id in seen:
            raise ValueError(f"标注 CSV 中 image_id 重复：{image_id}")
        if Path(image_id).name != image_id:
            raise ValueError(f"image_id 不能包含路径分隔符：{image_id}")
        seen.add(image_id)
        raw_path = Path(row.get("input_path", "")).expanduser()
        candidates = [raw_path, annotations_path.parent / raw_path] if str(raw_path) else []
        source = next((path.resolve() for path in candidates if path.is_file()), None)
        if source is None:
            matches = index.get(image_id, [])
            if len(matches) == 1:
                source = matches[0]
            elif len(matches) > 1:
                raise ValueError(f"{image_id} 在 --images-root 中匹配到多张图片")
        if source is None:
            raise FileNotFoundError(
                f"找不到 {image_id} 的 input_path={row.get('input_path', '')!r}；"
                "请提供 --images-root data/raw/images"
            )
        resolved.append((image_id, source))
    return resolved


def stage_annotated_sources(sources: list[tuple[str, Path]], directory: Path) -> None:
    for image_id, source in sources:
        suffix = source.suffix.lower() if source.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
        target = directory / f"{image_id}{suffix}"
        os.symlink(source, target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="annotate_web.py 生成的人工标注 CSV")
    parser.add_argument("--images-root", type=Path, help="input_path 失效时按 image_id 查找原图")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="评估批次输出父目录")
    parser.add_argument("--prefix", default="pipeline-annotated-eval")
    parser.add_argument("--sticker-method", choices=("agent", "local"), default="agent")
    parser.add_argument("--detector-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--count", type=int, help="只评估标注集中的前 N 张或随机 N 张")
    parser.add_argument("--selection", choices=("first", "random"), default="first")
    parser.add_argument("--seed", type=int, default=20260824, help="random 选择的随机种子")
    parser.add_argument("--agent-max-calls-per-image", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--input-price-per-million-cny", type=float)
    parser.add_argument("--output-price-per-million-cny", type=float)
    parser.add_argument(
        "--plate-ocr-cache",
        type=Path,
        default=None,
        help="独立模型 OCR JSON；默认自动读取 <annotations>.ocr.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    annotation_rows = select_annotation_rows(
        _read_source_rows(args.annotations),
        count=args.count,
        selection=args.selection,
        seed=args.seed,
    )
    default_ocr_cache = args.annotations.with_suffix(args.annotations.suffix + ".ocr.json")
    ocr_cache_path = args.plate_ocr_cache or (default_ocr_cache if default_ocr_cache.is_file() else None)
    plate_ocr_results = read_plate_ocr_cache(ocr_cache_path) if ocr_cache_path else None
    sources = resolve_annotated_sources(
        annotation_rows,
        annotations_path=args.annotations.resolve(),
        images_root=args.images_root.resolve() if args.images_root else None,
    )
    with tempfile.TemporaryDirectory(prefix="plate-annotated-eval-") as temporary:
        staging = Path(temporary)
        stage_annotated_sources(sources, staging)

        def progress(done: int, total: int, sample: dict[str, object]) -> None:
            print(f"[{done}/{total}] {sample.get('sample_id')} {sample.get('status')}", flush=True)

        run_dir, batch_report = run_batch(
            staging,
            args.output,
            count=len(sources),
            selection="first",
            prefix=args.prefix,
            detector_model=args.detector_model,
            sticker_method=args.sticker_method,
            agent_model=args.model,
            agent_base_url=args.base_url,
            agent_use_cache=not args.no_cache,
            agent_max_calls_per_image=args.agent_max_calls_per_image,
            workers=args.workers,
            progress_callback=progress,
            input_price_per_million_cny=args.input_price_per_million_cny,
            output_price_per_million_cny=args.output_price_per_million_cny,
            plate_ocr_results=plate_ocr_results,
        )

    all_annotations = read_annotations(args.annotations)
    evaluated_ids = {image_id for image_id, _ in sources}
    annotations = {key: value for key, value in all_annotations.items() if key in evaluated_ids}
    metrics = evaluate_reports(annotations, read_reports(run_dir))
    metrics["run_dir"] = str(run_dir)
    metrics["sticker_method"] = args.sticker_method
    metrics["cost"] = batch_report.get("cost", {})
    metrics["plate_ocr_quality_gate"] = batch_report.get("plate_ocr_quality_gate", {})
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"评估批次：{run_dir}")
    print(f"指标文件：{metrics_path}")
    return 0 if batch_report.get("failed_count", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
