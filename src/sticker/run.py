"""整体贴片检测 CLI：输入平面车牌，输出候选、最终标记图、掩膜和 JSON。"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sticker.agent import StickerAgentHarness
    from sticker.evidence import analyze_plate_evidence
    from sticker.output import make_local_decision, write_analysis_outputs
else:
    from .agent import StickerAgentHarness
    from .evidence import analyze_plate_evidence
    from .output import make_local_decision, write_analysis_outputs

try:
    from shape.geometry import read_image
except ModuleNotFoundError:  # pragma: no cover
    from src.shape.geometry import read_image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_plate_images(path: Path, start: int = 0, limit: int | None = None) -> list[Path]:
    if path.is_file():
        return [path]
    rectified = sorted(path.rglob("03_rectified.jpg"))
    if rectified:
        images = rectified
    else:
        images = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    images = images[start:]
    return images[:limit] if limit is not None else images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="平面车牌图片，或包含 03_rectified.jpg 的输出目录")
    parser.add_argument("--output", type=Path, default=Path("outputs/sticker"), help="输出根目录")
    parser.add_argument(
        "--method",
        choices=("agent", "local"),
        default="agent",
        help="贴片判断方法；默认使用云端多轮 agent，local 为确定性基线",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", help="云端视觉模型；默认读取 OPENAI_MODEL")
    parser.add_argument("--base-url", help="OpenAI 兼容接口；默认读取 OPENAI_BASE_URL")
    parser.add_argument("--no-cache", action="store_true", help="agent 模式禁用 API 响应缓存")
    return parser


def _stem_for(path: Path) -> str:
    return path.parent.name if path.name.lower() == "03_rectified.jpg" else path.stem


def main() -> int:
    args = build_parser().parse_args()
    images = iter_plate_images(args.input, args.start, args.limit)
    if not images:
        print(f"没有找到平面车牌图片：{args.input}", file=sys.stderr)
        return 2
    failures = 0
    for index, path in enumerate(images, start=1):
        stem = _stem_for(path)
        try:
            image = read_image(path)
            artifacts = analyze_plate_evidence(image)
            trajectory = None
            if args.method == "agent":
                cache_dir = args.output / ".api_cache"
                harness = StickerAgentHarness(
                    model=args.model,
                    base_url=args.base_url,
                    cache_dir=cache_dir,
                    use_cache=not args.no_cache,
                )
                decision, trajectory = harness.run(image, artifacts)
                trajectory["input_path"] = str(path)
                trajectory["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                decision = make_local_decision(artifacts)
            paths = write_analysis_outputs(image, artifacts, decision, args.output, stem, trajectory)
            print(
                f"[{index}/{len(images)}] {path}: {decision['decision']}, "
                f"selected={decision.get('selected_candidates', [])} -> {paths['final']}"
            )
        except Exception as exc:
            failures += 1
            print(f"[{index}/{len(images)}] {path}: FAILED: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
