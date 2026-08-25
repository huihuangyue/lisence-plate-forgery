"""命令行入口：输入图片/文件夹，输出点、框、平面号牌三张图。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from shape.classical import detect_plate_classical
    from shape.geometry import read_image, write_outputs
    from shape.llm import detect_plate_llm
    from shape.local_model import DEFAULT_MODEL, detect_plate_local
else:
    from .classical import detect_plate_classical
    from .geometry import read_image, write_outputs
    from .llm import detect_plate_llm
    from .local_model import DEFAULT_MODEL, detect_plate_local


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(path: Path, limit: int | None, start: int = 0) -> list[Path]:
    if path.is_file():
        return [path]
    images = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    images = images[start:]
    return images[:limit] if limit is not None else images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="输入图片或图片文件夹")
    parser.add_argument("--output", type=Path, default=Path("outputs/shape"), help="输出根目录")
    parser.add_argument("--method", choices=("local", "classical", "llm"), default="local")
    parser.add_argument("--limit", type=int, help="处理文件夹时最多处理多少张")
    parser.add_argument("--start", type=int, default=0, help="处理文件夹时跳过排序后的前 N 张")
    parser.add_argument("--model", help="LLM 模式的视觉模型名；也可设置 SHAPE_VISION_MODEL")
    parser.add_argument("--base-url", help="OpenAI 兼容接口地址；也可设置 OPENAI_BASE_URL")
    parser.add_argument("--detector-model", type=Path, default=DEFAULT_MODEL, help="local 模式的 ONNX 模型")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    images = iter_images(args.input, args.limit, args.start)
    if not images:
        print(f"没有找到图片：{args.input}", file=sys.stderr)
        return 2

    failures = 0
    for index, path in enumerate(images, start=1):
        try:
            image = read_image(path)
            if args.method == "local":
                detection = detect_plate_local(image, model_path=args.detector_model)
            elif args.method == "classical":
                detection = detect_plate_classical(image)
            else:
                detection = detect_plate_llm(image, model=args.model, base_url=args.base_url)
            paths = write_outputs(image, detection, args.output, path.stem)
            print(
                f"[{index}/{len(images)}] {path.name}: {detection.plate_type.value}, "
                f"confidence={detection.confidence:.3f} -> {paths['rectified']}"
            )
        except Exception as exc:  # 单张失败不阻塞批次，便于首轮审查全部样本。
            failures += 1
            print(f"[{index}/{len(images)}] {path.name}: FAILED: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
