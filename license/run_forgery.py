"""车牌伪造/变造鉴定 —— 单张图片版。

提示词从 forgery_prompt.md 最后一个代码块读取。

用法:
    source ~/swe-vision-venv/bin/activate && source local_env.sh
    python license/run_forgery.py data/raw/images/xxx.jpg [--save-trajectory 目录]
"""
import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swe_vision import VLMToolCallAgent


def load_prompt() -> str:
    md = (Path(__file__).parent / "forgery_prompt.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", md, re.S)
    assert blocks, "forgery_prompt.md 里没有找到提示词代码块"
    return blocks[-1].strip()  # 最后一个代码块 = 提示词正文


async def main():
    parser = argparse.ArgumentParser(description="车牌伪造/变造鉴定（单张）")
    parser.add_argument("image", help="待鉴定图片路径")
    parser.add_argument("--save-trajectory", default=None,
                        help="轨迹保存目录（默认自动存到 ./trajectories/）")
    args = parser.parse_args()

    agent = VLMToolCallAgent(
        system_prompt=load_prompt(),   # 模型走环境变量 OPENAI_MODEL 等
        reasoning=False,
        save_trajectory=args.save_trajectory,
        verbose=True,
    )
    try:
        answer = await agent.run(
            "请对这张照片中的车牌进行伪造/变造鉴定，并给出结论。",
            [args.image],
        )
        print("\n" + "=" * 60)
        print(answer)
        print("=" * 60)
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
