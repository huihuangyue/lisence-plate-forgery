#!/usr/bin/env python3
"""检查本地 OpenAI 兼容 VLM，并可用一张真实图片验证多模态请求。"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from pathlib import Path

from openai import OpenAI


def encode_image(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime_type};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("STICKER_AGENT_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("STICKER_AGENT_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("STICKER_AGENT_MODEL"))
    parser.add_argument("--image", type=Path, help="可选；提供后发送一次真实图像请求")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=300.0)
    available = [item.id for item in client.models.list().data]
    print(f"API_OK base_url={args.base_url}")
    print("available_models=" + ",".join(available))
    model = args.model or (available[0] if available else None)
    if not model:
        raise SystemExit("服务没有返回可用模型")
    if args.image is None:
        print(f"MODEL_READY model={model}; 如需验证视觉输入，请添加 --image <图片路径>")
        return 0
    if not args.image.is_file():
        raise SystemExit(f"图片不存在：{args.image}")

    image_url = encode_image(args.image)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=128,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": '观察图片，只返回 JSON：{"vision_ok":true,"brief":"不超过20字"}'},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print(f"VISION_RESPONSE model={model}")
    print(response.choices[0].message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
