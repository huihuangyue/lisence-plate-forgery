"""
Trajectory Recorder — saves the full agent trace to disk.

Output structure::

    <save_dir>/
    ├── trajectory.json          # full structured trajectory
    ├── messages_raw.json        # raw OpenAI messages list (base64 replaced by file refs)
    └── images/
        ├── user_input_0.png
        ├── step_2_tool_0.png
        └── ...
"""

import base64
import copy
import datetime
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from swe_vision.config import logger


class TrajectoryRecorder:
    """
    Records every step of the agentic loop and persists it to a local
    directory.

    ``trajectory.json`` contains::

        {
            "metadata": { model, start_time, end_time, total_iterations, query, ... },
            "steps": [
                {
                    "step":         int,
                    "role":         "user" | "assistant" | "tool",
                    "timestamp":    ISO-8601 string,
                    "content_text": str | null,
                    "tool_calls":   [ { name, arguments } ] | null,
                    "tool_call_id": str | null,
                    "code":         str | null,
                    "images":       [ "images/xxx.png" ],
                },
                ...
            ],
            "final_answer": str | null,
        }
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.image_dir = os.path.join(save_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        self.steps: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.final_answer: Optional[str] = None
        self._image_counter = 0

    # ── helpers ─────────────────────────────────────────────────────

    def _next_image_name(self, prefix: str, ext: str = "png") -> str:
        self._image_counter += 1
        return f"{prefix}_{self._image_counter}.{ext}"

    def _save_base64_image(self, b64_data: str, prefix: str) -> str:
        """Decode base64 image, save to disk, return path relative to save_dir."""
        fname = self._next_image_name(prefix)
        fpath = os.path.join(self.image_dir, fname)
        with open(fpath, "wb") as f:
            f.write(base64.b64decode(b64_data))
        return os.path.join("images", fname)

    def _save_image_file(self, src_path: str, prefix: str) -> str:
        """Copy an image file into the trajectory images dir."""
        ext = Path(src_path).suffix.lstrip(".") or "png"
        fname = self._next_image_name(prefix, ext)
        dst = os.path.join(self.image_dir, fname)
        shutil.copy2(src_path, dst)
        return os.path.join("images", fname)

    @staticmethod
    def _now_iso() -> str:
        return datetime.datetime.now().isoformat(timespec="milliseconds")

    # ── public recording API ───────────────────────────────────────

    def set_metadata(self, **kwargs):
        self.metadata.update(kwargs)

    def record_step(
        self,
        *,
        role: str,
        content_text: Optional[str] = None,
        reasoning_details: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        code: Optional[str] = None,
        images: Optional[List[str]] = None,
    ):
        """Append one step to the trajectory.  ``images`` are *relative* paths."""
        step = {
            "step": len(self.steps),
            "role": role,
            "timestamp": self._now_iso(),
            "content_text": content_text,
            "tool_calls": tool_calls,
            "tool_call_id": tool_call_id,
            "code": code,
            "images": images or [],
        }
        if reasoning_details:
            step["reasoning_details"] = reasoning_details
        self.steps.append(step)

    def record_user_step(
        self,
        query: str,
        image_paths: Optional[List[str]] = None,
    ):
        """Record the initial user message (text + optional images)."""
        saved_images = []
        for p in (image_paths or []):
            if os.path.exists(p):
                saved_images.append(self._save_image_file(p, "user_input"))
        self.record_step(role="user", content_text=query, images=saved_images)

    def record_assistant_step(
        self,
        content_text: Optional[str],
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        reasoning_details: Optional[str] = None,
    ):
        """Record an assistant message (text + optional tool calls + optional reasoning)."""
        simplified = None
        if tool_calls:
            simplified = []
            for tc in tool_calls:
                fn = tc.get("function", tc)
                simplified.append({
                    "id": tc.get("id"),
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments"),
                })
        self.record_step(
            role="assistant",
            content_text=content_text,
            reasoning_details=reasoning_details,
            tool_calls=simplified,
        )

    def record_tool_step(
        self,
        tool_call_id: str,
        tool_name: str,
        code: Optional[str],
        text_output: str,
        base64_images: Optional[List[str]] = None,
    ):
        """Record a tool execution result (code + output + images)."""
        saved_images = []
        step_idx = len(self.steps)
        for img_b64 in (base64_images or []):
            rel = self._save_base64_image(img_b64, f"step_{step_idx}_tool")
            saved_images.append(rel)
        self.record_step(
            role="tool",
            content_text=text_output,
            tool_call_id=tool_call_id,
            code=code,
            images=saved_images,
        )

    def record_finish(self, answer: str):
        self.final_answer = answer

    # ── persistence ────────────────────────────────────────────────

    def save(self):
        """Write trajectory.json to save_dir."""
        self.metadata.setdefault("end_time", self._now_iso())
        self.metadata.setdefault("total_steps", len(self.steps))

        traj = {
            "metadata": self.metadata,
            "steps": self.steps,
            "final_answer": self.final_answer,
        }

        traj_path = os.path.join(self.save_dir, "trajectory.json")
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(traj, f, ensure_ascii=False, indent=2)

        logger.info("Trajectory saved to %s (%d steps, %d images)",
                     self.save_dir, len(self.steps), self._image_counter)

    def save_messages_raw(self, messages: List[Dict[str, Any]]):
        """
        Save the raw OpenAI messages list, replacing inline base64 image
        data with file references so the JSON stays human-readable.
        """
        sanitized = sanitize_messages_for_save(messages, self.image_dir, self.save_dir)
        raw_path = os.path.join(self.save_dir, "messages_raw.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(sanitized, f, ensure_ascii=False, indent=2)


def sanitize_messages_for_save(
    messages: List[Dict[str, Any]],
    image_dir: str,
    save_dir: str,
) -> List[Dict[str, Any]]:
    """
    Deep-copy messages and replace base64 data URIs with saved file paths.
    """
    counter = [0]

    def _replace_b64(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "image_url":
                url = obj.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    match = re.match(r"data:([^;]+);base64,(.*)", url, re.DOTALL)
                    if match:
                        mime, b64 = match.group(1), match.group(2)
                        ext = mime.split("/")[-1]
                        if ext == "jpeg":
                            ext = "jpg"
                        counter[0] += 1
                        fname = f"msg_image_{counter[0]}.{ext}"
                        fpath = os.path.join(image_dir, fname)
                        if not os.path.exists(fpath):
                            with open(fpath, "wb") as fp:
                                fp.write(base64.b64decode(b64))
                        return {
                            "type": "image_url",
                            "image_url": {"url": os.path.join("images", fname)},
                        }
            return {k: _replace_b64(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_replace_b64(item) for item in obj]
        return obj

    return _replace_b64(copy.deepcopy(messages))
