"""
VLM Tool Call Agent — agentic VLM framework with Docker Jupyter notebook tool.

The agent loop:
1. Send user message (with optional images) to the VLM
2. If the model calls ``execute_code``, run the code in the Docker kernel
3. Feed results (text + images) back to the model
4. Repeat until the model calls ``finish`` or max iterations reached
"""

import asyncio
import datetime
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from swe_vision.config import (
    DEFAULT_MODEL,
    MAX_ITERATIONS,
    SYSTEM_PROMPT,
    TOOLS,
    logger,
    HOST_WORK_DIR,
    kernel_ports_for_slot,
)
from swe_vision.file_manager import NotebookFileManager
from swe_vision.image_utils import make_base64_image_content_part, make_image_content_part
from swe_vision.kernel import JupyterNotebookKernel
from swe_vision.trajectory import TrajectoryRecorder


class VLMToolCallAgent:
    """
    An agentic VLM framework that uses OpenAI's function calling to
    give a vision-language model access to a stateful Jupyter notebook
    running inside a Docker container.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_iterations: int = MAX_ITERATIONS,
        verbose: bool = True,
        save_trajectory: Optional[str] = None,
        reasoning: bool = True,
        finish_only_final_iteration: bool = False,
        finalization_grace_rounds: int = 0,
        include_budget_feedback: bool = False,
        kernel_slot: int = 0,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.reasoning = reasoning
        self.finish_only_final_iteration = finish_only_final_iteration
        self.finalization_grace_rounds = finalization_grace_rounds
        self.include_budget_feedback = include_budget_feedback
        self.kernel_slot = kernel_slot

        self._save_trajectory_dir = save_trajectory

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        elif os.environ.get("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]

        self.client = OpenAI(**client_kwargs)

        # 不输出 API key，避免批处理日志或终端历史泄露凭据。
        if self.verbose:
            print(f"Using model: {self.model}")
            print(f"Using base URL: {base_url or os.environ.get('OPENAI_BASE_URL')}")

        self.kernel: Optional[JupyterNotebookKernel] = None
        self.file_manager = NotebookFileManager()

        self.messages: List[Dict[str, Any]] = []

        # 每个 run 从零开始累计，供调用方记录实际多轮调用的 token。
        self.token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
            "calls": [],
        }

        self.trajectory: Optional[TrajectoryRecorder] = None

    def _configure_kernel(self) -> None:
        """Create the kernel configuration before copying user files to its mount."""
        if self.kernel is None:
            self.kernel = JupyterNotebookKernel(
                host_work_dir=os.path.join(HOST_WORK_DIR, f"worker_{self.kernel_slot}"),
                kernel_ports=kernel_ports_for_slot(self.kernel_slot),
            )
        self.file_manager.setup_work_dir(
            host_work_dir=self.kernel.host_work_dir,
            container_work_dir=self.kernel.container_work_dir,
        )

    async def _ensure_kernel(self):
        self._configure_kernel()
        if not self.kernel._started:
            await self.kernel.start()

    def _log(self, msg: str, *args, level: str = "info"):
        getattr(logger, level)(msg, *args)
        if self.verbose:
            formatted = msg % args if args else msg
            print(f"  [{level.upper()}] {formatted}", flush=True)

    def _build_user_message(
        self,
        query: str,
        image_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        content = []

        file_hints = []
        if image_paths:
            # The first model message is built before a tool call starts the kernel.
            # Configure its mount now so copied files reach the same worker directory.
            self._configure_kernel()
            basenames = [os.path.basename(os.path.abspath(p)) for p in image_paths]
            has_collision = len(basenames) != len(set(basenames))

            for idx, img_path in enumerate(image_paths):
                img_path = os.path.abspath(img_path)
                if not os.path.exists(img_path):
                    self._log("Warning: image not found: %s", img_path, level="warning")
                    continue
                content.append(make_image_content_part(img_path))
                dest_name = None
                if has_collision or len(image_paths) > 1:
                    base = os.path.basename(img_path)
                    name, ext = os.path.splitext(base)
                    dest_name = f"{idx}_{name}{ext}"
                container_path = self.file_manager.copy_file_to_workdir(
                    img_path, dest_name=dest_name,
                )
                file_hints.append(container_path)

        text = query
        if file_hints:
            paths_str = ", ".join(f"`{p}`" for p in file_hints)
            text += f"\n\n[Uploaded file(s) available at: {paths_str}]"
        content.insert(0, {"type": "text", "text": text})

        return {"role": "user", "content": content}

    async def _call_llm(
        self,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = "auto",
    ) -> Any:
        kwargs = dict(
            model=self.model,
            messages=self.messages,
            tools=tools if tools is not None else TOOLS,
            tool_choice=tool_choice,
        )
        if self.reasoning:
            kwargs["extra_body"] = {"reasoning": {"enabled": True, 'effort': 'xhigh'}}
            kwargs["reasoning_effort"] = 'xhigh'
        else:
            kwargs["extra_body"] = {"reasoning": {"enabled": False, 'effort': 'minimal'}}

        response = await asyncio.to_thread(self.client.chat.completions.create, **kwargs)
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if isinstance(prompt_tokens, int):
            self.token_usage["input_tokens"] += prompt_tokens
        if isinstance(completion_tokens, int):
            self.token_usage["output_tokens"] += completion_tokens
        if isinstance(total_tokens, int):
            self.token_usage["total_tokens"] += total_tokens
        self.token_usage["api_calls"] += 1
        self.token_usage["calls"].append({
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
        })
        return response

    async def _handle_execute_code(self, code: str) -> Dict[str, Any]:
        await self._ensure_kernel()

        self._log("Executing code in Docker Jupyter notebook:\n%s",
                   code[:200] + ("..." if len(code) > 200 else ""))

        result = await self.kernel.execute(code)

        text = result["text_output"]
        if result["status"] == "error":
            text = f"[Execution Error]\n{text}"

        image_parts = []
        for img_b64 in result["images"]:
            image_parts.append(make_base64_image_content_part(img_b64))

        return {
            "text_output": text,
            "image_parts": image_parts,
            "base64_images": result["images"],
        }

    def _init_trajectory(self, query: str, image_paths: Optional[List[str]]) -> TrajectoryRecorder:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self._save_trajectory_dir:
            save_dir = f"{self._save_trajectory_dir}_{ts}"
        else:
            project_root = Path(__file__).resolve().parents[2]
            save_dir = str(project_root / "trajectories" / f"run_{ts}")
        recorder = TrajectoryRecorder(save_dir)
        recorder.set_metadata(
            model=self.model,
            start_time=TrajectoryRecorder._now_iso(),
            query=query,
            image_paths=image_paths or [],
            max_iterations=self.max_iterations,
            system_prompt=self.system_prompt,
        )
        return recorder

    async def run(
        self,
        query: str,
        image_paths: Optional[List[str]] = None,
    ) -> str:
        """
        Run the agentic loop for a single user query.

        Returns the final answer string.
        """
        self.trajectory = self._init_trajectory(query, image_paths)

        self.messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        user_msg = self._build_user_message(query, image_paths)
        self.messages.append(user_msg)

        self.trajectory.record_user_step(query, image_paths)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"User Query: {query}")
            if image_paths:
                print(f"Images: {image_paths}")
            print(f"{'='*60}\n")

        final_answer = None
        try:
            final_answer = await self._run_loop()
        finally:
            if final_answer is not None:
                self.trajectory.record_finish(final_answer)
            self.trajectory.save()
            self.trajectory.save_messages_raw(self.messages)

        return final_answer

    async def _run_loop(self) -> str:
        """Core agentic loop."""
        total_iterations = self.max_iterations + (
            self.finalization_grace_rounds if self.finish_only_final_iteration else 0
        )
        for iteration in range(1, total_iterations + 1):
            if self.verbose:
                print(f"\n--- Iteration {iteration}/{total_iterations} ---")

            MAX_RETRIES = 10
            last_error: Optional[Exception] = None
            for retry in range(MAX_RETRIES):
                try:
                    # 最后一轮预留给结构化交付：不再允许执行代码，并强制 finish。
                    if self.finish_only_final_iteration and iteration >= self.max_iterations:
                        finish_tools = [
                            tool for tool in TOOLS
                            if tool.get("function", {}).get("name") == "finish"
                        ]
                        response = await self._call_llm(
                            tools=finish_tools,
                            tool_choice={"type": "function", "function": {"name": "finish"}},
                        )
                    else:
                        response = await self._call_llm()
                    break
                except Exception as exc:
                    last_error = exc
                    self._log("OpenAI API error: %s, retry %d/%d", str(exc), retry, MAX_RETRIES, level="error")

            if retry == MAX_RETRIES - 1:
                return f"[Error] Failed to call LLM: {last_error}"

            choice = response.choices[0]
            message = choice.message

            if hasattr(message, "to_dict"):
                assistant_msg = message.to_dict()
            elif hasattr(message, "model_dump"):
                assistant_msg = message.model_dump()
            else:
                assistant_msg = {"role": "assistant", "content": message.content}
            assistant_msg.setdefault("role", "assistant")
            self.messages.append(assistant_msg)

            tool_call_dicts = assistant_msg.get("tool_calls")
            reasoning_details = assistant_msg.get("reasoning_details")

            self.trajectory.record_assistant_step(
                message.content, tool_call_dicts, reasoning_details=reasoning_details,
            )

            try:
                if message.reasoning and self.verbose:
                    summary = message.reasoning if isinstance(message.reasoning, str) else ""
                    preview = summary[:300] + ("..." if len(summary) > 300 else "")
                    print(f"\n[Reasoning] {preview}")
            except Exception:
                try:
                    summary = message.reasoning_content[:300]
                except Exception:
                    pass

            if message.content:
                if self.verbose:
                    print(f"\n[Assistant] {message.content[:500]}")

            # 兼容 API 可能忽略 tool_choice，在最终轮仍返回普通文本或 execute_code。
            # 此时绝不能继续执行代码；给一次专门的、只允许 finish 的恢复轮。
            finalization_round = self.finish_only_final_iteration and iteration >= self.max_iterations
            if finalization_round and not any(
                call.function.name == "finish" for call in (message.tool_calls or [])
            ):
                if iteration < total_iterations:
                    if message.tool_calls:
                        for tool_call in message.tool_calls:
                            err_text = (
                                "[程序控制] 已到交付轮，未执行此工具。"
                                "下一轮必须调用 finish，并在 answer 中提供唯一合规 JSON。"
                            )
                            self.messages.append({
                                "role": "tool", "tool_call_id": tool_call.id, "content": err_text,
                            })
                            self.trajectory.record_tool_step(
                                tool_call_id=tool_call.id,
                                tool_name=tool_call.function.name,
                                code=None,
                                text_output=err_text,
                            )
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "【程序强制收尾】上一轮没有调用 finish，且不会再执行代码。"
                            "现在必须调用 finish；answer 必须只含一个符合既定协议的 JSON 对象。"
                        ),
                    })
                    continue
                # 恢复轮仍未 finish：若模型给出正文，交给上层 JSON 提取器；否则明确失败。
                return message.content or "[Error] Finalization recovery ended without finish or text answer."

            if not message.tool_calls:
                if choice.finish_reason == "stop":
                    self._log("Model stopped without calling finish tool.")
                    return message.content or "[No response]"
                continue

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    self._log("Failed to parse tool arguments: %s", e, level="error")
                    err_text = f"[Error] Invalid JSON arguments: {e}"
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": err_text,
                    })
                    self.trajectory.record_tool_step(
                        tool_call_id=tool_call.id,
                        tool_name=fn_name,
                        code=None,
                        text_output=err_text,
                    )
                    continue

                if fn_name == "finish":
                    answer = fn_args.get("answer", "")
                    if self.verbose:
                        print(f"\n{'='*60}")
                        print(f"[FINISH] Final Answer:")
                        print(answer)
                        print(f"{'='*60}\n")
                    return answer

                elif fn_name == "execute_code":
                    code = fn_args.get("code", "")
                    text_output = ""
                    image_parts: List[Dict[str, Any]] = []
                    base64_images: List[str] = []
                    try:
                        exec_result = await self._handle_execute_code(code)
                        text_output = exec_result["text_output"]
                        image_parts = exec_result["image_parts"]
                        base64_images = exec_result["base64_images"]
                    except Exception as e:
                        tb = traceback.format_exc()
                        self._log("Code execution failed: %s", e, level="error")
                        text_output = f"[Execution Error] {e}\n{tb}"

                    if self.include_budget_feedback:
                        remaining = self.max_iterations - iteration
                        text_output += (
                            f"\n\n【程序控制】本图已使用 {iteration}/{self.max_iterations} 次模型回复，"
                            f"剩余 {remaining} 次。请优先使用 execute_code 补足必要的图像证据，"
                            "特别是字符邻域的纹理、色差、边界、反光或局部形状；仅读取尺寸不足以形成结论。"
                            "证据充分后立即收敛，不要进行无关计算。"
                            "最终必须调用 finish，并只返回合规 JSON。"
                        )

                    if image_parts:
                        tool_content: Any = [
                            {"type": "text", "text": text_output},
                        ] + image_parts
                    else:
                        tool_content = text_output

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content,
                    })

                    self.trajectory.record_tool_step(
                        tool_call_id=tool_call.id,
                        tool_name=fn_name,
                        code=code,
                        text_output=text_output,
                        base64_images=base64_images,
                    )

                    if self.verbose:
                        print(f"\n[Code Output] {text_output[:500]}")
                        if image_parts:
                            print(f"  [{len(image_parts)} image(s) returned to model in tool message]")

                else:
                    self._log("Unknown tool: %s", fn_name, level="warning")
                    err_text = f"[Error] Unknown tool: {fn_name}"
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": err_text,
                    })
                    self.trajectory.record_tool_step(
                        tool_call_id=tool_call.id,
                        tool_name=fn_name,
                        code=None,
                        text_output=err_text,
                    )

        self._log("Max iterations reached (%d core + %d finalization recovery)", self.max_iterations, self.finalization_grace_rounds, level="warning")
        return "[Error] Max iterations reached without a final answer."

    async def run_interactive(self, image_paths: Optional[List[str]] = None):
        """
        Run in interactive mode — the user can keep asking questions
        and the kernel state is preserved.
        """
        print("\n" + "="*60)
        print("VLM Tool Call Agent - Interactive Mode (Docker Runtime)")
        print("Type 'quit' or 'exit' to stop.")
        print("Type 'image:<path>' to add an image to the next query.")
        print("="*60 + "\n")

        session_images = list(image_paths or [])

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            if user_input.lower().startswith("image:"):
                img_path = user_input[6:].strip()
                if os.path.exists(img_path):
                    session_images.append(img_path)
                    print(f"  Added image: {img_path}")
                else:
                    print(f"  Image not found: {img_path}")
                continue

            answer = await self.run(user_input, session_images if session_images else None)
            print(f"\nAnswer: {answer}\n")

            session_images = []

    async def cleanup(self):
        """Shut down the Docker kernel and clean up resources."""
        if self.kernel:
            await self.kernel.shutdown()
