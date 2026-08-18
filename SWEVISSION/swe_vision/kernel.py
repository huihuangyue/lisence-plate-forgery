"""
Docker Jupyter Notebook Kernel Manager.

Manages a persistent Jupyter kernel running inside a Docker container.
The container is built from env/Dockerfile and runs an IPython kernel
with pre-assigned ZMQ ports. The host connects via jupyter_client.
"""

import asyncio
import json
import os
import queue
import re
import shutil
import time
import uuid
from typing import Any, Dict, Optional

from swe_vision.config import (
    CELL_TIMEOUT,
    CONTAINER_WORK_DIR,
    DOCKER_IMAGE_NAME,
    DOCKERFILE_DIR,
    HOST_WORK_DIR,
    KERNEL_PORTS,
    MAX_OUTPUT_CHARS,
    logger,
)


class JupyterNotebookKernel:
    """
    Manages a persistent Jupyter kernel running inside a Docker container.

    The container is built from env/Dockerfile and runs an IPython kernel
    with pre-assigned ZMQ ports.  The host connects to it via
    jupyter_client using the same ports (forwarded from the container).

    A host directory is volume-mounted to /mnt/data inside the container
    so files can be shared between host and kernel.
    """

    def __init__(
        self,
        timeout: float = CELL_TIMEOUT,
        host_work_dir: str = HOST_WORK_DIR,
        container_work_dir: str = CONTAINER_WORK_DIR,
        docker_image: str = DOCKER_IMAGE_NAME,
        dockerfile_dir: str = DOCKERFILE_DIR,
        kernel_ports: Optional[Dict[str, int]] = None,
    ):
        self._timeout = timeout
        self._host_work_dir = host_work_dir
        self._container_work_dir = container_work_dir
        self._docker_image = docker_image
        self._dockerfile_dir = dockerfile_dir
        self._kernel_ports = kernel_ports or KERNEL_PORTS

        self._container = None  # docker.models.containers.Container
        self._docker_client = None  # docker.DockerClient
        self._kc = None  # jupyter_client.BlockingKernelClient
        self._started = False

        self._kernel_key = uuid.uuid4().hex

    @property
    def host_work_dir(self) -> str:
        return self._host_work_dir

    @property
    def container_work_dir(self) -> str:
        return self._container_work_dir

    # ── Docker helpers ─────────────────────────────────────────────

    def _build_image(self):
        import docker

        if self._docker_client is None:
            self._docker_client = docker.from_env()

        try:
            self._docker_client.images.get(self._docker_image)
            logger.info("Docker image '%s' already exists, skipping build.", self._docker_image)
            return
        except docker.errors.ImageNotFound:
            pass

        logger.info(
            "Building Docker image '%s' from %s ...",
            self._docker_image, self._dockerfile_dir,
        )
        image, build_logs = self._docker_client.images.build(
            path=self._dockerfile_dir,
            tag=self._docker_image,
            rm=True,
        )
        for chunk in build_logs:
            if "stream" in chunk:
                line = chunk["stream"].rstrip()
                if line:
                    logger.debug("  [docker build] %s", line)
        logger.info("Docker image '%s' built successfully.", self._docker_image)

    def _write_connection_file(self) -> str:
        conn = {
            "shell_port":   self._kernel_ports["shell_port"],
            "iopub_port":   self._kernel_ports["iopub_port"],
            "stdin_port":   self._kernel_ports["stdin_port"],
            "control_port": self._kernel_ports["control_port"],
            "hb_port":      self._kernel_ports["hb_port"],
            "ip": "0.0.0.0",
            "key": self._kernel_key,
            "transport": "tcp",
            "signature_scheme": "hmac-sha256",
            "kernel_name": "python3",
        }
        host_path = os.path.join(self._host_work_dir, ".kernel_connection.json")
        with open(host_path, "w") as f:
            json.dump(conn, f)
        logger.info("Wrote kernel connection file to %s", host_path)
        return os.path.join(self._container_work_dir, ".kernel_connection.json")

    def _start_container(self):
        import docker

        if self._docker_client is None:
            self._docker_client = docker.from_env()

        port_bindings = {
            f"{p}/tcp": ("127.0.0.1", p) for p in self._kernel_ports.values()
        }

        container_name = f"vlm-jupyter-{uuid.uuid4().hex[:8]}"

        logger.info(
            "Starting Docker container '%s' (image=%s, mount=%s -> %s) ...",
            container_name, self._docker_image,
            self._host_work_dir, self._container_work_dir,
        )

        self._container = self._docker_client.containers.run(
            image=self._docker_image,
            name=container_name,
            command="sleep infinity",
            ports=port_bindings,
            volumes={
                self._host_work_dir: {
                    "bind": self._container_work_dir,
                    "mode": "rw",
                },
            },
            working_dir=self._container_work_dir,
            detach=True,
            remove=False,
        )
        logger.info("Container '%s' started (id=%s).", container_name, self._container.short_id)

    def _start_kernel_in_container(self, connection_file: str):
        cmd = (
            f"python -m ipykernel_launcher -f {connection_file} "
            f"--IPKernelApp.matplotlib='inline'"
        )
        logger.info("Starting kernel inside container: %s", cmd)
        self._container.exec_run(
            cmd=["bash", "-c", cmd],
            detach=True,
            workdir=self._container_work_dir,
        )
        time.sleep(2)

    def _connect_client(self):
        from jupyter_client import BlockingKernelClient

        self._kc = BlockingKernelClient()
        self._kc.ip = "127.0.0.1"
        self._kc.shell_port = self._kernel_ports["shell_port"]
        self._kc.iopub_port = self._kernel_ports["iopub_port"]
        self._kc.stdin_port = self._kernel_ports["stdin_port"]
        self._kc.control_port = self._kernel_ports["control_port"]
        self._kc.hb_port = self._kernel_ports["hb_port"]
        self._kc.session.key = self._kernel_key.encode("utf-8")
        self._kc.start_channels()
        logger.info("Kernel client connected to 127.0.0.1 ports %s", list(self._kernel_ports.values()))

    # ── Public API ─────────────────────────────────────────────────

    async def start(self):
        if self._started:
            return

        os.makedirs(self._host_work_dir, exist_ok=True)
        self._build_image()
        conn_file = self._write_connection_file()
        self._start_container()
        self._start_kernel_in_container(conn_file)
        self._connect_client()

        try:
            self._kc.wait_for_ready(timeout=self._timeout)
        except RuntimeError:
            logger.warning("Kernel wait_for_ready timed out, retrying after 3s...")
            time.sleep(3)
            self._kc.wait_for_ready(timeout=self._timeout)

        test_result = await self._execute_raw("print('kernel_ready')")
        if "kernel_ready" not in test_result.get("stdout", ""):
            raise RuntimeError("Docker Jupyter kernel failed health check")

        await self._execute_raw("%config InlineBackend.figure_format = 'png'")

        self._started = True
        logger.info(
            "Docker Jupyter kernel started successfully "
            "(container=%s, work_dir=%s).",
            self._container.short_id, self._container_work_dir,
        )

    async def _execute_raw(self, code: str) -> Dict[str, Any]:
        cell_result: Dict[str, Any] = {
            "stdout": "",
            "stderr": "",
            "display": [],
            "error": [],
            "status": "ok",
        }

        def _sync_execute():
            msg_id = self._kc.execute(code)
            deadline = time.time() + self._timeout
            reply_received = False
            while time.time() < deadline:
                try:
                    msg = self._kc.get_iopub_msg(timeout=1)
                except queue.Empty:
                    if reply_received:
                        break
                    continue

                msg_type = msg["header"]["msg_type"]
                parent_id = msg.get("parent_header", {}).get("msg_id")
                if parent_id != msg_id:
                    continue

                content = msg["content"]

                if msg_type == "stream":
                    cell_result[content["name"]] += content["text"]
                elif msg_type in ("display_data", "execute_result"):
                    cell_result["display"].append(content["data"])
                elif msg_type == "error":
                    cell_result["error"].append(content)
                elif msg_type == "status" and content.get("execution_state") == "idle":
                    reply_received = True

            try:
                reply = self._kc.get_shell_msg(timeout=self._timeout)
                cell_result["status"] = reply["content"]["status"]
            except queue.Empty:
                cell_result["status"] = "error"

        await asyncio.get_event_loop().run_in_executor(None, _sync_execute)
        return cell_result

    async def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute code in the kernel and return structured results.

        Returns:
            {
                "text_output": str,
                "images": [str, ...],      # base64-encoded PNG images
                "status": "ok" | "error",
                "error_traceback": str | None,
            }
        """
        if not self._started:
            await self.start()

        raw = await self._execute_raw(code)

        text_parts = []
        if raw["stdout"]:
            text_parts.append(raw["stdout"])
        if raw["stderr"]:
            text_parts.append(f"[STDERR] {raw['stderr']}")
        if raw["error"]:
            for err in raw["error"]:
                tb_text = "\n".join(err.get("traceback", []))
                tb_text = re.sub(r"\x1b\[[0-9;]*m", "", tb_text)
                text_parts.append(f"[ERROR] {tb_text}")

        for display_item in raw["display"]:
            if "text/plain" in display_item and "image/png" not in display_item:
                text_parts.append(display_item["text/plain"])

        images = []
        for display_item in raw["display"]:
            if "image/png" in display_item:
                images.append(display_item["image/png"])
            elif "image/jpeg" in display_item:
                images.append(display_item["image/jpeg"])

        text_output = "\n".join(text_parts).strip()
        if not text_output and not images:
            text_output = "[No output produced. Use print() to see results.]"

        if len(text_output) > MAX_OUTPUT_CHARS:
            text_output = text_output[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"

        error_tb = None
        if raw["error"]:
            error_tb = "\n".join(
                "\n".join(e.get("traceback", [])) for e in raw["error"]
            )
            error_tb = re.sub(r"\x1b\[[0-9;]*m", "", error_tb)

        return {
            "text_output": text_output,
            "images": images,
            "status": raw["status"],
            "error_traceback": error_tb,
        }

    async def shutdown(self, cleanup_work_dir: bool = False):
        if self._kc is not None:
            try:
                self._kc.stop_channels()
            except Exception as e:
                logger.warning("Failed to stop kernel client channels: %s", e)
            self._kc = None

        if self._container is not None:
            try:
                logger.info(
                    "Stopping Docker container '%s' ...", self._container.short_id
                )
                self._container.stop(timeout=5)
                self._container.remove(force=True)
                logger.info("Container removed.")
            except Exception as e:
                logger.warning("Failed to stop/remove container: %s", e)
            self._container = None

        self._started = False

        if cleanup_work_dir and os.path.isdir(self._host_work_dir):
            try:
                shutil.rmtree(self._host_work_dir)
                logger.info("Cleaned up host work directory: %s", self._host_work_dir)
            except Exception as e:
                logger.warning("Failed to clean up work directory %s: %s", self._host_work_dir, e)
