"""
Configuration constants, logging setup, tool definitions, and system prompt.
"""

import datetime
import logging
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vlm_agent")

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
MAX_ITERATIONS = 100
CELL_TIMEOUT = 120.0
MAX_OUTPUT_CHARS = 50000

# Container-side working directory (visible to the kernel)
CONTAINER_WORK_DIR = "/mnt/data"

# Keep default runtime data inside this repository, independent of the launch CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWEVISSION_ROOT = Path(__file__).resolve().parents[1]


def _project_runtime_dir(env_name: str, default: Path) -> str:
    """仅接受项目根内的运行目录，避免遗留环境变量指向旧项目。"""
    configured = os.environ.get(env_name)
    if configured:
        candidate = Path(configured).expanduser().resolve()
        try:
            candidate.relative_to(PROJECT_ROOT)
        except ValueError:
            pass
        else:
            return str(candidate)
    return str(default)


# Host-side directory that is volume-mounted into the container.
_HOST_WORK_BASE = _project_runtime_dir(
    "VLM_HOST_WORK_DIR",
    PROJECT_ROOT / "runtime" / "swe-vision" / "workdir",
)
HOST_WORK_DIR = os.path.join(
    _HOST_WORK_BASE,
    datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
)

# Docker image / build settings
DOCKER_IMAGE_NAME = os.environ.get("VLM_DOCKER_IMAGE", "swe-vision:latest")

DOCKERFILE_DIR = os.environ.get(
    "VLM_DOCKERFILE_DIR",
    str(SWEVISSION_ROOT / "env"),
)

# Pre-assigned ZMQ ports for the Jupyter kernel inside the container
_KERNEL_BASE_PORT = 65500
KERNEL_PORTS = {
    "shell_port":   _KERNEL_BASE_PORT,
    "iopub_port":   _KERNEL_BASE_PORT + 1,
    "stdin_port":   _KERNEL_BASE_PORT + 2,
    "control_port": _KERNEL_BASE_PORT + 3,
    "hb_port":      _KERNEL_BASE_PORT + 4,
}


def kernel_ports_for_slot(slot: int) -> dict[str, int]:
    """Return a non-overlapping five-port range for one concurrent kernel."""
    if slot < 0:
        raise ValueError("kernel slot must be non-negative")
    base_port = _KERNEL_BASE_PORT + slot * 10
    return {
        "shell_port": base_port,
        "iopub_port": base_port + 1,
        "stdin_port": base_port + 2,
        "control_port": base_port + 3,
        "hb_port": base_port + 4,
    }

# ─────────────────────────────────────────────────────────────────────
# Tool Definitions (OpenAI function calling format)
# ─────────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": (
                "Execute Python code in a **stateful** Jupyter notebook environment. "
                "The kernel persists across calls, so variables, imports, and state are retained. "
                "Use this to process images, perform calculations, create visualizations, "
                "or run any Python code. "
                "Any images generated (e.g. via matplotlib plt.show() or PIL Image.save()) "
                "will be captured and returned as base64-encoded images."
                "Print statements and expression results are captured as text output. "
                "All uploaded files are available under /mnt/data/. "
                "The kernel's working directory is /mnt/data/."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "The Python code to execute. The code runs in a Jupyter kernel "
                            "so you can use magics, display(), etc. "
                            "Use print() for text output. "
                            "Images from matplotlib will be auto-captured."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call this tool when you have determined the final answer. "
                "This ends the agentic workflow and returns the answer to the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to the user's question.",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]

# ─────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert AI assistant with access to a **stateful Jupyter notebook** environment. \
You can execute Python code to help answer the user's questions.

## Available Tools

1. **execute_code**: Run Python code in a persistent Jupyter notebook. The kernel state \
(variables, imports, loaded data) is preserved between calls. Use this for:
   - Image processing and analysis (PIL/Pillow, OpenCV, skimage, etc.)
   - Data analysis and computation (numpy, pandas, scipy, etc.)
   - Visualization (matplotlib, seaborn, plotly, etc.)
   - Any Python computation

2. **finish**: Call this when you have the final answer. This ends the workflow.

## File System

- All uploaded files (images, data files, etc.) are placed in `/mnt/data/`.
- The Jupyter kernel's working directory is `/mnt/data/`, so you can reference files \
by their filename directly (e.g. `open('image.png')`) or by absolute path \
(e.g. `open('/mnt/data/image.png')`).
- Any files you create or save will also go into `/mnt/data/`.

## Guidelines

- When given an image, you can load it in the notebook using PIL or OpenCV. \
The image file will be available at `/mnt/data/<filename>`.
- You can call execute_code **multiple times** to iteratively explore and process data.
- Always use print() to output results you want to see.
- When you generate plots with matplotlib, use plt.show() — the plot image will be \
captured and returned to you.
- Think step by step. Examine intermediate results before giving a final answer.
- When you're confident in your answer, call the **finish** tool with your final response.
- If code produces an error, analyze the error and try a different approach.
"""
