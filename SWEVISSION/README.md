# SWE-Vision

<div align="center">
  <picture>
      <img src="./assets/logo_swev.png" width="30%">
  </picture>
</div>


<div align="center" style="line-height: 1;">

[![GITHUB](https://img.shields.io/badge/Github-24292F?style=for-the-badge&logo=github&logoColor=white)](https://github.com/UniPat-AI/SWE-Vision)
[![Blog](https://img.shields.io/badge/Blog-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](https://unipat.ai/blog/SWE-Vision)

</div>

An agentic VLM (Vision Language Model) framework that gives a language model access to a **stateful Jupyter notebook running inside a Docker container**. The agent can iteratively write and execute Python code to process images, run computations, and produce visualizations — all within a sandboxed environment.

## Project Structure

```
SWE-Vision/
├── swe_vision/                  # Core library
│   ├── __init__.py              # Package exports
│   ├── config.py                # Constants, logging, tool definitions, system prompt
│   ├── kernel.py                # JupyterNotebookKernel — Docker-based Jupyter runtime
│   ├── image_utils.py           # Image encoding, MIME detection, OpenAI content parts
│   ├── file_manager.py          # NotebookFileManager — host ↔ container file sharing
│   ├── trajectory.py            # TrajectoryRecorder — saves full agent traces to disk
│   ├── agent.py                 # VLMToolCallAgent — agentic loop with tool calling
│   ├── cli.py                   # CLI entry point
│   └── eval_utils.py            # LLM judge prompt, answer extraction utilities
│
├── apps/                        # Standalone applications
│   ├── web_app.py               # ChatGPT-style web UI (Flask + SSE streaming)
│   └── trajectory_viewer.py     # Trajectory visualization dashboard (Flask)
│
├── env/                         # Docker environment (Dockerfile for the kernel)
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"   # custom API endpoint
export OPENAI_MODEL="openai/gpt-5.2"                          # default model
```

### 3. Prepare the Docker environment

The agent runs code inside a Docker container. Make sure Docker is installed and running, then place a `Dockerfile` in the `env/` directory. A minimal example:

```bash
docker build -t swe-vision -f ./env/Dockerfile ./env
```


### 4. Run the agent (CLI)


We provide a script to run the agent with a single command.
```bash
bash run.sh
```


You can also run the agent manually.
```bash
# Single query with an image
python -m swe_vision.cli --image photo.png "What objects are in this image?"

# Multiple images
python -m swe_vision.cli -i img1.png -i img2.png "What is the difference between these two images?"
```


### 5. Run the Web UI

A ChatGPT-style interface with real-time streaming of the agent's reasoning, code execution, and results:

```bash
python apps/web_app.py --port 8080
# Open http://localhost:8080
```

![Web App Screenshot](./assets/web_app_screenshot.png)

### 6. View trajectories

Every agent run saves a trajectory (JSON + images) to `./trajectories/`. Browse them with the viewer:

```bash
python apps/trajectory_viewer.py --port 5050
# Open http://localhost:5050
```

![Trajectory Viewer Screenshot](./assets/traj_viewer_screenshot.png)



## Architecture

```
                                User Query (+ images)
                                        │
                                        ▼
                                ┌──────────────────────┐
                                │   LLM (e.g. GPT-5.2) │◄───────────────────────┐
                                │                      │                        │
                                │   Tool Calls:        │                        │
                                │   ┌────────────────┐ │     ┌──────────────┐   │
                                │   │  execute_code  │─┼────►│Jupyter Kernel│   │
                                │   └────────────────┘ │     │  (Docker)    │   │
                                │   ┌────────────────┐ │     └──────┬───────┘   │
                                │   │    finish      │─┼──► Answer  │ (Output)  │
                                │   └────────────────┘ │            │           │
                                └──────────────────────┘    text + images ──────┘
```

**Key components:**

| Module | Responsibility |
|---|---|
| `config.py` | All constants, tool schemas, system prompt |
| `kernel.py` | Builds Docker image, starts container, manages Jupyter kernel via ZMQ |
| `agent.py` | Orchestrates the agentic loop: LLM calls → tool dispatch → result collection |
| `trajectory.py` | Records every step with timestamps, code, images; saves to JSON |
| `image_utils.py` | Base64 encoding, compression, OpenAI content part builders |
| `file_manager.py` | Copies files into the Docker mount so the kernel can access them |

## CLI Options

```
usage: python -m swe_vision.cli [-h] [--image IMAGE] [--interactive]
                                [--model MODEL] [--api-key API_KEY]
                                [--base-url BASE_URL]
                                [--max-iterations MAX_ITERATIONS]
                                [--save-trajectory SAVE_TRAJECTORY]
                                [--verbose] [--quiet]
                                [--reasoning | --no-reasoning]
                                [query]
```

| Flag | Description |
|---|---|
| `--image, -i` | Image file path (repeatable) |
| `--interactive` | Multi-turn interactive mode |
| `--model, -m` | Model name (default: `gpt-4o` or `$OPENAI_MODEL`) |
| `--reasoning / --no-reasoning` | Enable/disable extended reasoning |
| `--save-trajectory` | Custom trajectory output directory |
| `--quiet, -q` | Minimal console output |

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | API key for the LLM provider | *(required)* |
| `OPENAI_BASE_URL` | Custom API base URL | OpenAI default |
| `OPENAI_MODEL` | Default model name | `gpt-4o` |
| `VLM_DOCKER_IMAGE` | Docker image name for the kernel | `swe-vision:latest` |
| `VLM_DOCKERFILE_DIR` | Path to the Dockerfile directory | `./env/` |
| `VLM_HOST_WORK_DIR` | Host-side working directory for file sharing | `<project>/runtime/swe-vision/workdir` |
| `VLM_WEB_SESSION_DIR` | Session storage for the web app | `<project>/runtime/swe-vision/web_sessions` |

## Programmatic Usage

```python
import asyncio
from swe_vision import VLMToolCallAgent

async def main():
    agent = VLMToolCallAgent(
        model="openai/gpt-5.2",
        api_key="sk-...",
        reasoning=True,
    )
    try:
        answer = await agent.run(
            "Analyze this chart and summarize the trends",
            image_paths=["chart.png"],
        )
        print(answer)
    finally:
        await agent.cleanup()

asyncio.run(main())
```

## License

MIT
