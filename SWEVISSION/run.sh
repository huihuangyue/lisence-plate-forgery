#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Environment ────────────────────────────────────────────────
export OPENAI_API_KEY="YOUR_API_KEY_HERE" # set your openai api key here
export OPENAI_BASE_URL="https://openrouter.ai/api/v1" # set your openai/openrouter/... base url here
export MODEL_NAME="${MODEL_NAME:-openai/gpt-5.2}" # set your model name here

# ─── Run agent with an image question ───────────────────────────
IMAGE_PATH="${1:-./assets/test_image.png}"
QUERY="${2:-What is the gap between GPT5.2 and 6-year-olds from the chart?}"

python -m swe_vision.cli \
    --image "$IMAGE_PATH" \
    --model "$MODEL_NAME" \
    --reasoning \
    "$QUERY"
