#!/usr/bin/env bash
set -euo pipefail

# WSL 将宿主机 NVIDIA 驱动库放在该目录；显式加入可避免安装或启动时误判为 CPU。
export PATH="/usr/lib/wsl/lib:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# 服务器只有CUDA运行库，没有nvcc；使用vLLM原生采样以避免FlashInfer首次JIT编译。
export VLLM_USE_FLASHINFER_SAMPLER="0"

model_dir="${QWEN35_MODEL_DIR:-/home/alex/models/Qwen3.5-4B}"
served_name="${QWEN35_SERVED_NAME:-qwen35-4b-local}"
listen_host="${QWEN35_HOST:-127.0.0.1}"
listen_port="${QWEN35_PORT:-8000}"
max_model_len="${QWEN35_MAX_MODEL_LEN:-16384}"
gpu_memory_utilization="${QWEN35_GPU_MEMORY_UTILIZATION:-0.88}"
cpu_offload_gb="${QWEN35_CPU_OFFLOAD_GB:-6}"
max_num_seqs="${QWEN35_MAX_NUM_SEQS:-1}"

if [[ ! -s "$model_dir/config.json" ]]; then
  echo "模型尚未下载完整：$model_dir/config.json" >&2
  exit 2
fi
if ! command -v vllm >/dev/null 2>&1; then
  echo "当前环境找不到 vllm；请先激活独立的 vLLM 虚拟环境。" >&2
  exit 2
fi

args=(
  serve "$model_dir"
  --served-model-name "$served_name"
  --host "$listen_host"
  --port "$listen_port"
  --dtype bfloat16
  --gpu-memory-utilization "$gpu_memory_utilization"
  --cpu-offload-gb "$cpu_offload_gb"
  --max-model-len "$max_model_len"
  --max-num-seqs "$max_num_seqs"
  --limit-mm-per-prompt '{"image":3,"video":0}'
  --reasoning-parser qwen3
  --enforce-eager
)

echo "model=$model_dir"
echo "served_name=$served_name"
echo "endpoint=http://$listen_host:$listen_port/v1"
echo "max_model_len=$max_model_len, cpu_offload_gb=$cpu_offload_gb, max_num_seqs=$max_num_seqs"
exec vllm "${args[@]}"
