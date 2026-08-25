# 车牌变造流水线：本地开源视觉模型部署指南

本指南用于在目标服务器上部署开源视觉语言模型，替代当前 DashScope 云端视觉模型调用。部署完成后，现有的车牌定位、透视矫正、本地物理证据、三阶段智能体判断、六图输出和人工标注评估流程都保留，只把智能体的模型端点从公网改为服务器本机。

本文按当前服务器环境编写：Ubuntu 22.04（WSL2）、NVIDIA RTX 4060 Ti 8GB、项目目录 `/home/alex/lisence-plate-forgery`。所有安装和下载命令都应在目标服务器的 WSL 中执行，不是在原电脑本地执行。

## 1. 推荐结论

主模型选择：[`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

- 它是视觉语言模型，能同时读取提示词、矫正车牌图、槽位图和诊断面板，并返回现有智能体需要的 JSON。
- 官方模型为 Apache-2.0 许可，约 2.13B 参数，ModelScope 标注的下载体积约 4.27GB。
- 官方模型页直接给出了 vLLM 和 OpenAI 兼容 Chat Completions 的用法。
- `Instruct` 版比 `Thinking` 版更适合当前固定协议、结构化 JSON 和低延迟任务。
- 它可以在当前 8GB 显存机器上作为合理的第一版，但必须限制上下文、多图数量和并发。

不把下列模型作为当前服务器首选：

- `Qwen3-VL-4B-Instruct` 的官方 BF16 权重约 8.89GB，仅权重就超过 8GB 显存，缺少 KV cache、视觉编码器中间量和 CUDA 工作区的空间。
- `Qwen3-VL-4B-Instruct-FP8` 仍会明显压缩 8GB 机器的运行余量，消费级 Ada 卡上的实际内核兼容性和峰值显存也需要另行验证。
- `Qwen3-VL-8B-Instruct` 更适合未来的 A6000，不适合作为当前 4060 Ti 8GB 服务器的第一落地点。
- YOLO 不能直接替代云端智能体。YOLO 是训练后输出固定类别/框的检测器，不读取当前多轮提示词，也不会直接生成本项目所需的解释和 JSON。以后可以训练 YOLO 作为快速候选器，但这与本次本地 VLM 替换是两个任务。

需要明确：2B 本地模型可以把流程离线跑通，但不能先验保证达到 `qwen3-vl-plus` 的准确率。是否可用必须在现有 100 张人工标注集上实测，不能只看模型通用榜单。

## 2. 替换后的信息流

```text
整车图片
  -> 本地 ONNX 车牌四点定位
  -> 透视矫正和平面车牌
  -> OpenCV/CIE Lab/边缘/材料差异等本地物理证据
  -> 本地 vLLM 服务上的 Qwen3-VL-2B-Instruct
  -> 调查、反证、裁决（默认每张 3 次）
  -> 字符槽位、变造字符、最终方框图和 JSON
```

项目中的 `StickerAgentHarness` 已经使用标准 OpenAI 多模态消息，且每轮发送 3 张图片，因此无需重写智能体。只要把下面三个环境变量改成本地服务即可：

```bash
OPENAI_API_KEY=EMPTY
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_MODEL=qwen3-vl-2b-local
```

## 3. 部署前检查

先 SSH 进入目标服务器 WSL，然后执行：

```bash
nvidia-smi
python3.10 --version
df -h /home/alex
free -h
```

验收条件：

- `nvidia-smi` 能看到 RTX 4060 Ti，且没有其他程序长期占满显存。
- Python 3.10 可用。
- `/home/alex` 建议至少预留 25GB 空间，用于约 4.27GB 模型、vLLM/PyTorch/CUDA Python 依赖、缓存和日志。
- 项目位于 `/home/alex/lisence-plate-forgery`。

不要继续使用之前的 Python 3.13 YOLO/vLLM 混合环境。vLLM 官方建议使用全新的虚拟环境，因为它的预编译 CUDA 内核与 PyTorch 版本强绑定。

## 4. 创建独立的 vLLM 环境

模型服务与项目流水线使用两个环境：

- `/home/alex/.venvs/lisence-plate-forgery-vllm`：只运行本地模型服务。
- `/home/alex/.venvs/lisence-plate-forgery`：继续运行现有项目。

在服务器执行：

```bash
python3.10 -m venv /home/alex/.venvs/lisence-plate-forgery-vllm
source /home/alex/.venvs/lisence-plate-forgery-vllm/bin/activate
python -m pip install --upgrade pip uv
uv pip install "vllm>=0.11.0" --torch-backend=auto
uv pip install "qwen-vl-utils==0.0.14" modelscope openai
```

Qwen 官方要求 Qwen3-VL 使用 `vllm>=0.11.0`；vLLM 官方推荐用 `uv ... --torch-backend=auto` 根据 NVIDIA 驱动选择 PyTorch 后端。不要先单独安装另一个版本的 Torch，再让 vLLM 覆盖它。

安装完成后检查：

```bash
python -c 'import torch, vllm; print("torch=", torch.__version__); print("cuda=", torch.cuda.is_available()); print("gpu=", torch.cuda.get_device_name(0)); print("vllm=", vllm.__version__)'
```

期望 `cuda=True`，GPU 名称为 RTX 4060 Ti。如果这里失败，不要开始下载或启动模型，先处理驱动/依赖问题。

## 5. 下载模型到服务器

国内网络优先使用 ModelScope。其官方 CLI 支持指定 `--local_dir`，下载中断后重复执行同一命令可继续补齐文件。

建议在 `tmux` 中下载，避免 SSH 断开导致前台进程结束：

```bash
tmux new -s qwen-download
source /home/alex/.venvs/lisence-plate-forgery-vllm/bin/activate
mkdir -p /home/alex/models/Qwen3-VL-2B-Instruct
modelscope download \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --local_dir /home/alex/models/Qwen3-VL-2B-Instruct
```

下载过程中按 `Ctrl-b`，松开后按 `d`，可退出 tmux 而不终止下载。重新查看：

```bash
tmux attach -t qwen-download
```

下载完成后检查：

```bash
test -f /home/alex/models/Qwen3-VL-2B-Instruct/config.json
find /home/alex/models/Qwen3-VL-2B-Instruct -maxdepth 1 -type f -printf '%f\n' | sort
du -sh /home/alex/models/Qwen3-VL-2B-Instruct
```

如果 ModelScope 不可用，可改用 Hugging Face 官方仓库：

```bash
source /home/alex/.venvs/lisence-plate-forgery-vllm/bin/activate
uv pip install huggingface_hub
hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir /home/alex/models/Qwen3-VL-2B-Instruct
```

两种下载方式任选一种，不要把同一模型重复下载到多个缓存目录。

## 6. 启动本地 OpenAI 兼容服务

第一次启动使用保守参数：单请求调度、最多 3 张图、8192 上下文，仅监听本机，不向局域网开放。

```bash
tmux new -s qwen3vl
source /home/alex/.venvs/lisence-plate-forgery-vllm/bin/activate
export CUDA_VISIBLE_DEVICES=0

vllm serve /home/alex/models/Qwen3-VL-2B-Instruct \
  --served-model-name qwen3-vl-2b-local \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":3,"video":0}' \
  --generation-config vllm
```

说明：

- 本项目每次智能体调用正好发送 3 张图，`image:3` 不能改成 1。
- `--max-num-seqs 1` 是 8GB 显存的保守起点。它不限制整批数量，只限制模型服务同一时刻处理的序列数。
- `--generation-config vllm` 避免模型仓库中的采样默认值覆盖服务端确定性设置；项目请求本身使用 `temperature=0`。
- 只监听 `127.0.0.1` 可避免模型服务被局域网其他机器直接调用。项目和模型服务在同一个 WSL 中运行，不需要开放防火墙端口。

看到服务开始监听 `http://127.0.0.1:8000` 后，可按 `Ctrl-b d` 离开 tmux。SSH 断开不会停止服务；Windows 重启或 `wsl --shutdown` 会停止服务，之后需要重新启动该 tmux 命令。

另开一个服务器终端检查：

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
nvidia-smi
```

响应中应包含 `qwen3-vl-2b-local`。

### 6.1 如果启动时显存不足

先检查并结束不需要的 GPU 程序，不要盲目删除环境：

```bash
nvidia-smi
```

然后用更保守参数重启：

```bash
source /home/alex/.venvs/lisence-plate-forgery-vllm/bin/activate
export CUDA_VISIBLE_DEVICES=0
vllm serve /home/alex/models/Qwen3-VL-2B-Instruct \
  --served-model-name qwen3-vl-2b-local \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.82 \
  --max-model-len 6144 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":3,"video":0}' \
  --generation-config vllm \
  --enforce-eager
```

若 6144 报输入过长，再恢复 8192；这表示问题是上下文容量，不是模型判断错误。若模型权重本身仍无法加载，保存完整启动日志再判断，不要直接换 8B 或增加并发。

## 7. 让项目改用本地模型

不要再 `source local_env.sh`，它会把端点重新指向 DashScope。现有代码优先读取 `SHAPE_VISION_MODEL`，因此还要显式清除这个旧变量。

在运行流水线的服务器终端执行：

```bash
cd /home/alex/lisence-plate-forgery
source /home/alex/.venvs/lisence-plate-forgery/bin/activate

unset SHAPE_VISION_MODEL
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_MODEL=qwen3-vl-2b-local

python -c 'import os; print(os.getenv("OPENAI_MODEL")); print(os.getenv("OPENAI_BASE_URL")); print(os.getenv("OPENAI_API_KEY"))'
```

`EMPTY` 不是云端密钥，只是满足 OpenAI Python 客户端的非空参数要求。图片不会离开服务器。

为了每次登录后快速加载，可在服务器创建个人配置文件：

```bash
mkdir -p /home/alex/.config/lisence-plate-forgery
nano /home/alex/.config/lisence-plate-forgery/local_vlm_env.sh
```

写入：

```bash
unset SHAPE_VISION_MODEL
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_MODEL=qwen3-vl-2b-local
```

保存后执行：

```bash
chmod 600 /home/alex/.config/lisence-plate-forgery/local_vlm_env.sh
source /home/alex/.config/lisence-plate-forgery/local_vlm_env.sh
```

## 8. 分级验证

### 8.1 一级：服务健康检查

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

这一步只证明服务活着，不证明它能正确处理项目的 3 张图和 JSON。

### 8.2 二级：3 张端到端冒烟测试

第一次只跑 3 张、单工作槽、每张 2 次调用。它用于发现接口、图片数量、上下文、JSON 解析和显存问题：

```bash
cd /home/alex/lisence-plate-forgery
source /home/alex/.venvs/lisence-plate-forgery/bin/activate
source /home/alex/.config/lisence-plate-forgery/local_vlm_env.sh

python -m src.pipeline.batch_test data/raw/images \
  --selection random \
  --seed 20260825 \
  --count 3 \
  --output outputs \
  --prefix pipeline-local-qwen3vl2b-smoke \
  --sticker-method agent \
  --model qwen3-vl-2b-local \
  --base-url http://127.0.0.1:8000/v1 \
  --agent-max-calls-per-image 2 \
  --workers 1 \
  --input-price-per-million-cny 0 \
  --output-price-per-million-cny 0
```

通过条件：`成功：3/3`，三个样本都生成六图、最终答案图和 JSON；服务终端没有 CUDA OOM。

### 8.3 三级：20 张运行稳定性测试

恢复正式的三阶段协议，但仍保持单工作槽：

```bash
python -m src.pipeline.batch_test data/raw/images \
  --selection random \
  --seed 20260826 \
  --count 20 \
  --output outputs \
  --prefix pipeline-local-qwen3vl2b-random20 \
  --sticker-method agent \
  --model qwen3-vl-2b-local \
  --base-url http://127.0.0.1:8000/v1 \
  --agent-max-calls-per-image 3 \
  --workers 1 \
  --input-price-per-million-cny 0 \
  --output-price-per-million-cny 0
```

外部模型费用应记录为 0 元。报告里仍可能出现“API 调用次数”，因为程序通过 HTTP API 调本机 vLLM；这不代表调用了公网或产生云端账单。

### 8.4 四级：100 张人工标注集验收

用已经合并的人工标注集做正式对比：

```bash
LABELS=data/annotations/plate_tamper_combined100_20260825.csv

python -m src.pipeline.evaluate_annotated "$LABELS" \
  --images-root data/raw/images \
  --output outputs \
  --prefix eval-combined100-local-qwen3vl2b \
  --sticker-method agent \
  --model qwen3-vl-2b-local \
  --base-url http://127.0.0.1:8000/v1 \
  --agent-max-calls-per-image 3 \
  --workers 1 \
  --no-cache \
  --input-price-per-million-cny 0 \
  --output-price-per-million-cny 0
```

命令结束后会打印：

```text
评估批次：outputs/eval-combined100-local-qwen3vl2b_时间戳
指标文件：outputs/eval-combined100-local-qwen3vl2b_时间戳/metrics.json
```

必须至少检查：

- 整牌层面的 precision、recall、accuracy，以及假阳性/假阴性数量。
- 字符槽层面的 precision、recall、F1。
- 整牌或槽位有一点不一致即算错的 exact-match。
- `unassessable` 是否仍只出现在不可见、严重模糊或定位失败样本。
- 100 张成功率、平均耗时、P95 耗时和吞吐量。
- 误报是否集中在 S6-S8、铆钉、车牌边框、正常渐变或字符本身边缘。

只有接口跑通但指标明显低于云端基线时，不能宣称替换完成。此时应先分析固定人工集上的错误类型，再决定是调整图像输入分辨率、提示信息流，还是用人工数据进行 LoRA 微调。

## 9. 并发和速度调优

当前 8GB 服务器从 `--workers 1` 开始。项目的 `workers` 是同时处理的图片数，而每张图片默认还会调用模型 3 次。

稳定跑完 20 张后，才尝试两路：

1. 停止 vLLM。
2. 把服务启动参数改为 `--max-num-seqs 2`。
3. 把流水线改为 `--workers 2`。
4. 用同一 20 张、同一随机种子比较总耗时和失败率。

如果出现显存不足、JSON 失败率上升或单张 P95 变差，退回单路。不要直接沿用云端实验的 `--workers 4`，云端并发能力不能代表 8GB 本地 GPU 的并发能力。

可用以下命令实时观察：

```bash
watch -n 1 nvidia-smi
```

本地模型的主要成本变成 GPU 时间和电费，不再是 Token 账单。当前脚本只统计外部模型 API 费用，因此本地实验显式传入两项 0 单价；如果以后需要机器成本，应另行按总运行时间和整机功耗计算，不能混进云端 Token 费用。

## 10. 日常启动、停止和重连

查看模型服务是否存在：

```bash
tmux ls
curl -s http://127.0.0.1:8000/v1/models
```

重新进入服务窗口：

```bash
tmux attach -t qwen3vl
```

离开但保持服务：按 `Ctrl-b`，松开，再按 `d`。

正常停止模型服务：

```bash
tmux attach -t qwen3vl
```

然后按 `Ctrl-c`。不要用 `kill -9` 作为日常停止方式。

如果 SSH 断开，只需重新 SSH 登录；tmux 中的下载或服务仍在。如果 Windows 重启或执行了 `wsl --shutdown`，WSL 中所有进程都会停止，需要重新创建 `qwen3vl` 会话并启动模型。

## 11. 常见故障

### `Connection refused` 或本地调用连续失败

```bash
curl -v http://127.0.0.1:8000/v1/models
tmux ls
```

没有监听说明 vLLM 没启动或已崩溃，先查看 `tmux attach -t qwen3vl` 中的错误。

### 程序仍然调用 DashScope

```bash
python -c 'import os; print("shape=", os.getenv("SHAPE_VISION_MODEL")); print("model=", os.getenv("OPENAI_MODEL")); print("url=", os.getenv("OPENAI_BASE_URL"))'
```

确保 `SHAPE_VISION_MODEL` 为空、URL 是 `127.0.0.1:8000/v1`，并且没有在本地配置之后再次 `source local_env.sh`。运行命令显式传 `--model` 和 `--base-url` 可以进一步避免环境污染。

### 报最多只允许 1 张图片

服务启动参数缺少或错误设置了多模态限制。当前流程每轮发送 3 张图，必须使用：

```bash
--limit-mm-per-prompt '{"image":3,"video":0}'
```

### 输入长度超过 `max_model_len`

优先恢复 `--max-model-len 8192`。不要先删掉诊断图，因为三张图是当前信息流的一部分。若 8192 导致显存不足，再考虑降低诊断面板分辨率，这需要作为代码变更单独评估。

### JSON 解析连续失败

先保存具体样本、三阶段原始响应和 vLLM 日志。小模型的结构化遵循能力可能弱于云端大模型；这属于模型能力差异，不是网络故障。不要只通过无限叠加提示词掩盖，应统计失败率并在固定标注集上验证修改。

### 安装时访问 `pypi.nvidia.com` 超时

不要继续在旧环境中反复安装 Torch。确认正在使用新的 vLLM 环境和官方推荐命令：

```bash
source /home/alex/.venvs/lisence-plate-forgery-vllm/bin/activate
uv pip install "vllm>=0.11.0" --torch-backend=auto
```

若仍超时，说明服务器到 Python/CUDA 包源的网络路径有问题。应先修复代理或在可联网机器下载匹配的 wheel 后转存；不要混用不匹配的 Torch、CUDA 和 vLLM 版本。

## 12. 未来迁移到 A6000

A6000 有更大的显存，可把模型升级为 `Qwen3-VL-8B-Instruct`，同时逐步增加上下文和并发。但迁移时仍保持同一接口名、同一 100 张人工集和同一三阶段协议，分别记录：

- 模型版本和权重来源。
- vLLM、Torch、CUDA 和驱动版本。
- 启动参数。
- 整牌、槽位和 exact-match 指标。
- 平均/P95 延迟与吞吐量。

不要因为 A6000 放得下更大模型，就跳过 2B、8B 在同一标注集上的对照。部署模型应由准确率、召回率、延迟和显存共同决定。

## 13. 官方资料

- [Qwen3-VL 官方仓库](https://github.com/QwenLM/Qwen3-VL)：官方推荐 `vllm>=0.11.0`，支持 vLLM/SGLang 和 OpenAI 风格 API。
- [Qwen3-VL-2B-Instruct 官方 Hugging Face 页面](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)：模型许可、参数和 vLLM 调用示例。
- [Qwen3-VL-2B-Instruct 官方 ModelScope 页面](https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct)：模型体积和国内下载入口。
- [ModelScope 官方下载文档](https://modelscope.cn/docs/models/download)：`modelscope download --model ... --local_dir ...` 用法。
- [vLLM 官方安装文档](https://docs.vllm.ai/en/latest/getting_started/quickstart/)：独立环境、`uv` 和 `--torch-backend=auto`。
- [vLLM 官方多模态输入文档](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)：OpenAI Chat Completions 的图片输入和多图限制。

