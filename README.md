# 车牌整体贴片变造：人工标注指南

本仓库提供从整车图片提取平面车牌、在浏览器中建立字符槽级人工真值、运行本地/云端兼容视觉模型，以及用人工真值评估检测结果的完整流程。第一次使用时只需要阅读“快速开始”和“页面操作”；后续章节解释 OCR、远程标注、机器预标注和评估。

需要在 RTX 4060 Ti 8GB 服务器上用开源视觉模型替代云端调用时，参见 [本地 Qwen3.5 下载、部署、接线与评估指南](LOCAL_VLM_DEPLOYMENT.md)。

当前标注对象仅限“整体贴片/磁贴覆盖字符”。增加笔画和消除笔画不属于本标注协议。

## 快速开始

人工标注器读取一个已经完成的流水线批次。首次使用先建立环境；仓库已经包含车牌四点定位所需的小型 ONNX 文件：

```bash
cd /path/to/lisence-plate-forgery
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r src/shape/requirements.txt
```

确认 `data/raw/images` 中有足够图片。下面用纯本地确定性路线先抽取40张，以便部分图片定位失败后仍能形成30张标注任务；该步骤不会调用大模型：

```bash

python -m src.pipeline.batch_test data/raw/images \
  --selection random \
  --seed 20260825 \
  --count 40 \
  --output outputs \
  --prefix annotation-pool-local \
  --sticker-method local \
  --workers 4
```

复制命令结束时打印的准确批次目录，不要在续标时重新猜“最新目录”。首次启动前检查成功样本不少于30张：

```bash
BATCH=outputs/annotation-pool-local_<实际时间戳>
python -c "import json; p=json.load(open('$BATCH/batch_report.json', encoding='utf-8')); print('成功样本:', p['successful_count'])"

LABELS=data/annotations/plate_tamper_random30_20260825.csv

python -m src.sticker.annotate_web "$BATCH" \
  --output "$LABELS" \
  --count 30 \
  --selection random \
  --seed 20260825 \
  --host 127.0.0.1 \
  --port 8765
```

在同一台电脑的浏览器打开 `http://127.0.0.1:8765`。每张图只需执行以下操作：

1. 有贴片：点击一个或多个 `S1..S7/S8` 字符按钮；
2. 正常：按 `C`；
3. 只有车牌不可见、严重模糊/曝光损坏，或提取结果确实无法用于判断时才按 `U`；若原图清晰但当前裁图错误，应优先重新定位；
4. 按 `Enter` 保存并进入下一张。

完整车牌文字不是必填项。页面若已显示 OCR 结果，只在识别错误且你希望保留正确文字时修改；字符是否变造始终以从左到右的槽位按钮为准。每次按 `Enter` 都会立即写入 `$LABELS`，中断后使用同一命令即可续标。

## 1. 标签定义

每张车牌只能使用以下三类整牌标签之一：

- `suspicious`：至少一个字符槽存在整体贴片变造，同时记录具体槽位；
- `clear`：未观察到整体贴片变造；
- `unassessable`：车牌不可见、严重模糊、严重过曝/欠曝，或车牌提取失败，因而无法完成判断。

`unassessable` 不是“证据不足”或“不确定”的替代标签。只要车牌清晰可见并且能够观察，就应在 `clear` 和 `suspicious` 之间作出标注。

字符槽从左到右按 `S1` 开始编号：

- 普通蓝牌通常为 `S1..S7`；
- 新能源绿牌通常为 `S1..S8`；
- 编号从 1 开始，不从 0 开始；
- 重复字符仍按位置选择。例如两个 `7` 分别位于 `S5`、`S6` 时，直接点击对应槽位，不需要用字符本身消歧。

完整车牌文字不是必填人工任务。页面中的模型识别结果只用于帮助槽位与可见字符对应：识别正确时无需填写；识别错误时可以直接修改；即使留空，也可以用 `S1..S7/S8` 完成变造槽位标注。

### 1.1 人工判断依据

本项目当前只标注“背景同色材料覆盖整个字符”的贴片或磁贴。判断时先找贴片本身的材料边界，再把边界归属到字符槽；不要只凭字符长得异常就判伪。

重点观察以下证据：

- 与车牌上下边或左右边近似平行的细直线、白边、暗缝或亮暗成对边缘；
- 字符周围出现完整矩形，或能够共同指向矩形的两三段局部边线；受打光、污渍或视角影响时，不要求四条边全部闭合；
- 局部贴片厚度、翘边、气泡、褶皱、贴合不平或残胶；
- 边线围住区域与相邻底材在颜色、反光、纹理、颗粒或老化程度上不连续；
- 新能源绿牌原有纵向渐变被局部打断，例如贴片内部异常均色、渐变方向或幅度与周围不一致；
- 上述较弱迹象在同一字符槽中同时出现，例如“轴对齐细缝 + 材料色差”。

以下现象单独出现时不能作为贴片真值：字符自身轮廓、号牌冲压边缘、外框、铆钉、污渍、划痕、局部反光、阴影、透视形变和 JPEG 压缩伪影。只有它们形成不属于车牌自然结构、并且能够解释为贴片边界或材料不连续的组合证据时，才选择相应字符槽。

## 2. 环境准备

以下命令均从仓库根目录执行。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r src/shape/requirements.txt
# 仅在需要运行仓库测试时安装
python -m pip install pytest
```

只做人工标注不需要大模型。如果需要 OCR 预填、`agent` 贴片复核或重新定位车牌，则需要加载一个 OpenAI 兼容视觉模型端点。使用云端端点会把相应图片发送给第三方服务，应只处理已获授权的数据；默认优先使用本地端点。运行结果和 OCR 缓存也可能含完整号牌文字与本机路径，均已按本地数据处理并排除在 Git 之外。

使用服务器本地 Qwen3.5 时，先按 [本地模型部署指南](LOCAL_VLM_DEPLOYMENT.md) 启动 vLLM，然后复制并加载本地配置：

```bash
cp local_vlm_env.sh.example local_vlm_env.sh
source ./local_vlm_env.sh
```

使用云端兼容端点时设置：

```bash
export OPENAI_API_KEY="你的密钥"
export OPENAI_BASE_URL="你的兼容接口地址"
export OPENAI_MODEL="你的视觉模型名"
```

可以把云端变量放在不会提交到 Git 的 `local_env.sh` 中，然后执行：

```bash
source ./local_env.sh
```

程序不会自动读取任何环境文件。`local_env.sh` 和 `local_vlm_env.sh` 均已被 `.gitignore` 排除，不要强制提交密钥文件。

## 3. 建立固定标注样本池

标注网页的输入不是原始图片目录，而是一次成功的流水线批次目录。该目录应包含 `batch_report.json`，并且每个成功样本目录中应有无预测框的 `03_rectified.jpg`。

如果还没有这样的批次，推荐先用纯本地路线建立样本池，避免在准备标注图片时产生贴牌判断的大模型费用：

```bash
python -m src.pipeline.batch_test data/raw/images \
  --selection random \
  --seed 20260824 \
  --count 100 \
  --output outputs \
  --prefix annotation-pool100-local \
  --sticker-method local \
  --workers 4
```

找到本次实际生成的时间戳目录：

```bash
ls -dt outputs/annotation-pool100-local_* | head -n 1
```

然后设置本轮使用的批次和人工标注文件。例如：

```bash
BATCH=outputs/annotation-pool100-local_20260824_190000
LABELS=data/annotations/plate_tamper_random30_20260824.csv
```

请把 `BATCH` 替换为实际目录。`LABELS` 文件名同时决定固定抽样清单和 OCR 缓存的文件名，因此开始标注后不要随意更换。

## 4. 可选：用视觉模型预填车牌文字

OCR 预填可减少人工输入，但不会生成变造真值，也不会把样本计为已完成人工标注。

本地模型和云端模型二选一：

```bash
# 本地 vLLM 已启动时
source ./local_vlm_env.sh

# 或者使用云端兼容端点
# source ./local_env.sh

python -m src.sticker.prefill_plate_ocr "$BATCH" \
  --annotation-output "$LABELS" \
  --count 30 \
  --selection random \
  --seed 20260824 \
  --workers 4
```

该步骤生成：

```text
<LABELS>.selection.json
<LABELS>.ocr.json
```

- `.selection.json` 固定本轮抽到的图片；
- `.ocr.json` 保存模型识别文字、调用状态和 Token；云端端点还会记录费用估算；
- 使用相同 `LABELS` 再次运行会复用既有清单并续跑未完成 OCR；
- `--force` 会忽略已有 OCR 结果并重新调用模型，通常不要使用；
- 想建立另一组随机样本时，应换一个新的 `LABELS` 文件名。

如果不需要 OCR，可以跳过本节，直接启动人工标注页面。车牌文字不是必填项。

## 5. 启动人工标注页面

### 5.1 标注程序运行在当前电脑

```bash
python -m src.sticker.annotate_web "$BATCH" \
  --output "$LABELS" \
  --count 30 \
  --selection random \
  --seed 20260824 \
  --host 127.0.0.1 \
  --port 8765
```

在当前电脑浏览器打开：

```text
http://127.0.0.1:8765
```

### 5.2 标注程序运行在远程服务器

先在服务器仓库根目录激活环境并启动标注程序：

```bash
source .venv/bin/activate

python -m src.sticker.annotate_web "$BATCH" \
  --output "$LABELS" \
  --count 30 \
  --selection random \
  --seed 20260824 \
  --host 127.0.0.1 \
  --port 8765
```

在本地电脑另开终端建立 SSH 端口转发：

```bash
ssh -N \
  -L 8765:127.0.0.1:8765 \
  -p <SSH_PORT> \
  <USER>@<SERVER_HOST>
```

SSH 命令停在前台且没有输出是正常现象。保持两个终端运行，在本地浏览器打开 `http://127.0.0.1:8765`。

如果 8765 已被占用，可把标注程序和 SSH 转发两边的端口同时改为 8766。检查占用：

```bash
ss -lntp '( sport = :8765 or sport = :8766 )'
pgrep -af 'src.sticker.annotate_web'
```

## 6. 页面操作

1. 观察页面中的无预测框平面车牌图。
2. 模型识别的车牌文字有误时，可在顶部文本框中直接修改。该字段不是必填项。
3. 点击存在整体贴片变造的字符槽，可多选；再次点击可取消。
4. 没有变造时点击“整牌正常”，或直接按 `C`。
5. 只有满足严格拒判条件时才点击“无法判断”，或直接按 `U`。
6. 按 `Enter` 或点击“保存并进入下一张”才会保存当前标注。

快捷键语义：

- `C`：等同点击“整牌正常”，只改变当前选择，不自动保存；
- `U`：等同点击“无法判断”，只改变当前选择，不自动保存；
- `Enter`：保存并进入下一张；
- 正在编辑车牌文本或备注时，`C/U` 作为普通文字输入，不触发快捷键；按 `Esc` 退出输入框后可恢复快捷键。

### 6.1 重新定位车牌

如果当前平面图裁错、缺少车牌边缘或提取成了非车牌区域，并且原始整车图仍然清晰，应先点击页面上的“云端重新定位车牌（会调用模型）”。按钮名称沿用旧界面，但实际会调用启动标注器时加载的兼容端点：

- 从 `input_path` 指向的原始整车图重新进行两阶段四角定位；
- 本地端点不会产生云端费用；云端端点可能产生费用；
- 成功后更新当前样本的 `01_points.jpg`、`02_quad.jpg`、`03_rectified.jpg` 和元数据；
- 把旧结果备份到当前样本的 `relocalization_backups/<时间戳>/`；
- 把操作记录写入 `relocalization_history.json`；
- 调用或写入失败时保留当前平面图。

该功能要求启动标注程序前已经加载可用的视觉模型环境，并且 CSV/批次报告中的 `input_path` 在当前机器上有效。重新定位成功后再按正常流程标注；只有重新定位不可用或仍失败、且当前结果确实无法判断时，提取失败才属于 `unassessable`。

## 7. 保存、断点续标与输出文件

每张图片在按下 `Enter` 后立即原子写入 CSV。浏览器关闭、SSH 断开或程序退出后，使用相同的 `BATCH`、`LABELS` 和参数重新启动，会从未完成图片继续。

同一组 `BATCH + LABELS` 同一时间只能由一个标注进程写入。多人标注时应给每人分配互不重叠的固定样本清单和不同 CSV，完成后按唯一 `image_id` 检查无重叠再合并；不要让两个浏览器或进程并发写同一个 CSV。

主要输出：

```text
data/annotations/plate_tamper_random30_20260824.csv
data/annotations/plate_tamper_random30_20260824.csv.selection.json
data/annotations/plate_tamper_random30_20260824.csv.ocr.json
```

CSV 主要字段：

- `image_id`：样本唯一编号，与批次中的样本目录名一致；
- `decision`：`suspicious`、`clear` 或 `unassessable`；
- `suspicious_slots`：1 起始的变造槽位，例如 `4` 或 `4;7`；
- `plate_text`：人工保留或修正后的可见车牌文字，可以为空；
- `ocr_plate_text`：视觉模型 OCR 的原始输出；
- `plate_text_corrected`：人工是否修改了 OCR 文字；
- `suspicious_characters`：与可见文字能够可靠映射时记录字符；
- `slot_count`：蓝牌为 7，绿牌为 8；
- `input_path`：原始整车图片路径，重新定位和端到端评估使用；
- `rectified_path`：人工标注时查看的平面车牌路径；
- `notes`：可选说明；
- `annotated_at`：保存时间。

完成后可执行基本读取检查：

```bash
python -c "from collections import Counter; from src.sticker.evaluate import read_annotations; a=read_annotations('$LABELS'); print('总数:', len(a)); print('标签:', Counter(x.decision for x in a.values()))"
```

不要直接拼接包含重复 `image_id` 的 CSV。合并多个人工批次时，应先确认样本互不重叠，并保留统一字段和 UTF-8 编码。

## 8. 使用模型识别伪造与生成机器标注

### 8.1 两种识别路线

完整流水线的输入是包含车牌的原始图片，处理顺序为：

```text
原始图片 → 车牌四点定位 → 透视校正 → 本地多通道证据 → 贴片判定 → 六张阶段图和结构化结果
```

贴片判定有两种模式：

- `agent`：默认路线。本地代码先生成候选和物理证据，再由 OpenAI 兼容视觉模型进行受控多轮复核；加载 `local_vlm_env.sh` 时使用服务器本地模型，加载 `local_env.sh` 时使用云端模型；
- `local`：纯确定性基线，不调用视觉大模型，适合低成本筛查和与 `agent` 路线做对照实验。

两种路线都使用本地 ONNX 关键点模型定位车牌。`agent` 并不是让模型直接在整车图上随意画框；最终字符槽与坐标仍由确定性代码管理。

### 8.2 识别单张图片

`agent` 路线（以下以服务器本地模型为例）：

```bash
source ./local_vlm_env.sh
# 若改用云端兼容端点，则改为：source ./local_env.sh

python -m src.pipeline.batch_test path/to/vehicle.jpg \
  --selection first \
  --count 1 \
  --output outputs \
  --prefix detect-one-agent \
  --sticker-method agent \
  --agent-max-calls-per-image 3 \
  --workers 1
```

纯本地路线：

```bash
python -m src.pipeline.batch_test path/to/vehicle.jpg \
  --selection first \
  --count 1 \
  --output outputs \
  --prefix detect-one-local \
  --sticker-method local \
  --workers 1
```

### 8.3 批量机器识别/机器标注

下面的命令从图片目录随机选择100张并固定随机种子。本地 vLLM 的图片工作槽不应超过服务端实际并发能力；当前单请求配置先使用 `--workers 1`：

```bash
source ./local_vlm_env.sh
# 若云端端点允许4路并发，可加载 local_env.sh 并把 workers 改为4

python -m src.pipeline.batch_test data/raw/images \
  --selection random \
  --seed 20260824 \
  --count 100 \
  --output outputs \
  --prefix machine-label-agent100 \
  --sticker-method agent \
  --agent-max-calls-per-image 3 \
  --workers 1
```

对应的纯本地基线：

```bash
python -m src.pipeline.batch_test data/raw/images \
  --selection random \
  --seed 20260824 \
  --count 100 \
  --output outputs \
  --prefix machine-label-local100 \
  --sticker-method local \
  --workers 4
```

在输入目录、`--selection`、`--seed` 和 `--count` 相同的情况下，两条路线会选择同一批图片，便于逐图比较。`--selection first` 则按文件名排序选择前 N 张，不使用随机种子。

这里的“机器标注”是模型预测，不是人工真值。可以用于预筛查、候选样本选择或人工复核起点，但不能直接当作测试集真值或未经复核的训练标签。

### 8.4 每张图片的六张输出

每个成功样本目录严格包含：

```text
<sample_id>/
├── 01_points.jpg             # 四个定位点
├── 02_quad.jpg               # 车牌四边形
├── 03_rectified.jpg          # 透视校正后的完整平面车牌
├── 04_candidate_overlay.jpg  # 本地候选字符槽
├── 05_final_marked.jpg       # 最终机器判定框
└── 06_candidate_mask.png     # 候选掩膜
```

批次根目录还包含：

```text
<run_dir>/
├── batch_report.json
├── selection_manifest.csv
├── results/
└── <sample_id>/...
```

- `results/`：每个成功样本只有一张最终答案 JPG，文件名包含整牌结论和槽位；
- `batch_report.json`：完整的逐图结构化预测、失败信息、耗时、Token、API 调用和费用；
- `selection_manifest.csv`：便于用表格软件查看本批样本和结论；
- 每个样本的六张图：用于定位错误、候选生成和最终判定的逐阶段复查。

`batch_report.json` 中最重要的逐图字段包括：

- `decision`：`suspicious`、`clear` 或 `unassessable`；
- `selected_candidates`：最终判定为贴片的候选编号；
- `uncertain_candidates`：需要复核但未计作阳性的候选；
- `tampered_characters`：最终贴片字符的槽位、可见字符、方框与置信等级；
- `uncertain_characters`：未达到最终阳性门槛的待复核字符；
- `quality`：图像是否可判别以及质量门依据；
- `llm_usage`：该图片的模型调用、缓存与 Token 信息，仅在 `agent` 路线出现。

模型输出的 `character` 表示照片中当前可见的字符，不推断贴片下面被遮挡的原字符。无法可靠读取时允许为 `null`，但槽位和方框仍可存在。

### 8.5 调用次数、并行和费用控制

`--agent-max-calls-per-image` 可取：

- `2`：较低消耗；
- `3`：默认，调查、反证、裁决；
- `4`：允许额外冲突复查，消耗最高。

`--workers N` 表示最多并行处理 N 张图片，不是每张图片调用 N 次。N 不应高于视觉模型服务端的有效并发，否则只会增加排队、显存压力或超时。总模型调用、输入/输出 Token 和估算费用会写入 `batch_report.json` 并在命令结束时打印；本地兼容端点的外部模型费用应为零。

默认 API 缓存位于输出根目录的 `.pipeline_api_cache`。只有图片、模型、提示词和协议均匹配时才会命中缓存。通常不要使用 `--no-cache`，否则重复实验可能再次产生费用。

如果服务商单价无法被程序识别，可以显式传入：

```bash
--input-price-per-million-cny <每百万输入Token单价> \
--output-price-per-million-cny <每百万输出Token单价>
```

### 8.6 把机器结果交给人工复核

机器批次本身已经符合人工网页标注器的输入格式。运行机器识别后，把实际时间戳目录作为新的 `BATCH`，并使用一个新的人工 CSV：

```bash
BATCH=outputs/machine-label-agent100_<实际时间戳>
LABELS=data/annotations/machine-label-agent100-human-review.csv

python -m src.sticker.annotate_web "$BATCH" \
  --output "$LABELS" \
  --count 100 \
  --selection first \
  --seed 20260824 \
  --host 127.0.0.1 \
  --port 8765
```

人工页面只显示无预测框的 `03_rectified.jpg`，不会把机器框直接展示给标注员，目的是减少模型结果对人工真值的诱导。人工标注完成后，再用第9节的评估命令对照同批机器结果。

如果需要在人工页面预填车牌号码，先针对这个 `BATCH` 和同一个 `LABELS` 运行第4节的 `prefill_plate_ocr`。

## 9. 使用人工标注评估模型

### 9.1 从人工标注中的原始图片重新运行完整 `agent` 流水线

```bash
source ./local_vlm_env.sh
# 若使用云端兼容端点，则改为：source ./local_env.sh

python -m src.pipeline.evaluate_annotated "$LABELS" \
  --images-root data/raw/images \
  --output outputs \
  --prefix eval-annotated-agent \
  --sticker-method agent \
  --agent-max-calls-per-image 3 \
  --workers 1
```

如果 CSV 中的 `input_path` 在当前机器失效，`--images-root` 会按 `image_id` 在原始图片目录中重新寻找唯一匹配。

### 9.2 评估纯本地路线

```bash
python -m src.pipeline.evaluate_annotated "$LABELS" \
  --images-root data/raw/images \
  --output outputs \
  --prefix eval-annotated-local \
  --sticker-method local \
  --workers 4
```

### 9.3 只评估已经存在的机器结果

```bash
RUN=outputs/<已有机器批次目录>

python -m src.sticker.evaluate "$LABELS" "$RUN" \
  --output "$RUN/metrics-from-human-labels.json"
```

人工标注与机器结果必须具有相同的 `image_id`。评估结果中的 `missing_reports` 和 `unlabelled_reports` 应先检查，避免把样本不匹配误认为模型错误。

## 10. 指标解释

评估输出同时包含两种层级：

- `plate_level`：整张车牌是否存在至少一个贴片变造；
- `character_slot_level`：具体 `S1..S7/S8` 是否定位正确。

因此，模型可能在整牌层面判断正确，但框错字符槽；这时整牌指标可能记为正确，字符槽指标仍会产生假阳性或假阴性。

人工真值为 `unassessable` 的样本不参与 `clear/suspicious` 准确率。机器输出 `unassessable` 时，还应同时查看 `plate_level_answered_only.coverage`，不能只报告已回答样本的准确率。

小样本指标的不确定性很高。30 张适合检查标注协议和发现明显错误，不适合宣称稳定性能；正式报告应同时给出样本量、标签分布、precision、recall、F1、specificity、coverage 和置信区间。

## 11. 常见问题

### 页面显示“Address already in use”

说明端口已有标注进程。先用 `ss` 和 `pgrep` 确认进程；继续原任务可直接访问原端口，另开任务则换一个未占用端口。

### 浏览器显示 `127.0.0.1 refused to connect`

确认标注程序仍在运行。远程标注还要确认 SSH 端口转发终端没有断开，并且浏览器访问的是本地转发端口。

### OCR 或重新定位提示缺少模型/密钥

程序不会自动加载环境文件。回到启动标注程序的终端，先执行 `source ./local_vlm_env.sh`（本地模型）或 `source ./local_env.sh`（云端模型），再重启程序。

### 修改 seed 后仍然出现同一组图片

同一个 `LABELS` 会复用既有 `.selection.json`。要建立新样本组，请使用新的 CSV 文件名；不要删除已完成任务的 selection 文件。

### 服务器与本地路径不同

`input_path` 可能记录另一台机器的绝对路径。人工页面仍可显示批次内的 `03_rectified.jpg`，但重新定位和端到端评估需要有效原图；评估时可通过 `--images-root` 重新解析原图。

## 12. 提交到 GitHub 前

- 确认 `local_env.sh`、`local_vlm_env.sh`、API 密钥和虚拟环境没有被提交；
- `outputs/`、`data/raw/`、`data/annotations/`、运行日志和下载的模型权重不作为代码提交；
- 如确需单独发布人工 CSV，应先根据数据授权脱敏 `plate_text`、`input_path` 和样本编号，并通过独立的数据发布流程处理；
- 至少运行相关测试：

```bash
python -m pytest -q src/shape/tests src/sticker/tests src/pipeline/tests
```
