# v3 使用指南

## 1. 进入项目并激活环境

```bash
# 在项目根目录运行；如虚拟环境不在默认位置，可设置 VIRTUAL_ENV_PATH。
source "${VIRTUAL_ENV_PATH:-$HOME/swe-vision-venv}/bin/activate"
source local_env.sh
```

该 `local_env.sh` 需要提供 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OPENAI_MODEL`。若未设置模型，程序默认使用 `qwen3-vl-plus`。

## 2. 先运行一张图片

```bash
python -m license.v3.run_forgery data/raw/images/AADY1439.JPG
```

运行流程为：本地 HSV 生成多个候选，按矩形边界、底色、字符状笔画、尺寸和位置做结构评分；仅在最高分可信时裁剪车牌，否则自动回退整图，再交给 SWE Agent 处理。初始模型请求只含本地确认的裁剪图或回退的原图；SWE 保留默认工具、关闭扩展推理，单图最多运行 10 轮。

裁剪完成后，程序会将该图实际的宽 × 高（像素）动态加入系统提示词。模型返回的 `bbox` 必须以这个尺寸的裁剪图左上角为原点。

单图最多 10 次模型回复：前 9 次仍可调用 `execute_code`，最后一次由程序移除该工具并强制调用 `finish`，以返回最终 JSON。若首次输出不合格，重试只能使用本图剩余的调用预算，绝不会超过 10 次。

若兼容 API 在第 10 次没有实际调用 `finish`，程序不会执行其工具调用，而是追加一次仅允许 `finish` 的恢复轮；因此极端情况下单图会有 11 次调用。

程序会严格验证最终 JSON、字段关系和异常框坐标范围。首次结果不合格时，会把错误反馈给模型并对同一张图最多重试一次；两次都失败时，`result.json` 会保留错误原因，不会把它标记为有效判定。

若自动定位的车牌框不正确，指定原图像素坐标：

```bash
python -m license.v3.run_forgery data/raw/images/AADY1439.JPG \
  --plate-box x,y,width,height
```

若本地裁剪不适用，可跳过裁剪并直接把原图交给模型；异常框坐标此时以原图为准：

```bash
python -m license.v3.run_forgery data/raw/images/AADY1439.JPG --no-crop-plate
```

## 3. 处理前 100 张数据

```bash
python -m license.v3.run_batch
```

默认读取 `data/manifest.csv` 的前 100 条。每张图的实际调用轮数由 SWE 中模型的工具决策决定；若要改变处理数量：

```bash
python -m license.v3.run_batch --limit 20 --workers 4
```

运行时终端会显示进度条、当前文件名，以及该文件的完成或失败状态。

批处理默认启动最多 4 个独立 Docker/Jupyter worker，整批结束后关闭。每个 worker 在自己的分组内复用内核，每张图的 SWE 消息对话会重置；`result.json` 记录该图耗时，`run.json` 汇总每张图、整批及各 worker 的 Docker 启动耗时。可用 `--workers 2` 等参数调整并发组数。

## 4. 查看结果

每次批处理会生成一个以时间戳命名的结果目录：

```text
runtime/license-v3/runs/<时间戳>/
├── run.json                    # 本批次的元数据与汇总
├── answer/                     # 本批次全部画框车牌图，直接可看
└── result/
    └── <原图名>/               # 单图三件套
        ├── plate.jpg
        ├── plate_annotated.jpg
        └── result.json
```

`run.json` 记录本批次的时间戳、模型、代码版本与哈希、提示词哈希、每张图的结果位置、累计 token、成本，以及完整的数据范围（默认 `manifest.csv` 前 100 条、成功数与失败数）。
