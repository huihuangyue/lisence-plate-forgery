# v2：简化提示词的车牌筛查工作流

完整操作步骤见 [USAGE.md](USAGE.md)。

v1 已原样归档在 `../v1(极繁)/`，不作为 v2 的运行入口。

## 设计边界

- 本地 HSV 颜色掩码定位并裁剪蓝/绿车牌；不为定位调用模型。
- 裁剪车牌交给 SWE Agent；保留默认工具、关闭扩展推理，单图最多运行 5 轮。
- v2 的系统提示词只有伪造特征清单与逐字符 JSON 输出规范。
- 裁剪完成后，程序会把该图的实际宽高动态写入提示词，作为异常框的像素坐标系。
- 单图最多 5 次模型回复；前 4 次可调用代码，最后一次程序只提供并强制调用 `finish`，为 JSON 结果预留交付轮次。
- 每次代码工具结果会追加程序生成的剩余轮次提示，要求模型基于已有证据收敛并交付 JSON。
- 程序会验证最终 JSON、字段语义和异常框是否位于裁剪图范围；不合格时仅对该图重试一次，并记录验证错误与额外 token。
- 初始请求只附带当前一张裁剪车牌，不附带原图或其他历史样本。若模型自行执行代码，工具结果会按 SWE 默认机制进入其后续上下文。
- 批处理默认启动最多 4 个独立 Docker/Jupyter worker，并在整批结束时关闭；每个 worker 在自己的分组内复用内核，每张图都会重置 SWE 消息对话。
- 单图 `result.json` 记录该图 `elapsed_seconds`；批次 `run.json` 记录单图耗时、整批耗时和 Docker 启动耗时。

## 下一步

## 运行

```bash
python -m license.v2.run_forgery data/raw/images/example.jpg
```

程序先在本地定位并裁剪车牌，再交给 SWE 处理；初始输入仅为该裁剪图。自动定位不正确时，可手动指定车牌框：

```bash
python -m license.v2.run_forgery image.jpg --plate-box x,y,width,height
```

默认启用本地车牌裁剪。若要跳过裁剪、将原图直接交给模型：

```bash
python -m license.v2.run_forgery image.jpg --no-crop-plate
```

单图运行的过程文件保存在 `runtime/license-v2/single/results/<原图名>/<时间戳>/`。批处理使用下方的统一运行目录结构。

默认批处理范围为 `data/manifest.csv` 的前 100 条：

```bash
python -m license.v2.run_batch
```

默认使用 4 个并行 worker；可用 `--workers` 调整，例如 `python -m license.v2.run_batch --workers 2`。

每次批处理生成一个以时间戳命名的结果目录：`runtime/license-v2/runs/<时间戳>/`。其中 `run.json` 记录整个批次的模型、代码版本、累计 token、成本和前 100 条数据范围；`answer/` 直接包含本批所有画框后的车牌图；`result/<原图名>/` 包含每张图的 `plate.jpg`、`plate_annotated.jpg` 和 `result.json`。
