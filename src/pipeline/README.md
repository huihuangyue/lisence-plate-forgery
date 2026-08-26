# 统一号牌物理变造筛查流水线

该模块把四关键点号牌定位、透视归一化、确定性物理证据和多轮视觉模型复核串联起来。默认 `agent` 路线覆盖整字符贴片、增加笔画、消除笔画和一增一消；`local` 仍只作为整字符贴片诊断基线。

每个成功样本目录严格包含六张图：

1. `01_points.jpg`：原图上的 TL/TR/BR/BL 四点；
2. `02_quad.jpg`：原图上的号牌闭合外框；
3. `03_rectified.jpg`：固定物理尺度的平面号牌；
4. `04_candidate_overlay.jpg`：字符槽及贴片候选分数；
5. `05_final_marked.jpg`：红框高风险、橙框待复核的最终答案图；
6. `06_candidate_mask.png`：只包含红框高风险区域的二值掩膜。

批量随机抽取 20 张：

```bash
source /home/huihuangyue/.venvs/lisence-plate-forgery/bin/activate
source local_env.sh
python -m src.pipeline.batch_test data/raw/images --count 20 --output outputs
```

高召回工作点默认启用，也可显式声明：

```bash
export STICKER_AGENT_DECISION_PROFILE=high_recall
```

它以召回率不低于95%、查准率尽量不降且底线约70%为校准目标。是否达到目标只能由独立人工标注集的 `metrics.json` 判定，不能由处理成功数或模型自报置信度推断。

指定随机种子可以精确复现抽样：

```bash
source local_env.sh
python -m src.pipeline.batch_test data/raw/images --count 20 --seed 20260821 --output outputs
```

按文件名排序处理前 100 张，使用云端多轮智能体判断贴牌，并记录大模型 Token、费用和分阶段耗时：

```bash
source local_env.sh
python -m src.pipeline.batch_test data/raw/images \
  --selection first \
  --count 100 \
  --output outputs \
  --prefix pipeline_first100_agent \
  --sticker-method agent \
  --agent-max-calls-per-image 3 \
  --workers 4
```

批处理默认同时处理4张图片，并显示 `已完成/总数`、速度、预计剩余时间和最近完成样本状态。`batch_report.json` 的 `parallelism` 会记录配置值与实际工作槽数。可用 `--workers 1` 临时恢复串行；日志或无人值守运行不需要进度条时可加 `--no-progress`。每个并发槽使用独立智能体实例，调用轨迹、Token与费用按图片隔离。

程序不会自行读取 `local_env.sh`；必须由操作者显式加载。当前 `local_env.sh` 的模型为 `qwen3-vl-plus`、端点为阿里云 DashScope，因此默认按阿里云官方实时推理阶梯原价计费：单次输入不超过32K Token时输入1元、输出10元/百万 Token；32K–128K为1.5/15元；128K–256K为3/30元。价格核查日期为2026-08-21，来源为[阿里云 qwen3-vl-plus 模型信息](https://help.aliyun.com/zh/model-studio/qwen3-vl-plus)。默认估算不含免费额度、活动折扣、上下文缓存优惠等账单调整，实际费用以服务商账单为准。如端点、模型或实际单价改变，可用 `--input-price-per-million-cny` 和 `--output-price-per-million-cny` 覆盖为固定单价。

`--agent-max-calls-per-image` 控制每张车牌最多进行多少次云端调用：

- `2`：调查员加最终裁决；
- `3`：调查员、独立反证审查员和最终裁决；
- `4`：在3次流程基础上，发生支持/反对冲突时增加一次复查。

默认值为 `3`，即调查、独立反证和最终裁决各一次，不执行额外冲突复查。需要冲突复查时可显式设为 `4`。

`batch_report.json` 中的 `timing` 包含批次处理时间、图片发现与选择耗时、目录初始化耗时、逐图循环耗时、平均值、P50、P95、最大逐图耗时、吞吐量和各图像阶段累计耗时。`cost` 只统计大模型 API 调用次数、缓存命中、非缓存输入/输出 Token 和大模型费用，不计算本地机器、电费或折旧。

输出目录名包含本地时区时间戳。批次根目录含每张图片的独立六图文件夹、`batch_report.json`、`selection_manifest.csv` 和 `results/`。其中 `results/` 只包含每个成功样本的 `05_final_marked.jpg` 副本，文件名同时写明 `clear`、`suspicious` 或 `unassessable` 以及高风险字符槽。

注意：项目现有贴片筛查阈值没有真实贴片真值校准，不能把批处理成功率解释为贴牌识别准确率。

如果只运行不调用云端的诊断基线，必须显式指定：

```bash
python -m src.pipeline.batch_test data/raw/images --count 20 --output outputs --sticker-method local --workers 4
```

2026-08-21 的一次授权云端冒烟测试处理了排序第一张真实图片：实际4次调用，输入18,593 Token、输出7,353 Token，按上述官方原价估算为0.092123元；总耗时215.287秒，其中云端贴牌判断210.913秒。该样本输出为 `unassessable`，这只验证了接线、六图、Token与费用记录，不代表识别结论正确。
