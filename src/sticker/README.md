# 车牌整体贴片检测模块

本模块输入第一阶段生成的固定尺寸 `03_rectified.jpg`，输出字符槽候选、证据图、最终标记图、掩膜和结构化报告。当前只处理使用字符贴、矩形底片或磁贴整体覆盖字符，不处理增加或消除笔画。

## 两种运行方式

默认使用受控多轮云端复核；本地确定性证据仍作为候选生成和最终物理门控。显式使用本地基线：

```bash
source /home/huihuangyue/.venvs/lisence-plate-forgery/bin/activate
python -m src.sticker.run outputs/shape_local_first10/AADY1439/03_rectified.jpg \
  --method local --output outputs/sticker_local
```

受控多轮云端复核：

```bash
source /home/huihuangyue/.venvs/lisence-plate-forgery/bin/activate
source local_env.sh
python -m src.sticker.run outputs/shape_local_first10/AADY1439/03_rectified.jpg \
  --method agent --output outputs/sticker_agent
```

程序不会主动读取 `local_env.sh`。只有操作者显式加载后，`agent` 模式才会使用其中的 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OPENAI_MODEL`。每张图片通常进行调查、反证和裁决三次调用；两方对候选发生支持/反对冲突时增加一次复查。三个审查阶段统一使用 `tamper_support`、`normal_support`、`uncertain`，标签含义不随角色反转。

候选 `Cn` 固定绑定从左到右第 `n` 个字符槽 `Sn`。模型同时接收独立槽位映射图。v6 在调用模型前先完成确定性材料归属：把候选逐行与同高度左右背景配对，再用整张号牌同一高度的多数槽位建立颜色参考。最终门控同时检查边界几何和候选自身材料离群，不能只凭提示词或一条共享接缝决定相邻字符。

输入目录时，程序优先递归查找所有 `03_rectified.jpg`：

```bash
python -m src.sticker.run outputs/shape_local_first10 --method local --limit 10
```

## 输出

```text
<image_id>/
├── 01_input_rectified.jpg
├── 02_candidate_overlay.jpg
├── 03_final_marked.jpg
├── candidate_mask.png
├── report.json
├── trajectory.json          # 仅 agent 模式
└── evidence/
    ├── edge_map.png
    ├── line_overlay.jpg
    ├── color_residual.jpg
    ├── bright_dark.jpg
    ├── clahe_and_inverted.jpg
    ├── slot_map.jpg
    ├── vertical_profiles.jpg
    ├── plate_row_reference_residual.jpg
    ├── diagnostic_panel.jpg
    ├── ink_mask.png
    ├── screw_mask.png
    └── evidence_sheet.jpg
```

云端模型只能选择本地生成的 `C1..C8` 或 `C1..C7` 候选，最终坐标由确定性代码导出。API 响应按输入图、提示词和模型哈希缓存在输出根目录的 `.api_cache` 中，以支持断点重跑；可用 `--no-cache` 禁用。

`report.json` 的 `tampered_characters` 只包含最终通过物理门槛的贴片字符。例如第5位可读为 `6` 时输出：

```json
{
  "tampered_characters": [
    {
      "candidate_id": "C5",
      "slot": 5,
      "character": "6",
      "recognition_status": "cross_review_consensus",
      "bbox": [550, 18, 649, 244],
      "confidence_level": "high"
    }
  ]
}
```

`character` 是贴片后照片中可见的字符，不推断被遮住的原字符。无法可靠读出时保留 `null` 并给出 `recognition_status`，不会猜字。橙框另列在 `uncertain_characters`，不会混入 `tampered_characters`。

证据页综合使用原始图、CLAHE 局部对比度、反色 CLAHE、Scharr/Canny、多尺度顶帽与黑帽、局部 Lab 照明平面残差、边界两侧 CIEDE2000 色差和亮暗成对边缘。v6 不要求矩形完整闭合：两条与号牌轴平行、闭合度较低的残边也可进入候选，但必须得到材料归属支持。材料归属由两级对照组成：第一级逐行比较候选与同高度左右邻域；第二级逐行比较该字符槽与整牌多数槽位。第二级保留绿牌上浅下绿的天然分布以及随高度变化的照明，因此相邻字符共用一条贴片边时，只有真正偏离整牌背景的槽位获得高归属分。首末槽只允许使用朝向号牌内部的局部对照，并要求整牌参考确认，避免号牌外部和相邻贴片污染。

`diagnostic_panel.jpg` 是固定 `1984 × 870` 的六联辅助图，依次包含字符槽定位、固定 `0..12` 量程的整牌同高度 ΔE76 参考残差、固定 `0..400` 量程的 Sobel 梯度、固定 `0..35` 量程的 2px 高频残差、各槽 `Expected | Actual` 纵向 Lab 颜色条和背景直线段。热图使用 ΔE76 做像素定位；JSON 中参与门控的逐行中位数和四分位数使用 CIEDE2000。固定量程保证不同图片之间可比较，伪彩色负责定位，结构化数值与原图负责成立判断。

## 当前限制

- 本地融合阈值和模型结论尚未使用真实贴片掩膜校准；输出是研究筛查结果，不是正确概率或司法结论。
- 字符槽来自固定物理模板；agent 路线通过多轮观察返回可见字符，本地路线不带 OCR，因此本地结果的 `character` 为 `null`。候选定位和贴片成立条件均不依赖字符识别。
- 当前门槛优先降低误伤；只有一条可见贴片边、成对竖缝不孤立或没有独立材料差异时仍会拒判。需要真实标注集评估召回损失。
- 单张普通照片只能把厚度和翘边作为弱光影证据。
- 必须建立真实贴片与困难负样本标注集后，才能报告 precision、recall、字符级 F1 或区域 IoU。

判定协议 v7 严格限制拒判：只有号牌不可见、有效尺寸过小、严重模糊、大面积过曝/欠曝或提取失败时允许输出 `unassessable`。独立车牌 OCR 是可评估性的正向证据：OCR 返回完整、`readable=true` 且字符数与蓝牌7位/绿牌8位一致时，即使本地图像阈值失败也允许继续判断；OCR 失败不能反过来否定本来合格的图像。只要综合质量门为 `assessable`，最终必须输出 `suspicious` 或 `clear`；证据不足或冲突的槽位保留在 `uncertain_candidates`，不得把整牌升级为拒判。

## 人工标注与评估 harness

推荐直接使用同窗口网页标注器，而不是在图片查看器和命令行之间切换。输入必须是一次已经完成的流水线批次根目录；界面只显示没有预测框的 `03_rectified.jpg`，避免模型结果诱导人工标签：

```bash
# 第一步：对固定 selection 清单做一次云端车牌 OCR；结果可断点续跑
source ./local_env.sh
python -m src.sticker.prefill_plate_ocr outputs/<批次目录> \
  --annotation-output data/annotations/plate_tamper_30.csv \
  --count 30 --selection random --seed 20260824 --workers 4

# 第二步：启动人工纠错页面；默认自动读取同名 .ocr.json
python -m src.sticker.annotate_web outputs/<批次目录> \
  --output data/annotations/plate_tamper_30.csv \
  --count 30 --selection random --seed 20260824
```

页面默认监听服务器 `127.0.0.1:8765`。通过 SSH 的本地端口转发后，在 Windows 浏览器打开 `http://127.0.0.1:8765`。页面把云端预识别号牌填进可编辑文本框，并把字符显示到对应 `S1..S8` 按钮；识别错误时直接改文本即可。贴牌纠错仍只需点击一个或多个字符槽，`C` 表示正常，`U` 表示无法判断，Enter 保存并进入下一张。`.ocr.json` 只保存模型预识别，不计作人工完成；`.selection.json` 固定本次30张清单；CSV 每张原子保存并同时保留 `ocr_plate_text` 与人工确认后的 `plate_text`。

CSV 字段含义：

- `image_id`：与输出目录名相同；
- `decision`：`suspicious`、`clear` 或 `unassessable`；
- `suspicious_slots`：被贴字符槽，例如 `2` 或 `2;3`；
- `plate_text`：人工读取的完整可见号牌文本，不含圆点；
- `ocr_plate_text`：云端模型的原始预识别值；
- `plate_text_corrected`：人工是否修改了模型预识别值；
- `suspicious_characters`：字符级可读标签；重复字符保存为 `7#1`、`7#2`，而判定主键仍是唯一槽位；
- `slot_count`：蓝牌7、绿牌8，用于正确计算槽位级真阴性和 accuracy；
- `input_path`：原始整车图路径，供固定标注集重新检测；
- `notes`：可选人工说明。

只评估已经存在的批次报告：

```bash
python -m src.sticker.evaluate data/sticker_annotations.csv outputs/sticker_agent \
  --output outputs/sticker_agent/metrics.json
```

用同一份人工标注原图重新运行默认云端 agent，然后立即评估：

```bash
source ./local_env.sh
python -m src.pipeline.evaluate_annotated data/annotations/plate_tamper_30.csv \
  --images-root data/raw/images \
  --output outputs \
  --prefix pipeline-annotated30-agent \
  --sticker-method agent \
  --agent-max-calls-per-image 3 \
  --workers 4
```

评估器读取单图 `report.json` 和批次 `batch_report.json`，报告车牌级与字符槽级 precision、recall、F1、accuracy、specificity、coverage、拒判率、Wilson 95% 区间和逐图错例。橙框候选不计作阳性；所有准确率必须来自人工标签。
