# 车牌平面化模块

本模块只完成流程第一步：从整车图片中定位最主要的一张单层号牌的最外侧金属边框四角，并输出固定物理尺度的平面图。它不判断号牌是否变造。

每个输入产生一个同名目录：

- `01_points.jpg`：原图上的 TL/TR/BR/BL 四点；
- `02_quad.jpg`：原图上的闭合四边形；
- `03_rectified.jpg`：透视校正后的固定尺寸号牌；
- `metadata.json`：四点、方法、置信度、正/逆单应矩阵、输出尺寸和名义占比。

## 两条实现路线

### 1. 本地小模型版（默认，推荐）

`local` 使用约 0.25M 参数的四关键点 ONNX 模型，整个推理与透视变换都在本机完成，不调用大模型或任何网络 API。模型直接输出 TL/TR/BR/BL，适合斜视车牌；HSV 只用于区分蓝牌与绿牌，不负责定位。

模型来源、提交版本、校验和与许可提醒见 [`models/README.md`](models/README.md)。

```bash
source /home/huihuangyue/.venvs/lisence-plate-forgery/bin/activate
python -m src.shape.run data/raw/images --method local --limit 10 --output outputs/shape_local_first10
```

### 2. 多模态大模型版（文字完整性兜底）

`llm` 采用两阶段定位：第一次在整车图中粗找号牌并把粗四边形透视展开，第二次在展开的大图上精确寻找金属牌四角并逆映射回原图。几何校验、定尺寸透视变换和写出仍在本地执行。

当前 `local_env.sh` 所配置模型可以识别牌型和粗位置。与本地版相同的前 10 张绿色新能源牌均完整保留全部车牌文字，因此满足云端兜底路线的当前验收要求；另有一张蓝牌在线对照也保留了完整文字。云端结果经常把黑色安装架或上下背景一并纳入输出，所以不能用其四点直接进行金属外框的像素级测量。前 10 张中只有 2 张通过相对本地结果的几何代理门槛，详细记录见 [`../../doc/shape_cloud_first10_evaluation.md`](../../doc/shape_cloud_first10_evaluation.md)。模型自报的 `0.95–0.98` 也未经过校准，不能解释为正确概率。

```bash
source /home/huihuangyue/.venvs/lisence-plate-forgery/bin/activate
source local_env.sh
python -m src.shape.run one.jpg --method llm --output outputs/shape_llm
```

程序不会自行读取或执行项目根目录的 `local_env.sh`，必须由操作者显式加载。云端路线只使用该文件提供的 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和模型变量；运行会把输入图及内部生成的粗展开图发送到该端点。

## 固定尺度与余量

统一采用 2 px/mm：

| 类型 | 物理尺寸 | 号牌主体像素 | 最终画布 | 名义主体占比 |
|---|---:|---:|---:|---:|
| 绿色新能源小型汽车牌 | 480×140 mm | 960×280 | 992×290 | 93.44% |
| 蓝色小型汽车牌 | 440×140 mm | 880×280 | 908×290 | 93.57% |

检测到的号牌外框四角被映射到画布内圈，画布四周保留少量原图背景。这样输出尺寸固定，完整号牌不会紧贴裁剪边，同时号牌名义面积保持在 90% 以上。这里的 93% 是几何模板约束；没有人工真值角点时，不能把它表述为逐图实测占比。

尺寸依据为 GA 36 相关政府采购材料和公安机关公开说明：普通小型汽车蓝牌 440×140 mm，新能源号牌 480×140 mm。

## 安装

项目专属环境：

```bash
source /home/huihuangyue/.venvs/lisence-plate-forgery/bin/activate
pip install -r src/shape/requirements.txt
```

也可从 `src` 布局运行：

```bash
PYTHONPATH=src python -m shape.run one.jpg --method local
```

处理文件夹中间一段可使用 `--start` 和 `--limit`：

```bash
python -m src.shape.run data/raw/images --method local --start 700 --limit 10
```

## 传统视觉诊断基线

`classical` 是不含任何学习权重的 HSV + 轮廓基线。它在前十张近距离绿牌上可用，但在远距离蓝牌上会把车漆反光或地面误当成候选，因此不作为正式默认路线：

```bash
python -m src.shape.run one.jpg --method classical
```

## 当前验证记录

- `data/raw/images` 按文件名排序的前 10 张：本地小模型 10/10 检出，逐张查看三类输出后，车牌与外框均完整；
- 额外蓝牌：近距离、远距离和明显斜视样本均已通过本地小模型复核；
- 大模型版已通过 `local_env.sh` 配置的唯一接口对前 10 张绿色新能源牌完成在线批测，调用与输出 10/10 成功，文字完整性 10/10；相对本地四点的平均 IoU 为 0.767、平均角点偏差为 0.200 个车牌短边，几何代理门槛仅通过 2/10。另测的一张蓝牌文字完整，但蓝牌尚未完成同规模批测。

以上是小规模人工冒烟测试，不等同于带真值的精度评估。后续应建立人工四角标注集，报告角点误差、四边形 IoU、完整边框保留率和拒判率。

`metadata.json` 中的 `confidence` 是检测器内部得分；本地模型分数和大模型自报分数均未用人工真值校准，不能直接解释为正确概率。

## 适用边界

当前只覆盖蓝色/绿色、单层小型汽车号牌，且默认返回最高置信度的一块号牌。原图已裁掉外框、严重遮挡、极小目标、严重失焦/运动模糊或多块同等显著车牌时，应返回失败或进入人工复核。黄色牌、白牌、双层牌和摩托车牌不在当前固定模板范围内。

## 参考工作

- [we0091234/yolov7_plate](https://github.com/we0091234/yolov7_plate)：框 + TL/TR/BR/BL 四关键点格式与轻量权重；
- [CCPD](https://github.com/detectRecog/CCPD)：文件名中包含车牌四顶点标注，并包含 CCPD-Green；
- [LPD-end-to-end](https://github.com/chensonglu/LPD-end-to-end)：用四角回归处理斜视车牌；
- [OpenCV perspective transforms](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)：四点单应变换接口。
