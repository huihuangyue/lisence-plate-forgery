# 本地四关键点模型来源

`yolov7-lite-t-plate-kpt.onnx` 来自公开仓库
[`we0091234/yolov7_plate`](https://github.com/we0091234/yolov7_plate) 的
`weights/yolov7-lite-t.pt`，固定到提交
`16f99565d1aa25afe0a0350c95e620b3ebf2aa8f`。

- 原始 PT SHA-256：`15ed98d11f39bc60aedcd4074e993cb6ee40cf51826c3d58fbb090b595b866a7`
- 当前 ONNX SHA-256：`b68007f9a6376b55a5ffa4f7388af73dc38c1c4672cb40ce7d88bfdde5f3eee5`
- 输入：`1×3×640×640` RGB float32
- 输出：`1×25200×19`，包含框、目标分数、单双层类别和四个关键点
- 导出环境：PyTorch `2.13.0+cpu`、ONNX `1.22.0`，最终合并为单文件 ONNX
- 参数量：上游导出日志报告 `250,922`

上游仓库在本次获取时没有提供 `LICENSE` 文件。因此该权重目前仅用于本项目内部研究验证；在发布、再分发或商用前，必须向上游确认授权，或用自有/明确许可的数据重新训练并替换它。
