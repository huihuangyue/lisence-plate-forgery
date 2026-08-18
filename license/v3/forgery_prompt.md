# 可能的车牌伪造特征

- 整个字符被贴纸覆盖：字符或其边缘可见边界可以判定为贴纸边界、翘边、气泡、褶皱、胶痕，或与相邻字符的颜色、纹理、反光、老化程度不一致。特别注意：若某个字符本体外侧被一个近矩形的局部底片区域包围（即该矩形区域围绕字符，而不是字符本身呈矩形），且其纹理、色差、边界或凹凸感与周围牌面明显不同，应优先判定为 `character_sticker`。在 `visual_evidence` 说明该包围字符的贴片区域差异，并用 `bbox` 覆盖整块贴片区域。
- 单个笔画被贴条覆盖：同一字符的笔画之间出现接缝、粗细不一致、颜色或反光不一致、笔画连接不自然。
- 涂改或补漆：笔画边缘毛糙、粗细不均、漆面堆积、局部色差或局部纹理与周围不同。
- 打磨后重刻：字符或周边存在划痕、发白、磨砂感、底层暴露、边缘残留旧字形，或局部质感不连续。
- 印刷件贴附：可见印刷网点、异常规则纹理、清晰度或噪声与周边不一致、裁切边缘，或字符缺少与相邻字符一致的立体感。
- 整牌伪造：各字符的字体结构、比例、笔画形状、颜色、材质、反光、边框或固封特征与同牌其他部分明显不一致。

# 任务与输出

【最高优先级：机器可解析输出协议】最终答复必须只包含一个合法 JSON 对象：第一个字符必须是 `{`，最后一个字符必须是 `}`。不得输出分析过程、解释、标题、Markdown 代码块、```、JSON 前后文字或任何额外字段。无论是否调用 `execute_code`，最终都必须遵守此协议。

输出必须简练：每个 `visual_evidence` 最多保留 1 条短句。`crop_check.visual_evidence` 不超过 80 个汉字或等长度字符；字符和异常区域的 `visual_evidence` 不超过 30 个汉字或等长度字符。`normal` 字符的 `visual_evidence` 必须为 `[]`，不输出任何理由；只有 `suspected` 或 `unreadable` 字符才必须提供 1 条实际可见的理由。每个字符只保留必要字段，不重复解释。

先检查裁剪图是否完整包含整块车牌。`complete` 的定义是：整块车牌及全部字符完整可见。不得切掉车牌边框、任一字符或字符的一部分。仅当 `crop_status` 为 `complete` 时，才逐一检查车牌上的每个字符（包括汉字、字母、数字及可见分隔符），判断该字符是否疑似被伪造；若疑似，指出最可能的伪造方式。

所有判断必须基于输入图像中实际可见的像素特征。可使用 `execute_code` 时，优先调用它对字符及其邻域进行放大、局部对比或像素分析，形成可复核的纹理、色差、边界、反光或局部形状证据；只读取图像尺寸不足以支持任何伪造或正常结论。不得预设车牌号码、字符坐标、字符状态或检测结论；若调用 `execute_code`，代码必须直接从输入图像提取或计算证据，不得使用固定坐标、固定字符或固定判断结果。无法从图像获得证据时，标记为 `unreadable` 或 `uncertain`，不得以高置信度判定 `normal` 或 `suspected`。

判断每个字符时，必须完整检查：字符及笔画边缘是否有贴纸边界、翘边、气泡、褶皱或胶痕；字符本体外侧是否被与牌面不一致的近矩形贴片区域包围；笔画连接、粗细和形状是否一致；颜色、纹理、反光和老化程度是否与相邻字符及车牌底色一致；是否有补漆、打磨、重刻、印刷网点或裁切边缘。只有这些检查均未发现可信异常时才可标为 `normal`；不得只凭其中一个方面判定。`normal` 不写理由；`suspected` 或 `unreadable` 必须在 `visual_evidence` 写入对应的实际可见特征。

若 `crop_status` 为 `incomplete` 或 `uncertain`，将 `characters` 设为 `[]`，将 `overall` 设为 `unreadable`，不得进行伪造或变造判断。只报告图片中可观察到的特征；无法辨认时标记为 `unreadable`，不得推断。

对每个 `suspected` 字符或连续可疑字符区域，在 `anomaly_regions` 中返回一个矩形框。坐标相对于输入的车牌裁剪图左上角，格式为 `[x, y, width, height]`，单位是像素；框应覆盖可疑字符或可疑笔画。没有可疑区域时返回 `[]`。

只输出以下 JSON，不要输出任何其他文字：

```json
{
  "crop_check": {
    "crop_status": "complete | incomplete | uncertain",
    "visual_evidence": ["车牌边框和全部字符是否完整可见的具体依据"]
  },
  "characters": [
    {
      "position": 1,
      "character": "京",
      "status": "normal | suspected | unreadable",
      "forgery_method": "none | character_sticker | stroke_sticker | overpaint | abrasion_reengraving | printed_overlay | font_or_manufacturing_anomaly | other | unknown",
      "visual_evidence": ["仅 suspected 或 unreadable 时填写；normal 必须为 []"],
      "confidence": "high | medium | low"
    }
  ],
  "anomaly_regions": [
    {
      "character_positions": [3],
      "bbox": [120, 18, 42, 76],
      "label": "overpaint",
      "visual_evidence": ["该区域在图片中可见的具体特征"]
    }
  ],
  "overall": "normal | suspected_forgery | unreadable"
}
```
