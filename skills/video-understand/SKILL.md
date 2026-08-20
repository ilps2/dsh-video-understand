---
name: video-understand
description: 低成本视频理解。用户要求理解/总结/分析视频（B站链接、BV号、本地视频路径）时使用——"理解这个视频"、"视频讲了什么"、"总结一下这个 B站视频"、"这个视频适合谁看"、"XX角色穿的什么/拿的什么"等。通过 video_understand 工具调用，一条命令出摘要+问答+成本报告。支持问题驱动动态路由分层（L0 ASR/L1 轨迹/L2 视觉）、语义层复用、预算上限。
---

# Video Understand（低成本视频理解）

用 AVIS 信息层（ASR + 场景结构 + 运动对象轨迹 + YOLO 语义，约 1k token）代替逐帧像素喂 LLM，显著降低单视频 LLM 调用成本（成本取决于所选模型定价，见 `engine/understand_video.py` 价格常量）。

## 触发场景

- 用户给 B站链接 / BV 号 / 本地视频路径，要求"理解/总结/分析/讲一下"
- 用户问"这个视频讲了什么/适合谁看/有什么亮点"
- **定点问题**：用户问某角色在某时刻的细节（穿什么/拿什么/在哪/什么时候）——自动走问题驱动路由

## 用法（工具，非命令）

调用 `video_understand` 工具：

```json
{
  "target": "BV1GJ411x7h7",
  "questions": ["讲了什么", "XX角色穿的什么衣服"],
  "noDownload": false,
  "level": "l1",
  "budgetCny": 0.01
}
```

参数：
- `target`：B站 URL / BV 号 / 本地视频绝对路径
- `questions`：自定义问题（默认 3 问；定点问题请明确写"XX穿的什么/拿的什么/什么时候"）
- `noDownload`：本地文件置 true
- `level`：l0(纯ASR) / l1(信息层+轨迹) / l2(视觉)；默认按问题意图自动路由
- `budgetCny`：单次问题预算上限（元）。视觉成本估算超预算时自动降级（拦截 L2 用 L0/L1 回答）
- `window`：L2 指定时间窗（如 "10-30"），定点问题时可不传（自动定位）

## 输出

返回 JSON，重点看：
- `answers[]`：直接呈现给用户
- `routing`：路由决策可观测——`question_intent`（问题意图）、`initial_layer`/`effective_layer`（初始/实际层级）、`escalation_reason`（升级理由）、`video_profile`（类型/ASR覆盖/轨迹数）、`budget_blocked`（预算拦截）、`frames_sent`（抽帧数）
- `avis.visual_notes`：VLM 视觉描述（L2 时）
- `layer_cached`/`suggest_layer`：完整语义层状态

## 动态路由分层（v0.4）

```
视频类型（classify 免费信号：帧差/色彩/ASR覆盖）
  + 问题意图（transcript_fact/summary/clothing/color/motion_event/object_presence/temporal_event...）
  + 已有证据质量（asr/ocr/obj_tracks/visual_l2）
  + 隐私模式（fully_local 禁 VLM）+ 预算上限
  → 动态决定层级
```

| 问题 | 意图 → 层级 | 说明 |
|---|---|---|
| 视频讲了什么/提到什么 | summary/transcript_fact → L0 | 纯 ASR 0 帧 |
| XX什么时候动了/走到哪 | motion_event → L1(+L2) | obj_tracks 轨迹定位 → VLM 确认 |
| XX穿的什么/什么颜色/长什么样 | clothing/color/appearance → L2 | 必须视觉 |
| 复合（举起斧子时穿的什么）| 拆子任务 | 运动定位 + 视觉确认 |

**speech_dense 视频不等于纯 ASR**：视频类型只调默认路线，不限制能力——衣着问题照样升级 L2。

## 语义层复用（建层一次，追问秒答）

- `--layer`（或 Node 后续支持）：建完整语义层（base 全量转写 + CLIP 视觉索引，约 2-4min）到缓存
- 建层后任何问题命中层直接回答（`answered_from_layer=True`），文本问题 0 帧、视觉问题定位窗口抽帧
- 未建层时 `suggest_layer=True` 提示可建

## 定点问题三坑（务必遵守）

1. **解说词滞后于画面**：解说视频台词先出、画面滞后 5~30s。按台词定位后**向后扩展窗口**（系统已 +5s），抽帧仍无目标就逐秒扫描。
2. **ASR 误转别名**：人名多变体（夏娃→下瓦/旨女儿，密客→祕刻/密克）。系统有别名表兜底，但答案若异常可人工核对转写。
3. **证据不足如实回答**：所有镜头都看不到目标细节（如脚）时，如实说"画面未展示"，不要用原片知识补答案。

详细踩坑记录见 avis 技能 `references/narration-video-factoid-qa-pitfalls.md`。

## 前置依赖

引擎已内含在插件 `engine/` 目录。环境异常时引导用户运行自检：

```bash
npx dsh-video-understand doctor        # 逐项检测 + 给出修复命令
```

常见修复：

- Python 依赖缺失：`pip install -r engine/requirements.txt`（语义层另需 `requirements-layer.txt`）
- ffmpeg 缺失：macOS `brew install ffmpeg` / Ubuntu `sudo apt install ffmpeg`
- API key 缺失：`export LLM_API_KEY=sk-xxxxx`，或 `~/.dsh/.credentials.yaml` 的 `XIAOMI_API_KEY`
- 自定义 Python：`VIDEO_UNDERSTAND_PYTHON` 环境变量（系统 3.13 含 torch/CLIP 时优先）

## 注意事项

- 处理耗时 2~4 分钟（下载 + ASR + YOLO + LLM），调用后告知用户"正在分析"
- 纯 BGM 无语音视频：依赖 YOLO 对象标签，描述到"对象+运动"层面
- 定点问题建议明确写角色名+细节（"XX穿的什么鞋/拿的什么武器"），路由更准
- 同一视频多次提问命中 ASR 内容哈希缓存，第二次显著更快
