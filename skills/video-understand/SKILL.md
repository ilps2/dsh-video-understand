---
name: video-understand
description: 低成本视频理解（token 压缩 99.95%+）。用户要求理解/总结/分析视频（B站链接、BV号、本地视频路径）时使用——"理解这个视频"、"视频讲了什么"、"总结一下这个 B站视频"、"这个视频适合谁看"等。通过 video_understand 工具调用，一条命令出摘要+3 个问答+成本报告。
---

# Video Understand（低成本视频理解）

用 AVIS 信息层（ASR + 场景结构 + 运动对象轨迹 + YOLO 语义，约 1k token）代替逐帧像素（约 540 万 token），token 压缩 **99.95%+**，单视频 LLM 成本约 0.006 元。

## 触发场景

- 用户给 B站链接 / BV 号 / 本地视频路径，要求"理解/总结/分析/讲一下"
- 用户问"这个视频讲了什么/适合谁看/有什么亮点"
- 需要批量理解多个视频（逐个调用）

## 用法（工具，非命令）

调用 `video_understand` 工具：

```json
{
  "target": "BV1GJ411x7h7",
  "questions": ["讲了什么", "适合谁看"],
  "noDownload": false
}
```

- `target`：B站 URL / BV 号 / 本地视频绝对路径
- `questions`：可选自定义问题（默认 3 问：核心内容/亮点/适合人群）
- `noDownload`：本地文件置 true

## 输出

返回 JSON：`video / duration_s / token_compression_pct / cost_cny / answers[]`。把 answers 的问答直接呈现给用户，并告知 token 压缩率与成本。

## 前置依赖

引擎已内含在插件 `engine/` 目录，无需克隆外部仓库。环境异常时引导用户运行自检：

```bash
npx dsh-video-understand doctor        # 逐项检测 + 给出修复命令
# 或 node <插件目录>/bin/doctor.mjs --json  # 机器可读，agent 可直接解析
```

常见修复（doctor 会按平台输出对应命令）：

- Python 依赖缺失：`pip install -r engine/requirements.txt`（语义层另需 `requirements-layer.txt`，体积大约 2GB，可选）
- ffmpeg 缺失：macOS `brew install ffmpeg` / Ubuntu `sudo apt install ffmpeg`
- API key 缺失：`export LLM_API_KEY=sk-xxxxx`，或写入 `~/.dsh/.credentials.yaml` 的 `XIAOMI_API_KEY`
- 自定义 Python 解释器：设置 `VIDEO_UNDERSTAND_PYTHON` 环境变量

## 注意事项

- 处理耗时 2~4 分钟（下载 + ASR + YOLO + LLM），调用后告知用户"正在分析"
- 纯 BGM 无语音视频：依赖 YOLO 对象标签，描述到"对象+运动"层面
- 需要精确型号/人名时建议 `--ask` 明确提问
