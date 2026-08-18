# dsh-video-understand

[![npm version](https://img.shields.io/npm/v/dsh-video-understand)](https://www.npmjs.com/package/dsh-video-understand)
[![GitHub stars](https://img.shields.io/github/stars/ilps2/dsh-video-understand)](https://github.com/ilps2/dsh-video-understand)

低成本视频理解插件：给 dsh agent 注册 `video_understand` 工具——B站链接 / BV 号 / 本地视频 → AVIS 信息层 → 摘要+问答（token 压缩 99.95%+，成本约 0.006 元/视频）。

## 安装

```bash
# npm 安装
npm install dsh-video-understand

# 或 dsh plugin add dsh-video-understand

# 安装 Python 依赖（可选，仅 ASR 功能需要）
pip3 install faster-whisper
```

**引擎已内含**，无需额外克隆外部仓库。

## ⚠️ 数据流披露

| 级别 | 数据流向 | 说明 |
|---|---|---|
| **L0**（默认） | **完全本地** | ASR + 场景分类 + 运动检测，不上传任何数据 |
| **L1** | MiMo API | 视频帧发送至 MiMo 服务器进行 VLM 分析 |
| **L2** | MiMo API | 视频帧发送至 MiMo 服务器进行 VLM 分析 |

- L0 级别（默认）仅使用本地 ASR + 场景分类 + 运动检测，不涉及云服务
- L1/L2 级别会将视频帧（JPEG 编码）发送至 MiMo API 进行视觉理解
- 帧数据仅用于单次 VLM 推理，不会被存储或用于训练

## 环境变量

| 变量 | 必需 | 说明 |
|---|---|---|
| `LLM_API_KEY` | ✅ | LLM API 密钥（支持 MiMo / DeepSeek / 任何 OpenAI 兼容 API） |
| `LLM_API_URL` | ❌ | API endpoint（默认 `https://api.deepseek.com/v1/chat/completions`） |
| `LLM_MODEL` | ❌ | 模型名（默认 `deepseek-chat`，MiMo 设为 `mimo-v2.5`） |
| `VIDEO_UNDERSTAND_PYTHON` | ❌ | Python 解释器路径（默认 `python3`） |
| `BILI_DOWNLOAD_SCRIPT` | ❌ | bilibili-downloader 脚本路径（下载 B站视频用） |

> 💡 向后兼容：`DEEPSEEK_API_KEY` 仍可使用，但推荐迁移到 `LLM_API_KEY`。

## 工具

`video_understand(target, questions?, noDownload?, level?, window?)`

| 参数 | 类型 | 说明 |
|---|---|---|
| target | string | B站 URL / BV 号 / 本地视频绝对路径 |
| questions | string[] | 可选，自定义问题（默认 3 问） |
| noDownload | boolean | 本地文件置 true |
| level | string | `l0`(默认) / `l1`(+3-5帧VLM视觉摘要) / `l2`(+时间窗密集帧证据) |
| window | string | L2 时间窗，如 `10-30` 或秒数（auto=轨迹最活跃30s） |

返回 JSON：`video / duration_s / token_compression_pct / cost_cny / answers[]`。

## 结构

```
dsh-video-understand/
├── package.json
├── cordis.patch.yml
├── dsh/
│   └── index.js          # host 端：注册 video_understand 工具
├── engine/               # 自包含引擎（无需外部依赖）
│   ├── understand_video.py
│   ├── avis.py
│   ├── visual_level.py
│   ├── frame_prep.py
│   └── livestream-highlight/
│       └── asr.py
└── skills/
    └── video-understand/SKILL.md
```

## 分级实测（2026-08）

| 层级 | 内容 | 数据流向 | 成本 |
|---|---|---|---|
| L0 信息层 | ASR+场景+轨迹+YOLO → 摘要/问答 | 本地 | ~0.006 元 |
| L1 视觉级 | 3-5 帧 VLM → 颜色/姿态/衣着 | MiMo API | +0.0005 元 |
| L2 证据级 | 时间窗密集帧 → 时间线 | MiMo API | 按窗长 |

实测：电影解说 L1 补出「白色立领衬衫/神情凝重/暗色调诊室」（L0 完全给不出）；舞蹈 L2 逐帧「头部转 15-20°→45°、口型开口→闭合→微笑」。

## 设计背景

本插件的核心目标是**低成本视频理解**（单视频 0.006 元起）。

- **LLM 选型**：最初选用 DeepSeek v4 Flash 因其性价比最优；近期调价后已迁移至 MiMo，持续追踪成本效益比
- **视觉级（L1/L2）**：DeepSeek 暂不支持多模态输入，故 L1/L2 使用 MiMo API 进行视觉分析（此前曾使用 DashScope 作为过渡方案）。未来若 DeepSeek 支持多模态或出现更优的本地推理方案，可进一步优化成本

## 原理

引擎把视频压缩成**信息层**（ASR 转写 + 场景结构 + 运动对象轨迹 + YOLO 语义，约 1k token）再喂 LLM，对比逐帧像素（540 万 token）压缩 99.95%+；同一视频重复理解时信息层命中上下文缓存（93%），每次成本约为首次的 1/20。

## License

MIT
