# dsh-video-understand

[![npm version](https://img.shields.io/npm/v/dsh-video-understand)](https://www.npmjs.com/package/dsh-video-understand)
[![GitHub stars](https://img.shields.io/github/stars/ilps2/dsh-video-understand)](https://github.com/ilps2/dsh-video-understand)

低成本视频理解插件：给 dsh agent 注册 `video_understand` 工具——B站链接 / BV 号 / 本地视频 → AVIS 信息层 → 摘要+问答（token 压缩 99.95%+，成本约 0.006 元/视频）。

## 安装

```bash
# 0. npm 一键安装（已发布 npm: dsh-video-understand@0.1.0）
npm install dsh-video-understand
# 或 dsh plugin add dsh-video-understand（若 dsh 支持 npm 源）

# 1. 前置：live-clip 引擎（understand_video.py + avis.py）
git clone git@github.com:ilps2/live-clip.git ~/Desktop/live-clip-repo
cd ~/Desktop/live-clip-repo && bash install_models.sh   # 模型走国内镜像，30 秒

# 2. profile 挂载：package.json 加依赖 + bundles 加 "dsh-video-understand"
pnpm install && dsh web   # 重启生效
```

## 工具

`video_understand(target, questions?, noDownload?, level?, window?)`

| 参数 | 类型 | 说明 |
|---|---|---|
| target | string | B站 URL / BV 号 / 本地视频绝对路径 |
| questions | string[] | 可选，自定义问题（默认 3 问） |
| noDownload | boolean | 本地文件置 true |
| level | string | `l0`(默认) / `l1`(+3-5帧VLM视觉摘要,+0.0005元) / `l2`(+时间窗密集帧证据) |
| window | string | L2 时间窗，如 `10-30` 或秒数（auto=轨迹最活跃30s） |

返回 JSON：`video / duration_s / token_compression_pct / cost_cny / answers[]`。

## 配置

| 配置项 / 环境变量 | 默认 | 说明 |
|---|---|---|
| `scriptPath` / `VIDEO_UNDERSTAND_SCRIPT` | `~/Desktop/live-clip-repo/understand_video.py` | 引擎脚本路径 |
| `pythonPath` / `VIDEO_UNDERSTAND_PYTHON` | `python3` | Python 解释器 |
| `toolName` | `video_understand` | 工具名（避免宿主冲突可改） |

## 结构

```
dsh-video-understand/
├── package.json        # dsh.bundle manifest
├── cordis.patch.yml    # 挂载插件
├── dsh/index.js        # host 端：注册 video_understand 工具（spawn 引擎）
└── skills/
    └── video-understand/SKILL.md   # 供 agent 了解触发场景
```

## 原理

引擎把视频压缩成**信息层**（ASR 转写 + 场景结构 + 运动对象轨迹 + YOLO 语义，约 1k token）再喂 LLM，对比逐帧像素（540 万 token）压缩 99.95%+；同一视频重复理解时信息层命中上下文缓存（93%），每次成本约为首次的 1/20。详见 live-clip 的 [README](https://github.com/ilps2/live-clip)。

## License

MIT
