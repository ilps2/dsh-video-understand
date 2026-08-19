# Changelog

## 0.5.0 (2026-08-20)
- feat: 首次调用自动创建插件内 `.venv` 并安装核心依赖（优先 uv，回退 venv+pip 清华镜像）——装完即用，零手动命令，不污染系统 Python
- feat: `npx dsh-video-understand doctor` 环境自检（Python/ffmpeg/yt-dlp/pip 依赖/API key/模型缓存），逐项给出修复命令，支持 `--fix` 一键修复与 `--json` 输出
- feat: 分层 requirements——L0 核心 ~300MB；语义层（torch/CLIP/YOLO ~2GB）可选，建层时再装
- fix: `detectPython()` 删除硬编码 macOS framework 路径，跨平台（env → 插件 .venv → 系统 python）
- fix: 补上 `package.json` 引用但缺失的 `engine/requirements.txt`（此前装完即报 ModuleNotFoundError）
- docs: SKILL.md 删除 live-clip 陈旧引用；新增 `experiments/` 对照实验（信息层 vs 字幕基线 vs 抽帧基线）

## 0.4.0 (2026-08-19)
- feat: 默认使用 MiMo v2.5（全模态推理模型），一个 API key 搞定
- feat: 自动从 `~/.dsh/.credentials.yaml` 读取 API key（无需手动配置环境变量）
- feat: 支持 B站 bangumi/番剧链接（直接走 yt-dlp，无需 BV 号）
- feat: Python 自动检测（优先 Framework 3.13 有依赖的版本）
- fix: MiMo v2.5 推理模型 max_tokens 调优（避免推理耗尽导致空内容）
- perf: LLM 调用 max_tokens 提升（locate 600 / keywords 400 / quality 300 / answer 1200）

## 0.1.1 (2026-08-17)
- feat: `video_understand` tool supports L1/L2 visual levels (`level`/`window` params) — on-demand frame sampling + qwen3-vl-flash, +0.0005 CNY for frame-level details
- test: tool registration + schema self-test (mock ctx)
- ci: run self-test on push/PR

## 0.1.0 (2026-08-17)
- feat: `video_understand` tool — Bilibili link / BV / local video → AVIS info layer → summary + Q&A
- token compression 99.95%+ vs frame sampling; ~0.006 CNY/video; repeat understanding ~1/20 via prompt cache
