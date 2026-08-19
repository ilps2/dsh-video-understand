# Changelog

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
