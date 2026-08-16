# Changelog

## 0.1.1 (unreleased)
- feat: `video_understand` tool supports L1/L2 visual levels (`level`/`window` params) — on-demand frame sampling + qwen3-vl-flash, +0.0005 CNY for frame-level details
- test: tool registration + schema self-test (mock ctx)
- ci: run self-test on push/PR

## 0.1.0 (2026-08-17)
- feat: `video_understand` tool — Bilibili link / BV / local video → AVIS info layer → summary + Q&A
- token compression 99.95%+ vs frame sampling; ~0.006 CNY/video; repeat understanding ~1/20 via prompt cache
