# Changelog

## 0.5.2 (2026-08-20)
- fix(security): **key/URL 配对缺陷**——`DEEPSEEK_API_KEY` 此前会发往默认的 `api.xiaomimimo.com`（mimo-v2.5）。三个配置加载点（stages.py / understand_video.py / visual_level.py）统一为配对逻辑：LLM_API_KEY 显式覆盖照用其 URL；DEEPSEEK_API_KEY 未设 LLM_API_URL 时配对 DeepSeek 端点与 deepseek-chat；credentials 文件 fallback 保持各自配对。规则：一把 key 绝不发往不是为它选定的主机。（PR #1204 审阅意见）
- fix(entry): 对外条目删除不可验证的三个数字（token 压缩 99.95%+ / 单视频 ~0.006 元 / 重复理解 ~1/20 成本）——README 顶部、package.json description、SKILL.md description、dsh/index.js 工具描述改为"它做什么"；成本测量保留在 docs/blog（含方法）。（PR #1204 审阅意见）
- test: 新增 TestLLMConfigPairing 4 用例（env 三场景 + credentials fallback），pytest fixture 隔离环境防污染

## 0.5.1 (2026-08-20)
- feat: **动态问题路由分层 v0.4**（`engine/router.py`）——视频类型做先验 + 问题意图决定入口 + 证据质量决定升级 + 隐私/预算决定上限
  - `classify_question`：15 类问题意图（规则优先，零模型成本）
  - `choose`：最终路由（意图/证据/隐私/预算四输入），`speech_dense` 不再限制视觉能力
  - `evidence_score`：已有证据充足度评估（or 型源任一命中即够）
  - `split_question`：复杂问题拆解（运动定位 + 视觉确认子任务）
- feat: **L1 obj_tracks 作为 L2 注意力引导器**——轨迹活跃窗口 → L2 只抽窗口帧
- feat: **预算上限 `--budget-cny`**——视觉成本估算超预算自动拦截 L2 降级 L0/L1
- feat: **语义层中间件**——`--layer` 建完整层（base 转写 + CLIP），建层后任何问题直接查层回答（`answered_from_layer`），文本问题 0 帧
- feat: 关键词定位别名表（ASR 误转：夏娃→下瓦/旨女儿），解说词滞后画面时窗口尾段 +5s 扩展
- feat: unknown class 拆分 `motion_confidence`/`class_confidence`（轨迹可信、类别不确定）
- fix: `answer_from_layer` visual_notes 未写回 ctx.avis → 结果缺视觉描述；assemble_result avis 补 `visual_notes` 输出
- fix: Node OUTPUT_SCHEMA.video string→object（Python 0.5.0 新格式），render 兼容双格式，`--l2-window`→`--window`
- perf: transcribe 内容寻址缓存（同一视频二次提问跳过 ASR，45s→0s）
- test: 103 passed（router 29 + 定位辅助 + 别名 + 预算）

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
