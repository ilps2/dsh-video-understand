// dsh-video-understand — host half.
//
// Registers a `video_understand` tool backed by the vendored
// understand_video.py pipeline. A registered tool schema reaches the model
// on every request (no trigger gamble, unlike prompt-triggered skills), so
// the agent reliably knows it can ask about a video.
//
// Pipeline (spawned; AVIS info layer instead of per-frame sampling):
//   target(B站URL/BV/本地路径) → 下载 → AVIS 信息层(MV/ASR/场景/YOLO轨迹)
//   → 融合 prompt → MiMo 摘要+问答 → JSON
//
// 数据流：L0 完全本地；L1/L2 使用 MiMo API 进行视觉分析（帧上传至 MiMo 服务器）。

import { spawn } from 'node:child_process'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

export const name = 'video-understand'
export const inject = ['tools']

// 引擎自包含：相对于本插件的 engine/ 目录
const __dirname = path.dirname(fileURLToPath(import.meta.url))
const DEFAULT_SCRIPT = path.join(__dirname, '..', 'engine', 'understand_video.py')

const OUTPUT_SCHEMA = {
  type: 'object',
  additionalProperties: true,
  properties: {
    // Python 0.5.0 起 video 为对象（VideoInfo）；旧版为字符串（basename）——
    // render 层兼容两种形态，这里按最新 schema 声明。
    video: {
      type: 'object',
      properties: {
        source: { type: 'string' },
        local_path: { type: 'string' },
        duration_s: { type: 'number' },
        width: { type: 'number' },
        height: { type: 'number' },
        fps: { type: 'number' },
        sha256: { type: 'string' },
      },
      additionalProperties: true,
    },
    duration_s: { type: 'number' },
    elapsed_s: { type: 'number' },
    info_tokens: { type: 'number' },
    orig_frame_tokens: { type: 'number' },
    token_compression_pct: { type: 'number' },
    cost_cny: { type: 'number' },
    prompt_cache_hit_tokens: { type: 'number' },
    layer_cached: { type: 'boolean' },
    suggest_layer: { type: 'boolean' },
    routing: {
      type: 'object',
      properties: {
        video_type: { type: 'string' },
        strategy: { type: 'string' },
        level: { type: 'string' },
        obj_tracks: { type: 'number' },
        visual_notes: { type: 'number' },
        cache_hit: { type: 'boolean' },
        question_intent: { type: 'string' },
        required_capability: { type: 'string' },
        initial_layer: { type: 'string' },
        effective_layer: { type: 'string' },
        upgrade_layer: { type: ['string', 'null'] },
        escalation_reason: { type: 'array', items: { type: 'string' } },
        evidence_score: { type: 'number' },
        evidence_sources: { type: 'array', items: { type: 'string' } },
        missing_evidence: { type: 'array', items: { type: 'string' } },
        frames_sent: { type: 'number' },
        video_profile: {
          type: 'object',
          properties: {
            asr_coverage: { type: 'number' },
            motion_score: { type: 'number' },
            static_ratio: { type: 'number' },
            ocr_text_count: { type: 'number' },
            track_count: { type: 'number' },
          },
          additionalProperties: true,
        },
        subtasks: { type: 'array', items: { type: 'object' } },
      },
      additionalProperties: true,
    },
    answers: {
      type: 'array',
      items: {
        type: 'object',
        properties: { question: { type: 'string' }, answer: { type: 'string' } },
      },
    },
  },
}

const TIMEOUT_MS = 15 * 60_000 // pipeline can take minutes (download + ASR + LLM)

function runScript(python, script, args, signal) {
  return new Promise((resolve, reject) => {
    const proc = spawn(python, [script, ...args], {
      env: { ...process.env },
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    proc.stdout.on('data', (d) => (stdout += d))
    proc.stderr.on('data', (d) => (stderr += d))
    proc.on('error', reject)
    proc.on('close', (code) => {
      if (code !== 0) reject(new Error(`video_understand exited ${code}: ${(stderr || stdout).trim().slice(0, 500)}`))
      else resolve(stdout.trim())
    })
    signal?.addEventListener('abort', () => proc.kill('SIGTERM'), { once: true })
  })
}

// Python 环境：检测 + 首次调用懒安装（共享实现见 dsh/env.mjs）
import { detectPython, ensurePythonEnv } from './env.mjs'

export function apply(ctx, config = {}) {
  const script = config.scriptPath || process.env.VIDEO_UNDERSTAND_SCRIPT || DEFAULT_SCRIPT
  const configuredPython = config.pythonPath

  const tool = (toolName) => ({
    name: toolName,
    description:
      '低成本理解一个视频：输入 B站链接 / BV 号 / 本地视频路径，返回摘要+问答（用 AVIS 信息层代替逐帧像素，LLM 调用仅需几千 token）。可选 level 参数升级视觉级（l1/l2）。返回中 suggest_layer=true 表示该视频尚未建完整语义层（base全量+CLIP，一次性2-4min，之后任何问题秒答）——若用户表示还会追问该视频其他问题，主动询问是否建层。' +
      '用户提到"理解这个视频/视频讲了什么/总结视频"或给出视频链接时使用。' +
      '可选 questions 数组自定义要问的问题（默认 3 问：核心内容/亮点/适合人群）。' +
      '可选 budgetCny 设定单次预算上限（元），超预算自动降级省成本。' +
      '数据流：L0 完全本地；L1/L2 使用 MiMo API 进行视觉分析（帧上传至 MiMo 服务器）。',
    parameters: {
      type: 'object',
      properties: {
        target: {
          type: 'string',
          description: 'B站链接、BV 号，或本地视频绝对路径',
        },
        questions: {
          type: 'array',
          items: { type: 'string' },
          description: '可选：要问的问题列表（默认 3 个预置问题）',
        },
        noDownload: {
          type: 'boolean',
          description: 'target 为本地文件时置 true，跳过下载',
        },
        level: {
          type: 'string',
          enum: ['l0', 'l1', 'l2'],
          description: 'l0=信息层(默认,全本地) l1=+3-5帧VLM视觉摘要(MiMo API) l2=+时间窗密集帧证据(MiMo API)',
        },
        window: {
          type: 'string',
          description: 'L2 时间窗，如 10-30 或秒数（auto=轨迹最活跃30s）',
        },
        budgetCny: {
          type: 'number',
          description: '单次问题预算上限（元）。视觉成本估算超预算时自动降级（拦截 L2，用 L0/L1 回答）',
        },
      },
      required: ['target'],
    },
    output: {
      schema: OUTPUT_SCHEMA,
      render: (_args, value) => {
        // 兼容 0.5.0 的对象 video（VideoInfo）与旧版字符串（basename）
        const video = value.video || {}
        const videoLabel =
          typeof video === 'string'
            ? video
            : video.local_path || video.source || 'unknown video'
        const lines = [`🎬 ${videoLabel}（${value.duration_s}s）`]
        for (const a of value.answers || []) {
          lines.push(`\n❓ ${a.question}\n${a.answer}`)
        }
        lines.push(`\n— token 压缩 ${value.token_compression_pct}% | 成本 ≈ ${value.cost_cny} 元 | 耗时 ${value.elapsed_s}s`)
        if (value.suggest_layer) {
          lines.push(`\n💡 该视频可建完整语义层（一次性 2-4min，之后追问秒答）——如用户还会问其他问题，可主动询问是否建层`)
        }
        return [{ type: 'text', text: lines.join('\n') }]
      },
    },
    timeoutMs: TIMEOUT_MS,
    isConcurrencySafe: () => false, // pipeline is CPU-heavy (ASR/MOG2/YOLO)
    presentCall: (args) => ({
      card: 'generic',
      title: toolName,
      kind: 'read',
      rawInput: args,
    }),
    async execute(args, exec) {
      if (typeof args?.target !== 'string' || args.target.trim() === '') {
        throw new Error(`${toolName} needs a non-empty "target" string.`)
      }
      const cliArgs = [args.target, '--json']
      if (args.noDownload) cliArgs.push('--no-download')
      if (typeof args.budgetCny === 'number') cliArgs.push('--budget-cny', String(args.budgetCny))
      if (args.level && args.level !== 'l0') {
        cliArgs.push('--level', args.level)
        if (args.level === 'l2' && args.window) {
          cliArgs.push('--window', args.window)
        }
      }
      for (const q of args.questions || []) {
        cliArgs.push('--ask', q)
      }
      // 首次调用时确保 Python 环境就绪（.venv 懒安装，仅一次）
      let python = configuredPython
      if (!python) {
        try {
          python = await ensurePythonEnv({ log: (m) => console.error(`[video-understand] ${m}`) })
        } catch (error) {
          throw new Error(`video_understand 环境准备失败: ${error.message}\n可运行 npx dsh-video-understand doctor 逐项排查。`)
        }
      }
      const stdout = await runScript(python, script, cliArgs, exec.signal)
      let parsed
      try {
        parsed = JSON.parse(stdout.slice(stdout.indexOf('{')))
      } catch {
        throw new Error(`video_understand produced no JSON: ${stdout.trim().slice(0, 300)}`)
      }
      return parsed
    },
  })

  try {
    ctx.tools.register(tool(config.toolName || 'video_understand'))
  } catch (error) {
    console.error(`[video-understand] tool registration skipped: ${error}`)
  }
}

// --self-test：不依赖 ctx/网络，仅验证工具 schema 与 render 对新旧两种
// video 格式（0.5.0 对象 / 旧版字符串）的兼容性。
// 运行：node dsh/index.js --self-test
const isMain =
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href

if (isMain && process.argv.includes('--self-test')) {
  const registered = []
  const ctx = { effect: () => {}, tools: { register: (t) => registered.push(t) } }
  apply(ctx, {})
  const tool = registered[0]
  if (!tool) throw new Error('no tool registered')

  const renderFor = (video) =>
    tool.output.render({}, { video, duration_s: 3, answers: [], token_compression_pct: 0, cost_cny: 0, elapsed_s: 0 })

  // 旧版字符串格式
  const legacy = renderFor('/tmp/old.mp4')
  if (!legacy[0].text.includes('/tmp/old.mp4')) throw new Error('legacy string video not rendered')

  // 0.5.0 对象格式（VideoInfo）
  const obj = {
    schema_version: '1',
    video: {
      source: 'local',
      local_path: '/tmp/test.mp4',
      duration_s: 3,
      width: 1280,
      height: 720,
      fps: 25,
      sha256: 'abc',
    },
    duration_s: 3,
    answers: [],
    warnings: [],
    errors: [],
  }
  if (typeof obj.video !== 'object' || obj.video === null) {
    throw new Error('video object shape rejected')
  }
  if (!obj.video.local_path) throw new Error('video.local_path missing')
  const rendered = renderFor(obj.video)
  if (!rendered[0].text.includes('/tmp/test.mp4')) throw new Error('object video not rendered')
  if (!rendered[0].text.includes('（3s）')) throw new Error('duration not rendered')

  // video 缺失时降级不崩溃
  const fallback = renderFor(undefined)
  if (!fallback[0].text.includes('unknown video')) throw new Error('missing video fallback broken')

  console.log(`PASS: ${name} --self-test (schema + render, video object/string compatible)`)
}
