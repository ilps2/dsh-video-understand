// dsh-video-understand — host half.
//
// Registers a `video_understand` tool backed by the live-clip
// understand_video.py pipeline. A registered tool schema reaches the model
// on every request (no trigger gamble, unlike prompt-triggered skills), so
// the agent reliably knows it can ask about a video.
//
// Pipeline (spawned, token compression 99.95%+ vs frame sampling):
//   target(B站URL/BV/本地路径) → 下载(360p) → AVIS 信息层(MV/ASR/场景/YOLO轨迹)
//   → 融合 prompt → DeepSeek 摘要+问答 → JSON
//
// The engine lives in the live-clip repo (or any dir via config.scriptPath /
// VIDEO_UNDERSTAND_SCRIPT env). Default paths mirror this machine's layout;
// override in the plugin config or env when vendoring elsewhere.

import { spawn } from 'node:child_process'
import os from 'node:os'
import path from 'node:path'

export const name = 'video-understand'
export const inject = ['tools']

const DEFAULT_SCRIPT = path.join(os.homedir(), 'Desktop', 'live-clip-repo', 'understand_video.py')

const OUTPUT_SCHEMA = {
  type: 'object',
  additionalProperties: true,
  properties: {
    video: { type: 'string' },
    duration_s: { type: 'number' },
    elapsed_s: { type: 'number' },
    info_tokens: { type: 'number' },
    orig_frame_tokens: { type: 'number' },
    token_compression_pct: { type: 'number' },
    cost_cny: { type: 'number' },
    prompt_cache_hit_tokens: { type: 'number' },
    layer_cached: { type: 'boolean' },
    suggest_layer: { type: 'boolean' },
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

export function apply(ctx, config = {}) {
  const script = config.scriptPath || process.env.VIDEO_UNDERSTAND_SCRIPT || DEFAULT_SCRIPT
  const python = config.pythonPath || process.env.VIDEO_UNDERSTAND_PYTHON || 'python3'

  const tool = (toolName) => ({
    name: toolName,
    description:
      '低成本理解一个视频：输入 B站链接 / BV 号 / 本地视频路径，返回摘要+问答（token 压缩 99.95%+）。可选 level 参数升级视觉级（l1/l2）。返回中 suggest_layer=true 表示该视频尚未建完整语义层（base全量+CLIP，一次性2-4min，之后任何问题秒答）——若用户表示还会追问该视频其他问题，主动询问是否建层。' +
      '用户提到"理解这个视频/视频讲了什么/总结视频"或给出视频链接时使用。' +
      '可选 questions 数组自定义要问的问题（默认 3 问：核心内容/亮点/适合人群）。' +
      '需要 live-clip 仓库（understand_video.py）与模型依赖（见 README，bash install_models.sh 一键装）。',
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
          description: 'l0=信息层(默认,~0.006元) l1=+3-5帧VLM视觉摘要(+0.0005元) l2=+时间窗密集帧证据',
        },
        window: {
          type: 'string',
          description: 'L2 时间窗，如 10-30 或秒数（auto=轨迹最活跃30s）',
        },
      },
      required: ['target'],
    },
    output: {
      schema: OUTPUT_SCHEMA,
      render: (_args, value) => {
        const lines = [`🎬 ${value.video}（${value.duration_s}s）`]
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
      if (args.level && args.level !== 'l0') {
        cliArgs.push('--level', args.level)
        if (args.level === 'l2' && args.window) {
          cliArgs.push('--l2-window', args.window)
        }
      }
      for (const q of args.questions || []) {
        cliArgs.push('--ask', q)
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
