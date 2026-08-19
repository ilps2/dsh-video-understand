#!/usr/bin/env node
// dsh-video-understand doctor — 一键环境自检。
//
// 逐项检测 Python / ffmpeg / yt-dlp / pip 依赖 / API key / 模型缓存，
// 每项失败时给出可直接复制的修复命令。核心项全部通过时退出码为 0。
//
// 用法：
//   npx dsh-video-understand doctor        # 人类可读报告
//   npx dsh-video-understand doctor --json # 机器可读（供 host/agent 排障）

import { spawnSync, execFileSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const PLUGIN_ROOT = path.join(__dirname, '..')
const IS_WIN = process.platform === 'win32'
const JSON_MODE = process.argv.includes('--json')

// ---------------------------------------------------------------- helpers

function which(cmd) {
  const r = spawnSync(IS_WIN ? 'where' : 'which', [cmd], { encoding: 'utf8' })
  return r.status === 0 ? r.stdout.split('\n')[0].trim() : null
}

function run(cmd, args) {
  const r = spawnSync(cmd, args, { encoding: 'utf8', timeout: 30_000 })
  return {
    ok: r.status === 0,
    stdout: (r.stdout || '').trim(),
    stderr: (r.stderr || '').trim(),
  }
}

/** 在指定 Python 中检查模块是否可导入，返回 { ok, version } */
function pyImport(python, mod) {
  const r = run(python, ['-c', `import ${mod}`])
  if (r.ok) return { ok: true }
  return { ok: false }
}

// ---------------------------------------------------------------- checks

const results = []
const CORE = 'core'
const OPTIONAL = 'optional'

function report(id, level, ok, label, fix = null, detail = '') {
  results.push({ id, level, ok, label, fix, detail })
  if (!JSON_MODE) {
    const mark = ok ? '✅' : level === CORE ? '❌' : '⚠️ '
    console.log(`${mark} ${label}${detail ? `  (${detail})` : ''}`)
    if (!ok && fix) console.log(`   修复: ${fix}`)
  }
}

// 1. Python：env → 插件 venv → python3
function detectPython() {
  const venvPy = IS_WIN
    ? path.join(PLUGIN_ROOT, '.venv', 'Scripts', 'python.exe')
    : path.join(PLUGIN_ROOT, '.venv', 'bin', 'python3')
  const candidates = []
  if (process.env.VIDEO_UNDERSTAND_PYTHON) candidates.push({ p: process.env.VIDEO_UNDERSTAND_PYTHON, why: 'VIDEO_UNDERSTAND_PYTHON' })
  candidates.push({ p: venvPy, why: '插件自带 .venv' })
  candidates.push({ p: 'python3', why: 'PATH' })
  if (IS_WIN) candidates.push({ p: 'python', why: 'PATH' })
  for (const c of candidates) {
    if (c.p.includes(path.sep) && !existsSync(c.p)) continue
    const r = run(c.p, ['-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'])
    if (r.ok) return { python: c.p, version: r.stdout, why: c.why }
  }
  return null
}

const py = detectPython()
if (!py) {
  report('python', CORE, false, '未找到可用的 Python',
    IS_WIN
      ? '安装 Python 3.10+：https://www.python.org/downloads/ （勾选 Add to PATH）'
      : 'macOS: brew install python@3.12 · Ubuntu: sudo apt install python3 python3-venv')
} else {
  const [maj, min] = py.version.split('.').map(Number)
  const ok = maj === 3 && min >= 9
  report('python', CORE, ok, `Python ${py.version}（${py.why}）${ok ? '' : ' — 版本过低'}`,
    ok ? null : '请使用 Python 3.9+，或设置 VIDEO_UNDERSTAND_PYTHON 指向新版解释器',
    py.python)
}

// 2. ffmpeg
const ffmpegPath = which('ffmpeg')
report('ffmpeg', CORE, !!ffmpegPath, ffmpegPath ? `ffmpeg 可用` : 'ffmpeg 未安装（ASR/抽帧必需）',
  IS_WIN ? 'winget install ffmpeg' : process.platform === 'darwin' ? 'brew install ffmpeg' : 'sudo apt install ffmpeg')

// 3. yt-dlp（B站下载）
const ytdlpPath = which('yt-dlp')
const ytdlpPy = py ? pyImport(py.python, 'yt_dlp') : { ok: false }
const ytdlpOk = !!ytdlpPath || ytdlpPy.ok
report('yt-dlp', CORE, ytdlpOk, ytdlpOk ? `yt-dlp 可用` : 'yt-dlp 未安装（B站视频下载必需）',
  py ? `${py.python} -m pip install yt-dlp` : 'pip install yt-dlp')

// 4. pip 依赖（在检测到的 Python 里）
const PIP_MIRROR = '-i https://pypi.tuna.tsinghua.edu.cn/simple'
const coreMods = [
  { mod: 'faster_whisper', pip: 'faster-whisper' },
  { mod: 'cv2', pip: 'opencv-python-headless' },
]
const layerMods = [
  { mod: 'torch', pip: 'torch' },
  { mod: 'transformers', pip: 'transformers' },
  { mod: 'ultralytics', pip: 'ultralytics' },
]
if (py) {
  const missingCore = []
  for (const m of coreMods) {
    const r = pyImport(py.python, m.mod)
    if (!r.ok) missingCore.push(m.pip)
  }
  report('pip-core', CORE, missingCore.length === 0,
    missingCore.length === 0 ? 'L0 核心依赖齐全（faster-whisper / opencv）' : `缺少核心依赖: ${missingCore.join(', ')}`,
    missingCore.length ? `${py.python} -m pip install ${missingCore.join(' ')} ${PIP_MIRROR}` : null)

  const missingLayer = layerMods.filter((m) => !pyImport(py.python, m.mod).ok).map((m) => m.pip)
  report('pip-layer', OPTIONAL, missingLayer.length === 0,
    missingLayer.length === 0 ? '语义层依赖齐全（CLIP / YOLO）' : `语义层依赖未装（可选，建层时才需要）: ${missingLayer.join(', ')}`,
    missingLayer.length ? `${py.python} -m pip install -r ${path.join(PLUGIN_ROOT, 'engine', 'requirements-layer.txt')} ${PIP_MIRROR}` : null)
}

// 5. API key
function findApiKey() {
  if (process.env.LLM_API_KEY || process.env.XIAOMI_API_KEY) return '环境变量'
  const cred = path.join(os.homedir(), '.dsh', '.credentials.yaml')
  if (existsSync(cred)) {
    const text = readFileSync(cred, 'utf8')
    if (/XIAOMI_API_KEY|DEEPSEEK_API_KEY|LLM_API_KEY/.test(text)) return '~/.dsh/.credentials.yaml'
  }
  return null
}
const keySrc = findApiKey()
report('api-key', CORE, !!keySrc, keySrc ? `LLM API key 已配置（${keySrc}）` : '未找到 LLM API key',
  'export LLM_API_KEY=sk-xxxxx  或写入 ~/.dsh/.credentials.yaml 的 XIAOMI_API_KEY')

// 6. 模型缓存（首次运行会下载，提示预热）
const hfCache = process.env.HF_HOME || path.join(os.homedir(), '.cache', 'huggingface')
const whisperCached = existsSync(hfCache) && (() => {
  try { return execFileSync('ls', [path.join(hfCache, 'hub')], { encoding: 'utf8' }).includes('faster-whisper') } catch { return false }
})()
report('models', OPTIONAL, whisperCached,
  whisperCached ? 'ASR 模型已缓存' : '模型未缓存 — 首次运行将自动下载（已内置 hf-mirror 国内镜像）',
  whisperCached ? null : `预热: ${py ? py.python : 'python3'} ${path.join(PLUGIN_ROOT, 'engine', 'livestream-highlight', 'asr.py')} --help`)

// ---------------------------------------------------------------- summary

const coreFailed = results.filter((r) => r.level === CORE && !r.ok)
if (JSON_MODE) {
  console.log(JSON.stringify({ ok: coreFailed.length === 0, python: py?.python || null, checks: results }, null, 2))
} else {
  console.log('')
  if (coreFailed.length === 0) {
    console.log('🎉 环境就绪，可以直接使用 video_understand。')
  } else {
    console.log(`❗ ${coreFailed.length} 项核心检查未通过，按上方"修复"命令逐项执行后重跑 doctor。`)
  }
}
process.exit(coreFailed.length === 0 ? 0 : 1)
