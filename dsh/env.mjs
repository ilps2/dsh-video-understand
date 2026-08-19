// dsh/env.mjs — Python 环境检测与懒安装（host 与 doctor 共用）。
//
// 策略：
//   1. VIDEO_UNDERSTAND_PYTHON 显式指定 → 直接使用（校验核心依赖）
//   2. 插件本地 .venv 已存在且依赖齐全 → 直接使用
//   3. 否则懒安装：优先 uv（快 ~10 倍，可自管 Python），
//      回退 python3 -m venv + pip（清华镜像，失败回退官方源）
//
// 懒安装只在首次调用 video_understand 时触发，不拖慢插件加载。

import { spawn, spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const PLUGIN_ROOT = path.join(__dirname, '..')
const IS_WIN = process.platform === 'win32'

export const VENV_PY = IS_WIN
  ? path.join(PLUGIN_ROOT, '.venv', 'Scripts', 'python.exe')
  : path.join(PLUGIN_ROOT, '.venv', 'bin', 'python3')

const REQUIREMENTS = path.join(PLUGIN_ROOT, 'engine', 'requirements.txt')
const CORE_MODS = ['faster_whisper', 'cv2', 'yt_dlp']
const PIP_MIRROR = 'https://pypi.tuna.tsinghua.edu.cn/simple'
const INSTALL_TIMEOUT_MS = 15 * 60_000

function which(cmd) {
  const r = spawnSync(IS_WIN ? 'where' : 'which', [cmd], { encoding: 'utf8' })
  return r.status === 0 ? r.stdout.split('\n')[0].trim() : null
}

/** 在指定 Python 中校验核心依赖是否可导入 */
export function hasCoreDeps(python) {
  const r = spawnSync(python, ['-c', CORE_MODS.map((m) => `import ${m}`).join('; ')], {
    encoding: 'utf8',
    timeout: 60_000,
  })
  return r.status === 0
}

/** 仅检测（不安装）：env → .venv → PATH。返回 python 路径或 null。 */
export function detectPython() {
  if (process.env.VIDEO_UNDERSTAND_PYTHON) return process.env.VIDEO_UNDERSTAND_PYTHON
  if (existsSync(VENV_PY)) return VENV_PY
  return IS_WIN ? 'python' : 'python3'
}

function runStreamed(cmd, args, log) {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, { stdio: ['ignore', 'ignore', 'pipe'] })
    let tail = ''
    proc.stderr.on('data', (d) => {
      tail = (tail + d).slice(-2000)
      log?.(d.toString().trimEnd())
    })
    proc.on('error', reject)
    proc.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`${cmd} exited ${code}: ${tail.slice(-300)}`))))
  })
}

async function installWithUv(uv, log) {
  log?.('使用 uv 创建隔离环境（通常 <1 分钟）…')
  await runStreamed(uv, ['venv', path.join(PLUGIN_ROOT, '.venv')], log)
  // uv pip 走默认源已很快；保留镜像参数以降低国内失败率
  await runStreamed(uv, ['pip', 'install', '--python', VENV_PY, '-r', REQUIREMENTS, '--index-url', PIP_MIRROR], log)
}

async function installWithVenv(basePython, log) {
  log?.('创建插件本地虚拟环境 .venv（首次约 1-3 分钟，仅一次）…')
  await runStreamed(basePython, ['-m', 'venv', path.join(PLUGIN_ROOT, '.venv')], log)
  try {
    await runStreamed(VENV_PY, ['-m', 'pip', 'install', '-r', REQUIREMENTS, '-i', PIP_MIRROR], log)
  } catch {
    log?.('清华镜像安装失败，回退 PyPI 官方源…')
    await runStreamed(VENV_PY, ['-m', 'pip', 'install', '-r', REQUIREMENTS], log)
  }
}

let ensuring = null // 并发去重：多个调用共享同一次安装

/**
 * 确保 Python 环境就绪，返回可用解释器路径。
 * 已就绪立即返回；否则在插件目录懒安装（隔离 .venv，不污染系统 Python）。
 * @param {{ log?: (msg: string) => void }} opts
 */
export function ensurePythonEnv(opts = {}) {
  if (ensuring) return ensuring
  ensuring = (async () => {
    const { log } = opts

    // 1. 显式指定：尊重但不代装
    if (process.env.VIDEO_UNDERSTAND_PYTHON) {
      const py = process.env.VIDEO_UNDERSTAND_PYTHON
      if (!hasCoreDeps(py)) {
        throw new Error(
          `VIDEO_UNDERSTAND_PYTHON (${py}) 缺少核心依赖，请执行: ${py} -m pip install -r "${REQUIREMENTS}"`,
        )
      }
      return py
    }

    // 2. .venv 已就绪
    if (existsSync(VENV_PY) && hasCoreDeps(VENV_PY)) return VENV_PY

    // 3. 懒安装
    log?.('首次使用，正在自动安装 Python 依赖（仅一次，装完即忘）…')
    const uv = which('uv')
    if (uv) {
      await installWithUv(uv, log)
    } else {
      const base = which(IS_WIN ? 'python' : 'python3')
      if (!base) {
        throw new Error(
          '未找到 Python。请先安装（macOS: brew install python@3.12 · Ubuntu: sudo apt install python3 python3-venv · Windows: python.org），或安装 uv: curl -LsSf https://astral.sh/uv/install.sh | sh',
        )
      }
      await installWithVenv(base, log)
    }

    if (!hasCoreDeps(VENV_PY)) {
      throw new Error(`依赖安装后校验失败，请运行 npx dsh-video-understand doctor 查看详情`)
    }
    log?.('环境就绪 ✅')
    return VENV_PY
  })().finally(() => {
    // 成功后下次走 .venv 快速路径；失败后允许重试
    setTimeout(() => { ensuring = null }, 1000).unref?.()
  })
  return ensuring
}
