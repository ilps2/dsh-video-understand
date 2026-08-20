// 自测：mock ctx 验证工具注册 + 参数 schema + render 对新旧 video 格式的兼容
import { apply, name, inject } from '../dsh/index.js'

const registered = []
const ctx = { effect: () => {}, tools: { register: (t) => registered.push(t) } }
apply(ctx, {})
const tool = registered[0]
if (!tool) throw new Error('no tool registered')
if (tool.name !== 'video_understand') throw new Error(`bad name: ${tool.name}`)
for (const k of ['target', 'questions', 'noDownload', 'level', 'window']) {
  if (!(k in tool.parameters.properties)) throw new Error(`missing param ${k}`)
}
if (typeof tool.execute !== 'function') throw new Error('execute missing')

// --- 回归测试：Python 0.5.0 起 video 为对象（VideoInfo），旧版为字符串 ---
if (!tool.output || !tool.output.render) throw new Error('output.render missing')
const render = (video) =>
  tool.output.render({}, { video, duration_s: 3, answers: [], token_compression_pct: 0, cost_cny: 0, elapsed_s: 0 })

// 1) 新结果格式（对象）
const value = {
  schema_version: '1',
  video: {
    source: 'local',
    local_path: '/tmp/test.mp4',
    duration_s: 3,
    width: 1280,
    height: 720,
    fps: 25,
  },
  duration_s: 3,
  answers: [],
  warnings: [],
  errors: [],
}

if (typeof value.video !== 'object' || value.video === null) {
  throw new Error('video object shape rejected')
}
if (!value.video.local_path) throw new Error('video.local_path missing')

const objRendered = render(value.video)[0].text
if (!objRendered.includes('/tmp/test.mp4')) throw new Error('object video label not rendered')
if (!objRendered.includes('（3s）')) throw new Error('object duration not rendered')

// 2) 旧版字符串格式仍兼容
const legacyRendered = render('/tmp/legacy.mp4')[0].text
if (!legacyRendered.includes('/tmp/legacy.mp4')) throw new Error('legacy string video not rendered')

// 3) video 缺失时降级不崩溃
const fallbackRendered = render(undefined)[0].text
if (!fallbackRendered.includes('unknown video')) throw new Error('missing video fallback broken')

console.log(`PASS: ${name} (inject: ${inject.join(',')}), params=${Object.keys(tool.parameters.properties).join(',')}`)
console.log('PASS: render compatible with video object (0.5.0) + legacy string')
