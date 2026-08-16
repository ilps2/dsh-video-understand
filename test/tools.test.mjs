// 自测：mock ctx 验证工具注册 + 参数 schema
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
console.log(`PASS: ${name} (inject: ${inject.join(',')}), params=${Object.keys(tool.parameters.properties).join(',')}`)
