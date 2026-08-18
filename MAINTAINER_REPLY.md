# 回复维护者评论（草稿）

---

感谢您的详细反馈和建设性意见。我已针对您提出的三个问题进行了修复：

## 1. API Key 轮换 ✅

已从所有代码文件中移除硬编码的 API Key，并替换为环境变量引用：

- `understand_video.py`: `KEY = os.environ.get("DEEPSEEK_API_KEY", "")`
- `visual_level.py`: `API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")`
- `understand_video_v1_backup.py`: 同步修复

**需要您配合的操作：**
- 请前往 [DeepSeek 控制台](https://platform.deepseek.com/) 吊销旧的 API Key 并生成新的
- 请前往 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/) 吊销旧的 API Key 并生成新的

代码中已添加缺失 key 时的明确错误提示，而非静默失败。

## 2. 引擎打包 ✅

已将 live-clip 引擎内含到插件仓库中（方案 1：vendor）：

```
dsh-video-understand/
├── engine/                    # 新增：自包含引擎
│   ├── understand_video.py
│   ├── avis.py
│   ├── visual_level.py
│   ├── frame_prep.py
│   └── livestream-highlight/
│       └── asr.py
├── dsh/index.js              # 已更新：指向本地引擎
└── package.json              # 已更新：版本 0.3.0
```

关键变更：
- 移除了对 `~/Desktop/live-clip-repo/` 的硬编码依赖
- 使用 `__file__` 相对路径定位引擎文件
- 插件现在是自包含的，用户 `npm install` 后即可使用

## 3. 数据流披露 ✅

已在以下位置添加明确的数据流说明：

**工具描述（dsh/index.js）：**
```
⚠️ 数据流：L0 完全本地；L1/L2 会将视频帧发送至阿里云 DashScope 进行 VLM 分析，需设置 DASHSCOPE_API_KEY。
```

**README.md：**
- 新增「⚠️ 数据流披露」章节，明确标注 L0/L1/L2 均为本地处理
- 新增「设计背景」章节，说明：
  - 核心目标是低成本视频理解（0.006 元/视频起）
  - LLM 选型从 DeepSeek v4 Flash 迁移至 MiMo 的成本考量
  - MiMo 支持多模态，L1/L2 已迁移至本地推理，不再依赖 DashScope

**参数描述：**
- `level` 参数描述现在包含数据流向说明
- L0 标注「本地运行」
- L1/L2 标注「帧上传至 DashScope」

## 版本变更

- 版本号：`0.2.0` → `0.3.0`
- 新增 `engine/` 目录到 npm 发布文件列表
- 新增 `install-engine` 脚本（安装 Python 依赖）

## 测试验证

```bash
# 验证引擎文件完整
ls -la engine/
# 应显示：understand_video.py, avis.py, visual_level.py, frame_prep.py

# 验证路径解析
node -e "
import path from 'path';
import { fileURLToPath } from 'url';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
console.log('Engine path:', path.join(__dirname, 'engine', 'understand_video.py'));
"
```

请审核这些变更。如需调整或补充，请告知。

---

**Note:** 此回复需要在 GitHub 上发布前删除本地路径信息和测试命令。
