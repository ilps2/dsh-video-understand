#!/usr/bin/env bash
# run.sh — 一键跑完对照实验 5 组 × 3 视频，原始数据存 experiments/results/
#
# 用法:
#   bash experiments/run.sh <解说BV或URL> <舞蹈BV或URL> <教程BV或URL>
#
# 组别:
#   A = 引擎默认问题（信息层主线）
#   B = 引擎 + 视觉细节问题（触发视觉聚焦，检验 L2 视觉证据能力）
#   C = B站字幕直取基线
#   D = 均匀抽帧直发 VLM 基线
#   E = MiMo 原生视频理解基线（video_url 直发）
#
# 说明:
# - 引擎 stdout 含进度行，JSON 在末尾——本脚本自动提取
# - 失败的组不中断整体，stderr/stdout 分别落盘
# - 打分别在这里做——原始数据收集与盲评分开
set -uo pipefail
cd "$(dirname "$0")/.."

if [ $# -ne 3 ]; then
  echo "用法: bash experiments/run.sh <解说BV> <舞蹈BV> <教程BV>" >&2
  exit 1
fi

# 优先用插件 .venv（依赖齐全），否则回退系统 python3
PY=".venv/bin/python3"; [ -x "$PY" ] || PY="python3"
mkdir -p experiments/results

# 从引擎 stdout（进度行 + 末尾 JSON）提取 JSON 并注入 elapsed_s
extract_json() { # <raw文件> <out文件> <elapsed>
  "$PY" - "$1" "$2" "$3" <<'EOF'
import json, sys
raw, out, elapsed = sys.argv[1], sys.argv[2], int(sys.argv[3])
text = open(raw).read()
start = text.find('{')
if start < 0:
    sys.exit(1)
d, _ = json.JSONDecoder().raw_decode(text, start)
d.setdefault("elapsed_s", elapsed)
json.dump(d, open(out, "w"), ensure_ascii=False, indent=2)
EOF
}

run_group() { # <视频别名> <target> <组名> <命令...>
  alias="$1"; target="$2"; group="$3"; shift 3
  out="experiments/results/${alias}_${group}.json"
  raw="${out}.raw"
  echo "▶ [${alias}/${group}] $*"
  t0=$SECONDS
  "$@" "$target" --json > "$raw" 2> "experiments/results/${alias}_${group}.stderr"
  rc=$?
  elapsed=$((SECONDS - t0))
  if [ "$rc" -eq 0 ] && extract_json "$raw" "$out" "$elapsed"; then
    rm -f "$raw"
    echo "  ✅ ${elapsed}s → $out"
  else
    echo "  ❌ 失败（rc=${rc}），原始输出: ${raw}，日志: experiments/results/${alias}_${group}.stderr"
  fi
}

# B 组：视觉细节问题（触发引擎视觉聚焦，对应实验设计的 L1/L2 视觉级检验）
VISUAL_Q=(--ask "视频中人物的穿着/外观有什么特点" --ask "画面的色调和场景氛围是怎样的" --ask "描述一个具体动作或画面细节")

# A 组追问（验证缓存带来的边际成本优势）
FOLLOWUP=(--ask "这个视频里最实用的一个信息是什么" --ask "有哪些容易被忽略的细节" --ask "如果要向朋友推荐，一句话怎么说")

ALIASES=(解说 舞蹈 教程)
TARGETS=("$1" "$2" "$3")

for idx in 0 1 2; do
  alias="${ALIASES[$idx]}"; target="${TARGETS[$idx]}"
  echo "===== 视频 $((idx+1))/3：${alias}（${target}）====="
  run_group "$alias" "$target" A_l0 "$PY" engine/understand_video.py
  run_group "$alias" "$target" B_visual "$PY" engine/understand_video.py "${VISUAL_Q[@]}"
  run_group "$alias" "$target" C_subtitle "$PY" experiments/baseline_c_subtitle.py
  run_group "$alias" "$target" D_frames "$PY" experiments/baseline_d_frames.py
  run_group "$alias" "$target" E_native "$PY" experiments/baseline_e_native.py
  run_group "$alias" "$target" A_followup "$PY" engine/understand_video.py "${FOLLOWUP[@]}"
done

echo ""
echo "===== 完成 ====="
echo "原始数据: experiments/results/ （$(ls experiments/results/*.json 2>/dev/null | wc -l | tr -d ' ') 个文件）"
echo "下一步: 把各组 answer(s) 去掉组标签后盲评打分，填入 experiments/benchmark.md 的记录表"
