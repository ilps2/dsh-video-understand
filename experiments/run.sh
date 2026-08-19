#!/usr/bin/env bash
# run.sh — 一键跑完对照实验 4 组 × 3 视频，原始数据存 experiments/results/
#
# 用法:
#   bash experiments/run.sh <解说BV或URL> <舞蹈BV或URL> <教程BV或URL>
#
# 例:
#   bash experiments/run.sh BV1xx411c7mD BV1yy4y1y7Ab BV1zz5z1z8Cd
#
# 说明:
# - A/B 组直接调引擎 understand_video.py（与 dsh 工具同一条流水线）
# - C/D 组跑基线脚本；需要 ffmpeg、yt-dlp 与 LLM_API_KEY（或 ~/.dsh/.credentials.yaml）
# - 每组计时写入 results/<视频>_<组>.json 的 elapsed_s 字段
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

run_group() { # <视频别名> <target> <组名> <命令...>
  local alias="$1" target="$2" group="$3"; shift 3
  local out="experiments/results/${alias}_${group}.json"
  echo "▶ [${alias}/${group}] $*"
  local t0=$SECONDS
  "$@" "$target" --json > "$out" 2> "experiments/results/${alias}_${group}.stderr"
  local rc=$?
  local elapsed=$((SECONDS - t0))
  if [ $rc -eq 0 ]; then
    # 注入耗时字段（若 JSON 已有 elapsed_s 则保留原值）
    "$PY" - "$out" "$elapsed" <<'EOF'
import json, sys
p, elapsed = sys.argv[1], int(sys.argv[2])
d = json.load(open(p))
d.setdefault("elapsed_s", elapsed)
json.dump(d, open(p, "w"), ensure_ascii=False, indent=2)
EOF
    echo "  ✅ ${elapsed}s → $out"
  else
    echo "  ❌ 失败（rc=$rc），见 ${out%.json}.stderr"
  fi
}

FOLLOWUP=(--ask "这个视频里最实用的一个信息是什么" --ask "有哪些容易被忽略的细节" --ask "如果要向朋友推荐，一句话怎么说")

ALIASES=(解说 舞蹈 教程)
TARGETS=("$1" "$2" "$3")

for idx in 0 1 2; do
  alias="${ALIASES[$idx]}"; target="${TARGETS[$idx]}"
  echo "===== 视频 $((idx+1))/3：${alias}（${target}）====="
  run_group "$alias" "$target" A_l0 "$PY" engine/understand_video.py
  run_group "$alias" "$target" B_l1 "$PY" engine/understand_video.py --level l1
  run_group "$alias" "$target" C_subtitle "$PY" experiments/baseline_c_subtitle.py
  run_group "$alias" "$target" D_frames "$PY" experiments/baseline_d_frames.py
  # A 组追问（验证缓存带来的边际成本优势）
  echo "▶ [${alias}/A_followup] 追问 3 个新问题"
  t0=$SECONDS
  "$PY" engine/understand_video.py "$target" --json "${FOLLOWUP[@]}" \
    > "experiments/results/${alias}_A_followup.json" 2> "experiments/results/${alias}_A_followup.stderr"
  echo "  ✅ $((SECONDS - t0))s（成本见 JSON 内 cost_cny / prompt_cache_hit_tokens）"
done

echo ""
echo "===== 完成 ====="
echo "原始数据: experiments/results/ （$(ls experiments/results/*.json 2>/dev/null | wc -l | tr -d ' ') 个文件）"
echo "下一步: 把各组 answer(s) 去掉组标签后盲评打分，填入 experiments/benchmark.md 的记录表"
