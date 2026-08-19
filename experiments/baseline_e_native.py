#!/usr/bin/env python3
# baseline_e_native.py — 对照组 E：MiMo 原生视频理解（零预处理，视频直发）。
#
# 这是"范式威胁"检验组：不抽帧、不转写、不建信息层，
# 把视频本体通过 video_url 直接发给 mimo-v2.5（官方多模态格式，
# 见 https://mimo.mi.com/models/zh-CN/mimo-v2.5）。
#
# 用法: python3 baseline_e_native.py <B站URL/BV/本地路径> [--fps 2] [--ask 问题]... [--json]
# 依赖: yt-dlp（URL 输入时取直链）；LLM 配置同引擎（LLM_API_KEY 或 ~/.dsh/.credentials.yaml）
import argparse, base64, json, subprocess, sys, urllib.request
from pathlib import Path

from baseline_c_subtitle import load_llm_conf  # 复用同一份 LLM 配置加载


def resolve_video_url(target):
    """返回 (video_url, note)。B站 → yt-dlp 直链；本地文件 → data URI（≤20MB）。"""
    p = Path(target)
    if p.exists():
        size_mb = p.stat().st_size / 1e6
        if size_mb > 20:
            raise SystemExit(f"本地视频 {size_mb:.0f}MB 超过内联上限（20MB），请改用 B站链接或自行托管后传 URL")
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f"data:video/mp4;base64,{b64}", f"本地文件内联 {size_mb:.1f}MB"
    page = target if target.startswith("http") else f"https://www.bilibili.com/video/{target}"
    # 取 480p 以下直链：控制 MiMo 服务端拉流体积，也对齐 D 组的分辨率条件
    r = subprocess.run(["yt-dlp", "-f", "best[height<=480]/best", "-g", page],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not r.stdout.strip():
        raise SystemExit(f"yt-dlp 取直链失败: {r.stderr[-300:]}")
    return r.stdout.strip().splitlines()[0], "B站 CDN 直链（480p）"


def ask_native(key, url, model, video_url, questions, fps):
    content = [
        {"type": "text", "text":
            "请观看这个视频并回答问题。基于视频实际内容作答，不确定的请明说，不要编造。\n\n"
            + "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))},
        {"type": "video_url", "video_url": {"url": video_url},
         "fps": fps, "media_resolution": "default"},
    ]
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": content}],
                       "max_tokens": 2000}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"], data.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--fps", type=int, default=2, help="MiMo 端抽帧率（官方参数，默认 2fps）")
    ap.add_argument("--ask", action="append", default=[])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    questions = args.ask or ["视频核心内容是什么", "有哪些亮点", "适合什么人看"]

    key, url, model = load_llm_conf()
    video_url, note = resolve_video_url(args.target)
    print(f"[E组] 视频来源: {note}；MiMo 抽帧 {args.fps}fps", file=sys.stderr)
    answer, usage = ask_native(key, url, model, video_url, questions, args.fps)
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0)
    # MiMo 价目：输入(缓存命中) ¥0.02/百万，输入(未命中) ¥1/百万，输出 ¥2/百万
    cached_tok = usage.get("prompt_cache_hit_tokens") or usage.get("cached_tokens") or 0
    cost = ((in_tok - cached_tok) * 1 + cached_tok * 0.02 + out_tok * 2) / 1_000_000
    result = {"group": "E", "video": args.target, "source": note, "fps": args.fps,
              "prompt_tokens": in_tok, "completion_tokens": out_tok,
              "cache_hit_tokens": cached_tok, "cost_cny": round(cost, 4), "answer": answer}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(answer)
        print(f"\n— 原生视频 | token {in_tok}+{out_tok} | 成本 ≈ {cost:.4f} 元", file=sys.stderr)


if __name__ == "__main__":
    main()
