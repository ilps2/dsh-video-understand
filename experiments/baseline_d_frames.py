#!/usr/bin/env python3
# baseline_d_frames.py — 对照组 D：无信息层，均匀抽帧直发 MiMo VLM。
#
# 用法: python3 baseline_d_frames.py <视频路径或B站URL> [--fps 1/30] [--ask 问题]... [--json]
# 依赖: ffmpeg、yt-dlp（URL 输入时）；LLM 配置同引擎（LLM_API_KEY 或 ~/.dsh/.credentials.yaml）
import argparse, base64, json, os, subprocess, sys, tempfile, urllib.request
from pathlib import Path

from baseline_c_subtitle import load_llm_conf  # 复用同一份 LLM 配置加载


def download(url_or_path, workdir):
    if Path(url_or_path).exists():
        return Path(url_or_path)
    target = url_or_path if url_or_path.startswith("http") else f"https://www.bilibili.com/video/{url_or_path}"
    out = Path(workdir) / "video.mp4"
    # B站 DASH 音视频分离：优先 ≤480p 视频+音频合并，无匹配回退默认最优
    r = subprocess.run(["yt-dlp", "-f", "bv*[height<=480]+ba/b[height<=480]/b",
                        "-o", str(out), target],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise SystemExit(f"yt-dlp 下载失败: {r.stderr[-300:]}")
    return out


def extract_frames(video, workdir, fps):
    pattern = str(Path(workdir) / "f_%04d.jpg")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video),
                    "-vf", f"fps={fps},scale=768:-1", "-q:v", "5", pattern],
                   check=True, timeout=600)
    frames = sorted(Path(workdir).glob("f_*.jpg"))
    return frames[:60]  # 上限 60 帧，控制 token 爆炸


def ask_vlm(key, url, model, frames, questions):
    content = [{"type": "text", "text":
        "以下是一个视频按时间顺序均匀抽取的帧。请基于这些帧回答问题；帧里看不到的信息请明说，不要编造。\n\n"
        + "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))}]
    for f in frames:
        b64 = base64.b64encode(f.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": content}],
                       "max_tokens": 2000}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"], data.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--fps", default="1/30", help="抽帧频率，默认每 30 秒 1 帧")
    ap.add_argument("--ask", action="append", default=[])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    questions = args.ask or ["视频核心内容是什么", "有哪些亮点", "适合什么人看"]

    key, url, model = load_llm_conf()
    with tempfile.TemporaryDirectory() as td:
        video = download(args.target, td)
        frames = extract_frames(video, td, args.fps)
        answer, usage = ask_vlm(key, url, model, frames, questions)
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0)
    # MiMo 价目：输入(缓存命中) ¥0.02/百万，输入(未命中) ¥1/百万，输出 ¥2/百万
    cached_tok = usage.get("prompt_cache_hit_tokens") or usage.get("cached_tokens") or 0
    cost = ((in_tok - cached_tok) * 1 + cached_tok * 0.02 + out_tok * 2) / 1_000_000
    result = {"group": "D", "video": args.target, "frames": len(frames),
              "prompt_tokens": in_tok, "completion_tokens": out_tok,
              "cache_hit_tokens": cached_tok, "cost_cny": round(cost, 4), "answer": answer}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(answer)
        print(f"\n— {len(frames)} 帧 | token {in_tok}+{out_tok} | 成本 ≈ {cost:.4f} 元", file=sys.stderr)


if __name__ == "__main__":
    main()
