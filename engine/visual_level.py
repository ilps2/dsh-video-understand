#!/usr/bin/env python3
"""
AVIS L1/L2 视觉级理解：按需抽帧 + VLM（MiMo v2.5 多模态）

L1 视觉摘要：信息层（L0）盲区补充 —— 颜色/姿态/衣着/型号/文字
  从 AVIS 目录选 3-5 个代表时间点（轨迹活跃 + 场景边界）→ 抽帧 → 多图合并 VLM → 画面摘要
L2 时间窗证据：对指定片段密集抽帧 → 时间线证据链（"第X秒出现Y"）

用法:
  visual_level.py l1 <video> <avis_dir> [--frames 5] [--question "重点看什么"] [--json]
  visual_level.py l2 <video> <avis_dir> [--window 10-30] [--step 2] [--json]
  visual_level.py l2 <video> <avis_dir> --window auto   # 自动选轨迹最活跃 30s

⚠️ 数据流：L1/L2 级别会将视频帧（JPEG 编码）发送至 MiMo API 进行视觉理解。
   帧数据仅用于单次 VLM 推理，不会被存储或用于训练。
成本: MiMo v2.5 ≈ 253 tok/帧(图) → L1 单次 ≈ 0.001 元
"""
import argparse, base64, json, os, subprocess, sys, urllib.request
import ssl
# 系统代理(Clash MITM)用自签名证书 → 指向 macOS 系统证书链（双保险）
if os.path.exists("/etc/ssl/cert.pem"):
    os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/cert.pem")
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile="/etc/ssl/cert.pem")

# VLM 配置：环境变量 → DSH credentials 文件 → 默认值
def _load_vlm_config():
    """配对规则：一把 key 绝不能被送到不是为它选定的主机上。
      - LLM_API_KEY（显式覆盖）：key 是用户选的，他设的 URL/model 也照用
      - DEEPSEEK_API_KEY（且未设 LLM_API_URL）：配对 DeepSeek 端点与模型
      - 未设任何 key → 读 DSH credentials 文件（xiaomi/deepseek 各自配对）"""
    url = os.environ.get("LLM_API_URL", "")
    model = os.environ.get("VLM_MODEL", "")
    # 显式覆盖：用户同时指定了 key 与（可选的）url/model —— 一律照用
    if os.environ.get("LLM_API_KEY"):
        return (os.environ["LLM_API_KEY"],
                url or "https://api.xiaomimimo.com/v1/chat/completions",
                model or "mimo-v2.5")
    # DeepSeek key：未显式指定 LLM_API_URL 时，绝不发往默认的小米端点
    if os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("LLM_API_URL"):
        return (os.environ["DEEPSEEK_API_KEY"],
                "https://api.deepseek.com/v1/chat/completions",
                "deepseek-chat")
    if os.environ.get("DEEPSEEK_API_KEY"):
        # 显式 LLM_API_URL 存在：尊重用户设置（他们选了 DeepSeek key + 自定义端点）
        return (os.environ["DEEPSEEK_API_KEY"],
                url or "https://api.deepseek.com/v1/chat/completions",
                model or "deepseek-chat")
    # Fallback: DSH credentials file
    cred = os.path.expanduser("~/.dsh/.credentials.yaml")
    if os.path.exists(cred):
        try:
            keys = {}
            with open(cred) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("XIAOMI_API_KEY:"):
                        keys["xiaomi"] = line.split(":", 1)[1].strip()
                    elif line.startswith("DEEPSEEK_API_KEY:"):
                        keys["deepseek"] = line.split(":", 1)[1].strip()
            if "xiaomi" in keys:
                return keys["xiaomi"], "https://api.xiaomimimo.com/v1/chat/completions", "mimo-v2.5"
            if "deepseek" in keys:
                return keys["deepseek"], "https://api.deepseek.com/v1/chat/completions", "deepseek-chat"
        except Exception:
            pass
    return "", url, model

API_KEY, API_URL, MODEL = _load_vlm_config()

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

def probe_dur(video):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=duration", "-of", "csv=p=0", video])
    return float(r.stdout.strip() or 0)

def extract_frame(video, t, out):
    run(["ffmpeg", "-y", "-v", "error", "-ss", str(t), "-i", video,
         "-frames:v", "1", "-q:v", "3", out])

def vlm_frames(frame_paths, question):
    """多图合并一次 VLM 调用（⚠️ 帧将上传至 VLM 服务器进行视觉分析）。"""
    if not API_KEY:
        raise SystemExit("错误: 未设置 LLM_API_KEY 环境变量。请运行:\n  export LLM_API_KEY='your-key-here'")
    content = []
    for p in frame_paths:
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": question})
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "max_tokens": 2048, "temperature": 0.3}
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        r = json.loads(resp.read())
    msg = r["choices"][0]["message"]["content"]
    if isinstance(msg, list):
        msg = " ".join(msg)
    u = r.get("usage", {})
    return msg, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

def pick_times_l1(avis_dir, dur, n):
    """L1 代表时间点：轨迹活跃 + 场景边界 + 均匀分布。"""
    cands = []
    tr_path = os.path.join(avis_dir, "obj_tracks.jsonl")
    if os.path.exists(tr_path):
        for line in open(tr_path, encoding="utf-8"):
            if line.strip():
                o = json.loads(line)
                cands.append(o.get("appear_t", 0))
                cands.append((o.get("appear_t", 0) + o.get("disappear_t", 0)) / 2)
    sc_path = os.path.join(avis_dir, "scenes.csv")
    if os.path.exists(sc_path):
        rows = [l.strip() for l in open(sc_path, encoding="utf-8") if l.strip() and not l.startswith("sec")]
        labels = [r.split(",")[1].strip() for r in rows if "," in r]
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                cands.append(i)
    # 均匀补充
    for i in range(n):
        cands.append(dur * (i + 0.5) / n)
    # 去重 + 排序 + 取 n 个（均匀采样保持分布）
    cands = sorted(set(round(c, 1) for c in cands if 0 <= c < dur))
    if len(cands) <= n:
        return cands
    step = (len(cands) - 1) / (n - 1)
    return [cands[int(i * step)] for i in range(n)]

def pick_times_l2(dur, window, step):
    """L2 时间窗密集帧。"""
    if window == "auto":
        return []  # 自动模式由调用方决定
    if "-" in window:
        a, b = (float(x) for x in window.split("-"))
    else:
        a, b = 0, min(float(window), dur)
    b = min(b, dur)
    return [round(t, 1) for t in range(int(a), int(b) + 1, max(1, step))]

def main():
    ap = argparse.ArgumentParser(description="AVIS L1/L2 视觉级理解")
    sub = ap.add_subparsers(dest="cmd", required=True)

    l1 = sub.add_parser("l1")
    l1.add_argument("video"); l1.add_argument("avis_dir")
    l1.add_argument("--frames", type=int, default=5)
    l1.add_argument("--question", default="描述这些画面：有什么人/物、在做什么、什么颜色/姿态/衣着、场景如何？按帧顺序说明。\n\n⭐重要：优先读取画面中的文字/字幕/水印标注——它们可能是关键信息（如\"正片在XX:XX\"、价格、名称、警告语）。单独列出每帧出现的文字内容。")
    l1.add_argument("--json", action="store_true")

    l2 = sub.add_parser("l2")
    l2.add_argument("video"); l2.add_argument("avis_dir")
    l2.add_argument("--window", default="auto")
    l2.add_argument("--step", type=int, default=2)
    l2.add_argument("--question", default="按时间顺序描述这些帧：每帧发生了什么、对象/动作/变化。这是同一段视频按时间采样的帧。\n\n⭐重要：优先读取画面中的文字/字幕/水印标注——它们可能是关键信息（如\"正片在XX:XX\"、价格、名称、警告语）。单独列出每帧出现的文字内容，特别是时间格式的标注。")
    l2.add_argument("--json", action="store_true")

    args = ap.parse_args()
    dur = probe_dur(args.video)
    work = "/tmp/avis_visual"
    os.makedirs(work, exist_ok=True)

    if args.cmd == "l1":
        times = pick_times_l1(args.avis_dir, dur, args.frames)
    else:
        times = pick_times_l2(dur, args.window, args.step)
        if not times:  # auto：取轨迹最活跃窗口
            tr_path = os.path.join(args.avis_dir, "obj_tracks.jsonl")
            segs = {}
            if os.path.exists(tr_path):
                for line in open(tr_path, encoding="utf-8"):
                    if line.strip():
                        o = json.loads(line)
                        a, b = o.get("appear_t", 0), o.get("disappear_t", 0)
                        for s in range(int(a), min(int(b) + 1, int(dur))):
                            segs[s] = segs.get(s, 0) + 1
            if segs:
                # 找累计活跃最高的 30s 窗口
                keys = sorted(segs)
                best, best_t = 0, 0
                for start in range(0, int(dur) - 29):
                    w = sum(segs.get(s, 0) for s in range(start, start + 30))
                    if w > best:
                        best, best_t = w, start
                times = [round(t, 1) for t in range(best_t, min(best_t + 30, int(dur)) + 1, args.step)]
            else:
                times = [round(t, 1) for t in range(0, int(dur), args.step)]

    if not times:
        print("无可用帧时间点"); sys.exit(1)

    frames = []
    for i, t in enumerate(times):
        fp = os.path.join(work, f"f{i:02d}_{t:.0f}s.jpg")
        extract_frame(args.video, t, fp)
        frames.append(fp)

    print(f"抽 {len(frames)} 帧 @ {[f'{t:.0f}s' for t in times]} → VLM...", flush=True)
    desc, pin, pout = vlm_frames(frames, args.question)

    if args.json:
        print(json.dumps({"level": args.cmd, "frames": times, "tokens": {"in": pin, "out": pout},
                          "description": desc}, ensure_ascii=False, indent=2))
    else:
        print(f"\n[L{args.cmd} 视觉描述] ({pin} in / {pout} out tok)\n{desc}")

if __name__ == "__main__":
    main()
