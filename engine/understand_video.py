#!/usr/bin/env python3
"""
问题驱动视频理解（query-driven）：带着问题找答案
  B站链接/BV/本地 → 下载 → tiny ASR 全文索引 → LLM 定位答案窗口
  → 聚焦分析（局部 base 重转写 / L2 抽帧）→ 综合回答 → 质量自评

用法:
  python3 understand_video.py BV1GJ411x7h7
  python3 understand_video.py "链接" --ask "博主点了哪些荤菜"
  python3 understand_video.py /path/local.mp4 --no-download
成本: ~0.01 元/视频（信息层 1k-3k tok vs 原始逐帧 540万-3800万 tok）
"""
import argparse, glob, json, os, re, subprocess, sys, time, urllib.request
import os
import ssl
# 系统代理(Clash MITM)用自签名证书 → 指向 macOS 系统证书链（双保险）
if os.path.exists("/etc/ssl/cert.pem"):
    os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/cert.pem")
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile="/etc/ssl/cert.pem")
def _detect_python():
    """Auto-detect Python with dependencies: prefer framework 3.13 (has all deps)."""
    env = os.environ.get("VIDEO_UNDERSTAND_PYTHON")
    if env:
        return env
    candidates = [
        "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
        "python3",
    ]
    for p in candidates:
        if "/" in p and not os.path.exists(p):
            continue
        return p
    return "python3"

PY = _detect_python()
BILI = os.environ.get("BILI_DOWNLOAD_SCRIPT",
                      os.path.expanduser("~/.agents/skills/bilibili-downloader/scripts/bili_download.py"))
# 引擎自包含路径（相对于本文件）
_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
AVIS = os.path.join(_ENGINE_DIR, "avis.py")
ASR = os.path.join(_ENGINE_DIR, "livestream-highlight", "asr.py")
# 脚本直跑（python engine/understand_video.py）时把仓库根目录加入 sys.path，
# 使 engine 包内模块（cache_content 等）可以 from engine.xxx import
_REPO_ROOT = os.path.dirname(_ENGINE_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# LLM 配置：环境变量 → DSH credentials 文件 → 默认值
def _load_llm_config():
    """Load LLM API key: env var → DSH credentials file → empty."""
    key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    url = os.environ.get("LLM_API_URL", "https://api.xiaomimimo.com/v1/chat/completions")
    model = os.environ.get("LLM_MODEL", "mimo-v2.5")
    if key:
        return key, url, model
    # Fallback: DSH credentials file (~/.dsh/.credentials.yaml)
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

KEY, URL, MODEL = _load_llm_config()
DEFAULT_QUESTIONS = [
    "这段视频的核心内容是什么？用 3-5 句话概括。",
    "视频中有哪些关键细节或亮点？",
    "这段视频适合什么场景/人群使用？",
]

def run(cmd, timeout=1800):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def llm(messages, max_tokens=16384):
    if not KEY:
        raise SystemExit("错误: 未设置 LLM_API_KEY 环境变量。请运行:\n  export LLM_API_KEY='your-key-here'")
    body = {"model": MODEL, "messages": messages,
            "max_tokens": max_tokens, "temperature": 0.3}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        r = json.loads(resp.read())
    return r["choices"][0]["message"]["content"], r.get("usage", {})

def parse_json_obj(text):
    try:
        return json.loads(text[text.index("{"):text.rindex("}") + 1])
    except Exception:
        return {}

def fetch_title(url_or_bv):
    try:
        bv = re.search(r"BV[0-9A-Za-z]{10}", url_or_bv)
        if bv:
            req = urllib.request.Request(f"https://api.bilibili.com/x/web-interface/view?bvid={bv.group(0)}",
                headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                d = json.loads(resp.read())
            if d.get("code") == 0:
                return d["data"].get("title", "")
    except Exception:
        pass
    return ""

def download(url_or_bv, outdir):
    print(f"⬇️  下载 {url_or_bv} → 360p...", flush=True)
    # Full URLs (including bangumi): use yt-dlp directly
    if url_or_bv.strip().lower().startswith(("http://", "https://")):
        r = run([PY, "-m", "yt_dlp",
                 "-f", "worst[ext=mp4]/worst",
                 "--no-playlist",
                 "-o", os.path.join(outdir, "%(title)s.%(ext)s"),
                 url_or_bv])
    else:
        r = run([PY, BILI, "download", url_or_bv, "--quality", "360", "-o", outdir])
    if r.returncode != 0:
        raise SystemExit(f"下载失败: {r.stderr[-300:]}")
    mp4s = sorted(glob.glob(os.path.join(outdir, "*.mp4")), key=os.path.getmtime)
    if not mp4s:
        raise SystemExit("下载后未找到 mp4")
    return mp4s[-1], fetch_title(url_or_bv)

def encode(video_path, workdir, asr_model="tiny", sub="avis_tiny"):
    out = os.path.join(workdir, sub)
    os.makedirs(out, exist_ok=True)
    r = run([PY, AVIS, "encode", video_path, "-o", out, "--obj-tracks",
             "--asr-model", asr_model, "--skip-mv"])
    if r.returncode != 0:
        raise SystemExit(f"encode 失败: {r.stderr[-400:]}")
    stem = os.path.splitext(os.path.basename(video_path))[0] + "_avis"
    return os.path.join(out, stem)

def encode_cached(video_path, workdir, asr_model="tiny", use_clip=False):
    """带缓存的 encode：按视频内容哈希分层缓存（tiny/base/base+clip）。
    同一视频二次调用命中缓存，跳过重新提取（语义层复用）。
    
    使用 ContentAddressableCache 实现跨路径/跨设备缓存复用。
    """
    from engine.cache_content import ContentAddressableCache
    
    # 创建缓存实例
    cache = ContentAddressableCache(cache_root=os.path.join(workdir, "content_cache"))
    
    # 检查缓存是否命中
    cached_avis = cache.get(video_path)
    if cached_avis and os.path.exists(os.path.join(cached_avis, "avis.json")):
        tag = asr_model + ("_clip" if use_clip else "")
        print(f"♻️  语义层缓存命中: {tag}（内容哈希）", flush=True)
        cache.close()
        return cached_avis, cache.compute_hash(video_path)
    
    # 缓存未命中，执行 encode
    tag = asr_model + ("_clip" if use_clip else "")
    content_hash = cache.compute_hash(video_path)
    cache_dir = os.path.join(workdir, "avis_cache", content_hash, tag)
    stem = os.path.splitext(os.path.basename(video_path))[0] + "_avis"
    avis_dir = os.path.join(cache_dir, stem)
    
    os.makedirs(cache_dir, exist_ok=True)
    cmd = [PY, AVIS, "encode", video_path, "-o", cache_dir, "--obj-tracks",
           "--asr-model", asr_model, "--skip-mv"]
    if use_clip:
        cmd.append("--clip")
    r = run(cmd)
    if r.returncode != 0:
        cache.close()
        raise SystemExit(f"encode 失败: {r.stderr[-400:]}")
    
    # 写入缓存
    cache.put(video_path, avis_dir)
    cache.close()
    
    print(f"✅ {tag} 语义层已构建并缓存", flush=True)
    return avis_dir, content_hash

def layer_cache_status(video_path, workdir):
    """检查视频已有哪些语义层缓存。返回 (has_tiny, has_full)。"""
    import hashlib
    st = os.stat(video_path)
    h = hashlib.md5(f"{video_path}:{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:10]
    base = os.path.join(workdir, "avis_cache", h)
    has_tiny = os.path.exists(os.path.join(base, "tiny"))
    has_full = os.path.exists(os.path.join(base, "base_clip"))
    return has_tiny, has_full

def load_transcript(avis_dir):
    tr = os.path.join(avis_dir, "transcript.jsonl")
    segs = []
    if os.path.exists(tr):
        for line in open(tr, encoding="utf-8"):
            if line.strip():
                segs.append(json.loads(line))
    return segs

def transcript_text(avis_dir, limit=500):
    segs = load_transcript(avis_dir)
    return "\n".join(f"[{s.get('start', 0):.0f}s] {s.get('text', '').strip()}" for s in segs[:limit])

def locate(question, avis_dir, dur, title=""):
    """LLM 定位器：读 tiny 全文 → 候选窗口 + 缺口。返回 (windows, gap, reason)。"""
    full = transcript_text(avis_dir)
    body = {"model": MODEL,
            "messages": [{"role": "system",
                          "content": "你是视频内容定位器。给你视频语音转写全文（带时间戳）和用户问题，"
                          "找出最可能包含答案的 1-2 个时间段（秒），并判断缺口："
                          "asr=语音转写精度不足需局部重转写, visual=答案在画面里需抽帧看, none=转写已足够。"},
                         {"role": "user",
                          "content": f"视频标题: {title or '未知'}\n用户问题: {question}\n视频时长: {int(dur)}s\n\n语音转写全文:\n{full}\n\n"
                          "输出 JSON: {\"windows\": [\"30-90\"], \"gap\": \"asr|visual|none\", \"reason\": \"20字内说明\"}\n"
                          "windows 是 1-2 个时间段（秒，闭区间），gap 单选。"}],
            "max_tokens": 2048, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        r = json.loads(resp.read())
    d = parse_json_obj(r["choices"][0]["message"]["content"])
    wins = d.get("windows") or []
    gap = d.get("gap") or "asr"
    print(f"  [定位] 窗口={wins} 缺口={gap} 原因={d.get('reason', '')}", flush=True)
    return wins, gap, d.get("reason", "")

def asr_coverage(avis_dir):
    """ASR 覆盖度：返回转写总字数（纯视觉视频≈0）。"""
    segs = load_transcript(avis_dir)
    return sum(len(s.get("text", "")) for s in segs)

TIMING_WORDS = ("正片", "片头", "广告", "预告", "几点开始", "正片开始", "从.*分钟开始")
# 注意：宽泛的"什么时候/第几分钟"是剧情事件定位（如"主角什么时候变异"），
# 应走 ASR 文本定位找事件关键词，而不是视觉读标注。

def is_timing_question(question):
    """识别'正片起点/片头多长'类视频结构时间问题（需视觉读画面标注）。
    剧情事件时间（"什么时候变成X"）不算——走文本定位。"""
    return any(re.search(w, question) for w in TIMING_WORDS)

def clip_search(avis_dir, query, top_k=3):
    """调 avis.py search_avis 返回 [(timestamp, score)]。"""
    code = (f"import sys, pathlib; sys.path.insert(0, {os.path.dirname(AVIS)!r}); "
            f"from avis import search_avis; "
            f"print(repr(search_avis(pathlib.Path({str(avis_dir)!r}), {query!r}, {top_k})))")
    r = run([PY, "-c", code], timeout=180)
    for line in reversed(r.stdout.strip().splitlines()):
        if line.startswith("["):
            try:
                return json.loads(line)
            except Exception:
                continue
    print(f"  ⚠️ CLIP search 解析失败: {r.stdout[-150:]}")
    return []

def locate_visual(question, video_path, avis_dir, workdir, dur, title=""):
    """纯视觉视频定位：问题 → CLIP 视觉查询词 → 检索帧 → 窗口。返回 (windows, gap, reason)。"""
    print("  [纯视觉] ASR 覆盖≈0，启用 CLIP 视觉检索定位...", flush=True)
    # 1. 确保 clip 层存在（没有则构建）
    clip_path = os.path.join(avis_dir, "clip.npz")
    if not os.path.exists(clip_path):
        print("  ⚠️ 缺 CLIP 层，构建中（--clip，约 1-3min）...", flush=True)
        avis_dir2, _ = encode_cached(video_path, workdir, "tiny", use_clip=True)
        avis_dir = avis_dir2
    # 2. LLM 提取英文视觉查询词
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": "你是视觉检索词提取器。把用户问题转成 2-3 个英文视觉关键词"
                          "（CLIP 语义检索用，覆盖主要视觉元素/动作/场景）。只输出 JSON 数组。"},
                         {"role": "user", "content": f"视频标题: {title or '未知'}\n用户问题: {question}\n"
                          "输出: [\"keyword1\", \"keyword2\", \"keyword3\"]"}],
            "max_tokens": 1024, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        r = json.loads(resp.read())
    try:
        queries = parse_json_obj(r["choices"][0]["message"]["content"])
        if isinstance(queries, dict):
            queries = list(queries.values())
        queries = [q for q in (queries or []) if isinstance(q, str) and q.strip()][:3]
    except Exception:
        queries = []
    if not queries:
        queries = ["person", "action", "sports"]
    print(f"  [检索词] {queries}", flush=True)
    # 3. CLIP 检索
    hits = []
    for q in queries[:3]:
        for ts, sc in clip_search(avis_dir, q, 3):
            hits.append((ts, sc, q))
    if not hits:
        return [], "visual", "CLIP 无命中"
    # 4. 聚合窗口：命中帧 ±12s，相邻合并
    hits.sort()
    windows = []
    cur_a = cur_b = None
    for ts, _, _ in hits:
        a, b = max(0, ts - 12), ts + 12
        if cur_a is None or a > cur_b:
            windows.append([cur_a, cur_b]) if cur_a is not None else None
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_a is not None:
        windows.append([cur_a, cur_b])
    wins = [f"{int(a)}-{int(min(b, dur))}" for a, b in windows[:2]]
    reason = f"CLIP 命中 {len(hits)} 帧: {[(int(t), q) for t, s, q in hits[:4]]}"
    print(f"  [定位] 纯视觉窗口={wins} 缺口=visual 原因={reason}", flush=True)
    return wins, "visual", reason

def retranscribe_window(video_path, avis_dir, window, model="base"):
    """局部重转写：裁音频 → base 转写 → 时间戳偏移 → 替换 transcript 对应段。返回 (新段数, 耗时)。"""
    a, b = (int(x) for x in window.split("-"))
    t0 = time.time()
    wav = f"/tmp/avis_rt_{a}_{b}.wav"
    out = f"/tmp/avis_rt_{a}_{b}.jsonl"
    run(["ffmpeg", "-y", "-v", "error", "-ss", str(a), "-to", str(b), "-i", video_path,
         "-vn", "-ac", "1", "-ar", "16000", wav])
    r = run([PY, ASR, "--video", wav, "--out", out, "--model", model, "--device", "auto"], timeout=600)
    if r.returncode != 0:
        print(f"  ⚠️ 局部重转写失败: {r.stderr[-200:]}")
        return 0, 0
    new_segs = []
    for line in open(out, encoding="utf-8"):
        if line.strip():
            s = json.loads(line)
            s["start"] = round(s["start"] + a, 2)
            s["end"] = round(s["end"] + a, 2)
            new_segs.append(s)
    # 替换 transcript.jsonl 中窗口内的段
    tr = os.path.join(avis_dir, "transcript.jsonl")
    old_segs = [s for s in load_transcript(avis_dir) if not (a <= s.get("start", 0) <= b)]
    merged = old_segs + new_segs
    merged.sort(key=lambda s: s.get("start", 0))
    with open(tr, "w", encoding="utf-8") as f:
        for s in merged:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  ✅ 局部重转写 {window}s [{model}]：{len(new_segs)} 段（{time.time() - t0:.0f}s）", flush=True)
    return len(new_segs), time.time() - t0

def run_visual(level, video, avis_dir, window=None):
    vl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visual_level.py")
    cmd = [PY, vl, level, video, avis_dir, "--json"]
    if level == "l2" and window:
        cmd += ["--window", str(window), "--step", "5"]
    vr = run(cmd)
    try:
        vd = json.loads(vr.stdout[vr.stdout.index("{"):])
        desc = vd.get("description", "")
        pin, pout = vd.get("tokens", {}).get("in", 0), vd.get("tokens", {}).get("out", 0)
        cost = pin / 1e6 * 0.2 + pout / 1e6 * 0.7
        note = f"\n## 视觉补充（{window}s L2 抽帧）\n{desc}\n"
        print(f"  ✅ 视觉 {len(desc)} 字 | VLM {pin}+{pout} tok ≈ {cost:.4f} 元", flush=True)
        return note, cost
    except Exception as e:
        print(f"  ⚠️ 视觉级失败: {e}")
        return "", 0.0

def quality_check(question, answer):
    body = {"model": MODEL,
            "messages": [{"role": "system",
                          "content": "你是严格的质量评估器。回答含「无法确认/识别错误/信息不足/不确定/缺失」等表述时应给低分（<6）；"
                          "用户问具体内容（物品/数量/价格/名称）时，回答缺少具体名称、数量、价格即为不足。"},
                         {"role": "user",
                          "content": f"用户问题: {question}\n模型回答: {answer[:600]}\n\n"
                          "评估回答的信息充分度（0-10，<7 为不足）和主要缺口"
                          "（asr=语音转写不清/缺失, visual=缺画面细节, none=已充分, other=其他）。"
                          "严格输出 JSON: {\"score\": 0-10, \"gap\": \"asr|visual|none|other\"}"}],
            "max_tokens": 1024, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        r = json.loads(resp.read())
    d = parse_json_obj(r["choices"][0]["message"]["content"])
    return int(d.get("score", 0)), d.get("gap", "other")

# ── 成本核算（v4-flash 空闲价 + qwen3-vl-flash 近似）──
PRICE_IN_HIT = float(os.environ.get("AVIS_PRICE_IN_HIT", 0.05))
PRICE_IN_MISS = float(os.environ.get("AVIS_PRICE_IN", 1.5))
PRICE_OUT = float(os.environ.get("AVIS_PRICE_OUT", 4.5))

def calc_cost(usage):
    hit = usage.get("prompt_cache_hit_tokens", 0) or 0
    miss = usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit) or 0
    out = usage.get("completion_tokens", 0) or 0
    return hit / 1e6 * PRICE_IN_HIT + miss / 1e6 * PRICE_IN_MISS + out / 1e6 * PRICE_OUT, hit, miss

def main():
    ap = argparse.ArgumentParser(description="视频理解（统一 Pipeline 入口）")
    ap.add_argument("target", help="B站 URL / BV 号 / 本地视频路径")
    ap.add_argument("--ask", action="append", default=[], help="自定义问题（可多次）")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--workdir", default="/tmp/avis_qd", help="工作目录")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--max-rounds", type=int, default=3, help="最多轮次（定位聚焦循环）")
    ap.add_argument("--ask-layer", action="store_true",
                    help="回答后询问是否提取完整语义层（base全量+CLIP，后续问题秒答）")
    ap.add_argument("--layer", action="store_true", help="直接提取完整语义层（不询问）")
    # 新增参数
    ap.add_argument("--level", default="l0", choices=["l0", "l1", "l2"],
                    help="理解级别：l0(默认)/l1(+VLM视觉)/l2(+时间窗)")
    ap.add_argument("--window", help="L2 时间窗，如 10-30")
    ap.add_argument("--privacy-mode", default="remote_answer",
                    choices=["local_extract", "remote_answer", "remote_visual", "fully_local"],
                    help="隐私模式")
    ap.add_argument("--asr-backend", default="whisper",
                    choices=["whisper", "sensevoice"],
                    help="ASR 后端")
    ap.add_argument("--ocr", default="auto", choices=["auto", "on", "off"],
                    help="OCR 模式")
    ap.add_argument("--vlm", default="none",
                    choices=["none", "florence2", "qwen2vl", "remote"],
                    help="VLM 后端")
    args = ap.parse_args()

    # 检查是否使用新 Pipeline
    use_new_pipeline = args.level != "l0" or args.privacy_mode != "remote_answer" or args.asr_backend != "whisper"

    if use_new_pipeline:
        # 使用新 Pipeline
        from engine.context import create_context_from_request
        from engine.pipeline import run_pipeline
        from engine.result_schema import validate_result

        request = {
            "target": args.target,
            "questions": args.ask or DEFAULT_QUESTIONS,
            "noDownload": args.no_download,
            "level": args.level,
            "window": args.window,
            "privacy_mode": args.privacy_mode,
        }

        ctx = create_context_from_request(request)
        ctx.work_dir = args.workdir

        result = run_pipeline(ctx)

        # Schema 验证
        errors = validate_result(result)
        if errors:
            print(f"⚠️ Schema 验证警告: {errors}", file=sys.stderr)

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    t0 = time.time()
    os.makedirs(args.workdir, exist_ok=True)
    qs = args.ask or DEFAULT_QUESTIONS
    q_block = "\n\n".join(f"问题{i + 1}: {q}" for i, q in enumerate(qs))

    # 1. 获取视频
    if args.no_download:
        video_path = args.target
        title = os.path.splitext(os.path.basename(video_path))[0]
    else:
        video_path, title = download(args.target, args.workdir)
    print(f"🎬 {os.path.basename(video_path)}" + (f"（{title}）" if title else ""), flush=True)

    # 2. tiny ASR 全文索引（带缓存：同一视频二次提问跳过提取）
    print("🔍 tiny ASR 全文索引...", flush=True)
    avis_dir, _vh = encode_cached(video_path, args.workdir, "tiny")
    dur = float(run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=duration", "-of", "csv=p=0", video_path]).stdout.strip() or 0)

    # 3. 定位 → 聚焦计划队列 → 执行 → 自评
    visual_note, visual_cost, llm_cost = "", 0.0, 0.0
    total_hit = total_miss = total_out = 0
    rounds, upgrades = 0, []
    answers = []
    loc_gaps = []

    # 3a. 定位一次，生成聚焦计划
    #     ASR 覆盖极低（纯视觉视频）→ CLIP 视觉检索定位；否则文本定位
    #     '正片起点'类问题 → 强制视觉优先（读画面文字标注）
    if is_timing_question(qs[0]):
        wins, gap, reason = ["0-360"], "visual", "timing问题：需视觉读画面文字标注（正片在XX:XX）"
        print(f"  [定位] 时间类问题 → 视觉优先 窗口={wins} 缺口=visual（读标注）", flush=True)
    elif asr_coverage(avis_dir) < 60:
        wins, gap, reason = locate_visual(qs[0], video_path, avis_dir, args.workdir, dur, title)
    else:
        wins, gap, reason = locate(qs[0], avis_dir, dur, title)
    loc_gaps.append(gap)
    if not wins:
        wins = [f"0-{min(60, int(dur))}"]
    # 计划：按缺口优先排列（base 后补 visual，因为 base 可能仍不够；visual 后视情况）
    focus_plan = []
    is_pure_visual = asr_coverage(avis_dir) < 60
    if gap != "none":
        for w in wins:
            if gap == "asr":
                focus_plan.append(("base", w))
                focus_plan.append(("visual", w))   # base 后补视觉（菜单/实物常在画面）
            elif is_pure_visual:
                focus_plan.append(("visual", w))   # 纯视觉视频无音频，只排视觉
            else:
                focus_plan.append(("visual", w))
                focus_plan.append(("base", w))
    print(f"  [聚焦计划] {' → '.join(f'{k}@{w}' for k, w in focus_plan) or '无（转写已够）'}", flush=True)

    while focus_plan or rounds == 0:
        # 3b. 执行一个聚焦动作
        if focus_plan:
            kind, w = focus_plan.pop(0)
            if kind == "base":
                n, _ = retranscribe_window(video_path, avis_dir, w, "base")
                if n:
                    upgrades.append(f"base@{w}")
            else:
                note, vc = run_visual("l2", video_path, avis_dir, w)
                visual_note += note; visual_cost += vc
                upgrades.append(f"L2@{w}")

        # 3c. 回答
        p = run([PY, AVIS, "prompt", avis_dir]).stdout
        if title:
            p = f"# 视频标题：{title}\n（标题可能与内容不符，请结合内容判断）\n\n" + p
        sys_msg = ("你是视频内容分析助手。基于信息层（语音转写+场景结构+运动对象轨迹"
                   + ("+视觉帧描述" if visual_note else "") + "）回答。直接给答案，不要复述问题。"
                   "⭐重要规则：视觉描述中的画面文字标注（如'正片在XX:XX'、时间水印）是制作者给出的权威信息，"
                   "优先采信为答案，不要当作装饰水印忽略。")
        print(f"🤖 回答（第 {rounds + 1} 轮）...", flush=True)
        msg, usage = llm([{"role": "system", "content": sys_msg},
                          {"role": "user", "content": p + visual_note + "\n\n" + q_block +
                           "\n\n请按 '问题N: 回答' 格式逐条回答。"}], max_tokens=16384)
        c, h, m = calc_cost(usage)
        llm_cost += c; total_hit += h; total_miss += m; total_out += usage.get("completion_tokens", 0)
        answers = []
        for i, q in enumerate(qs, 1):
            mm = re.search(rf"问题{i}\s*[:：]\s*(.*?)(?=问题{i + 1}\s*[:：]|\Z)", msg, re.S)
            answers.append(mm.group(1).strip() if mm else f"(未能拆分) {msg[:200]}")
        for i, (q, a) in enumerate(zip(qs, answers), 1):
            print(f"\n❓ Q{i} {q}\n💬 {a}\n", flush=True)

        # 3d. 自评：逐题评估，取最差题；发现缺口必须生成补救计划，否则自评无意义
        rounds += 1
        if rounds >= args.max_rounds:
            break
        score, sgap = 10, "none"
        for q, a in zip(qs, answers):
            s_i, g_i = quality_check(q, a)
            if s_i < score:
                score, sgap = s_i, g_i
        print(f"  [自评] 充分度 {score}/10 | 缺口 {sgap} | 轮次 {rounds}/{args.max_rounds}", flush=True)
        if score >= 7:
            break
        loc_gaps.append(sgap)
        # 自评发现缺口 → 补一个聚焦动作（此前只记录缺口不行动，导致视觉问题答"信息层未提供"后直接退出）
        if sgap in ("visual", "asr"):
            done = set(upgrades)
            w = wins[0] if wins else f"0-{min(30, int(dur))}"
            kind = "visual" if sgap == "visual" else "base"
            tag = f"{'L2' if kind == 'visual' else 'base'}@{w}"
            if tag not in done:
                focus_plan.append((kind, w))
                print(f"  [补救] 自评缺口 {sgap} → 追加聚焦 {tag}", flush=True)

    # 4. 汇总
    elapsed = time.time() - t0
    total_cost = llm_cost + visual_cost
    tr_path = os.path.join(avis_dir, "transcript.jsonl")
    asr_tok = sum(len(json.loads(l).get("text", "")) // 2 for l in open(tr_path, encoding="utf-8") if l.strip()) if os.path.exists(tr_path) else 0
    n_tracks = sum(1 for l in open(os.path.join(avis_dir, "obj_tracks.jsonl"), encoding="utf-8") if l.strip()) if os.path.exists(os.path.join(avis_dir, "obj_tracks.jsonl")) else 0
    info_tok = asr_tok + n_tracks * 60 + 150
    orig_tok = int(dur * 30 * 1000)

    # 4b. 懒加载语义层：问用户要不要建完整层（base 全量 + CLIP 视觉索引）
    has_tiny, has_full = layer_cache_status(video_path, args.workdir)
    layer_built = has_full
    if args.layer or args.ask_layer:
        if not has_full:
            if args.layer or (not args.json):
                if args.layer:
                    resp = "y"
                else:
                    print("\n💡 建议：提取完整语义层（base 全量转写 + CLIP 视觉索引，约 2-4min）——"
                          "之后问这个视频任何问题都秒答、更准。")
                    resp = input("要提取吗？(y/N): ").strip().lower()
                if resp in ("y", "yes"):
                    t1 = time.time()
                    base_dir, _ = encode_cached(video_path, args.workdir, "base", use_clip=True)
                    layer_built = True
                    print(f"✅ 完整语义层已建（{time.time() - t1:.0f}s）→ {os.path.dirname(base_dir)}", flush=True)
                    upgrades.append("layer:base+clip")
            # json 模式不阻塞：标记 suggest_layer 由 agent 决定
        elif args.ask_layer:
            print("♻️  完整语义层已在缓存中，直接复用", flush=True)

    result = {
        "video": os.path.basename(video_path), "title": title, "duration_s": round(dur),
        "elapsed_s": round(elapsed), "info_tokens": info_tok, "orig_frame_tokens": orig_tok,
        "token_compression_pct": round(100 * (1 - info_tok / orig_tok), 2),
        "cost_cny": round(total_cost, 5), "visual_cost_cny": round(visual_cost, 5),
        "rounds": rounds, "upgrades": upgrades, "locator_gaps": loc_gaps,
        "layer_cached": has_full,
        "suggest_layer": (not has_full) and (not layer_built),
        "answers": [{"question": q, "answer": a} for q, a in zip(qs, answers)],
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print("=" * 70)
    print(f"✅ 完成 | {dur:.0f}s 视频 | {elapsed:.0f}s | 信息层 {info_tok} tok vs 逐帧 {orig_tok:,} tok | 压缩 {result['token_compression_pct']}%")
    print(f"💵 成本 ≈ {total_cost:.4f} 元（LLM {llm_cost:.4f} + 视觉 {visual_cost:.4f}）| 定位缺口: {'→'.join(loc_gaps)} | 升级: {'→'.join(upgrades) or '无'}")
    report = os.path.join(args.workdir, "report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"# 视频理解报告（问题驱动）\n\n- {os.path.basename(video_path)} ({title})\n- 时长 {dur:.0f}s | 信息层 {info_tok} tok | 压缩 {result['token_compression_pct']}%\n")
        f.write(f"- 成本 {total_cost:.4f} 元 | 定位缺口 {'→'.join(loc_gaps)} | 升级 {'→'.join(upgrades) or '无'}\n\n")
        for i, (q, a) in enumerate(zip(qs, answers), 1):
            f.write(f"## Q{i} {q}\n\n{a}\n\n")
    print(f"📄 报告: {report}")

if __name__ == "__main__":
    main()
