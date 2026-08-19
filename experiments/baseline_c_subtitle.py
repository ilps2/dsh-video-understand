#!/usr/bin/env python3
# baseline_c_subtitle.py — 对照组 C：B站字幕直取 + 纯文本 LLM 问答。
#
# 用法: python3 baseline_c_subtitle.py <B站URL或BV> [--ask 问题]... [--json]
# 依赖: yt-dlp（pip install yt-dlp）；LLM 配置同引擎（LLM_API_KEY 或 ~/.dsh/.credentials.yaml）
import argparse, json, os, re, subprocess, sys, tempfile, urllib.request
from pathlib import Path


def load_llm_conf():
    key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    url = os.environ.get("LLM_API_URL", "https://api.xiaomimimo.com/v1/chat/completions")
    model = os.environ.get("LLM_MODEL", "mimo-v2.5")
    if not key:
        cred = Path.home() / ".dsh" / ".credentials.yaml"
        if cred.exists():
            for line in cred.read_text().splitlines():
                if line.startswith(("XIAOMI_API_KEY:", "DEEPSEEK_API_KEY:")):
                    key = line.split(":", 1)[1].strip().strip("'\"")
                    break
    if not key:
        raise SystemExit("错误: 未设置 LLM_API_KEY")
    return key, url, model


def fetch_subtitle(url_or_bv, workdir):
    """下载 CC/自动字幕，返回纯文本。无字幕返回 None。"""
    target = url_or_bv if url_or_bv.startswith("http") else f"https://www.bilibili.com/video/{url_or_bv}"
    out = str(Path(workdir) / "sub")
    for sub_langs in ("zh-CN,zh-Hans,zh", "zh-Hans,zh"):  # CC 字幕优先，自动字幕兜底
        subprocess.run(
            ["yt-dlp", "--write-subs" if "CN" in sub_langs else "--write-auto-subs",
             "--sub-langs", sub_langs, "--skip-download", "-o", out, target],
            capture_output=True, timeout=300)
        files = list(Path(workdir).glob("sub*.vtt")) + list(Path(workdir).glob("sub*.srt"))
        if files:
            text = files[0].read_text(errors="ignore")
            text = re.sub(r"<[^>]+>", "", text)  # 去 vtt 标签
            lines = [l.strip() for l in text.splitlines()
                     if l.strip() and "-->" not in l and not l.strip().isdigit()
                     and not l.startswith(("WEBVTT", "Kind:", "Language:"))]
            # 去重相邻重复行（自动字幕滚动重复）
            dedup = [l for i, l in enumerate(lines) if i == 0 or l != lines[i - 1]]
            return "\n".join(dedup)
    return None


def ask(key, url, model, subtitle, questions):
    prompt = (
        "以下是视频字幕文本，请基于它回答问题。若字幕中找不到依据，请明确说'字幕未覆盖'，不要编造。\n\n"
        f"【字幕】\n{subtitle[:12000]}\n\n【问题】\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    usage = data.get("usage", {})
    return data["choices"][0]["message"]["content"], usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--ask", action="append", default=[])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    questions = args.ask or ["视频核心内容是什么", "有哪些亮点", "适合什么人看"]

    key, url, model = load_llm_conf()
    with tempfile.TemporaryDirectory() as td:
        subtitle = fetch_subtitle(args.target, td)
    if not subtitle:
        result = {"group": "C", "video": args.target, "subtitle_available": False,
                  "note": "该视频无 CC/自动字幕，基线 C 不适用", "cost_cny": 0.0}
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["note"])
        return
    answer, usage = ask(key, url, model, subtitle, questions)
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0)
    # MiMo 单价按引擎 understand_video.py 的口径估算；如不同请改这里
    # MiMo 价目：输入(缓存命中) ¥0.02/百万，输入(未命中) ¥1/百万，输出 ¥2/百万
    cached_tok = usage.get("prompt_cache_hit_tokens") or usage.get("cached_tokens") or 0
    cost = ((in_tok - cached_tok) * 1 + cached_tok * 0.02 + out_tok * 2) / 1_000_000
    result = {"group": "C", "video": args.target, "subtitle_available": True,
              "subtitle_chars": len(subtitle), "prompt_tokens": in_tok,
              "completion_tokens": out_tok, "cache_hit_tokens": cached_tok, "cost_cny": round(cost, 4), "answer": answer}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(answer)
        print(f"\n— token {in_tok}+{out_tok} | 成本 ≈ {cost:.4f} 元", file=sys.stderr)


if __name__ == "__main__":
    main()
