#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""asr.py — 视频音频转录（faster-whisper，支持 CPU/GPU）。

用法:
    python asr.py --video results/live.mp4 --out results/transcript.jsonl \
        --model small --lang zh --device auto

输出 JSONL 每行（句级）:
    {"start": 12.3, "end": 15.1, "text": "...", "words": [[w, ws, we], ...]}

设备选择:
    - auto: 自动检测 (cuda > mps > cpu)
    - cuda: NVIDIA GPU
    - mps: Apple Silicon GPU
    - cpu: CPU
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def extract_audio(video: Path, wav: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[asr] ffmpeg 未找到", file=sys.stderr)
        return False
    cmd = [ffmpeg, "-y", "-i", str(video), "-vn",
           "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not wav.exists():
        print(f"[asr] 音频提取失败: {r.stderr.decode()[-300:]}", file=sys.stderr)
        return False
    return True


def get_auto_device():
    """自动检测最佳可用设备。faster-whisper 不支持 MPS，回退 CPU。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # MPS not supported by faster-whisper 1.x — fall through to cpu
    except ImportError:
        pass
    return "cpu"


def transcribe(wav: Path, model_size: str, lang: str, device: str = "cpu"):
    """返回 segments 列表；模型下载失败抛异常由调用方处理。"""
    from faster_whisper import WhisperModel
    compute_type = "float16" if device != "cpu" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segs, _info = model.transcribe(str(wav), language=lang,
                                   word_timestamps=True, vad_filter=True)
    out = []
    for s in segs:
        words = [[w.word, round(w.start, 2), round(w.end, 2)]
                 for w in (s.words or [])]
        out.append({"start": round(s.start, 2), "end": round(s.end, 2),
                    "text": s.text.strip(), "words": words})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="results/transcript.jsonl")
    ap.add_argument("--model", default="small", help="tiny/base/small/medium")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--device", default="auto",
                    help="auto/cpu/cuda/mps (auto=自动检测)")
    args = ap.parse_args()

    # 解析设备
    device = args.device
    if device == "auto":
        device = get_auto_device()
        print(f"[asr] Auto-selected device: {device}")

    video = Path(args.video)
    if not video.exists():
        print(f"[asr] 视频不存在: {video}", file=sys.stderr)
        sys.exit(2)
    wav = video.with_suffix(".16k.wav")
    if not extract_audio(video, wav):
        sys.exit(2)
    try:
        segs = transcribe(wav, args.model, args.lang, device=device)
    except Exception as e:
        print(f"[asr] 转录失败（可能是模型下载受限）: {e}", file=sys.stderr)
        sys.exit(2)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in segs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[asr] done: {len(segs)} segments -> {args.out}")


if __name__ == "__main__":
    main()
