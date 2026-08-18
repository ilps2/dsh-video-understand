#!/usr/bin/env python3
"""
frame_prep.py — 视频喂 VLM 前的智能帧预处理

双模式:
  contiguous (默认) — 时间窗口内高帧率连续帧, 保持时序完整性
    适用: Kimi K3, Gemini 2.5+ (内部有时空压缩, 需连续帧)
    
    K3 MoonViT-V2 内部做 sd2_tpool 压缩:
      帧间 Mean-Pool (chunk=4) × 2×2 Pixel Shuffle = 16× 压缩
    300帧10秒视频最终只产生 19,200 个视觉 token (1M上下文的2%)
    所以外部不应做抽帧/去重——编码器自己会压缩。

  sparse — 均匀稀疏抽帧 + 去重
    适用: Claude, GPT-5 (逐帧独立处理, 无时间维度)

时间窗对齐: 给定 N 个时间窗口, 只输出窗口内的帧块/帧。

输出:
  contiguous: <out>/blocks/block_NNN/frame_MMM.jpg  + blocks.csv
  sparse:     <out>/final/frame_NNN.jpg              + frames.csv

用法:
  # 连续帧块 (K3 视频理解 — 15fps, 让 MoonViT-V2 自己压缩)
  python3 frame_prep.py video.mp4 --mode contiguous --block-fps 15
  python3 frame_prep.py video.mp4 --windows "12-35,65-120" --block-fps 15

  # 稀疏抽帧 (Claude/GPT)
  python3 frame_prep.py video.mp4 --mode sparse --fps 1 --clip-clusters 20

依赖: ffmpeg (系统), pillow
     CLIP 模式需要: open-clip-torch torch scikit-learn
"""

import argparse, csv, json, subprocess, sys, os
from pathlib import Path
from collections import defaultdict


# ══════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════

def probe_duration(video_path):
    r = subprocess.run(
        f"ffprobe -v error -show_entries format=duration -of csv=p=0 '{video_path}'",
        shell=True, capture_output=True, text=True)
    return float(r.stdout.strip()) if r.stdout.strip() else 0


def parse_windows(windows_str):
    """'12.0-35.0,65.0-120.0' → [(12.0, 35.0), (65.0, 120.0)]"""
    if not windows_str: return None
    windows = []
    for part in windows_str.split(","):
        part = part.strip()
        if "-" in part:
            s, e = part.split("-")
            windows.append((float(s), float(e)))
    return windows or None


def frame_diff_hash(img_path, hash_size=16):
    """pHash: 感知哈希, 用于帧间相似度比较。"""
    from PIL import Image
    img = Image.open(img_path).convert("L").resize((hash_size + 1, hash_size))
    pixels = list(img.getdata())
    result = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            result = (result << 1) | (1 if left > right else 0)
    return result


def hamming_distance(h1, h2, bits=256):
    return bin(h1 ^ h2).count("1") / bits


# ══════════════════════════════════════════════════════════
# 模式 1: contiguous — 时间窗口内连续帧块
# ══════════════════════════════════════════════════════════

def extract_contiguous_blocks(video_path, windows, out_dir, block_fps=8,
                               max_block_sec=15, max_dim=512):
    """
    在每个时间窗口内提取连续帧。保持帧间时序关系。

    窗口内的帧按 max_block_sec 切成块, 块之间可以重叠也可以紧密拼接。
    每个块的帧是连续的, 保留帧间运动信息供 MoonViT-V2 时间注意力使用。

    返回: [(block_id, start_sec, end_sec, [frame_paths]), ...]
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    total_dur = probe_duration(video_path)
    if total_dur <= 0:
        print("ERROR: cannot probe video duration"); sys.exit(1)

    if not windows:
        windows = [(0, total_dur)]

    all_blocks = []
    block_id = 0

    for ws, we in windows:
        ws = max(0, ws); we = min(total_dur, we)
        if we - ws < 1: continue

        # 切成 max_block_sec 的块
        cursor = ws
        while cursor < we:
            block_end = min(cursor + max_block_sec, we)
            block_dir = out_dir / f"block_{block_id:03d}"
            block_dir.mkdir(parents=True, exist_ok=True)

            dur = block_end - cursor
            # frame_count based on actual duration at block_fps
            frame_count = max(3, int(dur * block_fps))

            cmd = (
                f"ffmpeg -y -v error "
                f"-ss {cursor} -t {dur} "
                f"-i '{video_path}' "
                f"-vf 'fps={frame_count}/{dur}:round=down,"
                f"scale=w={max_dim}:h={max_dim}:force_original_aspect_ratio=decrease' "
                f"-q:v 3 '{block_dir}/frame_%03d.jpg'"
            )
            subprocess.run(cmd, shell=True, capture_output=True)
            frames = sorted(block_dir.glob("frame_*.jpg"))

            if frames:
                all_blocks.append({
                    "block_id": block_id,
                    "start_sec": round(cursor, 2),
                    "end_sec": round(block_end, 2),
                    "duration_sec": round(dur, 2),
                    "frame_count": len(frames),
                    "dir": str(block_dir),
                    "frames": [str(f) for f in frames],
                })
                block_id += 1

            cursor = block_end

    return all_blocks


def detect_scene_changes(video_path, sample_fps=2, threshold=0.30):
    """
    扫描全视频, 检测场景切换点 (用于自动切分 contiguous blocks)。
    返回: [切换时间秒, ...]
    """
    total_dur = probe_duration(video_path)
    if total_dur <= 0: return []

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 抽低 fps 帧做扫描
        cmd = (
            f"ffmpeg -y -v error "
            f"-i '{video_path}' "
            f"-vf 'fps={sample_fps}:round=down,scale=256:256' "
            f"-q:v 5 '{td}/scan_%04d.jpg'"
        )
        subprocess.run(cmd, shell=True, capture_output=True)
        scan_frames = sorted(td.glob("scan_*.jpg"))
        if len(scan_frames) < 3: return []

        hashes = [frame_diff_hash(str(f)) for f in scan_frames]
        cuts = []
        for i in range(1, len(hashes)):
            dist = hamming_distance(hashes[i-1], hashes[i])
            if dist > threshold:
                t = i / sample_fps
                cuts.append(round(t, 1))

        # Merge nearby cuts (< 3s apart)
        merged = []
        for c in cuts:
            if not merged or c - merged[-1] > 3:
                merged.append(c)
        return merged


# ══════════════════════════════════════════════════════════
# 模式 2: sparse — 均匀稀疏抽帧 + 去重 (原版逻辑)
# ══════════════════════════════════════════════════════════

def extract_uniform_frames(video_path, out_dir, fps=1.0, max_dim=512):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("raw_*.jpg"): old.unlink()
    cmd = (
        f"ffmpeg -y -v error -i '{video_path}' "
        f"-vf 'fps={fps}:round=down,scale=w={max_dim}:h={max_dim}:force_original_aspect_ratio=decrease' "
        f"-q:v 3 '{out_dir}/raw_%04d.jpg'")
    subprocess.run(cmd, shell=True, capture_output=True)
    return sorted(out_dir.glob("raw_*.jpg"))


def dedup_by_diff(frames, threshold=0.90):
    """相邻帧差异去重, 相似度 ≥ threshold 则丢弃。"""
    if len(frames) <= 1: return frames
    hashes = [frame_diff_hash(f) for f in frames]
    kept = [frames[0]]
    last_hash = hashes[0]
    for i in range(1, len(frames)):
        sim = 1.0 - hamming_distance(last_hash, hashes[i])
        if sim < threshold:
            kept.append(frames[i])
            last_hash = hashes[i]
    return kept


def dedup_by_clip(frames, n_clusters=20):
    """CLIP K-means 聚类, 每簇保留质心最近帧。"""
    try:
        import torch, open_clip, numpy as np
    except ImportError:
        print("  ⚠️ open-clip-torch 未安装, 跳过 CLIP 聚类")
        return frames
    if len(frames) <= n_clusters: return frames

    print("  加载 CLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval()

    embeddings = []
    from PIL import Image
    for i in range(0, len(frames), 32):
        batch = frames[i:i+32]
        images = torch.stack([preprocess(Image.open(f).convert("RGB")) for f in batch])
        with torch.no_grad():
            emb = model.encode_image(images)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=min(n_clusters, len(frames)), random_state=42, n_init=3)
    labels = kmeans.fit_predict(embeddings)

    kept_indices = []
    for c in range(kmeans.n_clusters):
        mask = labels == c
        cluster_embs = embeddings[mask]
        centroid = kmeans.cluster_centers_[c]
        sims = np.dot(cluster_embs, centroid)
        kept_indices.append(int(np.where(mask)[0][sims.argmax()]))
    kept_indices.sort()
    return [frames[i] for i in kept_indices]


def filter_by_windows(frames, windows, fps):
    """保留时间窗口内的帧。"""
    if not windows: return frames
    kept = []
    for f in frames:
        try:
            fn = int(f.stem.split("_")[-1]) - 1  # 0-indexed
        except ValueError:
            kept.append(f); continue
        t = fn / fps
        for ws, we in windows:
            if ws <= t <= we:
                kept.append(f); break
    return kept


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="视频 → VLM 喂入前的智能帧预处理 (双模式)")
    p.add_argument("video", help="视频文件路径")
    p.add_argument("--out", default="./frames_out", help="输出目录")
    p.add_argument("--mode", choices=["contiguous", "sparse"], default="contiguous",
                   help="contiguous=连续帧块(temporal模型) | sparse=稀疏抽帧(frame模型)")
    p.add_argument("--windows", help="时间窗: '12-35,65-120' (秒)")
    p.add_argument("--max-dim", type=int, default=512, help="帧最大边长")

    # Contiguous mode
    p.add_argument("--block-fps", type=float, default=15,
                   help="contiguous模式: 连续帧块内的帧率 (默认15fps). "
                        "K3用15让MoonViT-V2自己压缩; Gemini可选5")

    p.add_argument("--max-block-sec", type=float, default=15,
                   help="contiguous模式: 单块最大时长(秒)")
    p.add_argument("--auto-scenes", action="store_true",
                   help="contiguous模式: 自动检测场景切换来切分块")

    # Sparse mode
    p.add_argument("--fps", type=float, default=1.0, help="sparse模式抽帧率")
    p.add_argument("--sim-threshold", type=float, default=0.90,
                   help="sparse模式相似度阈值")
    p.add_argument("--skip-diff-dedup", action="store_true")
    p.add_argument("--clip-clusters", type=int, default=0, help="CLIP聚类数, 0=跳过")
    args = p.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"ERROR: 视频不存在: {video}"); sys.exit(1)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    total_dur = probe_duration(str(video))
    windows = parse_windows(args.windows)

    print("=" * 60)
    print(f"🎬 frame_prep — {args.mode} 模式")
    print(f"   视频: {video.name} ({total_dur:.0f}s)")
    print(f"   输出: {out_dir}/")
    if args.mode == "contiguous":
        print(f"   块内帧率: {args.block_fps}fps | 块上限: {args.max_block_sec}s")
        if args.auto_scenes: print(f"   自动场景检测: 开")
    if windows:
        print(f"   时间窗: {len(windows)} 个")
    print("=" * 60)

    # ═══════════════════════════════
    # CONTIGUOUS 模式
    # ═══════════════════════════════
    if args.mode == "contiguous":
        # 自动场景检测
        if args.auto_scenes and not windows:
            print("\n🔍 扫描场景切换...")
            cuts = detect_scene_changes(str(video))
            if cuts:
                # 构建场景窗口
                scene_windows = []
                prev = 0
                for cut in cuts:
                    scene_windows.append((prev, cut))
                    prev = cut
                scene_windows.append((prev, total_dur))
                windows = scene_windows
                print(f"  检测到 {len(cuts)} 个切换点 → {len(windows)} 个场景")
                for i, (ws, we) in enumerate(windows[:10]):
                    print(f"    场景{i}: {ws:.0f}s - {we:.0f}s ({we-ws:.0f}s)")

        if not windows:
            windows = [(0, total_dur)]

        print(f"\n📦 提取连续帧块 (block_fps={args.block_fps})...")
        blocks = extract_contiguous_blocks(
            str(video), windows, out_dir / "blocks",
            block_fps=args.block_fps,
            max_block_sec=args.max_block_sec,
            max_dim=args.max_dim)

        if not blocks:
            print("ERROR: 未提取到任何帧"); sys.exit(1)

        # 保存块索引
        csv_path = out_dir / "blocks.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "block_id", "start_sec", "end_sec", "duration_sec",
                "frame_count", "dir"])
            writer.writeheader()
            for b in blocks:
                writer.writerow({k: b[k] for k in writer.fieldnames})

        total_frames = sum(b["frame_count"] for b in blocks)
        print(f"\n  完成: {len(blocks)} 块, {total_frames} 帧")
        print(f"  索引: {csv_path}")
        for b in blocks[:10]:
            print(f"    block_{b['block_id']:03d}  "
                  f"{b['start_sec']:.0f}s-{b['end_sec']:.0f}s  "
                  f"({b['frame_count']}帧)")

        print(f"\n{'=' * 60}")
        print(f"✅ 完成: {len(blocks)} 连续帧块, {total_frames} 帧")
        print(f"   可直接喂入 K3 / Gemini 等时序模型")
        print(f"   输出: {out_dir}/blocks/")
        print(f"{'=' * 60}")
        return

    # ═══════════════════════════════
    # SPARSE 模式
    # ═══════════════════════════════
    print(f"\n[L1] 均匀抽帧 (fps={args.fps})...")
    raw_dir = out_dir / "_raw"
    frames = extract_uniform_frames(str(video), raw_dir, fps=args.fps, max_dim=args.max_dim)
    print(f"  → {len(frames)} 帧")

    if not frames:
        print("ERROR: 未抽到帧"); sys.exit(1)

    if not args.skip_diff_dedup:
        print(f"\n[L2] 帧差异去重 (threshold={args.sim_threshold})...")
        before = len(frames)
        frames = dedup_by_diff(frames, threshold=args.sim_threshold)
        print(f"  → {before} → {len(frames)} ({before - len(frames)} removed)")

    if windows:
        before = len(frames)
        frames = filter_by_windows(frames, windows, args.fps)
        print(f"\n[L2.5] 时间窗对齐: {before} → {len(frames)}")

    if args.clip_clusters > 0:
        print(f"\n[L3] CLIP 聚类 ({args.clip_clusters} 簇)...")
        before = len(frames)
        frames = dedup_by_clip(frames, n_clusters=args.clip_clusters)
        print(f"  → {before} → {len(frames)}")

    # Save
    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    import shutil
    csv_rows = []
    for i, f in enumerate(frames):
        new = final_dir / f"frame_{i:03d}.jpg"
        shutil.copy(f, new)
        try: t = (int(f.stem.split("_")[-1]) - 1) / args.fps
        except: t = 0
        csv_rows.append({"frame_file": new.name, "time_sec": round(t, 2)})

    csv_path = out_dir / "frames.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_file", "time_sec"])
        w.writeheader(); w.writerows(csv_rows)

    reduction = (1 - len(frames) / (total_dur * args.fps)) * 100 if total_dur else 0
    print(f"\n{'=' * 60}")
    print(f"✅ 完成: {len(frames)} 稀疏帧 | 压缩率 {reduction:.0f}%")
    print(f"   适用: Claude / GPT-5 等逐帧模型")
    print(f"   输出: {final_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
