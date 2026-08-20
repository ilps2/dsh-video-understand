"""
场景边界检测 + 场景类型分类

混合方案：
  1. 基于 PyAV 码流 MV 的场景边界检测（零成本，从 H.264 码流直接读取编码器运动矢量）
  2. 基于帧差的轻量级分类（无需 PyAV，fallback）
  3. 用于问题路由（ASR 稀疏时辅助判断）和全局扫描抽帧策略

输入：视频文件路径
输出：场景边界时间戳 + 逐秒场景标签 + 兴趣权重

兼容旧版 SceneClassifier API（class boundaries/per_second/weights），同时支持新路径直接传视频文件。
"""
import subprocess
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


def detect_scene_boundaries(video_path: str, sample_fps: float = 1.0,
                            threshold: float = 0.4, min_gap: int = 2) -> List[float]:
    """
    基于 PyAV 码流运动向量的场景边界检测（零成本）。
    从 H.264/H.265 码流直接读取编码器 MV（无需额外计算），
    相邻帧 MV 幅度均值差 > threshold → 场景切换点。

    Args:
        video_path: 视频文件路径
        sample_fps: 采样帧率（1fps 足够检测场景边界）
        threshold: MV 变化阈值（越大越不敏感）
        min_gap: 边界间最小间隔秒数（合并过近的边界）

    Returns:
        场景边界时间戳列表（秒），已合并间隔 < min_gap 的点
    """
    try:
        import av
    except ImportError:
        return _detect_boundaries_fallback(video_path, sample_fps, threshold, min_gap)

    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        # 使用 export_mvs 让 ffmpeg 导出编码器运动矢量
        stream.side_data_type = "MOTION_VECTORS"
        target_rate = max(1, int(stream.average_rate / sample_fps))
        frame_idx = 0
        mv_mags = []  # 每帧 MV 幅度均值

        for frame in container.decode(video=0):
            if frame_idx % target_rate == 0:
                if hasattr(frame, 'side_data') and hasattr(frame.side_data, 'motion_vectors'):
                    mvs = frame.side_data.motion_vectors
                    if mvs:
                        mags = np.hypot(mvs["dst_x"].astype(float) - mvs["src_x"].astype(float),
                                        mvs["dst_y"].astype(float) - mvs["src_y"].astype(float))
                        mv_mags.append(float(np.mean(mags)))
                    else:
                        mv_mags.append(0.0)
                else:
                    mv_mags.append(0.0)
            frame_idx += 1

        container.close()

        if len(mv_mags) < 2:
            return []

        # 场景边界检测：相邻 MV 幅度差 > threshold
        mags = np.array(mv_mags)
        diffs = np.abs(np.diff(mags))
        frame_period = target_rate / stream.average_rate
        times = [i * frame_period for i in range(len(mv_mags))]
        bounds = [times[i] for i in range(1, len(diffs)) if diffs[i - 1] > threshold]

        return _merge_bounds(bounds, min_gap, dur=float(times[-1]) if times else 0)

    except Exception as e:
        print(f"  ⚠ PyAV MV 检测失败，回退帧差方式: {e}")
        return _detect_boundaries_fallback(video_path, sample_fps, threshold, min_gap)


def _detect_boundaries_fallback(video_path: str, sample_fps: float = 1.0,
                                 threshold: float = 0.4, min_gap: int = 2) -> List[float]:
    """帧差 fallback：每秒抽一帧，计算相邻帧的灰度差异均值，差 > threshold → 场景边界。"""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(src_fps / sample_fps))
    prev_gray = None
    diffs = []
    times = []

    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_gray is not None:
                diff = np.abs(gray - prev_gray).mean() / 255.0  # 归一化到 0-1
                diffs.append(diff)
                times.append(frame_i / src_fps)
            prev_gray = gray
        frame_i += 1

    cap.release()

    if not diffs:
        return []

    diffs_arr = np.array(diffs)
    thresh = max(threshold, float(np.mean(diffs_arr) + 2 * np.std(diffs_arr)))
    bounds = [times[i] for i in range(1, len(diffs_arr)) if diffs_arr[i] > thresh]
    return _merge_bounds(bounds, min_gap)


def _merge_bounds(bounds: List[float], min_gap: int, dur: float = 0) -> List[float]:
    """合并间隔 < min_gap 的边界，返回去重排序结果。"""
    if not bounds:
        return []
    merged = [bounds[0]]
    for b in bounds[1:]:
        if b - merged[-1] >= min_gap:
            merged.append(b)
    return merged


def classify_scenes_by_motion(video_path: str, sample_fps: float = 1.0) -> Dict[int, str]:
    """
    基于帧差的轻量场景类型分类（无需 PyAV，无需 MV）。

    Args:
        video_path: 视频文件路径
        sample_fps: 采样帧率

    Returns:
        {秒数: 场景类型} 映射（static / motion_low / motion_high / transition）
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(src_fps / sample_fps))
    prev_gray = None
    result = {}
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            sec = int(frame_i / src_fps)
            if prev_gray is not None:
                diff = float(np.abs(gray - prev_gray).mean())
                if diff < 1.0:
                    result[sec] = "static"
                elif diff < 4.0:
                    result[sec] = "motion_low"
                else:
                    result[sec] = "motion_high"
            else:
                result[sec] = "motion_low"
            prev_gray = gray
        frame_i += 1

    cap.release()

    # 标记场景边界处为 transition
    bounds = detect_scene_boundaries(video_path, sample_fps=1.0)
    for b in bounds:
        sec = int(b)
        if sec in result:
            result[sec] = "transition"

    return result


def get_scene_info(video_path: str) -> Dict:
    """
    获取视频场景信息（边界 + 分类 + 统计），供路由和抽帧策略使用。

    Returns:
        {
            "boundaries": [4.0, 12.5, ...],
            "n_boundaries": int,
            "scene_distribution": {"static": 40, "motion_low": 120, ...},
            "sample_fps": float,
            "duration": float
        }
    """
    bounds = detect_scene_boundaries(video_path)
    scenes = classify_scenes_by_motion(video_path)
    from collections import Counter
    dist = Counter(scenes.values())
    # 时长（秒）
    cap_dist = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True, timeout=30)
    dur = 0.0
    try:
        dur = float(json.loads(cap_dist.stdout)["format"]["duration"])
    except Exception:
        pass

    return {
        "boundaries": bounds,
        "n_boundaries": len(bounds),
        "scene_distribution": dict(dist),
        "duration": dur,
    }
