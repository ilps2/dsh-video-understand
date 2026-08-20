"""
视频理解 pipeline 阶段模块

定义各个处理阶段的函数。
"""
import os
import sys
import json
import re
import time
import subprocess
import hashlib
import urllib.request
import ssl
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .context import ProcessingContext
from .result_schema import ErrorCode

# 系统代理(Clash MITM)用自签名证书 → 指向 macOS 系统证书链（双保险）
if os.path.exists("/etc/ssl/cert.pem"):
    os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/cert.pem")
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile="/etc/ssl/cert.pem")


# ── LLM 配置：环境变量 → DSH credentials 文件 → 默认值（与 understand_video.py 一致）──
def _load_llm_config():
    """Load LLM API key: env var → DSH credentials file → empty.

    配对规则：一把 key 绝不能被送到不是为它选定的主机上。
      - LLM_API_KEY（显式覆盖）：key 是用户选的，他设的 URL/model 也照用
      - DEEPSEEK_API_KEY（且未设 LLM_API_URL）：配对 DeepSeek 端点与模型
      - 未设任何 key → 读 DSH credentials 文件（xiaomi/deepseek 各自配对）
    """
    url = os.environ.get("LLM_API_URL", "")
    model = os.environ.get("LLM_MODEL", "")
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


def _llm(messages, max_tokens=16384):
    """文本 LLM 调用（与 understand_video.py 相同的 API 契约）。"""
    if not KEY:
        raise RuntimeError("未设置 LLM_API_KEY 环境变量")
    body = {"model": MODEL, "messages": messages,
            "max_tokens": max_tokens, "temperature": 0.3}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        r = json.loads(resp.read())
    return r["choices"][0]["message"]["content"], r.get("usage", {})


def _privacy_allows_visual(ctx: ProcessingContext) -> bool:
    """视觉级（帧上传 MiMo）是否被隐私模式允许。
    fully_local / local_extract 禁止远程 VLM；remote_answer / remote_visual 允许
    （用户显式请求 l1/l2 即视为同意帧上传，与工具描述一致）。"""
    mode = (ctx.privacy_mode or "remote_answer").lower()
    return mode not in ("fully_local", "local_extract")


def resolve_target(ctx: ProcessingContext) -> bool:
    """
    解析目标：本地文件或 URL
    
    Args:
        ctx: 处理上下文
        
    Returns:
        是否成功
    """
    if ctx.is_cancelled():
        ctx.add_error(ErrorCode.CANCELLED.value, "处理已取消", stage="request")
        return False
    
    target = ctx.target
    
    # 检查是否是本地文件
    if os.path.isfile(target):
        ctx.local_video_path = os.path.abspath(target)
        ctx.video_metadata["source"] = "local"
        ctx.video_metadata["local_path"] = ctx.local_video_path
        return True
    
    # 检查是否是 B站 URL 或 BV 号
    import re
    bv_match = re.search(r"BV[0-9A-Za-z]{10}", target)
    if bv_match or target.startswith("http"):
        ctx.video_metadata["source"] = "bilibili"
        ctx.video_metadata["url"] = target
        return True
    
    # 尝试作为本地路径
    if os.path.exists(target):
        ctx.local_video_path = os.path.abspath(target)
        ctx.video_metadata["source"] = "local"
        ctx.video_metadata["local_path"] = ctx.local_video_path
        return True
    
    ctx.add_error(ErrorCode.TARGET_NOT_FOUND.value, f"找不到目标: {target}", stage="request")
    return False


def probe_media(ctx: ProcessingContext) -> bool:
    """
    探测媒体信息
    
    Args:
        ctx: 处理上下文
        
    Returns:
        是否成功
    """
    if ctx.is_cancelled():
        ctx.add_error(ErrorCode.CANCELLED.value, "处理已取消", stage="probe")
        return False
    
    if not ctx.local_video_path:
        ctx.add_error(ErrorCode.MEDIA_PROBE_FAILED.value, "没有本地视频文件", stage="probe")
        return False
    
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            ctx.local_video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            ctx.add_error(ErrorCode.MEDIA_PROBE_FAILED.value, 
                         f"ffprobe 失败: {result.stderr[:500]}", 
                         stage="probe")
            return False
        
        info = json.loads(result.stdout)
        
        # 提取视频信息
        duration = float(info.get("format", {}).get("duration", 0))
        ctx.video_metadata["duration_s"] = duration
        ctx.video_metadata["duration"] = duration
        
        # 提取流信息
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                ctx.video_metadata["width"] = int(stream.get("width", 0))
                ctx.video_metadata["height"] = int(stream.get("height", 0))
                
                # 解析 fps
                fps_str = stream.get("r_frame_rate", "30/1")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    ctx.video_metadata["fps"] = float(num) / float(den)
                else:
                    ctx.video_metadata["fps"] = float(fps_str)
        
        return True
        
    except subprocess.TimeoutExpired:
        ctx.add_error(ErrorCode.MEDIA_PROBE_FAILED.value, "ffprobe 超时", stage="probe")
        return False
    except Exception as e:
        ctx.add_error(ErrorCode.MEDIA_PROBE_FAILED.value, 
                     f"探测失败: {str(e)}", 
                     stage="probe")
        return False


def extract_audio(ctx: ProcessingContext) -> bool:
    """
    提取音频
    
    Args:
        ctx: 处理上下文
        
    Returns:
        是否成功
    """
    if ctx.is_cancelled():
        ctx.add_error(ErrorCode.CANCELLED.value, "处理已取消", stage="extract_audio")
        return False
    
    if not ctx.local_video_path:
        ctx.add_error(ErrorCode.FFMPEG_FAILED.value, "没有本地视频文件", stage="extract_audio")
        return False
    
    try:
        audio_dir = ctx.create_work_dir("audio")
        ctx.audio_path = str(audio_dir / "audio.wav")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", ctx.local_video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            ctx.audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            ctx.add_error(ErrorCode.FFMPEG_FAILED.value, 
                         f"音频提取失败: {result.stderr[:500]}", 
                         stage="extract_audio")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        ctx.add_error(ErrorCode.FFMPEG_FAILED.value, "音频提取超时", stage="extract_audio")
        return False
    except Exception as e:
        ctx.add_error(ErrorCode.FFMPEG_FAILED.value, 
                     f"音频提取失败: {str(e)}", 
                     stage="extract_audio")
        return False


def _video_content_hash(video_path, sample_bytes=4 * 1024 * 1024):
    """按视频内容哈希（xxhash 前 4MB + 文件大小）——跨路径/跨设备稳定。"""
    import xxhash
    h = xxhash.xxh64()
    with open(video_path, "rb") as f:
        h.update(f.read(sample_bytes))
    return f"{h.hexdigest()}_{os.path.getsize(video_path)}"


def _transcript_cache_path(ctx, model="tiny"):
    """转写缓存路径：cache_dir/transcripts/{content_hash}_{model}.json"""
    if not ctx.local_video_path or not os.path.exists(ctx.local_video_path):
        return None
    try:
        h = _video_content_hash(ctx.local_video_path)
    except Exception:
        return None
    cache_dir = os.path.join(ctx.cache_dir or "", "transcripts")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{h}_{model}.json")


def _load_transcript_cache(ctx, model="tiny"):
    """从内容寻址缓存加载转写。命中返回 segments 列表，未命中返回 None。"""
    path = _transcript_cache_path(ctx, model)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_transcript_cache(ctx, segments, model="tiny"):
    """保存转写到内容寻址缓存。"""
    path = _transcript_cache_path(ctx, model)
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False)
    except Exception as e:
        print(f"  ⚠️ 转写缓存写入失败: {e}", flush=True)


def transcribe(ctx: ProcessingContext) -> bool:
    """
    ASR 转写（带内容寻址缓存：同一视频二次提问跳过 ASR）

    使用 faster-whisper 或 SenseVoice 进行转写。
    转写结果按视频内容哈希缓存到 cache_dir/transcripts/，
    同一视频的任何后续问题直接命中缓存，不再重复转写。

    Args:
        ctx: 处理上下文
        
    Returns:
        是否成功
    """
    if ctx.is_cancelled():
        ctx.add_error(ErrorCode.CANCELLED.value, "处理已取消", stage="transcribe")
        return False
    
    # 无音频时返回空 transcript（不报错）
    if not ctx.audio_path or not os.path.exists(ctx.audio_path):
        ctx.avis["transcript"] = []
        ctx.add_warning("NO_AUDIO_TRACK", "视频没有音轨", stage="transcribe")
        return True
    
    try:
        transcript_dir = ctx.create_work_dir("transcript")
        ctx.transcript_path = str(transcript_dir / "transcript.jsonl")
        
        # 根据 asr_backend 选择 ASR 后端
        asr_backend = ctx.video_metadata.get("asr_backend", "whisper")
        model_size = ctx.video_metadata.get("asr_model", "tiny")
        
        # ── 内容寻址缓存：同一视频命中缓存跳过 ASR ──
        cached = _load_transcript_cache(ctx, model_size)
        if cached is not None:
            ctx.avis["transcript"] = cached
            with open(ctx.transcript_path, "w", encoding="utf-8") as f:
                for seg in cached:
                    f.write(json.dumps(seg, ensure_ascii=False) + "\n")
            ctx.cache_hit = True
            print(f"  ♻️  [ASR] 语义层缓存命中（内容哈希）: {len(cached)} segments，跳过转写", flush=True)
            return True
        
        if asr_backend == "sensevoice":
            # 使用 SenseVoice
            try:
                from engine.asr_sensevoice import SenseVoiceASR
                asr = SenseVoiceASR(device=ctx.video_metadata.get("device", "auto"))
                segments = asr.transcribe(ctx.audio_path, ctx.transcript_path)
                ctx.avis["transcript"] = segments
                _save_transcript_cache(ctx, segments, model_size)
                return True
            except Exception as e:
                print(f"  ⚠ SenseVoice ASR 失败: {e}")
                asr_backend = "whisper"
        
        if asr_backend == "whisper":
            # 使用 faster-whisper
            from faster_whisper import WhisperModel
            
            device = ctx.video_metadata.get("device", "auto")
            
            # 自动检测设备
            if device == "auto":
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"
                except ImportError:
                    device = "cpu"
            
            # MPS 不支持 faster-whisper，回退到 CPU
            if device == "mps":
                device = "cpu"
            
            print(f"  [ASR] 加载 faster-whisper {model_size} (device: {device})")
            
            # 设置离线模式
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            # 使用缓存的模型
            compute_type = "float16" if device != "cpu" else "int8"
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            
            # 转写
            segments_gen, info = model.transcribe(
                ctx.audio_path,
                language="zh",
                word_timestamps=True,
                vad_filter=True,
            )
            
            # 转换为标准格式
            segments = []
            for seg in segments_gen:
                words = [[w.word, round(w.start, 2), round(w.end, 2)]
                         for w in (seg.words or [])]
                segments.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                    "language": info.language,
                    "confidence": seg.avg_logprob if hasattr(seg, 'avg_logprob') else None,
                    "source": f"faster-whisper-{model_size}",
                })
            
            # 写入输出文件 + 缓存
            with open(ctx.transcript_path, "w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(json.dumps(seg, ensure_ascii=False) + "\n")
            _save_transcript_cache(ctx, segments, model_size)
            
            ctx.avis["transcript"] = segments
            
            print(f"  [ASR] 转写完成: {len(segments)} segments")
            return True
        
        return False
        
    except Exception as e:
        ctx.add_error(ErrorCode.ASR_FAILED.value, 
                     f"转写失败: {str(e)}", 
                     stage="transcribe")
        return False


def build_avis(ctx: ProcessingContext) -> bool:
    """
    构建 AVIS 信息层（问题路由分层的基础层）

    完成三件事：
      1. classify 类型路由（免费信号：帧差/色彩/ASR覆盖）→ 视频类型 + 推荐策略
      2. obj_tracks 对象轨迹层（L1 语义层：MOG2 运动对象 + IoU 追踪 + YOLO 标签）
         —— 主对象时间线运动轨迹 + 少移动对象的记录
      3. 组装 manifest

    Args:
        ctx: 处理上下文
        
    Returns:
        是否成功
    """
    if ctx.is_cancelled():
        ctx.add_error(ErrorCode.CANCELLED.value, "处理已取消", stage="build_avis")
        return False
    
    try:
        avis_dir = ctx.create_work_dir("avis")
        ctx.avis_dir = str(avis_dir)
        
        # ── 1. classify 类型路由（免费信号，零模型成本，失败不阻断）──
        video_type = ctx.video_metadata.get("video_type")
        if not video_type and ctx.local_video_path and os.path.exists(ctx.local_video_path):
            try:
                from .avis import classify_video
                print("  [路由] 免费信号类型探测...", flush=True)
                c = classify_video(Path(ctx.local_video_path), use_asr=bool(ctx.avis.get("transcript")))
                if c and isinstance(c, dict):
                    video_type = c.get("type")
                    ctx.video_metadata["video_type"] = video_type
                    ctx.video_metadata["video_strategy"] = c.get("recommended_strategy") or c.get("strategy")
                    # 扩展 video_profile 信号
                    sig = c.get("signals", {})
                    ctx.video_metadata["asr_coverage"] = sig.get("speech_ratio", 0)
                    ctx.video_metadata["motion_score"] = sig.get("motion_score", 0)
                    ctx.video_metadata["static_ratio"] = sig.get("static_ratio", 0)
                    print(f"  [路由] 类型={video_type} → 策略={ctx.video_metadata.get('video_strategy', '')}", flush=True)
            except Exception as e:
                print(f"  ⚠️ classify 失败（继续）: {e}", flush=True)
        
        # ASR 覆盖率（已有转写时直接算，classify 未给时兜底）
        if "asr_coverage" not in ctx.video_metadata or not ctx.video_metadata.get("asr_coverage"):
            tr = ctx.avis.get("transcript", [])
            dur = float(ctx.video_metadata.get("duration_s", 0) or 0)
            if tr and dur > 0:
                covered = sum(min(s.get("end", s.get("start", 0) + 1), dur) - s.get("start", 0)
                              for s in tr if str(s.get("text", "")).strip())
                ctx.video_metadata["asr_coverage"] = round(min(1.0, covered / dur), 3)
            else:
                ctx.video_metadata["asr_coverage"] = 0.0
        
        # ── 2. obj_tracks 对象轨迹层（L1 语义层）──
        #    主对象时间线运动轨迹 + 少移动对象记录；失败不阻断（降级为无轨迹）
        n_tracks = 0
        if ctx.local_video_path and os.path.exists(ctx.local_video_path):
            try:
                from .avis import extract_obj_tracks
                n_tracks = extract_obj_tracks(Path(ctx.local_video_path), avis_dir,
                                              fps_target=5, max_objects=12, min_track_sec=0.6)
            except Exception as e:
                print(f"  ⚠️ obj_tracks 提取失败（继续）: {e}", flush=True)
        if n_tracks:
            try:
                tracks = []
                trk_path = avis_dir / "obj_tracks.jsonl"
                if trk_path.exists():
                    for line in open(trk_path, encoding="utf-8"):
                        if not line.strip():
                            continue
                        o = json.loads(line)
                        # unknown class 拆分：运动轨迹质量高、类别识别质量低
                        # （避免因 YOLO 标签 unknown 就把整个轨迹层判为无效）
                        cls = o.get("class", "unknown")
                        o["class_confidence"] = 0.0 if cls in ("unknown", None) else 0.6
                        o["motion_confidence"] = min(
                            1.0, (o.get("n_frames", 0) / 20.0) *
                            (1.0 if o.get("motion", "") not in ("", "static") else 0.5))
                        tracks.append(o)
                ctx.avis["objects"] = tracks
            except Exception as e:
                print(f"  ⚠️ obj_tracks 解析失败: {e}", flush=True)
        ctx.video_metadata["track_count"] = n_tracks
        
        # ── 场景边界检测（PyAV MV / 帧差 fallback，零模型成本）──
        scene_boundaries = []
        if ctx.local_video_path and os.path.exists(ctx.local_video_path):
            try:
                from .scene_classifier import detect_scene_boundaries
                scene_boundaries = detect_scene_boundaries(ctx.local_video_path)
                ctx.video_metadata["scene_boundaries"] = scene_boundaries
                print(f"  [场景] 边界检测: {len(scene_boundaries)} 个变换点", flush=True)
            except Exception as e:
                print(f"  ⚠️ 场景边界检测失败（继续）: {e}", flush=True)
        
        # ── OCR 文字计数（画面文字，free 信号；失败不阻断）──
        ocr_count = 0
        try:
            ocr_path = avis_dir / "ocr_text.jsonl"
            if ocr_path.exists():
                ocr_count = sum(1 for _ in open(ocr_path, encoding="utf-8"))
        except Exception:
            pass
        ctx.video_metadata["ocr_text_count"] = ocr_count
        
        # ── 3. 组装 manifest (v2) ──
        manifest = {
            "avis_version": "2",
            "video": ctx.video_metadata,
            "video_type": video_type,
            "transcript": ctx.avis.get("transcript", []),
            "text_tracks": ctx.avis.get("text_tracks", []),
            "visual_observations": ctx.avis.get("visual_observations", []),
            "scenes": ctx.avis.get("scenes", []),
            "motion": ctx.avis.get("motion", []),
            "objects": ctx.avis.get("objects", []),
            "metadata": ctx.avis.get("metadata", {}),
        }
        
        # 保存 manifest
        manifest_path = avis_dir / "avis.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        ctx.avis["manifest_path"] = str(manifest_path)
        
        return True
        
    except Exception as e:
        ctx.add_error(ErrorCode.AVIS_FAILED.value, 
                     f"AVIS 构建失败: {str(e)}", 
                     stage="build_avis")
        return False


def select_visual_evidence(ctx: ProcessingContext, focus_round: int = 1) -> bool:
    """
    选择视觉证据：动态问题路由分层。

    视频类型做先验，问题意图决定入口，L1 轨迹负责定位，L2 VLM 负责确认：
      - router.choose() 决定 effective_layer（问题意图 + 视频先验 + 已有证据 + 隐私）
      - L1 轨迹定位器：obj_tracks 活跃窗口 / 关键词窗口 / LLM 定位窗口
      - L2 只在需要视觉能力时抽帧（L1 轨迹窗口内的低成本关键帧）
      - speech_dense 只改变默认路线，不限制能力（视觉问题照样升级 L2）

    Args:
        ctx: 处理上下文
        focus_round: 聚焦轮次（多轮循环中递增，抽帧密度随轮次增加）

    Returns:
        是否成功（视觉失败不阻断 pipeline，仅加 warning）
    """
    if ctx.is_cancelled():
        ctx.add_error(ErrorCode.CANCELLED.value, "处理已取消", stage="visual_analysis")
        return False

    # L0 不需要视觉证据
    if ctx.level == "l0":
        return True

    # 隐私模式检查：帧上传 MiMo 被禁止时跳过并告警
    if not _privacy_allows_visual(ctx):
        ctx.add_warning("VISUAL_SKIPPED_BY_PRIVACY",
                        f"隐私模式 {ctx.privacy_mode} 禁止上传帧，跳过 L{ctx.level} 视觉证据",
                        stage="visual_analysis", retryable=False)
        return True

    if not ctx.local_video_path or not os.path.exists(ctx.local_video_path):
        ctx.add_warning("VISUAL_NO_VIDEO", "没有本地视频文件，跳过视觉证据",
                        stage="visual_analysis", retryable=False)
        return True

    try:
        # 延迟导入：视觉级依赖（VLM key）仅在需要时加载
        from .visual_level import pick_times_l1, pick_times_l2, extract_frame, vlm_frames
        from .router import choose as route_choose

        dur = float(ctx.video_metadata.get("duration_s", 0) or 0)
        if dur <= 0:
            dur = float(probe_dur(ctx.local_video_path))

        frames_dir = ctx.create_work_dir("visual_frames")
        avis_dir = ctx.avis_dir or ""
        transcript = ctx.avis.get("transcript", [])
        video_type = ctx.video_metadata.get("video_type", "mixed")

        # ── 问题驱动定位（首个问题；失败/无问题时不阻断）──
        located_windows = []
        loc_gap = "none"
        first_q = (ctx.questions or [""])[0]
        if first_q.strip():
            try:
                located_windows, loc_gap, _reason = _locate_windows(
                    first_q, transcript, dur,
                    title=str(ctx.video_metadata.get("title", "")))
            except Exception as e:
                print(f"  ⚠️ 定位失败（回退均匀抽帧）: {e}", flush=True)
                located_windows, loc_gap = [], "none"
            # 兜底：LLM 定位空窗/失败时用关键词定位（转写含问题关键词的段）
            if not located_windows:
                kw_wins = _keyword_locate(first_q, transcript, dur)
                if kw_wins:
                    located_windows = kw_wins
                    loc_gap = "visual"  # 关键词命中 → 抽帧验证
                    print(f"  [定位] 关键词兜底命中: {kw_wins}", flush=True)
        # 保存定位窗口供聚焦循环的 asr 补救复用
        ctx.avis["_loc_windows"] = located_windows

        # ── L1 轨迹定位器：obj_tracks 活跃窗口（运动问题的主定位源）──
        #    轨迹活跃窗口作为 L2 的注意力引导器（"L1 负责定位，L2 负责确认"）
        track_windows = _track_active_windows(ctx.avis.get("objects", []), dur)
        if track_windows and not located_windows:
            located_windows = track_windows[:1]
            print(f"  [定位] obj_tracks 轨迹活跃窗口: {located_windows}", flush=True)

        per_window = 3 + max(0, focus_round - 1)  # 多轮抽帧更密

        # ── 动态路由：问题意图 + 视频先验 + 已有证据 + 隐私 ──
        available = {
            "asr": bool(transcript),
            "ocr": ctx.video_metadata.get("ocr_text_count", 0) > 0,
            "obj_tracks": bool(ctx.avis.get("objects")),
            "scenes": bool(ctx.avis.get("scenes")),
            "visual_l1": bool(ctx.avis.get("visual_notes")),
            "visual_l2": bool(ctx.avis.get("visual_notes")),
        }
        decision = route_choose(
            video_profile=ctx.video_metadata,
            question=first_q,
            available=available,
            level_hint=ctx.level if ctx.level in ("l0", "l1", "l2") else None,
            privacy_mode=ctx.privacy_mode,
            max_frames=per_window * 2,
            budget_cny=getattr(ctx, "budget_cny", None),
        )
        ctx.avis["_routing"] = decision
        effective = decision["effective_layer"]
        ctx.avis["_effective_layer"] = effective
        reason = decision["escalation_reason"]
        budget_note = f" | 预算拦截={decision.get('budget_blocked')} 估算={decision.get('estimated_cost_cny')}元" if ctx.budget_cny is not None else ""
        print(f"  [路由] 意图={decision['intent']} 初始L{decision['initial_layer']} → "
              f"有效L{effective} | 理由: {'; '.join(reason) or '默认'}{budget_note}", flush=True)

        # 预算拦截：禁止 L2 → 不抽帧（L0/L1 结论即可）
        if decision.get("budget_blocked"):
            print(f"  [路由] 预算不足，跳过视觉（降级 L0/L1）", flush=True)
            return True

        # 意图不需要视觉 → 不抽帧（L0/L1 结论即可）
        if not decision.get("visual_required") and effective in ("l0", "l1"):
            print(f"  [路由] 意图={decision['intent']} 不需视觉，跳过抽帧", flush=True)
            return True

        # 视觉被隐私禁止 → 跳过
        if not decision.get("visual_allowed", True):
            return True

        # ── 抽帧时间点：根据 scan_mode 走不同策略 ──
        scan_mode = decision.get("scan_mode")
        intent_q = decision["intent"]

        if scan_mode == "global_scan":
            # 全局扫描模式：分批调用 VLM（每批 ≤7 帧，避免单张图过大）
            scan_step = max(10, int(dur / 15))  # 15 帧以内，最多每 10s 一帧
            all_times = [round(t, 1) for t in np.arange(0, dur, scan_step)]
            question = (
                f"这是视频剪辑的一部分帧，按时间顺序排列。"
                f"用户问题：{first_q}\n"
                "逐帧识别：画面中的角色/物体、场景类型（动作/变身/战斗/对话等）、"
                "文字标注。如有变身请标注时间段。"
            )
            # 分批调用（每批 7 帧，避免拼接图过大导致 VLM 返回空内容）
            BATCH = 7
            all_descs = []
            all_pin = all_pout = 0
            for start in range(0, len(all_times), BATCH):
                batch_times = all_times[start:start + BATCH]
                batch_frames = []
                for i, t in enumerate(batch_times):
                    idx = start + i
                    fp = str(frames_dir / f"scan_{idx:02d}_{int(t)}s.jpg")
                    extract_frame(ctx.local_video_path, t, fp)
                    batch_frames.append(fp)
                desc, pin, pout = vlm_frames(batch_frames, question)
                if desc:
                    all_descs.append(desc)
                all_pin += pin
                all_pout += pout
                times = all_times  # 后续 evidence 构建用

            # 合并描述
            desc = "\n\n".join(all_descs) if all_descs else ""
            pin, pout = all_pin, all_pout
            frame_paths = [str(frames_dir / f"scan_{i:02d}_{int(t)}s.jpg")
                           for i, t in enumerate(all_times)]
            print(f"  [模式] 全局扫描（ASR 稀疏）→ {len(all_times)} 帧，"
                  f"{(len(all_times)+BATCH-1)//BATCH} 批", flush=True)

            # 全局扫描已完成 VLM 调用，直接写入结果
            ctx.evidence = []
            for i, t in enumerate(all_times):
                ctx.evidence.append({
                    "start_s": float(t),
                    "end_s": float(t) + scan_step,
                    "source": "visual_l2",
                    "ref": frame_paths[i] if i < len(frame_paths) else None,
                    "reason": f"全局扫描 @ {t:.0f}s（ASR 稀疏模式）",
                    "confidence": None,
                })
            if desc:
                note = f"\n## 视觉补充（全局扫描 {len(all_times)} 帧 @ {[f'{t:.0f}s' for t in all_times]}）\n{desc}\n"
                ctx.avis["visual_notes"] = ctx.avis.get("visual_notes", []) + [note]
                ctx.avis["metadata"] = dict(ctx.avis.get("metadata", {}))
                ctx.avis["metadata"]["visual_tokens"] = {"in": pin, "out": pout}
            print(f"  ✅ 全局扫描 {len(desc)} 字 | VLM {pin}+{pout} tok", flush=True)
            return True

        elif ctx.window and ctx.window != "auto":
            times = pick_times_l2(dur, ctx.window, step=max(2, 5 - (focus_round - 1) * 2))
            question = _visual_prompt_for_intent(intent_q, first_q)
        elif located_windows and loc_gap in ("visual", "asr"):
            times = _frames_in_windows(located_windows, dur, per_window=per_window)
            question = _visual_prompt_for_intent(intent_q, first_q)
        elif track_windows:
            times = _frames_in_windows(track_windows[:2], dur, per_window=per_window)
            question = _visual_prompt_for_intent(intent_q, first_q)
        elif effective == "l2":
            times = pick_times_l2(dur, "auto", step=5)
            question = _visual_prompt_for_intent(intent_q, first_q)
        else:
            times = pick_times_l1(avis_dir, dur, 5)
            question = _visual_prompt_for_intent(intent_q, first_q)

        if not times:
            ctx.add_warning("VISUAL_NO_FRAMES", "没有可用的帧时间点", stage="visual_analysis")
            return True

        # 抽帧
        frame_paths = []
        for i, t in enumerate(times):
            fp = str(frames_dir / f"f{i:02d}_{int(t)}s.jpg")
            extract_frame(ctx.local_video_path, t, fp)
            frame_paths.append(fp)

        print(f"  [视觉] L{effective} 抽 {len(frame_paths)} 帧 @ {[f'{t:.0f}s' for t in times]} → VLM...", flush=True)

        # VLM 多图描述（⚠️ 帧将上传至 MiMo 服务器）
        desc, pin, pout = vlm_frames(frame_paths, question)

        # 结构化证据
        ctx.evidence = []
        for i, t in enumerate(times):
            ctx.evidence.append({
                "start_s": float(t),
                "end_s": float(t) + 1.0,
                "source": f"visual_l{effective[-1]}",
                "ref": frame_paths[i] if i < len(frame_paths) else None,
                "reason": f"L{effective} 抽帧 @ {t:.0f}s（意图 {intent_q}）",
                "confidence": None,
            })

        # 文本描述供 answer_questions 融合
        loc_tag = f"（定位窗 {','.join(located_windows)}）" if located_windows else ""
        note = f"\n## 视觉补充（L{effective}{loc_tag} 抽帧 {len(times)} 张 @ {[f'{t:.0f}s' for t in times]}）\n{desc}\n"
        ctx.avis["visual_notes"] = ctx.avis.get("visual_notes", []) + [note]
        ctx.avis["metadata"] = dict(ctx.avis.get("metadata", {}))
        ctx.avis["metadata"]["visual_tokens"] = {"in": pin, "out": pout}
        ctx.avis["metadata"]["locator_gaps"] = ctx.avis["metadata"].get("locator_gaps", []) + [loc_gap]

        print(f"  ✅ 视觉 {len(desc)} 字 | VLM {pin}+{pout} tok", flush=True)
        return True

    except Exception as e:
        ctx.add_warning("VISUAL_FAILED", f"视觉级失败: {str(e)}",
                        stage="visual_analysis", retryable=True)
        return True


def _track_active_windows(objects, dur, window_s=30, min_activity=2):
    """obj_tracks → 最活跃的 30s 窗口（L1 定位器 → L2 注意力引导）。"""
    if not objects or dur <= 0:
        return []
    import collections
    segs = collections.defaultdict(int)
    for o in objects:
        a, b = float(o.get("appear_t", 0)), float(o.get("disappear_t", 0))
        for s in range(int(a), min(int(b) + 1, int(dur))):
            segs[s] += 1
    if not segs:
        return []
    best, best_t = 0, 0
    for start in range(0, max(1, int(dur) - int(window_s) + 1)):
        w = sum(segs.get(s, 0) for s in range(start, start + int(window_s)))
        if w > best:
            best, best_t = w, start
    if best < min_activity:
        return []
    return [f"{best_t}-{min(best_t + int(window_s), int(dur))}"]


def _visual_prompt_for_intent(intent, question):
    """按问题意图生成针对性的 VLM 问题描述。"""
    base = "按时间顺序描述这些帧。\n⭐重要：优先读取画面中的文字/字幕/水印标注——它们可能是关键信息。"
    prompts = {
        "clothing": f"描述画面中人物的衣着：颜色/款式/材质/配饰（用户问题：{question}）。" + base,
        "color": f"描述画面中的主要颜色（用户问题：{question}）。" + base,
        "appearance": f"描述画面中人物/物体的外观：长相/形状/特征（用户问题：{question}）。" + base,
        "pose": f"描述画面中人物的姿态/动作/位置（用户问题：{question}）。" + base,
        "fine_visual_detail": f"仔细描述画面中的细节：品牌/型号/文字/标志（用户问题：{question}）。" + base,
        "object_presence": f"描述画面中出现的物体/人物及其位置（用户问题：{question}）。" + base,
        "object_count": f"数出画面中的人物/物体数量并描述（用户问题：{question}）。" + base,
        "motion_event": f"描述画面中的运动/动作：谁在动、怎么动、何时发生（用户问题：{question}）。" + base,
        "spatial_relation": f"描述画面中各对象的位置关系：谁在左边/右边/旁边/上方（用户问题：{question}）。" + base,
        "temporal_event": f"按时间顺序描述事件发生的时刻（用户问题：{question}）。" + base,
        "ocr_fact": f"逐帧读出画面中的文字/字幕/水印内容（用户问题：{question}）。" + base,
    }
    return prompts.get(intent, f"描述这些画面：有什么人/物、在做什么、什么颜色/姿态/衣着、场景如何？（用户问题：{question}）" + base)


def probe_dur(video):
    """探测视频时长（秒）。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=duration", "-of", "csv=p=0", video],
            capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip() or 0)
    except Exception:
        return 0.0


def _visual_queries(question, default=("person", "action", "woman")):
    """LLM 把中文问题转成 2-3 个英文视觉关键词（CLIP 语义检索用）。"""
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": "你是视觉检索词提取器。把用户问题转成 2-3 个英文视觉关键词"
                          "（CLIP 语义检索用，覆盖主要视觉元素/动作/场景/衣着）。只输出 JSON 数组。"},
                         {"role": "user", "content": f"用户问题: {question}\n"
                          "输出: [\"keyword1\", \"keyword2\", \"keyword3\"]"}],
            "max_tokens": 1024, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        r = json.loads(resp.read())
    d = _parse_json_obj(r["choices"][0]["message"]["content"])
    if isinstance(d, list):
        queries = d
    else:
        queries = list(d.values()) if isinstance(d, dict) else []
    queries = [q for q in (queries or []) if isinstance(q, str) and q.strip()][:3]
    return queries or list(default)


def _keyword_locate(question, transcript, dur, pad=20):
    """
    关键词定位兜底：从问题中提取关键词（去停用词），在转写中找包含关键词的段，
    取其时间窗（±pad 秒，相邻合并）。LLM 定位失败/空窗时使用，更稳。

    Args:
        question: 用户问题
        transcript: 转写段列表
        dur: 视频时长（秒）
        pad: 窗口扩展秒数

    Returns:
        windows: List[str] "a-b"
    """
    # 提取候选关键词：中文按 2-3 字滑动子串切分（避免整句吞掉），英文整词
    stop = {"什么", "怎么", "为什么", "如何", "哪些", "谁", "是不是", "有没有", "多少", "何时", "哪里",
            "扮演", "穿", "穿什么", "是", "在", "的", "了", "吗", "呢", "啊", "这个", "那个", "一个", "视频"}
    # ASR 误转别名：问题中的词 → 转写中可能出现的变体（提高命中）
    ALIASES = {
        "夏娃": ("夏娃", "下瓦", "旨女儿", "执女儿", "下娃"),
        "侄女": ("侄女", "旨女儿", "执女儿"),
    }
    tokens = set()
    for m in re.finditer(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,20}", question):
        chunk = m.group(0)
        if re.match(r"^[A-Za-z]+$", chunk):
            if chunk.lower() not in stop:
                tokens.add(chunk.lower())
            continue
        # 中文块：2-3 字滑动子串
        for n in (3, 2):
            for i in range(len(chunk) - n + 1):
                sub = chunk[i:i + n]
                if sub not in stop and not any(s in sub for s in ("视频", "电影", "里面")):
                    tokens.add(sub)
    # 别名展开
    expanded = set()
    for t in tokens:
        expanded.add(t)
        for alias_group in ALIASES.values():
            if t in alias_group:
                expanded.update(alias_group)
    tokens = expanded
    if not tokens:
        return []
    # 在转写中找命中段
    hits = []
    for seg in transcript:
        text = str(seg.get("text", ""))
        for t in tokens:
            if t in text:
                hits.append((float(seg.get("start", 0)), float(seg.get("end", 0))))
                break
    if not hits:
        return []
    # 窗口合并
    hits.sort()
    windows = []
    cur_a, cur_b = hits[0][0], hits[0][1]
    for a, b in hits[1:]:
        if a <= cur_b + pad:
            cur_b = max(cur_b, b)
        else:
            windows.append([max(0, cur_a - pad), min(dur, cur_b + pad)])
            cur_a, cur_b = a, b
    windows.append([max(0, cur_a - pad), min(dur, cur_b + pad)])
    return [f"{int(a)}-{int(b)}" for a, b in windows[:2]]


def _transcript_text(transcript, limit=500):
    """转写段 → 带时间戳文本（供定位器 / 回答 prompt 使用）。"""
    return "\n".join(
        f"[{s.get('start', 0):.0f}s] {str(s.get('text', '')).strip()}"
        for s in transcript[:limit] if str(s.get("text", "")).strip()
    )


def _parse_json_obj(text):
    """健壮 JSON 提取：容忍 markdown 围栏 / 前后杂质。"""
    if not text:
        return {}
    # 去 markdown 围栏
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text)
    try:
        return json.loads(text[text.index("{"):text.rindex("}") + 1])
    except Exception:
        pass
    # 退路：正则找 windows 数组 / gap
    try:
        m = re.search(r'"windows"\s*:\s*(\[[^\]]*\])', text)
        g = re.search(r'"gap"\s*:\s*"([^"]+)"', text)
        d = {}
        if m:
            d["windows"] = json.loads(m.group(1))
        if g:
            d["gap"] = g.group(1)
        return d
    except Exception:
        return {}


def _locate_windows(question, transcript, dur, title=""):
    """
    LLM 定位器：读转写全文 → 候选答案窗口 + 缺口类型。
    与旧路径 locate() 同款逻辑（问题驱动）。

    Args:
        question: 用户问题
        transcript: 转写段列表
        dur: 视频时长（秒）
        title: 视频标题（可选）

    Returns:
        (windows: List[str] "a-b", gap: "asr|visual|none", reason: str)
    """
    full = _transcript_text(transcript)
    if not full.strip():
        return [], "visual", "无转写内容，需视觉定位"
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
    content = r["choices"][0]["message"]["content"]
    d = _parse_json_obj(content)
    wins = d.get("windows") or []
    gap = d.get("gap") or "asr"
    print(f"  [定位] 窗口={wins} 缺口={gap} 原因={d.get('reason', '')}", flush=True)
    return [w for w in wins if isinstance(w, str)], gap, d.get("reason", "")


def _frames_in_windows(windows, dur, per_window=3):
    """定位窗口内均匀抽帧时间点（每窗 per_window 个）。"""
    times = []
    for w in windows[:2]:
        try:
            if "-" in w:
                a, b = (float(x) for x in w.split("-"))
            else:
                a, b = 0.0, min(float(w), dur)
            b = min(b, dur)
            if b <= a:
                continue
            for i in range(per_window):
                times.append(round(a + (b - a) * (i + 0.5) / per_window, 1))
        except Exception:
            continue
    return sorted(set(t for t in times if 0 <= t < dur))


def answer_questions(ctx: ProcessingContext) -> bool:
    """
    回答用户问题：融合 AVIS 信息层（转写 + 视觉帧描述）调用 LLM。

    Args:
        ctx: 处理上下文

    Returns:
        是否成功
    """
    if ctx.is_cancelled():
        ctx.add_error(ErrorCode.CANCELLED.value, "处理已取消", stage="llm")
        return False

    try:
        # 组装信息层 prompt
        parts = ["# 视频信息层（AVIS + 视觉补充）"]
        dur = ctx.video_metadata.get("duration_s", 0)
        parts.append(f"- 时长 {dur:.0f}s（{dur / 60:.1f}min）")
        video_type = ctx.video_metadata.get("video_type", "mixed")
        if video_type:
            parts.append(f"- 视频类型: {video_type}（classify 免费信号路由）")
        parts.append("")

        # 转写
        transcript = ctx.avis.get("transcript", [])
        if transcript:
            parts.append("## 说了什么（语音转写，时间戳）")
            for seg in transcript[:200]:
                t0 = seg.get("start", 0)
                txt = seg.get("text", "").strip()
                if txt:
                    parts.append(f"- [{t0:.1f}s] {txt}")
            if len(transcript) > 200:
                parts.append(f"- …（共 {len(transcript)} 段，已截前 200）")
            parts.append("")

        # ── L1 语义层：obj_tracks 对象轨迹（主对象时间线运动轨迹 + 少移动对象记录）──
        objects = ctx.avis.get("objects", [])
        if objects:
            parts.append("## 对象轨迹（obj_tracks：主对象时间线运动轨迹 + 少移动对象记录）")
            for o in objects[:15]:
                cls = o.get("class", "unknown")
                motion = o.get("motion", "")
                # 类别置信度低时提示"运动轨迹可信、类别不确定"
                cls_note = "" if o.get("class_confidence", 0) >= 0.5 else "（类别未确认，运动轨迹可信）"
                parts.append(
                    f"- 对象#{o.get('obj_id')} [{cls}{cls_note}] 出现 {o.get('appear_t')}s–{o.get('disappear_t')}s "
                    f"({o.get('duration_sec')}s) 运动:{motion} 速度:{o.get('speed_px_s')}px/s "
                    f"运动置信度:{o.get('motion_confidence', '?')}"
                )
            parts.append("")

        # 视觉描述
        visual_notes = ctx.avis.get("visual_notes", [])
        vis_text = "\n".join(visual_notes)

        sys_msg = ("你是视频内容分析助手。基于信息层（语音转写+场景结构+运动对象轨迹"
                   + ("+视觉帧描述" if vis_text else "") + "）回答。直接给答案，不要复述问题。"
                   "⭐重要规则：视觉描述中的画面文字标注（如'正片在XX:XX'、时间水印）是制作者给出的权威信息，"
                   "优先采信为答案，不要当作装饰水印忽略。")

        q_block = "\n\n".join(f"问题{i + 1}: {q}" for i, q in enumerate(ctx.questions))
        prompt = "\n".join(parts) + vis_text + "\n\n" + q_block + "\n\n请按 '问题N: 回答' 格式逐条回答。"

        print(f"🤖 回答 {len(ctx.questions)} 个问题（L{ctx.level}）...", flush=True)
        msg, usage = _llm([{"role": "system", "content": sys_msg},
                           {"role": "user", "content": prompt}], max_tokens=16384)

        # 解析 "问题N: 回答" 格式
        answers = []
        for i, q in enumerate(ctx.questions, 1):
            mm = re.search(rf"问题{i}\s*[:：]\s*(.*?)(?=问题{i + 1}\s*[:：]|\Z)", msg, re.S)
            answers.append({
                "question": q,
                "answer": mm.group(1).strip() if mm else f"(未能拆分) {msg[:200]}",
                "answer_status": "answered",
                "confidence": None,
            })

        ctx.avis["answers"] = answers
        ctx.avis["metadata"] = dict(ctx.avis.get("metadata", {}))
        ctx.avis["metadata"]["llm_usage"] = usage

        for i, (q, a) in enumerate(zip(ctx.questions, answers), 1):
            print(f"  ❓ Q{i} {q}\n  💬 {a['answer'][:150]}\n", flush=True)
        return True

    except Exception as e:
        ctx.add_error(ErrorCode.LLM_REQUEST_FAILED.value,
                      f"回答失败: {str(e)}", stage="llm", retryable=True)
        return False


def quality_check(question, answer):
    """
    LLM 自评：回答信息充分度 0-10 + 主要缺口（asr/visual/none/other）。
    与旧路径 quality_check() 同款逻辑。
    """
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
    d = _parse_json_obj(r["choices"][0]["message"]["content"])
    try:
        score = int(d.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    return score, d.get("gap", "other")


def retranscribe_window(ctx: ProcessingContext, window: str, model: str = "base") -> int:
    """
    局部 base 重转写：裁音频 → base 转写 → 时间戳偏移 → 替换 transcript 对应段。

    Args:
        ctx: 处理上下文
        window: 窗口 "a-b"（秒）
        model: 重转写模型（base 默认）

    Returns:
        新增段数（0 表示失败/无新增）
    """
    try:
        a, b = (int(x) for x in window.split("-"))
    except Exception:
        return 0
    if not ctx.local_video_path or not os.path.exists(ctx.local_video_path):
        return 0

    wav = os.path.join(ctx.work_dir, f"rt_{a}_{b}.wav")
    out = os.path.join(ctx.work_dir, f"rt_{a}_{b}.jsonl")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(a), "-to", str(b),
                    "-i", ctx.local_video_path, "-vn", "-ac", "1", "-ar", "16000", wav],
                   capture_output=True, timeout=120)

    asr_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "livestream-highlight", "asr.py")
    r = subprocess.run([sys.executable, asr_script, "--video", wav, "--out", out,
                        "--model", model, "--device", "auto"],
                       capture_output=True, timeout=600)
    if r.returncode != 0 or not os.path.exists(out):
        print(f"  ⚠️ 局部重转写失败: {r.stderr[-200:]}")
        return 0

    new_segs = []
    for line in open(out, encoding="utf-8"):
        if line.strip():
            s = json.loads(line)
            s["start"] = round(s["start"] + a, 2)
            s["end"] = round(s["end"] + a, 2)
            new_segs.append(s)

    # 替换窗口内的段（去重合并）
    old_segs = [s for s in ctx.avis.get("transcript", [])
                if not (a <= s.get("start", 0) <= b)]
    merged = old_segs + new_segs
    merged.sort(key=lambda s: s.get("start", 0))
    ctx.avis["transcript"] = merged

    print(f"  ✅ 局部重转写 {window}s [{model}]：{len(new_segs)} 段", flush=True)
    return len(new_segs)


def focused_answer_loop(ctx: ProcessingContext, max_rounds: int = 3) -> bool:
    """
    多轮聚焦回答循环（问题驱动路由）：

    1. 定位 → 视觉证据（L1/L2）→ LLM 回答
    2. 自评（quality_check）：最差答案 < 7 分时按缺口补动作再重答
       - visual 缺口 → 下一轮用定位窗口补更密抽帧
       - asr 缺口 → 局部 base 重转写替换 transcript
    3. 最多 max_rounds 轮；L0 单轮即返（无视觉）

    Args:
        ctx: 处理上下文
        max_rounds: 最多轮数（默认 3）

    Returns:
        是否成功
    """
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        if ctx.is_cancelled():
            ctx.add_error(ErrorCode.CANCELLED.value, "处理已取消", stage="focus_loop")
            return False

        # L0 无视觉，单轮问答即可
        if ctx.level == "l0":
            return answer_questions(ctx)

        # 视觉证据（内部含问题驱动定位；多轮时抽帧更密）
        if not select_visual_evidence(ctx, focus_round=rounds):
            return False

        # 回答
        if not answer_questions(ctx):
            return False

        answers = ctx.avis.get("answers", [])
        if not answers:
            return True

        # 自评：取最差题
        score, gap = 10, "none"
        for a in answers:
            try:
                s, g = quality_check(a.get("question", ""), a.get("answer", ""))
            except Exception as e:
                print(f"  ⚠️ 自评失败: {e}")
                s, g = 10, "none"
            if s < score:
                score, gap = s, g

        print(f"  [自评] 充分度 {score}/10 | 缺口 {gap} | 轮次 {rounds}/{max_rounds}", flush=True)
        if score >= 7:
            return True
        if gap == "other":
            # 无明确缺口但低分：尝试补一轮视觉（抽帧密度随轮次递增）
            # 最后一轮不再补救
            if rounds >= max_rounds:
                return True
            ctx.avis["_focus_round"] = rounds
            continue
        if gap not in ("visual", "asr"):
            return True
        if rounds >= max_rounds:
            return True

        # 补救动作
        if gap == "asr":
            wins = ctx.avis.get("_loc_windows", [])
            w = wins[0] if wins else f"0-{min(60, int(ctx.video_metadata.get('duration_s', 0) or 60))}"
            retranscribe_window(ctx, w, "base")
        # visual 缺口：不在此处理，下一轮 select_visual_evidence(focus_round=rounds+1)
        # 会以更密抽帧（per_window 随轮次递增）重新取证
        ctx.avis["_focus_round"] = rounds

    return True


def assemble_result(ctx: ProcessingContext) -> Dict:
    """
    组装结果
    
    Args:
        ctx: 处理上下文
        
    Returns:
        结果字典
    """
    from datetime import datetime
    
    # 计算耗时
    if ctx.started_at and ctx.finished_at:
        elapsed_ms = (ctx.finished_at - ctx.started_at).total_seconds() * 1000
    else:
        elapsed_ms = 0
    
    # ── token 统计与成本（与旧路径口径一致）──
    transcript = ctx.avis.get("transcript", [])
    asr_tok = sum(len(str(s.get("text", ""))) // 2 for s in transcript)
    n_evidence = len(ctx.evidence)
    info_tok = asr_tok + n_evidence * 60 + 150
    dur = float(ctx.video_metadata.get("duration_s", 0) or 0)
    orig_tok = int(dur * 30 * 1000)

    # 视觉 token（VLM）
    vis_tok = ctx.avis.get("metadata", {}).get("visual_tokens", {}) or {}
    vis_in, vis_out = vis_tok.get("in", 0), vis_tok.get("out", 0)
    # LLM token
    llm_usage = ctx.avis.get("metadata", {}).get("llm_usage", {}) or {}
    hit = llm_usage.get("prompt_cache_hit_tokens", 0) or 0
    miss = llm_usage.get("prompt_cache_miss_tokens", llm_usage.get("prompt_tokens", 0) - hit) or 0
    llm_out = llm_usage.get("completion_tokens", 0) or 0
    # 成本：文本 LLM v4-flash 空闲价 + VLM qwen3-vl-flash 近似
    cost_llm = hit / 1e6 * 0.05 + miss / 1e6 * 1.5 + llm_out / 1e6 * 4.5
    cost_visual = vis_in / 1e6 * 0.2 + vis_out / 1e6 * 0.7

    # ── 完整语义层状态（base+clip 懒加载提示）──
    # 有 answers（本次已产出内容）且尚未建完整层 → suggest_layer=True
    # （Node 端据此提示用户可建完整层，之后任何问题秒答、更准）
    has_answers = bool(ctx.avis.get("answers"))
    layer_cached = _has_full_layer_cache(ctx)
    suggest_layer = bool(has_answers and not layer_cached and ctx.level != "l0")

    # 路由分层信息（视频先验 + 问题意图 + 证据层次 + 升级路径）
    decision = ctx.avis.get("_routing", {})
    routing = {
        "video_type": ctx.video_metadata.get("video_type", "mixed"),
        "strategy": ctx.video_metadata.get("video_strategy", ""),
        "level": ctx.level,
        "obj_tracks": len(ctx.avis.get("objects", [])),
        "visual_notes": len(ctx.avis.get("visual_notes", [])),
        "cache_hit": ctx.cache_hit,
    }
    # 动态路由决策（问题意图驱动）
    if decision:
        routing.update({
            "question_intent": decision.get("intent"),
            "required_capability": decision.get("required_capability"),
            "initial_layer": decision.get("initial_layer"),
            "effective_layer": decision.get("effective_layer"),
            "upgrade_layer": decision.get("upgrade_layer"),
            "escalation_reason": decision.get("escalation_reason", []),
            "evidence_score": decision.get("evidence_score"),
            "evidence_sources": decision.get("evidence_sources", []),
            "missing_evidence": decision.get("missing_evidence", []),
            "frames_sent": len(ctx.evidence),
            "subtasks": decision.get("subtasks", []),
            "budget_blocked": decision.get("budget_blocked", False),
            "estimated_cost_cny": decision.get("estimated_cost_cny"),
            "budget_cny": decision.get("budget_cny"),
        })
    # 视频 profile（基础信号层）
    routing["video_profile"] = {
        "asr_coverage": ctx.video_metadata.get("asr_coverage", 0),
        "motion_score": ctx.video_metadata.get("motion_score", 0),
        "static_ratio": ctx.video_metadata.get("static_ratio", 0),
        "ocr_text_count": ctx.video_metadata.get("ocr_text_count", 0),
        "track_count": ctx.video_metadata.get("track_count", 0),
    }

    result = {
        "schema_version": "1",
        "video": {
            "source": ctx.video_metadata.get("source", "unknown"),
            "local_path": ctx.video_metadata.get("local_path"),
            "duration_s": ctx.video_metadata.get("duration_s", 0),
            "width": ctx.video_metadata.get("width", 0),
            "height": ctx.video_metadata.get("height", 0),
            "fps": ctx.video_metadata.get("fps", 0),
        },
        "duration_s": ctx.video_metadata.get("duration_s", 0),
        "processing": {
            "started_at": ctx.started_at.isoformat() if ctx.started_at else None,
            "finished_at": ctx.finished_at.isoformat() if ctx.finished_at else None,
            "elapsed_ms": elapsed_ms,
            "level": ctx.level,
            "privacy_mode": ctx.privacy_mode,
            "cache_hit": ctx.cache_hit,
        },
        "avis": {
            "transcript": ctx.avis.get("transcript", []),
            "scenes": ctx.avis.get("scenes", []),
            "motion": ctx.avis.get("motion", []),
            "objects": ctx.avis.get("objects", []),
            "visual_notes": ctx.avis.get("visual_notes", []),
            "metadata": ctx.avis.get("metadata", {}),
        },
        "answers": ctx.avis.get("answers", []),
        "warnings": ctx.warnings,
        "errors": ctx.errors,
        "routing": routing,
        # 兼容旧格式
        "elapsed_s": elapsed_ms / 1000,
        "info_tokens": info_tok,
        "orig_frame_tokens": orig_tok,
        "token_compression_pct": round(100 * (1 - info_tok / orig_tok), 2) if orig_tok else 0,
        "cost_cny": round(cost_llm + cost_visual, 5),
        "layer_cached": layer_cached,
        "suggest_layer": suggest_layer,
    }

    return result


def _has_full_layer_cache(ctx: ProcessingContext) -> bool:
    """检查完整语义层（base 全量 + CLIP）缓存是否已存在。
    兼容两种缓存布局：本引擎 cache_dir/layers/{hash}_base_clip 与旧路径 workdir/avis_cache/{hash}/base_clip。"""
    if not ctx.local_video_path or not os.path.exists(ctx.local_video_path):
        return False
    try:
        h = _video_content_hash(ctx.local_video_path)
    except Exception:
        return False
    candidates = []
    if ctx.cache_dir:
        candidates.append(os.path.join(ctx.cache_dir, "layers", f"{h}_base_clip"))
    if ctx.work_dir:
        candidates.append(os.path.join(ctx.work_dir, "avis_cache", h, "base_clip"))
    return any(os.path.exists(p) for p in candidates)


def _layer_dir(ctx: ProcessingContext):
    """完整语义层目录（含 avis.json 的目录；不存在时返回 None）。"""
    if not ctx.local_video_path or not os.path.exists(ctx.local_video_path):
        return None
    try:
        h = _video_content_hash(ctx.local_video_path)
    except Exception:
        return None
    for cand in (
        os.path.join(ctx.cache_dir or "", "layers", f"{h}_base_clip"),
        os.path.join(ctx.work_dir or "", "avis_cache", h, "base_clip"),
    ):
        if not os.path.exists(cand):
            continue
        # encode_video 输出为 {base}/video_avis/（含 avis.json）；直接命中则用根目录
        direct = os.path.join(cand, "avis.json")
        if os.path.exists(direct):
            return cand
        # 子目录 {stem}_avis/ 内含 avis.json
        for sub in sorted(os.listdir(cand)):
            sub_path = os.path.join(cand, sub)
            if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "avis.json")):
                return sub_path
    return None


def build_semantic_layer(ctx: ProcessingContext, force: bool = False) -> Optional[str]:
    """
    构建完整语义层（中间件）：base 全量转写 + CLIP 视觉索引 + 对象轨迹。
    输出到 cache_dir/layers/{hash}_base_clip/，之后任何问题直接查层回答。

    复用 avis.py 的 encode_video（ASR base + clip + obj_tracks），
    失败不阻断（返回 None 并加 warning）。

    Args:
        ctx: 处理上下文
        force: 强制重建（默认 False：已有层直接返回）

    Returns:
        语义层目录路径；失败返回 None
    """
    if not ctx.local_video_path or not os.path.exists(ctx.local_video_path):
        ctx.add_warning("LAYER_NO_VIDEO", "没有本地视频文件，无法建语义层", stage="layer")
        return None

    existing = _layer_dir(ctx)
    if existing and not force:
        print(f"  ♻️  [语义层] 完整层已存在（复用）: {existing}", flush=True)
        return existing

    try:
        h = _video_content_hash(ctx.local_video_path)
        out_root = os.path.join(ctx.cache_dir or "", "layers")
        os.makedirs(out_root, exist_ok=True)
        layer_out = os.path.join(out_root, f"{h}_base_clip")

        from .avis import encode_video
        # encode_video 输出为 {out_root}/{h}_base_clip/{stem}_avis → 把输出直接放到层目录下
        # 但 _has_full_layer_cache 检查 {h}_base_clip 目录本身存在即可
        t0 = time.time()
        print(f"  ⏳ [语义层] 构建 base 全量转写 + CLIP 索引（约 2-4min）...", flush=True)
        avis_dir = encode_video(
            Path(ctx.local_video_path),
            output_dir=Path(layer_out),
            asr_model="base",
            use_clip=True,
            use_obj_tracks=True,
            device="auto",
        )
        if avis_dir is None or not os.path.exists(avis_dir):
            ctx.add_warning("LAYER_BUILD_FAILED", "语义层构建失败（encode_video 返回空）", stage="layer")
            return None
        print(f"  ✅ [语义层] 构建完成（{time.time() - t0:.0f}s）→ {avis_dir}", flush=True)
        return str(avis_dir)
    except Exception as e:
        ctx.add_warning("LAYER_BUILD_FAILED", f"语义层构建失败: {str(e)}", stage="layer", retryable=True)
        print(f"  ⚠️ [语义层] 构建失败: {e}", flush=True)
        return None


def answer_from_layer(ctx: ProcessingContext) -> bool:
    """
    基于完整语义层（中间件）回答：直接查层找答案，不重跑理解管线。

    流程：
      1. 用 build_fused_prompt（ASR 全量 + 场景 + 对象轨迹）作为信息层
      2. CLIP 检索定位答案窗口（纯视觉问题）
      3. 定位窗口抽帧 VLM 视觉描述
      4. LLM 综合回答

    Args:
        ctx: 处理上下文

    Returns:
        是否成功（无层/失败时 False，由调用方决定是否降级）
    """
    layer = _layer_dir(ctx)
    if not layer:
        return False

    if ctx.is_cancelled():
        ctx.add_error(ErrorCode.CANCELLED.value, "处理已取消", stage="layer_answer")
        return False

    try:
        from .avis import build_fused_prompt, search_avis
        from .visual_level import extract_frame, vlm_frames

        dur = float(ctx.video_metadata.get("duration_s", 0) or 0)
        if dur <= 0:
            dur = float(probe_dur(ctx.local_video_path))

        # 1. 信息层 prompt（层内 ASR 全量 + 场景 + 对象轨迹）
        prompt = build_fused_prompt(Path(layer), with_tracks=True)
        print("  ♻️  [语义层] 命中完整层，直接查层回答（不重跑管线）", flush=True)

        # 1b. 层内 base 转写（比 tiny 更准，优先用于定位）
        layer_transcript = []
        try:
            tr_path = os.path.join(layer, "transcript.jsonl")
            if os.path.exists(tr_path):
                layer_transcript = [json.loads(l) for l in open(tr_path, encoding="utf-8") if l.strip()]
                if layer_transcript:
                    ctx.avis["transcript"] = layer_transcript
                    print(f"  [语义层] base 转写 {len(layer_transcript)} 段（复用层内 ASR）", flush=True)
        except Exception as e:
            print(f"  ⚠️ 层内转写加载失败: {e}", flush=True)

        # 合并 tiny（缓存）+ base（层内）转写用于定位：两份 ASR 各有盲区
        # （如 base 把"侄女"误转为"旨女儿"，tiny 则准确），合并可互补命中
        tiny_transcript = ctx.avis.get("transcript", [])
        if layer_transcript and tiny_transcript:
            merged_for_locate = layer_transcript + [
                s for s in tiny_transcript
                if str(s.get("text", "")).strip()
            ]
        else:
            merged_for_locate = layer_transcript or tiny_transcript

        # 2. 问题意图路由：决定是否需要抽帧 VLM（视觉问题才抽，纯文本直接层内回答）
        from .router import choose as route_choose
        first_q = (ctx.questions or [""])[0]
        decision = route_choose(
            video_profile=ctx.video_metadata,
            question=first_q,
            available={
                "asr": bool(layer_transcript or tiny_transcript),
                "ocr": ctx.video_metadata.get("ocr_text_count", 0) > 0,
                "obj_tracks": bool(ctx.avis.get("objects")),
                "scenes": bool(ctx.avis.get("scenes")),
                "visual_l1": False,
                "visual_l2": False,
            },
            level_hint=ctx.level if ctx.level in ("l0", "l1", "l2") else None,
            privacy_mode=ctx.privacy_mode,
            max_frames=6,
            budget_cny=getattr(ctx, "budget_cny", None),
        )
        ctx.avis["_routing"] = decision
        need_visual = decision["visual_required"] and decision["effective_layer"] == "l2" \
            and not decision.get("budget_blocked") and _privacy_allows_visual(ctx)
        print(f"  [语义层路由] 意图={decision['intent']} → 有效L{decision['effective_layer']} "
              f"| 需视觉={need_visual} | 理由: {'; '.join(decision['escalation_reason']) or '默认'}", flush=True)

        # 2b. 定位窗口（视觉问题才需要精确定位；文本问题直接用层内全文）
        windows, visual_notes = [], []
        if need_visual and first_q.strip():
            # 2b-1. LLM 定位（只用 base 转写：更准、无重复段干扰）
            locate_tr = layer_transcript or ctx.avis.get("transcript", [])
            try:
                located, gap, _ = _locate_windows(first_q, locate_tr, dur)
                windows = located
                if windows:
                    print(f"  [定位] LLM 定位命中: {windows}（缺口 {gap}）", flush=True)
            except Exception as e:
                print(f"  ⚠️ LLM 定位失败: {e}", flush=True)
            # 2b-2. 关键词兜底（合并 tiny+base 转写：两份 ASR 互补）
            if not windows:
                kw = _keyword_locate(first_q, merged_for_locate, dur)
                if kw:
                    windows = kw
                    print(f"  [定位] 关键词兜底命中: {kw}", flush=True)
            # 2b-3. CLIP 视觉检索补充（仅当转写定位失败且层内有 CLIP）
            if not windows:
                try:
                    clip_path = os.path.join(layer, "clip.npz")
                    if os.path.exists(clip_path):
                        queries = _visual_queries(first_q)
                        hits = []
                        for q in queries:
                            for ts, sc in search_avis(Path(layer), q, top_k=3):
                                hits.append((ts, sc))
                        if hits:
                            hits.sort()
                            wins = []
                            cur_a, cur_b = hits[0][0], hits[0][0]
                            for ts, _sc in hits[1:]:
                                if ts <= cur_b + 12:
                                    cur_b = max(cur_b, ts)
                                else:
                                    wins.append([max(0, cur_a - 12), min(dur, cur_b + 12)])
                                    cur_a, cur_b = ts, ts
                            wins.append([max(0, cur_a - 12), min(dur, cur_b + 12)])
                            windows = [f"{int(a)}-{int(b)}" for a, b in wins[:2]]
                            print(f"  [定位] CLIP 检索命中（{queries}）: {windows}", flush=True)
                except Exception as e:
                    print(f"  ⚠️ CLIP 检索失败（跳过）: {e}", flush=True)
            if not windows:
                kw = _keyword_locate(first_q, ctx.avis.get("transcript", []), dur)
                if kw:
                    windows = kw
                    print(f"  [定位] 关键词兜底命中: {kw}", flush=True)

        # 3. 定位窗口抽帧 VLM（仅当意图需要视觉且有窗口）
        if need_visual and windows:
            try:
                # 窗口尾段扩展 +5s：动作常发生在台词之后（如"举起斧子"在"举动吓人"台词后）
                extended = []
                for w in windows:
                    try:
                        a, b = (float(x) for x in w.split("-"))
                        extended.append(f"{int(a)}-{min(int(b) + 5, int(dur))}")
                    except Exception:
                        extended.append(w)
                times = _frames_in_windows(extended or windows, dur, per_window=3)
                frames_dir = ctx.create_work_dir("layer_frames")
                frame_paths = []
                for i, t in enumerate(times):
                    fp = str(frames_dir / f"lf{i:02d}_{int(t)}s.jpg")
                    extract_frame(ctx.local_video_path, t, fp)
                    frame_paths.append(fp)
                if frame_paths:
                    desc, pin, pout = vlm_frames(
                        frame_paths,
                        f"按时间顺序描述这些帧（用户问题聚焦：{first_q}）："
                        "每帧发生了什么、对象/动作/变化、颜色/衣着/文字标注？")
                    # 写回 ctx.avis：供结果输出与后续聚焦循环复用
                    note = f"\n## 视觉补充（语义层定位 {windows}）\n{desc}\n"
                    ctx.avis["visual_notes"] = ctx.avis.get("visual_notes", []) + [note]
                    visual_notes = [note]
                    # 结构化证据（供 routing.frames_sent 统计）
                    for i, t in enumerate(times):
                        ctx.evidence.append({
                            "start_s": float(t),
                            "end_s": float(t) + 1.0,
                            "source": "visual_l2",
                            "ref": frame_paths[i] if i < len(frame_paths) else None,
                            "reason": f"语义层 L2 抽帧 @ {t:.0f}s（意图 {decision['intent']}）",
                            "confidence": None,
                        })
                    ctx.avis["metadata"] = dict(ctx.avis.get("metadata", {}))
                    ctx.avis["metadata"]["visual_tokens"] = {"in": pin, "out": pout}
                    print(f"  ✅ 视觉 {len(desc)} 字 | VLM {pin}+{pout} tok", flush=True)
            except Exception as e:
                print(f"  ⚠️ 层内视觉补充失败: {e}", flush=True)
            except Exception as e:
                print(f"  ⚠️ 层内视觉补充失败: {e}", flush=True)

        # 4. LLM 综合回答
        vis_text = "\n".join(visual_notes)
        sys_msg = ("你是视频内容分析助手。基于完整语义层（语音转写全文+场景结构+对象轨迹"
                   + ("+视觉帧描述" if vis_text else "") + "）回答。直接给答案，不要复述问题。"
                   "⭐重要规则：视觉描述中的画面文字标注是制作者给出的权威信息，优先采信。")
        q_block = "\n\n".join(f"问题{i + 1}: {q}" for i, q in enumerate(ctx.questions))
        print(f"🤖 层内回答 {len(ctx.questions)} 个问题...", flush=True)
        msg, usage = _llm([{"role": "system", "content": sys_msg},
                           {"role": "user", "content": prompt + vis_text + "\n\n" + q_block +
                            "\n\n请按 '问题N: 回答' 格式逐条回答。"}], max_tokens=16384)

        answers = []
        for i, q in enumerate(ctx.questions, 1):
            mm = re.search(rf"问题{i}\s*[:：]\s*(.*?)(?=问题{i + 1}\s*[:：]|\Z)", msg, re.S)
            answers.append({
                "question": q,
                "answer": mm.group(1).strip() if mm else f"(未能拆分) {msg[:200]}",
                "answer_status": "answered",
                "confidence": None,
            })
        ctx.avis["answers"] = answers
        ctx.avis["metadata"] = dict(ctx.avis.get("metadata", {}))
        ctx.avis["metadata"]["llm_usage"] = usage
        ctx.avis["metadata"]["answered_from_layer"] = True
        return True
    except Exception as e:
        ctx.add_error(ErrorCode.LLM_REQUEST_FAILED.value,
                      f"层内回答失败: {str(e)}", stage="layer_answer", retryable=True)
        return False