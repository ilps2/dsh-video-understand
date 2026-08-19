"""
视频理解 pipeline 阶段模块

定义各个处理阶段的函数。
"""
import os
import json
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .context import ProcessingContext
from .result_schema import ErrorCode


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


def transcribe(ctx: ProcessingContext) -> bool:
    """
    ASR 转写
    
    使用 faster-whisper 或 SenseVoice 进行转写。
    
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
        
        if asr_backend == "sensevoice":
            # 使用 SenseVoice
            try:
                from engine.asr_sensevoice import SenseVoiceASR
                asr = SenseVoiceASR(device=ctx.video_metadata.get("device", "auto"))
                segments = asr.transcribe(ctx.audio_path, ctx.transcript_path)
                ctx.avis["transcript"] = segments
                return True
            except Exception as e:
                print(f"  ⚠ SenseVoice ASR 失败: {e}")
                print("  回退到 faster-whisper...")
                asr_backend = "whisper"
        
        if asr_backend == "whisper":
            # 使用 faster-whisper
            from faster_whisper import WhisperModel
            
            model_size = ctx.video_metadata.get("asr_model", "tiny")
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
            
            # 写入输出文件
            with open(ctx.transcript_path, "w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(json.dumps(seg, ensure_ascii=False) + "\n")
            
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
    构建 AVIS 信息层
    
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
        
        # 构建 AVIS manifest
        manifest = {
            "avis_version": "0.1.0",
            "video": ctx.video_metadata,
            "signals": {
                "transcript": ctx.avis.get("transcript", []),
                "scenes": ctx.avis.get("scenes", []),
                "motion": ctx.avis.get("motion", []),
                "objects": ctx.avis.get("objects", []),
            }
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


def select_visual_evidence(ctx: ProcessingContext) -> bool:
    """
    选择视觉证据
    
    Args:
        ctx: 处理上下文
        
    Returns:
        是否成功
    """
    if ctx.is_cancelled():
        ctx.add_error(ErrorCode.CANCELLED.value, "处理已取消", stage="visual_analysis")
        return False
    
    # L0 不需要视觉证据
    if ctx.level == "l0":
        return True
    
    # L1/L2 需要视觉证据
    # 这里简化处理，实际应该实现帧选择逻辑
    ctx.evidence = []
    
    return True


def answer_questions(ctx: ProcessingContext) -> bool:
    """
    回答问题
    
    Args:
        ctx: 处理上下文
        
    Returns:
        是否成功
    """
    if ctx.is_cancelled():
        ctx.add_error(ErrorCode.CANCELLED.value, "处理已取消", stage="llm")
        return False
    
    # 这里简化处理，实际应该调用 LLM
    # 暂时返回占位答案
    ctx.avis["answers"] = []
    
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
            "metadata": ctx.avis.get("metadata", {}),
        },
        "answers": ctx.avis.get("answers", []),
        "warnings": ctx.warnings,
        "errors": ctx.errors,
        # 兼容旧格式
        "elapsed_s": elapsed_ms / 1000,
        "info_tokens": 0,
        "orig_frame_tokens": 0,
        "token_compression_pct": 0,
        "cost_cny": 0,
        "layer_cached": False,
        "suggest_layer": False,
    }
    
    return result