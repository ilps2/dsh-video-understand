"""
SenseVoice ASR Stage

使用阿里 SenseVoice 进行语音识别，替代 faster-whisper。
优势：中文优化、支持情感识别、推理速度快。
"""
import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

from ..context import ProcessingContext
from ..result_schema import ErrorCode


class SenseVoiceASR:
    """SenseVoice ASR 封装"""
    
    MODELS = {
        "tiny": "iic/SenseVoiceSmall",
        "base": "iic/SenseVoiceSmall",
        "small": "iic/SenseVoiceSmall",
    }
    
    def __init__(self, model_dir: str = "~/.cache/sensevoice", device: str = "auto"):
        """
        初始化 SenseVoice ASR
        
        Args:
            model_dir: 模型缓存目录
            device: 设备类型 (auto/cpu/cuda/mps)
        """
        self.model_dir = Path(model_dir).expanduser()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._auto_device() if device == "auto" else device
        self.model = None
        self._model_name = None
    
    def _auto_device(self) -> str:
        """自动检测最佳可用设备"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    def _ensure_model(self, model_name: str = "iic/SenseVoiceSmall"):
        """确保模型已加载"""
        if self.model is not None and self._model_name == model_name:
            return
        
        try:
            from funasr import AutoModel
            
            print(f"[SenseVoice] 加载模型: {model_name} (device: {self.device})")
            
            self.model = AutoModel(
                model=model_name,
                trust_remote_code=True,
                device=self.device if self.device != "mps" else "cpu",
            )
            self._model_name = model_name
            print(f"[SenseVoice] 模型加载完成")
            
        except ImportError:
            raise RuntimeError("funasr 未安装，请运行: pip install funasr modelscope")
        except Exception as e:
            raise RuntimeError(f"funasr 加载模型失败: {e}")
    
    def transcribe(self, audio_path: str, output_path: str, 
                   model: str = "tiny", lang: str = "zh") -> List[Dict]:
        """
        转写音频
        
        Args:
            audio_path: 音频文件路径
            output_path: 输出 jsonl 文件路径
            model: 模型大小
            lang: 语言代码
            
        Returns:
            转写结果列表
        """
        model_name = self.MODELS.get(model, "iic/SenseVoiceSmall")
        self._ensure_model(model_name)
        
        try:
            result = self.model.generate(
                input=audio_path,
                language=lang,
                use_itn=True,
            )
            
            # 转换为标准格式
            segments = []
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        text = item.get("text", "").strip()
                        timestamp = item.get("timestamp", [])
                        
                        if timestamp and len(timestamp) >= 2:
                            start = timestamp[0] / 1000.0
                            end = timestamp[-1] / 1000.0
                        else:
                            start = 0.0
                            end = 0.0
                        
                        segments.append({
                            "start": round(start, 2),
                            "end": round(end, 2),
                            "text": text,
                        })
            
            # 写入输出文件
            with open(output_path, "w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(json.dumps(seg, ensure_ascii=False) + "\n")
            
            return segments
            
        except Exception as e:
            raise RuntimeError(f"SenseVoice 转写失败: {e}")


def transcribe_with_sensevoice(ctx: ProcessingContext) -> bool:
    """
    使用 SenseVoice 进行转写
    
    Args:
        ctx: 处理上下文
        
    Returns:
        是否成功
    """
    if ctx.is_cancelled():
        ctx.add_error(ErrorCode.CANCELLED.value, "处理已取消", stage="transcribe")
        return False
    
    if not ctx.audio_path or not os.path.exists(ctx.audio_path):
        ctx.add_error(ErrorCode.ASR_FAILED.value, "没有音频文件", stage="transcribe")
        return False
    
    try:
        transcript_dir = ctx.create_work_dir("transcript")
        ctx.transcript_path = str(transcript_dir / "transcript.jsonl")
        
        # 使用 SenseVoice
        asr = SenseVoiceASR(device=ctx.video_metadata.get("device", "auto"))
        segments = asr.transcribe(
            ctx.audio_path,
            ctx.transcript_path,
            model=ctx.video_metadata.get("asr_model", "tiny"),
        )
        
        # 更新 AVIS
        ctx.avis["transcript"] = segments
        
        return True
        
    except Exception as e:
        ctx.add_error(ErrorCode.ASR_FAILED.value, 
                     f"SenseVoice 转写失败: {str(e)}", 
                     stage="transcribe")
        return False


def extract_audio_for_sensevoice(ctx: ProcessingContext) -> bool:
    """
    为 SenseVoice 提取音频（16kHz 单声道 WAV）
    
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
        ctx.audio_path = str(audio_dir / "audio_16k.wav")
        
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