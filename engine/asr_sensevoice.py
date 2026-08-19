"""
SenseVoice ASR 模块

使用阿里 SenseVoice 进行语音识别，替代 faster-whisper。
按照升级规划 v2.1 Task 1.2 实现。

优势：
- 专为中文优化，多语言支持
- 支持语音情感识别和音频事件检测
- 推理速度快（与 whisper base 相当）
- Apache 2.0 开源，可本地部署
"""
import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional


class SenseVoiceASR:
    """SenseVoice ASR 封装：替代 faster-whisper 作为默认 ASR"""
    
    MODELS = {
        "tiny": "iic/SenseVoiceSmall",
        "base": "iic/SenseVoiceSmall",
        "small": "iic/SenseVoiceSmall",
    }
    
    def __init__(self, 
                 model_dir: str = "~/.cache/sensevoice",
                 device: str = "auto"):
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
        """确保模型已下载并加载"""
        if self.model is not None and self._model_name == model_name:
            return
        
        try:
            from funasr import AutoModel
            
            print(f"[SenseVoice] 加载模型: {model_name} (device: {self.device})")
            
            # 创建 pipeline
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
        转写音频，输出与 faster-whisper 兼容的 jsonl 格式
        
        Args:
            audio_path: 音频文件路径 (wav/mp3)
            output_path: 输出 jsonl 文件路径
            model: 模型大小 (tiny/base/small)
            lang: 语言代码
            
        Returns:
            转写结果列表
        """
        # 获取模型名称
        model_name = self.MODELS.get(model, "iic/SenseVoiceSmall")
        
        # 确保模型已加载
        self._ensure_model(model_name)
        
        # 执行转写
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
                        
                        # 转换时间戳格式
                        if timestamp and len(timestamp) >= 2:
                            start = timestamp[0] / 1000.0  # 毫秒转秒
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
            
            print(f"[SenseVoice] 转写完成: {len(segments)} segments -> {output_path}")
            return segments
            
        except Exception as e:
            raise RuntimeError(f"SenseVoice 转写失败: {e}")
    
    def extract_audio(self, video_path: str, output_wav: str) -> bool:
        """
        从视频中提取音频并转换为 16kHz 单声道 WAV
        
        Args:
            video_path: 视频文件路径
            output_wav: 输出 WAV 文件路径
            
        Returns:
            是否成功
        """
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            output_wav
        ]
        
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                print(f"[SenseVoice] 音频提取失败: {r.stderr[:300]}")
                return False
            return os.path.exists(output_wav)
        except subprocess.TimeoutExpired:
            print("[SenseVoice] 音频提取超时")
            return False
        except Exception as e:
            print(f"[SenseVoice] 音频提取失败: {e}")
            return False
    
    def transcribe_video(self, video_path: str, output_path: str,
                         model: str = "tiny", lang: str = "zh") -> List[Dict]:
        """
        从视频提取音频并转写
        
        Args:
            video_path: 视频文件路径
            output_path: 输出 jsonl 文件路径
            model: 模型大小
            lang: 语言代码
            
        Returns:
            转写结果列表
        """
        # 提取音频
        video = Path(video_path)
        wav = video.with_suffix(".16k.wav")
        
        if not self.extract_audio(video_path, str(wav)):
            raise RuntimeError(f"音频提取失败: {video_path}")
        
        try:
            # 转写
            return self.transcribe(str(wav), output_path, model, lang)
        finally:
            # 清理临时音频文件
            if wav.exists():
                wav.unlink()


def transcribe_with_sensevoice(video_path: str, output_path: str,
                                model: str = "tiny", lang: str = "zh",
                                device: str = "auto") -> List[Dict]:
    """
    使用 SenseVoice 转写视频的便捷函数
    
    Args:
        video_path: 视频文件路径
        output_path: 输出 jsonl 文件路径
        model: 模型大小
        lang: 语言代码
        device: 设备类型
        
    Returns:
        转写结果列表
    """
    asr = SenseVoiceASR(device=device)
    return asr.transcribe_video(video_path, output_path, model, lang)