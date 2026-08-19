"""
真实 ASR 阶段测试
"""
import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.context import ProcessingContext
from engine.stages import transcribe, extract_audio, probe_media


# 测试视频路径
FIXTURES_DIR = Path(__file__).parent / "fixtures"
BLUE_3S_VIDEO = FIXTURES_DIR / "blue-3s.mp4"
RED_3S_VIDEO = FIXTURES_DIR / "red-3s-noaudio.mp4"


class TestRealASR:
    """真实 ASR 测试"""
    
    def test_transcribe_with_audio(self):
        """测试有音频的视频转写"""
        if not BLUE_3S_VIDEO.exists():
            pytest.skip("测试视频不存在")
        
        ctx = ProcessingContext(
            target=str(BLUE_3S_VIDEO),
            no_download=True,
        )
        ctx.local_video_path = str(BLUE_3S_VIDEO)
        
        # 探测媒体
        probe_media(ctx)
        
        # 提取音频
        extract_audio(ctx)
        
        # 转写
        result = transcribe(ctx)
        
        assert result == True
        assert ctx.transcript_path is not None
        assert os.path.exists(ctx.transcript_path)
    
    def test_transcribe_without_audio(self):
        """测试无音频的视频转写"""
        if not RED_3S_VIDEO.exists():
            pytest.skip("测试视频不存在")
        
        ctx = ProcessingContext(
            target=str(RED_3S_VIDEO),
            no_download=True,
        )
        ctx.local_video_path = str(RED_3S_VIDEO)
        
        # 探测媒体
        probe_media(ctx)
        
        # 提取音频（应该失败或为空）
        extract_audio(ctx)
        
        # 转写（应该处理无音频情况）
        result = transcribe(ctx)
        
        # 无音频时应该成功但 transcript 为空
        assert result == True
    
    def test_transcribe_output_format(self):
        """测试转写输出格式"""
        if not BLUE_3S_VIDEO.exists():
            pytest.skip("测试视频不存在")
        
        ctx = ProcessingContext(
            target=str(BLUE_3S_VIDEO),
            no_download=True,
        )
        ctx.local_video_path = str(BLUE_3S_VIDEO)
        
        # 探测媒体
        probe_media(ctx)
        
        # 提取音频
        extract_audio(ctx)
        
        # 转写
        transcribe(ctx)
        
        # 验证输出格式
        if ctx.transcript_path and os.path.exists(ctx.transcript_path):
            import json
            with open(ctx.transcript_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        seg = json.loads(line)
                        assert "start" in seg
                        assert "end" in seg
                        assert "text" in seg


class TestASRBackend:
    """ASR 后端测试"""
    
    def test_whisper_backend(self):
        """测试 whisper 后端"""
        if not BLUE_3S_VIDEO.exists():
            pytest.skip("测试视频不存在")
        
        ctx = ProcessingContext(
            target=str(BLUE_3S_VIDEO),
            no_download=True,
        )
        ctx.local_video_path = str(BLUE_3S_VIDEO)
        ctx.video_metadata["asr_backend"] = "whisper"
        ctx.video_metadata["asr_model"] = "tiny"
        
        # 探测媒体
        probe_media(ctx)
        
        # 提取音频
        extract_audio(ctx)
        
        # 转写
        result = transcribe(ctx)
        
        assert result == True