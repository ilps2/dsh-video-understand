"""
Pipeline 测试
"""
import pytest
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.context import ProcessingContext, create_context_from_request
from engine.pipeline import run_pipeline
from engine.stages import (
    resolve_target,
    probe_media,
    extract_audio,
    transcribe,
    build_avis,
    select_visual_evidence,
    answer_questions,
    assemble_result,
)


class TestContext:
    """上下文测试"""
    
    def test_create_context(self):
        """测试创建上下文"""
        ctx = ProcessingContext(target="test.mp4")
        assert ctx.target == "test.mp4"
        assert ctx.level == "l0"
        assert ctx.privacy_mode == "remote_answer"
    
    def test_create_from_request(self):
        """测试从请求创建上下文"""
        request = {
            "target": "test.mp4",
            "questions": ["test question"],
            "level": "l1",
        }
        ctx = create_context_from_request(request)
        assert ctx.target == "test.mp4"
        assert ctx.questions == ["test question"]
        assert ctx.level == "l1"
    
    def test_add_warning(self):
        """测试添加警告"""
        ctx = ProcessingContext(target="test.mp4")
        ctx.add_warning("TEST", "Test warning", stage="test")
        assert len(ctx.warnings) == 1
        assert ctx.warnings[0]["code"] == "TEST"
    
    def test_add_error(self):
        """测试添加错误"""
        ctx = ProcessingContext(target="test.mp4")
        ctx.add_error("TEST", "Test error", stage="test")
        assert len(ctx.errors) == 1
        assert ctx.errors[0]["code"] == "TEST"
    
    def test_cancel(self):
        """测试取消"""
        ctx = ProcessingContext(target="test.mp4")
        assert not ctx.is_cancelled()
        ctx.cancel()
        assert ctx.is_cancelled()


class TestStages:
    """阶段测试"""
    
    def test_resolve_target_local(self):
        """测试解析本地目标"""
        # 使用测试视频
        test_video = Path(__file__).parent / "fixtures" / "blue-3s.mp4"
        if test_video.exists():
            ctx = ProcessingContext(target=str(test_video))
            result = resolve_target(ctx)
            assert result == True
            assert ctx.local_video_path is not None
    
    def test_resolve_target_not_found(self):
        """测试解析不存在的目标"""
        ctx = ProcessingContext(target="nonexistent.mp4")
        result = resolve_target(ctx)
        assert result == False
        assert len(ctx.errors) > 0
    
    def test_probe_media(self):
        """测试探测媒体"""
        test_video = Path(__file__).parent / "fixtures" / "blue-3s.mp4"
        if test_video.exists():
            ctx = ProcessingContext(target=str(test_video))
            ctx.local_video_path = str(test_video)
            result = probe_media(ctx)
            assert result == True
            assert "duration_s" in ctx.video_metadata
    
    def test_probe_media_no_file(self):
        """测试探测没有文件"""
        ctx = ProcessingContext(target="test.mp4")
        result = probe_media(ctx)
        assert result == False
    
    def test_extract_audio(self):
        """测试提取音频"""
        test_video = Path(__file__).parent / "fixtures" / "blue-3s.mp4"
        if test_video.exists():
            ctx = ProcessingContext(target=str(test_video))
            ctx.local_video_path = str(test_video)
            result = extract_audio(ctx)
            assert result == True
            assert ctx.audio_path is not None
    
    def test_select_visual_evidence_l0(self):
        """测试 L0 不需要视觉证据"""
        ctx = ProcessingContext(target="test.mp4", level="l0")
        result = select_visual_evidence(ctx)
        assert result == True
    
    def test_select_visual_evidence_l1(self):
        """测试 L1 需要视觉证据"""
        ctx = ProcessingContext(target="test.mp4", level="l1")
        result = select_visual_evidence(ctx)
        assert result == True


class TestPipeline:
    """Pipeline 测试"""
    
    def test_run_pipeline_local_video(self):
        """测试运行本地视频 pipeline"""
        test_video = Path(__file__).parent / "fixtures" / "blue-3s.mp4"
        if test_video.exists():
            ctx = ProcessingContext(target=str(test_video))
            result = run_pipeline(ctx)
            
            assert "schema_version" in result
            assert result["schema_version"] == "1"
            assert "video" in result
            assert "processing" in result
            assert "answers" in result
    
    def test_run_pipeline_not_found(self):
        """测试运行不存在的视频"""
        ctx = ProcessingContext(target="nonexistent.mp4")
        result = run_pipeline(ctx)
        
        assert "errors" in result
        assert len(result["errors"]) > 0


class TestAssembleResult:
    """组装结果测试"""
    
    def test_assemble_result(self):
        """测试组装结果"""
        ctx = ProcessingContext(target="test.mp4")
        ctx.video_metadata = {
            "source": "local",
            "duration_s": 10.0,
        }
        result = assemble_result(ctx)
        
        assert result["schema_version"] == "1"
        assert result["video"]["source"] == "local"
        assert result["duration_s"] == 10.0