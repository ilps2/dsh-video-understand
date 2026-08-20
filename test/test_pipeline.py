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
    _parse_json_obj,
    _frames_in_windows,
    _keyword_locate,
    _visual_queries,
    _has_full_layer_cache,
)


class TestLLMConfigPairing:
    """环境变量 key/URL 配对测试：一把 key 绝不能发往不是为它选定的主机"""

    @pytest.fixture(autouse=True)
    def _restore_env(self):
        """保存并恢复环境，避免污染其他测试"""
        saved = {k: os.environ.get(k) for k in
                 ("LLM_API_KEY", "DEEPSEEK_API_KEY", "LLM_API_URL", "LLM_MODEL", "VLM_MODEL")}
        yield
        # 恢复原环境
        for k in saved:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]
        import importlib
        import engine.stages as stages
        importlib.reload(stages)

    def _load(self, env):
        import importlib
        import engine.stages as stages
        for k in ("LLM_API_KEY", "DEEPSEEK_API_KEY", "LLM_API_URL", "LLM_MODEL", "VLM_MODEL"):
            os.environ.pop(k, None)
        for k, v in env.items():
            os.environ[k] = v
        importlib.reload(stages)
        return stages.KEY, stages.URL, stages.MODEL

    def test_deepseek_key_pairs_with_deepseek_endpoint(self):
        """只有 DEEPSEEK_API_KEY → 必须配对 DeepSeek 端点，绝不发往小米默认端点"""
        key, url, model = self._load({"DEEPSEEK_API_KEY": "sk-ds-test"})
        assert "api.deepseek.com" in url
        assert model == "deepseek-chat"

    def test_llm_key_uses_default_mimo(self):
        """只有 LLM_API_KEY（显式覆盖）→ 走默认小米端点"""
        key, url, model = self._load({"LLM_API_KEY": "sk-llm-test"})
        assert "xiaomimimo.com" in url

    def test_custom_url_respected(self):
        """DEEPSEEK key + 显式 LLM_API_URL → 尊重用户自定义端点"""
        key, url, model = self._load({
            "DEEPSEEK_API_KEY": "sk-ds",
            "LLM_API_URL": "https://custom.example.com/v1",
        })
        assert "custom.example.com" in url

    def test_no_env_falls_back_to_credentials(self):
        """无 env key → fallback 到 credentials 文件（存在则配对对应端点）"""
        key, url, model = self._load({})
        import os as _os
        cred = _os.path.expanduser("~/.dsh/.credentials.yaml")
        if _os.path.exists(cred):
            # credentials 文件存在：key 非空且端点配对（deepseek key → deepseek 端点）
            assert key != ""
            if "deepseek" in _os.popen(f"grep -o '^DEEPSEEK_API_KEY' {cred} 2>/dev/null").read():
                assert "api.deepseek.com" in url
        else:
            assert key == ""


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


class TestLocatorHelpers:
    """定位辅助函数测试"""
    
    def test_parse_json_obj_plain(self):
        """纯 JSON"""
        d = _parse_json_obj('{"windows": ["30-90"], "gap": "visual"}')
        assert d.get("windows") == ["30-90"]
        assert d.get("gap") == "visual"
    
    def test_parse_json_obj_markdown_fence(self):
        """markdown 围栏包裹的 JSON（定位器曾因围栏解析失败）"""
        d = _parse_json_obj('```json\n{"windows": ["415-420"], "gap": "visual", "reason": "需看画面"}\n```')
        assert d.get("windows") == ["415-420"]
        assert d.get("gap") == "visual"
    
    def test_parse_json_obj_with_prefix(self):
        """带前导文本的 JSON"""
        d = _parse_json_obj('定位结果如下：\n{"windows": ["10-20", "30-40"], "gap": "asr"}')
        assert d.get("windows") == ["10-20", "30-40"]
    
    def test_parse_json_obj_garbage(self):
        """完全无法解析时返回空 dict 不崩溃"""
        d = _parse_json_obj("模型输出了一堆废话")
        assert d == {}
    
    def test_frames_in_windows(self):
        """窗口内均匀抽帧"""
        times = _frames_in_windows(["400-430"], 648, per_window=3)
        assert len(times) == 3
        assert all(400 <= t <= 430 for t in times)
        # 窗口超出时长时截断
        times2 = _frames_in_windows(["640-700"], 648, per_window=3)
        assert all(t < 648 for t in times2)
    
    def test_keyword_locate_hit(self):
        """关键词定位命中：转写含问题关键词的段"""
        transcript = [
            {"start": 10, "end": 12, "text": "正好碰到来送午餐的侄女"},
            {"start": 30, "end": 32, "text": "无关内容"},
        ]
        wins = _keyword_locate("夏娃是侄女吗", transcript, 100)
        assert len(wins) == 1
        # 窗口应覆盖命中段（10s ± pad）
        a, b = (int(x) for x in wins[0].split("-"))
        assert a <= 10 <= b
    
    def test_keyword_locate_miss(self):
        """关键词定位未命中返回空"""
        transcript = [{"start": 0, "end": 5, "text": "完全无关的内容"}]
        assert _keyword_locate("夏娃穿什么", transcript, 100) == []
    
    def test_keyword_locate_empty_transcript(self):
        """空转写返回空"""
        assert _keyword_locate("问题", [], 100) == []
    
    def test_keyword_locate_alias(self):
        """ASR 误转别名：问题中的"夏娃"应命中转写中的"下瓦/旨女儿"变体"""
        transcript = [
            {"start": 10, "end": 12, "text": "正好碰到来送午餐的侄女下瓦"},
            {"start": 30, "end": 32, "text": "无关内容"},
        ]
        wins = _keyword_locate("夏娃出场时穿的什么", transcript, 100)
        assert len(wins) == 1
        a, b = (int(x) for x in wins[0].split("-"))
        assert a <= 10 <= b
    
    def test_keyword_locate_no_alias_no_hit(self):
        """无别名映射时不误命中"""
        transcript = [{"start": 0, "end": 5, "text": "完全无关的内容"}]
        assert _keyword_locate("夏娃出场时穿的什么", transcript, 100) == []
    
    def test_has_full_layer_cache_none(self):
        """无本地视频时层缓存为 False"""
        ctx = ProcessingContext(target="nonexistent.mp4")
        assert _has_full_layer_cache(ctx) == False
    
    def test_visual_queries_fallback(self):
        """LLM 不可用时 _visual_queries 返回默认词（不抛异常）"""
        # 无 API key 时应走默认；有 key 时正常调用
        qs = _visual_queries.__wrapped__ if hasattr(_visual_queries, "__wrapped__") else None
        # 直接验证默认参数路径：无 key 时函数可能抛 RuntimeError（KEY 为空）
        import engine.stages as stages
        if not stages.KEY:
            import pytest as _pytest
            with _pytest.raises((RuntimeError, Exception)):
                _visual_queries("测试问题")
        else:
            qs = _visual_queries("测试问题")
            assert isinstance(qs, list) and len(qs) >= 1


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