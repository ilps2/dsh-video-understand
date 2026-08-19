"""
Schema 验证测试
"""
import pytest
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.result_schema import (
    validate_request,
    validate_result,
    create_error,
    create_warning,
    AnswerStatus,
    ErrorCode,
    PrivacyMode,
    Level,
)


class TestRequestValidation:
    """请求验证测试"""
    
    def test_valid_minimal_request(self):
        """测试合法最小请求"""
        request = {"target": "test.mp4"}
        errors = validate_request(request)
        assert errors == []
    
    def test_valid_full_request(self):
        """测试完整合法请求"""
        request = {
            "target": "BV1GJ411x7h7",
            "questions": ["讲了什么", "适合谁看"],
            "noDownload": False,
            "level": "l1",
            "window": "10-30",
            "privacy_mode": "remote_answer",
        }
        errors = validate_request(request)
        assert errors == []
    
    def test_missing_target(self):
        """测试缺少 target"""
        request = {"questions": ["test"]}
        errors = validate_request(request)
        assert len(errors) == 1
        assert "target" in errors[0]
    
    def test_empty_target(self):
        """测试空 target"""
        request = {"target": ""}
        errors = validate_request(request)
        assert len(errors) == 1
    
    def test_target_too_long(self):
        """测试 target 过长"""
        request = {"target": "a" * 2001}
        errors = validate_request(request)
        assert len(errors) == 1
        assert "2000" in errors[0]
    
    def test_invalid_level(self):
        """测试无效 level"""
        request = {"target": "test.mp4", "level": "l3"}
        errors = validate_request(request)
        assert len(errors) == 1
        assert "level" in errors[0]
    
    def test_invalid_window_format(self):
        """测试无效 window 格式"""
        request = {"target": "test.mp4", "window": "abc"}
        errors = validate_request(request)
        assert len(errors) == 1
        assert "window" in errors[0]
    
    def test_window_start_ge_end(self):
        """测试 window 开始 >= 结束"""
        request = {"target": "test.mp4", "window": "30-10"}
        errors = validate_request(request)
        assert len(errors) == 1
        assert "开始" in errors[0] or "结束" in errors[0]
    
    def test_too_many_questions(self):
        """测试问题过多"""
        request = {"target": "test.mp4", "questions": ["q"] * 21}
        errors = validate_request(request)
        assert len(errors) == 1
        assert "20" in errors[0]
    
    def test_question_too_long(self):
        """测试问题过长"""
        request = {"target": "test.mp4", "questions": ["a" * 2001]}
        errors = validate_request(request)
        assert len(errors) == 1
    
    def test_invalid_privacy_mode(self):
        """测试无效 privacy_mode"""
        request = {"target": "test.mp4", "privacy_mode": "invalid"}
        errors = validate_request(request)
        assert len(errors) == 1
        assert "privacy_mode" in errors[0]


class TestResultValidation:
    """结果验证测试"""
    
    def test_valid_minimal_result(self):
        """测试合法最小结果"""
        result = {
            "schema_version": "1",
            "video": {"source": "test.mp4"},
            "duration_s": 10,
            "processing": {},
            "avis": {},
            "answers": [],
            "warnings": [],
            "errors": [],
        }
        errors = validate_result(result)
        assert errors == []
    
    def test_missing_required_field(self):
        """测试缺少必填字段"""
        result = {"schema_version": "1"}
        errors = validate_result(result)
        assert len(errors) >= 1
        assert any("video" in e for e in errors)
    
    def test_invalid_schema_version(self):
        """测试无效 schema_version"""
        result = {
            "schema_version": "2",
            "video": {},
            "duration_s": 10,
            "processing": {},
            "avis": {},
            "answers": [],
            "warnings": [],
            "errors": [],
        }
        errors = validate_result(result)
        assert len(errors) == 1
        assert "schema_version" in errors[0]
    
    def test_invalid_answer_status(self):
        """测试无效 answer_status"""
        result = {
            "schema_version": "1",
            "video": {},
            "duration_s": 10,
            "processing": {},
            "avis": {},
            "answers": [
                {"question": "test", "answer": "test", "answer_status": "invalid"}
            ],
            "warnings": [],
            "errors": [],
        }
        errors = validate_result(result)
        assert len(errors) == 1
        assert "answer_status" in errors[0]
    
    def test_confidence_out_of_range(self):
        """测试 confidence 越界"""
        result = {
            "schema_version": "1",
            "video": {},
            "duration_s": 10,
            "processing": {},
            "avis": {},
            "answers": [
                {"question": "test", "answer": "test", "answer_status": "answered", "confidence": 1.5}
            ],
            "warnings": [],
            "errors": [],
        }
        errors = validate_result(result)
        assert len(errors) == 1
        assert "confidence" in errors[0]


class TestErrorCreation:
    """错误创建测试"""
    
    def test_create_error(self):
        """测试创建错误"""
        error = create_error(
            ErrorCode.INVALID_REQUEST,
            "Invalid request",
            stage="request",
            retryable=False,
        )
        assert error["code"] == "INVALID_REQUEST"
        assert error["message"] == "Invalid request"
        assert error["stage"] == "request"
        assert error["retryable"] == False
    
    def test_create_warning(self):
        """测试创建警告"""
        warning = create_warning(
            "DEPRECATED",
            "This feature is deprecated",
            stage="runtime",
            retryable=True,
        )
        assert warning["code"] == "DEPRECATED"
        assert warning["message"] == "This feature is deprecated"
        assert warning["stage"] == "runtime"
        assert warning["retryable"] == True


class TestEnums:
    """枚举测试"""
    
    def test_answer_status(self):
        """测试 AnswerStatus 枚举"""
        assert AnswerStatus.ANSWERED.value == "answered"
        assert AnswerStatus.PARTIALLY_ANSWERED.value == "partially_answered"
        assert AnswerStatus.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"
        assert AnswerStatus.UNSUPPORTED_QUESTION.value == "unsupported_question"
        assert AnswerStatus.FAILED.value == "failed"
    
    def test_error_code(self):
        """测试 ErrorCode 枚举"""
        assert ErrorCode.INVALID_REQUEST.value == "INVALID_REQUEST"
        assert ErrorCode.TARGET_NOT_FOUND.value == "TARGET_NOT_FOUND"
        assert ErrorCode.LLM_TIMEOUT.value == "LLM_TIMEOUT"
    
    def test_privacy_mode(self):
        """测试 PrivacyMode 枚举"""
        assert PrivacyMode.LOCAL_EXTRACT.value == "local_extract"
        assert PrivacyMode.REMOTE_ANSWER.value == "remote_answer"
        assert PrivacyMode.REMOTE_VISUAL.value == "remote_visual"
        assert PrivacyMode.FULLY_LOCAL.value == "fully_local"
    
    def test_level(self):
        """测试 Level 枚举"""
        assert Level.L0.value == "l0"
        assert Level.L1.value == "l1"
        assert Level.L2.value == "l2"