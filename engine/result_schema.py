"""
视频理解结果 schema 验证模块

定义请求/结果/错误 schema，提供验证函数。
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class AnswerStatus(str, Enum):
    """答案状态枚举"""
    ANSWERED = "answered"
    PARTIALLY_ANSWERED = "partially_answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED_QUESTION = "unsupported_question"
    FAILED = "failed"


class ErrorCode(str, Enum):
    """错误码枚举"""
    INVALID_REQUEST = "INVALID_REQUEST"
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    TARGET_UNSUPPORTED = "TARGET_UNSUPPORTED"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    MEDIA_PROBE_FAILED = "MEDIA_PROBE_FAILED"
    FFMPEG_FAILED = "FFMPEG_FAILED"
    ASR_FAILED = "ASR_FAILED"
    AVIS_FAILED = "AVIS_FAILED"
    VISUAL_ANALYSIS_FAILED = "VISUAL_ANALYSIS_FAILED"
    LLM_AUTH_MISSING = "LLM_AUTH_MISSING"
    LLM_REQUEST_FAILED = "LLM_REQUEST_FAILED"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    CACHE_READ_FAILED = "CACHE_READ_FAILED"
    CACHE_WRITE_FAILED = "CACHE_WRITE_FAILED"
    CANCELLED = "CANCELLED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class PrivacyMode(str, Enum):
    """隐私模式枚举"""
    LOCAL_EXTRACT = "local_extract"
    REMOTE_ANSWER = "remote_answer"
    REMOTE_VISUAL = "remote_visual"
    FULLY_LOCAL = "fully_local"


class Level(str, Enum):
    """理解级别枚举"""
    L0 = "l0"
    L1 = "l1"
    L2 = "l2"


@dataclass
class Evidence:
    """证据"""
    start_s: float
    end_s: float
    source: str  # asr, scene, motion, object, visual_l1, visual_l2
    ref: Optional[str] = None
    reason: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class Answer:
    """答案"""
    question: str
    answer: str
    answer_status: AnswerStatus
    confidence: Optional[float] = None
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class Warning:
    """警告"""
    code: str
    message: str
    stage: Optional[str] = None
    retryable: bool = False


@dataclass
class Error:
    """错误"""
    code: ErrorCode
    message: str
    stage: Optional[str] = None
    retryable: bool = False
    details: Optional[Dict[str, Any]] = None


@dataclass
class VideoInfo:
    """视频信息"""
    source: str
    local_path: Optional[str] = None
    duration_s: float = 0
    width: int = 0
    height: int = 0
    fps: float = 0
    sha256: Optional[str] = None


@dataclass
class ProcessingInfo:
    """处理信息"""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    elapsed_ms: float = 0
    level: Level = Level.L0
    privacy_mode: PrivacyMode = PrivacyMode.REMOTE_ANSWER
    cache_hit: bool = False


@dataclass
class AVISInfo:
    """AVIS 信息"""
    transcript: List[Dict] = field(default_factory=list)
    scenes: List[Dict] = field(default_factory=list)
    motion: List[Dict] = field(default_factory=list)
    objects: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class VideoUnderstandResult:
    """视频理解结果"""
    schema_version: str = "1"
    video: VideoInfo = field(default_factory=VideoInfo)
    duration_s: float = 0
    processing: ProcessingInfo = field(default_factory=ProcessingInfo)
    avis: AVISInfo = field(default_factory=AVISInfo)
    answers: List[Answer] = field(default_factory=list)
    warnings: List[Warning] = field(default_factory=list)
    errors: List[Error] = field(default_factory=list)
    
    # 兼容旧格式的字段
    video_info: Optional[Dict] = None
    elapsed_s: float = 0
    info_tokens: int = 0
    orig_frame_tokens: int = 0
    token_compression_pct: float = 0
    cost_cny: float = 0
    layer_cached: bool = False
    suggest_layer: bool = False


def validate_request(request: Dict) -> List[str]:
    """
    验证请求
    
    Args:
        request: 请求字典
        
    Returns:
        错误消息列表，空列表表示验证通过
    """
    errors = []
    
    # 检查必填字段
    if "target" not in request:
        errors.append("缺少必填字段: target")
    elif not isinstance(request["target"], str) or len(request["target"]) == 0:
        errors.append("target 必须是非空字符串")
    elif len(request["target"]) > 2000:
        errors.append("target 长度不能超过 2000")
    
    # 检查 questions
    if "questions" in request:
        questions = request["questions"]
        if not isinstance(questions, list):
            errors.append("questions 必须是数组")
        elif len(questions) > 20:
            errors.append("questions 不能超过 20 个")
        else:
            for i, q in enumerate(questions):
                if not isinstance(q, str):
                    errors.append(f"questions[{i}] 必须是字符串")
                elif len(q) > 2000:
                    errors.append(f"questions[{i}] 长度不能超过 2000")
    
    # 检查 level
    if "level" in request:
        level = request["level"]
        if level not in ["l0", "l1", "l2"]:
            errors.append(f"level 必须是 l0, l1, l2 之一，收到: {level}")
    
    # 检查 window
    if "window" in request:
        window = request["window"]
        if not isinstance(window, str):
            errors.append("window 必须是字符串")
        elif window != "auto":
            # 验证窗口格式
            import re
            if not re.match(r"^(\d+)-(\d+)$", window) and not re.match(r"^\d+$", window):
                errors.append(f"window 格式无效: {window}")
            elif re.match(r"^(\d+)-(\d+)$", window):
                start, end = map(int, window.split("-"))
                if start >= end:
                    errors.append(f"window 开始必须小于结束: {window}")
    
    # 检查 privacy_mode
    if "privacy_mode" in request:
        mode = request["privacy_mode"]
        if mode not in [m.value for m in PrivacyMode]:
            errors.append(f"privacy_mode 无效: {mode}")
    
    return errors


def validate_result(result: Dict) -> List[str]:
    """
    验证结果
    
    Args:
        result: 结果字典
        
    Returns:
        错误消息列表，空列表表示验证通过
    """
    errors = []
    
    # 检查必填字段
    required_fields = ["schema_version", "video", "duration_s", "processing", "avis", "answers", "warnings", "errors"]
    for field in required_fields:
        if field not in result:
            errors.append(f"缺少必填字段: {field}")
    
    # 检查 schema_version
    if "schema_version" in result and result["schema_version"] != "1":
        errors.append(f"schema_version 必须是 '1'，收到: {result['schema_version']}")
    
    # 检查 answers
    if "answers" in result:
        if not isinstance(result["answers"], list):
            errors.append("answers 必须是数组")
        else:
            for i, answer in enumerate(result["answers"]):
                if not isinstance(answer, dict):
                    errors.append(f"answers[{i}] 必须是对象")
                else:
                    if "question" not in answer:
                        errors.append(f"answers[{i}] 缺少 question")
                    if "answer" not in answer:
                        errors.append(f"answers[{i}] 缺少 answer")
                    if "answer_status" not in answer:
                        errors.append(f"answers[{i}] 缺少 answer_status")
                    elif answer["answer_status"] not in [s.value for s in AnswerStatus]:
                        errors.append(f"answers[{i}] answer_status 无效: {answer['answer_status']}")
                    
                    # 检查 confidence
                    if "confidence" in answer:
                        conf = answer["confidence"]
                        if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                            errors.append(f"answers[{i}] confidence 必须是 0-1 之间的数字")
    
    # 检查 warnings 和 errors
    for field_name in ["warnings", "errors"]:
        if field_name in result:
            if not isinstance(result[field_name], list):
                errors.append(f"{field_name} 必须是数组")
    
    return errors


def create_error(code: ErrorCode, message: str, stage: Optional[str] = None, 
                 retryable: bool = False, details: Optional[Dict] = None) -> Dict:
    """
    创建错误对象
    
    Args:
        code: 错误码
        message: 错误消息
        stage: 错误阶段
        retryable: 是否可重试
        details: 错误详情
        
    Returns:
        错误字典
    """
    error = {
        "code": code.value,
        "message": message,
        "retryable": retryable,
    }
    if stage:
        error["stage"] = stage
    if details:
        error["details"] = details
    return error


def create_warning(code: str, message: str, stage: Optional[str] = None,
                   retryable: bool = False) -> Dict:
    """
    创建警告对象
    
    Args:
        code: 警告码
        message: 警告消息
        stage: 警告阶段
        retryable: 是否可重试
        
    Returns:
        警告字典
    """
    warning = {
        "code": code,
        "message": message,
        "retryable": retryable,
    }
    if stage:
        warning["stage"] = stage
    return warning


def result_to_dict(result: VideoUnderstandResult) -> Dict:
    """
    将结果转换为字典
    
    Args:
        result: 结果对象
        
    Returns:
        结果字典
    """
    return {
        "schema_version": result.schema_version,
        "video": {
            "source": result.video.source,
            "local_path": result.video.local_path,
            "duration_s": result.video.duration_s,
            "width": result.video.width,
            "height": result.video.height,
            "fps": result.video.fps,
            "sha256": result.video.sha256,
        },
        "duration_s": result.duration_s,
        "processing": {
            "started_at": result.processing.started_at,
            "finished_at": result.processing.finished_at,
            "elapsed_ms": result.processing.elapsed_ms,
            "level": result.processing.level.value,
            "privacy_mode": result.processing.privacy_mode.value,
            "cache_hit": result.processing.cache_hit,
        },
        "avis": {
            "transcript": result.avis.transcript,
            "scenes": result.avis.scenes,
            "motion": result.avis.motion,
            "objects": result.avis.objects,
            "metadata": result.avis.metadata,
        },
        "answers": [
            {
                "question": a.question,
                "answer": a.answer,
                "confidence": a.confidence,
                "evidence": [
                    {
                        "start_s": e.start_s,
                        "end_s": e.end_s,
                        "source": e.source,
                        "ref": e.ref,
                        "reason": e.reason,
                        "confidence": e.confidence,
                    }
                    for e in a.evidence
                ],
                "answer_status": a.answer_status.value,
            }
            for a in result.answers
        ],
        "warnings": result.warnings,
        "errors": result.errors,
        # 兼容旧格式
        "video_info": result.video_info,
        "elapsed_s": result.elapsed_s,
        "info_tokens": result.info_tokens,
        "orig_frame_tokens": result.orig_frame_tokens,
        "token_compression_pct": result.token_compression_pct,
        "cost_cny": result.cost_cny,
        "layer_cached": result.layer_cached,
        "suggest_layer": result.suggest_layer,
    }