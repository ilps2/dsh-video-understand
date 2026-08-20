"""
视频理解处理上下文模块

定义 ProcessingContext，包含处理过程中的所有状态和数据。
"""
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ProcessingContext:
    """
    视频理解处理上下文
    
    包含处理过程中的所有状态和数据，用于阶段间传递信息。
    """
    # 输入参数
    target: str
    questions: List[str] = field(default_factory=list)
    no_download: bool = False
    level: str = "l0"
    window: Optional[str] = None
    privacy_mode: str = "remote_answer"
    max_rounds: int = 3
    build_layer: bool = False
    ask_layer: bool = False
    budget_cny: Optional[float] = None
    
    # 工作目录
    work_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    
    # 视频元数据
    video_metadata: Dict = field(default_factory=dict)
    
    # AVIS 数据
    avis: Dict = field(default_factory=dict)
    
    # 证据
    evidence: List[Dict] = field(default_factory=list)
    
    # 警告和错误
    warnings: List[Dict] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)
    
    # 取消事件
    cancel_event: threading.Event = field(default_factory=threading.Event)
    
    # 处理状态
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    cache_hit: bool = False
    
    # 中间结果
    local_video_path: Optional[str] = None
    audio_path: Optional[str] = None
    transcript_path: Optional[str] = None
    avis_dir: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.work_dir is None:
            self.work_dir = tempfile.mkdtemp(prefix="video_understand_")
        if self.cache_dir is None:
            cache_base = os.environ.get("VIDEO_UNDERSTAND_CACHE_DIR", 
                                        os.path.expanduser("~/.dsh/cache/video-understand"))
            self.cache_dir = cache_base
        
        # 设置默认问题
        if not self.questions:
            self.questions = [
                "这段视频的核心内容是什么？用 3-5 句话概括。",
                "视频中有哪些关键细节或亮点？",
                "这段视频适合什么场景/人群使用？",
            ]
    
    def create_work_dir(self, subdir: str = "") -> Path:
        """创建工作子目录"""
        path = Path(self.work_dir) / subdir if subdir else Path(self.work_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def add_warning(self, code: str, message: str, stage: Optional[str] = None,
                    retryable: bool = False):
        """添加警告"""
        self.warnings.append({
            "code": code,
            "message": message,
            "stage": stage,
            "retryable": retryable,
        })
    
    def add_error(self, code: str, message: str, stage: Optional[str] = None,
                  retryable: bool = False, details: Optional[Dict] = None):
        """添加错误"""
        error = {
            "code": code,
            "message": message,
            "stage": stage,
            "retryable": retryable,
        }
        if details:
            error["details"] = details
        self.errors.append(error)
    
    def is_cancelled(self) -> bool:
        """检查是否已取消"""
        return self.cancel_event.is_set()
    
    def cancel(self):
        """取消处理"""
        self.cancel_event.set()
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "target": self.target,
            "questions": self.questions,
            "no_download": self.no_download,
            "level": self.level,
            "window": self.window,
            "privacy_mode": self.privacy_mode,
            "video_metadata": self.video_metadata,
            "avis": self.avis,
            "evidence": self.evidence,
            "warnings": self.warnings,
            "errors": self.errors,
            "cache_hit": self.cache_hit,
        }


def create_context_from_request(request: Dict) -> ProcessingContext:
    """
    从请求创建处理上下文
    
    Args:
        request: 请求字典
        
    Returns:
        处理上下文
    """
    return ProcessingContext(
        target=request.get("target", ""),
        questions=request.get("questions", []),
        no_download=request.get("noDownload", False),
        level=request.get("level", "l0"),
        window=request.get("window"),
        privacy_mode=request.get("privacy_mode", "remote_answer"),
        max_rounds=int(request.get("max_rounds", 3) or 3),
        build_layer=bool(request.get("build_layer", False)),
        ask_layer=bool(request.get("ask_layer", False)),
        budget_cny=request.get("budget_cny"),
    )