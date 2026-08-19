"""
隐私模式和数据流控制模块

实现四种隐私模式，控制视频数据流向。
按照升级规划 v2.1 Task 5 实现。
"""
import os
from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass


class PrivacyMode(str, Enum):
    """隐私模式枚举"""
    LOCAL_EXTRACT = "local_extract"      # 本地抽取 AVIS，禁止远程 LLM/VLM
    REMOTE_ANSWER = "remote_answer"      # 原视频不上传，只发送 transcript/AVIS/问题
    REMOTE_VISUAL = "remote_visual"      # 只发送选定 JPEG 帧，禁止整段视频
    FULLY_LOCAL = "fully_local"          # 全本地；本地模型不可用时失败


@dataclass
class DataFlow:
    """数据流向记录"""
    video_uploaded: bool = False
    audio_uploaded: bool = False
    transcript_uploaded: bool = False
    avis_uploaded: bool = False
    frames_uploaded: bool = False
    provider: Optional[str] = None


class PrivacyController:
    """隐私控制器"""
    
    def __init__(self, mode: PrivacyMode = PrivacyMode.REMOTE_ANSWER):
        """
        初始化隐私控制器
        
        Args:
            mode: 隐私模式
        """
        self.mode = mode
        self.data_flow = DataFlow()
    
    def can_use_remote_llm(self) -> bool:
        """
        是否可以使用远程 LLM
        
        Returns:
            是否允许
        """
        if self.mode == PrivacyMode.LOCAL_EXTRACT:
            return False
        if self.mode == PrivacyMode.FULLY_LOCAL:
            return False
        return True
    
    def can_use_remote_vlm(self) -> bool:
        """
        是否可以使用远程 VLM
        
        Returns:
            是否允许
        """
        if self.mode == PrivacyMode.LOCAL_EXTRACT:
            return False
        if self.mode == PrivacyMode.FULLY_LOCAL:
            return False
        if self.mode == PrivacyMode.REMOTE_ANSWER:
            return False
        return True
    
    def can_upload_video(self) -> bool:
        """
        是否可以上传原始视频
        
        Returns:
            是否允许
        """
        return False  # 任何模式都不允许上传原始视频
    
    def can_upload_frames(self) -> bool:
        """
        是否可以上传帧图片
        
        Returns:
            是否允许
        """
        if self.mode == PrivacyMode.LOCAL_EXTRACT:
            return False
        if self.mode == PrivacyMode.FULLY_LOCAL:
            return False
        if self.mode == PrivacyMode.REMOTE_ANSWER:
            return False
        return True
    
    def can_upload_transcript(self) -> bool:
        """
        是否可以上传转写文本
        
        Returns:
            是否允许
        """
        if self.mode == PrivacyMode.LOCAL_EXTRACT:
            return False
        if self.mode == PrivacyMode.FULLY_LOCAL:
            return False
        return True
    
    def record_upload(self, data_type: str, provider: Optional[str] = None):
        """
        记录上传行为
        
        Args:
            data_type: 数据类型
            provider: 提供商
        """
        if data_type == "video":
            self.data_flow.video_uploaded = True
        elif data_type == "audio":
            self.data_flow.audio_uploaded = True
        elif data_type == "transcript":
            self.data_flow.transcript_uploaded = True
        elif data_type == "avis":
            self.data_flow.avis_uploaded = True
        elif data_type == "frames":
            self.data_flow.frames_uploaded = True
        
        if provider:
            self.data_flow.provider = provider
    
    def get_data_flow(self) -> Dict:
        """
        获取数据流向记录
        
        Returns:
            数据流字典
        """
        return {
            "video_uploaded": self.data_flow.video_uploaded,
            "audio_uploaded": self.data_flow.audio_uploaded,
            "transcript_uploaded": self.data_flow.transcript_uploaded,
            "avis_uploaded": self.data_flow.avis_uploaded,
            "frames_uploaded": self.data_flow.frames_uploaded,
            "provider": self.data_flow.provider,
            "mode": self.mode.value,
        }
    
    def validate_request(self, request: Dict) -> list[str]:
        """
        验证请求是否符合隐私模式
        
        Args:
            request: 请求字典
            
        Returns:
            错误消息列表
        """
        errors = []
        
        level = request.get("level", "l0")
        
        # 检查级别是否符合隐私模式
        if self.mode == PrivacyMode.LOCAL_EXTRACT:
            if level in ["l1", "l2"]:
                errors.append(f"隐私模式 {self.mode.value} 不支持 {level} 级别")
        
        if self.mode == PrivacyMode.FULLY_LOCAL:
            if level in ["l1", "l2"]:
                errors.append(f"隐私模式 {self.mode.value} 不支持 {level} 级别")
        
        if self.mode == PrivacyMode.REMOTE_ANSWER:
            if level in ["l1", "l2"]:
                errors.append(f"隐私模式 {self.mode.value} 不支持 {level} 级别")
        
        return errors


def get_privacy_mode(mode_str: str) -> PrivacyMode:
    """
    获取隐私模式枚举
    
    Args:
        mode_str: 模式字符串
        
    Returns:
        隐私模式枚举
    """
    try:
        return PrivacyMode(mode_str)
    except ValueError:
        return PrivacyMode.REMOTE_ANSWER


def create_privacy_controller(mode_str: str = "remote_answer") -> PrivacyController:
    """
    创建隐私控制器
    
    Args:
        mode_str: 模式字符串
        
    Returns:
        隐私控制器
    """
    mode = get_privacy_mode(mode_str)
    return PrivacyController(mode)