"""
隐私模式测试
"""
import pytest
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.privacy import (
    PrivacyMode,
    PrivacyController,
    get_privacy_mode,
    create_privacy_controller,
)


class TestPrivacyMode:
    """隐私模式测试"""
    
    def test_mode_values(self):
        """测试模式值"""
        assert PrivacyMode.LOCAL_EXTRACT.value == "local_extract"
        assert PrivacyMode.REMOTE_ANSWER.value == "remote_answer"
        assert PrivacyMode.REMOTE_VISUAL.value == "remote_visual"
        assert PrivacyMode.FULLY_LOCAL.value == "fully_local"
    
    def test_get_privacy_mode(self):
        """测试获取隐私模式"""
        assert get_privacy_mode("local_extract") == PrivacyMode.LOCAL_EXTRACT
        assert get_privacy_mode("remote_answer") == PrivacyMode.REMOTE_ANSWER
        assert get_privacy_mode("remote_visual") == PrivacyMode.REMOTE_VISUAL
        assert get_privacy_mode("fully_local") == PrivacyMode.FULLY_LOCAL
        assert get_privacy_mode("invalid") == PrivacyMode.REMOTE_ANSWER


class TestPrivacyController:
    """隐私控制器测试"""
    
    def test_local_extract_mode(self):
        """测试 LOCAL_EXTRACT 模式"""
        controller = PrivacyController(PrivacyMode.LOCAL_EXTRACT)
        
        assert controller.can_use_remote_llm() == False
        assert controller.can_use_remote_vlm() == False
        assert controller.can_upload_video() == False
        assert controller.can_upload_frames() == False
        assert controller.can_upload_transcript() == False
    
    def test_remote_answer_mode(self):
        """测试 REMOTE_ANSWER 模式"""
        controller = PrivacyController(PrivacyMode.REMOTE_ANSWER)
        
        assert controller.can_use_remote_llm() == True
        assert controller.can_use_remote_vlm() == False
        assert controller.can_upload_video() == False
        assert controller.can_upload_frames() == False
        assert controller.can_upload_transcript() == True
    
    def test_remote_visual_mode(self):
        """测试 REMOTE_VISUAL 模式"""
        controller = PrivacyController(PrivacyMode.REMOTE_VISUAL)
        
        assert controller.can_use_remote_llm() == True
        assert controller.can_use_remote_vlm() == True
        assert controller.can_upload_video() == False
        assert controller.can_upload_frames() == True
        assert controller.can_upload_transcript() == True
    
    def test_fully_local_mode(self):
        """测试 FULLY_LOCAL 模式"""
        controller = PrivacyController(PrivacyMode.FULLY_LOCAL)
        
        assert controller.can_use_remote_llm() == False
        assert controller.can_use_remote_vlm() == False
        assert controller.can_upload_video() == False
        assert controller.can_upload_frames() == False
        assert controller.can_upload_transcript() == False
    
    def test_record_upload(self):
        """测试记录上传"""
        controller = PrivacyController(PrivacyMode.REMOTE_VISUAL)
        
        controller.record_upload("transcript", "mimo")
        data_flow = controller.get_data_flow()
        
        assert data_flow["transcript_uploaded"] == True
        assert data_flow["provider"] == "mimo"
    
    def test_validate_request(self):
        """测试验证请求"""
        controller = PrivacyController(PrivacyMode.LOCAL_EXTRACT)
        
        # L0 应该通过
        errors = controller.validate_request({"level": "l0"})
        assert len(errors) == 0
        
        # L1 应该失败
        errors = controller.validate_request({"level": "l1"})
        assert len(errors) > 0


class TestCreatePrivacyController:
    """创建隐私控制器测试"""
    
    def test_create_controller(self):
        """测试创建控制器"""
        controller = create_privacy_controller("remote_answer")
        assert controller.mode == PrivacyMode.REMOTE_ANSWER
    
    def test_create_default(self):
        """测试默认创建"""
        controller = create_privacy_controller()
        assert controller.mode == PrivacyMode.REMOTE_ANSWER