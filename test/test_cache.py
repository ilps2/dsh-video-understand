"""
缓存模块测试
"""
import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.cache import VideoUnderstandCache
from engine.media_fingerprint import compute_media_fingerprint, compute_request_hash


class TestMediaFingerprint:
    """媒体指纹测试"""
    
    def test_compute_file_hash(self):
        """测试计算文件哈希"""
        from engine.media_fingerprint import compute_file_hash
        
        test_video = Path(__file__).parent / "fixtures" / "blue-3s.mp4"
        if test_video.exists():
            h = compute_file_hash(str(test_video))
            assert len(h) == 64  # SHA-256
            assert h == compute_file_hash(str(test_video))  # 相同文件相同哈希
    
    def test_compute_media_fingerprint(self):
        """测试计算媒体指纹"""
        test_video = Path(__file__).parent / "fixtures" / "blue-3s.mp4"
        if test_video.exists():
            fp = compute_media_fingerprint(str(test_video))
            assert "file_size" in fp
            assert "sha256" in fp
            assert "duration_s" in fp
            assert fp["file_size"] > 0
    
    def test_compute_request_hash(self):
        """测试计算请求哈希"""
        request1 = {"target": "test.mp4", "level": "l0"}
        request2 = {"target": "test.mp4", "level": "l0"}
        request3 = {"target": "test.mp4", "level": "l1"}
        
        h1 = compute_request_hash(request1)
        h2 = compute_request_hash(request2)
        h3 = compute_request_hash(request3)
        
        assert h1 == h2  # 相同请求相同哈希
        assert h1 != h3  # 不同请求不同哈希


class TestVideoUnderstandCache:
    """缓存测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.test_dir = tempfile.mkdtemp(prefix="cache_test_")
        self.cache = VideoUnderstandCache(cache_dir=self.test_dir)
    
    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_init(self):
        """测试初始化"""
        assert self.cache.cache_dir.exists()
        assert self.cache.db_path.exists()
    
    def test_put_and_get(self):
        """测试写入和读取缓存"""
        test_video = Path(__file__).parent / "fixtures" / "blue-3s.mp4"
        if not test_video.exists():
            pytest.skip("测试视频不存在")
        
        request = {"target": str(test_video), "level": "l0"}
        result = {"schema_version": "1", "video": {}, "answers": []}
        
        # 写入
        self.cache.put(str(test_video), request, result)
        
        # 读取
        cached = self.cache.get(str(test_video), request)
        assert cached is not None
        assert cached["schema_version"] == "1"
    
    def test_cache_hit(self):
        """测试缓存命中"""
        test_video = Path(__file__).parent / "fixtures" / "blue-3s.mp4"
        if not test_video.exists():
            pytest.skip("测试视频不存在")
        
        request = {"target": str(test_video), "level": "l0"}
        result = {"schema_version": "1", "answers": []}
        
        # 写入两次
        self.cache.put(str(test_video), request, result)
        self.cache.put(str(test_video), request, result)
        
        # 读取
        cached = self.cache.get(str(test_video), request)
        assert cached is not None
        
        # 检查统计
        stats = self.cache.stats()
        assert stats["total_access_count"] >= 1
    
    def test_same_content_different_path(self):
        """测试同内容不同路径命中缓存"""
        test_video = Path(__file__).parent / "fixtures" / "blue-3s.mp4"
        if not test_video.exists():
            pytest.skip("测试视频不存在")
        
        # 复制到不同路径
        tmp_dir = tempfile.mkdtemp()
        copy_path = Path(tmp_dir) / "copy.mp4"
        shutil.copy(test_video, copy_path)
        
        try:
            request = {"target": str(test_video), "level": "l0"}
            result = {"schema_version": "1", "answers": []}
            
            # 写入
            self.cache.put(str(test_video), request, result)
            
            # 从不同路径读取
            cached = self.cache.get(str(copy_path), request)
            assert cached is not None
        finally:
            shutil.rmtree(tmp_dir)
    
    def test_invalidate(self):
        """测试缓存失效"""
        test_video = Path(__file__).parent / "fixtures" / "blue-3s.mp4"
        if not test_video.exists():
            pytest.skip("测试视频不存在")
        
        request = {"target": str(test_video), "level": "l0"}
        result = {"schema_version": "1", "answers": []}
        
        # 写入
        self.cache.put(str(test_video), request, result)
        
        # 失效
        count = self.cache.invalidate(str(test_video))
        assert count == 1
        
        # 读取应该返回 None
        cached = self.cache.get(str(test_video), request)
        assert cached is None
    
    def test_clear(self):
        """测试清空缓存"""
        test_video = Path(__file__).parent / "fixtures" / "blue-3s.mp4"
        if not test_video.exists():
            pytest.skip("测试视频不存在")
        
        request = {"target": str(test_video), "level": "l0"}
        result = {"schema_version": "1", "answers": []}
        
        # 写入
        self.cache.put(str(test_video), request, result)
        
        # 清空
        count = self.cache.clear()
        assert count == 1
        
        # 统计应该为 0
        stats = self.cache.stats()
        assert stats["total_entries"] == 0
    
    def test_stats(self):
        """测试统计"""
        stats = self.cache.stats()
        assert "total_entries" in stats
        assert "total_access_count" in stats
        assert "cache_dir" in stats