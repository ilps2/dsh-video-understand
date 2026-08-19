"""
内容寻址缓存系统

基于 xxhash 内容哈希，支持跨路径/跨设备复用 AVIS 信息层。
按照升级规划 v2.1 Task 1.1 实现。
"""
import os
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import xxhash


class ContentAddressableCache:
    """
    内容寻址缓存：支持跨路径/跨设备复用 AVIS 信息层
    
    使用 xxhash 对文件内容进行哈希，替代原有的 md5(path:size:mtime) 方案。
    相同内容的文件无论放在哪个路径/设备，都能命中缓存。
    """
    
    def __init__(self, cache_root: str = "~/.cache/dsh-video-understand"):
        """
        初始化缓存
        
        Args:
            cache_root: 缓存根目录
        """
        self.root = Path(cache_root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.db"
        self.db = sqlite3.connect(str(self.db_path))
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS avis_cache (
                content_hash TEXT PRIMARY KEY,
                video_path TEXT NOT NULL,
                file_size INTEGER,
                avis_dir TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP
            )
        """)
        self.db.commit()
    
    def compute_hash(self, video_path: str, sample_bytes: int = 4 * 1024 * 1024) -> str:
        """
        计算文件内容哈希
        
        使用 xxhash 对文件前 4MB 内容进行哈希，同时纳入文件大小作为二次校验。
        这样既保证速度（只读部分文件），又保证足够的唯一性。
        
        Args:
            video_path: 视频文件路径
            sample_bytes: 采样字节数，默认 4MB
            
        Returns:
            格式为 "{xxhash}_{file_size}" 的哈希字符串
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {video_path}")
        
        h = xxhash.xxh64()
        
        # 读取文件前 sample_bytes 字节
        with open(video_path, 'rb') as f:
            data = f.read(sample_bytes)
            h.update(data)
        
        # 获取文件大小
        file_size = path.stat().st_size
        
        # 返回格式: {hash}_{size}
        return f"{h.hexdigest()}_{file_size}"
    
    def get(self, video_path: str) -> Optional[str]:
        """
        获取缓存的 avis_dir
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            缓存的 avis_dir 路径，如果不存在返回 None
        """
        content_hash = self.compute_hash(video_path)
        
        cursor = self.db.execute(
            "SELECT avis_dir FROM avis_cache WHERE content_hash = ?",
            (content_hash,)
        )
        row = cursor.fetchone()
        
        if row:
            # 更新访问计数和时间
            self.db.execute(
                "UPDATE avis_cache SET access_count = access_count + 1, last_accessed = ? WHERE content_hash = ?",
                (datetime.now().isoformat(), content_hash)
            )
            self.db.commit()
            return row[0]
        
        return None
    
    def put(self, video_path: str, avis_dir: str) -> None:
        """
        写入缓存索引
        
        Args:
            video_path: 视频文件路径
            avis_dir: AVIS 信息层目录路径
        """
        content_hash = self.compute_hash(video_path)
        file_size = Path(video_path).stat().st_size
        
        # 使用 INSERT OR REPLACE 处理重复
        self.db.execute(
            """INSERT OR REPLACE INTO avis_cache 
               (content_hash, video_path, file_size, avis_dir, created_at, access_count, last_accessed)
               VALUES (?, ?, ?, ?, ?, 1, ?)""",
            (content_hash, video_path, file_size, avis_dir, datetime.now().isoformat(), datetime.now().isoformat())
        )
        self.db.commit()
    
    def invalidate(self, video_path: str) -> bool:
        """
        使缓存失效
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            是否成功删除
        """
        content_hash = self.compute_hash(video_path)
        cursor = self.db.execute("DELETE FROM avis_cache WHERE content_hash = ?", (content_hash,))
        self.db.commit()
        return cursor.rowcount > 0
    
    def clear(self) -> int:
        """
        清空所有缓存
        
        Returns:
            删除的缓存条目数
        """
        cursor = self.db.execute("DELETE FROM avis_cache")
        self.db.commit()
        return cursor.rowcount
    
    def stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            包含缓存统计的字典
        """
        cursor = self.db.execute("SELECT COUNT(*), SUM(file_size) FROM avis_cache")
        count, total_size = cursor.fetchone()
        
        cursor = self.db.execute("SELECT SUM(access_count) FROM avis_cache")
        total_access = cursor.fetchone()[0] or 0
        
        return {
            "total_entries": count,
            "total_size_bytes": total_size or 0,
            "total_size_mb": (total_size or 0) / (1024 * 1024),
            "total_access_count": total_access,
            "cache_dir": str(self.root),
        }
    
    def close(self):
        """关闭数据库连接"""
        if self.db:
            self.db.close()
            self.db = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False