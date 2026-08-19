"""
视频理解缓存模块

实现内容寻址缓存，支持幂等处理。
"""
import os
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

from .media_fingerprint import (
    compute_media_fingerprint,
    compute_request_hash,
    generate_cache_key,
    get_cache_path,
)


class VideoUnderstandCache:
    """
    视频理解缓存
    
    支持：
    - 内容寻址：相同视频 + 相同请求 = 缓存命中
    - 幂等处理：重复请求直接返回缓存结果
    - 原子写入：使用临时文件 + rename
    - 损坏恢复：缓存损坏自动重建
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化缓存
        
        Args:
            cache_dir: 缓存目录，默认 ~/.dsh/cache/video-understand/
        """
        if cache_dir is None:
            cache_dir = os.environ.get(
                "VIDEO_UNDERSTAND_CACHE_DIR",
                os.path.expanduser("~/.dsh/cache/video-understand")
            )
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 SQLite 数据库
        self.db_path = self.cache_dir / "index.db"
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_index (
                cache_key TEXT PRIMARY KEY,
                video_path TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                result_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fingerprint_index (
                video_path TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def get_fingerprint(self, video_path: str) -> Dict:
        """
        获取视频指纹（优先从缓存读取）
        
        Args:
            video_path: 视频路径
            
        Returns:
            指纹字典
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute(
            "SELECT fingerprint FROM fingerprint_index WHERE video_path = ?",
            (video_path,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                pass
        
        # 计算指纹并缓存
        fingerprint = compute_media_fingerprint(video_path)
        
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT OR REPLACE INTO fingerprint_index (video_path, fingerprint, updated_at) VALUES (?, ?, ?)",
            (video_path, json.dumps(fingerprint), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        
        return fingerprint
    
    def get(self, video_path: str, request: Dict) -> Optional[Dict]:
        """
        获取缓存结果
        
        Args:
            video_path: 视频路径
            request: 请求字典
            
        Returns:
            缓存结果，未命中返回 None
        """
        fingerprint = self.get_fingerprint(video_path)
        request_hash = compute_request_hash(request)
        cache_key = generate_cache_key(fingerprint, request_hash)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute(
            "SELECT result_path FROM cache_index WHERE cache_key = ?",
            (cache_key,)
        )
        row = cursor.fetchone()
        
        if row:
            result_path = row[0]
            if os.path.exists(result_path):
                # 更新访问计数
                conn.execute(
                    "UPDATE cache_index SET access_count = access_count + 1, last_accessed = ? WHERE cache_key = ?",
                    (datetime.now().isoformat(), cache_key)
                )
                conn.commit()
                conn.close()
                
                try:
                    with open(result_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError):
                    # 缓存损坏，删除
                    os.unlink(result_path)
                    conn = sqlite3.connect(str(self.db_path))
                    conn.execute("DELETE FROM cache_index WHERE cache_key = ?", (cache_key,))
                    conn.commit()
            else:
                # 文件不存在，删除索引
                conn.execute("DELETE FROM cache_index WHERE cache_key = ?", (cache_key,))
                conn.commit()
        
        conn.close()
        return None
    
    def put(self, video_path: str, request: Dict, result: Dict) -> str:
        """
        写入缓存
        
        使用原子写入：临时文件 + rename。
        
        Args:
            video_path: 视频路径
            request: 请求字典
            result: 结果字典
            
        Returns:
            缓存文件路径
        """
        fingerprint = self.get_fingerprint(video_path)
        request_hash = compute_request_hash(request)
        cache_key = generate_cache_key(fingerprint, request_hash)
        
        # 获取缓存路径
        result_path = get_cache_path(str(self.cache_dir), cache_key, "result.json")
        
        # 原子写入
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(result_path.parent),
            suffix=".tmp"
        )
        
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # rename 实现原子写入
            os.replace(tmp_path, str(result_path))
            
            # 更新索引
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                """INSERT OR REPLACE INTO cache_index 
                   (cache_key, video_path, request_hash, result_path, created_at, access_count, last_accessed) 
                   VALUES (?, ?, ?, ?, ?, 0, ?)""",
                (cache_key, video_path, request_hash, str(result_path), 
                 datetime.now().isoformat(), datetime.now().isoformat())
            )
            conn.commit()
            conn.close()
            
            return str(result_path)
            
        except Exception:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    
    def invalidate(self, video_path: str) -> int:
        """
        使视频的所有缓存失效
        
        Args:
            video_path: 视频路径
            
        Returns:
            删除的缓存数量
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute(
            "SELECT cache_key, result_path FROM cache_index WHERE video_path = ?",
            (video_path,)
        )
        rows = cursor.fetchall()
        
        count = 0
        for cache_key, result_path in rows:
            if os.path.exists(result_path):
                os.unlink(result_path)
            count += 1
        
        conn.execute("DELETE FROM cache_index WHERE video_path = ?", (video_path,))
        conn.execute("DELETE FROM fingerprint_index WHERE video_path = ?", (video_path,))
        conn.commit()
        conn.close()
        
        return count
    
    def clear(self) -> int:
        """
        清空所有缓存
        
        Returns:
            删除的缓存数量
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("SELECT result_path FROM cache_index")
        rows = cursor.fetchall()
        
        count = 0
        for (result_path,) in rows:
            if os.path.exists(result_path):
                os.unlink(result_path)
            count += 1
        
        conn.execute("DELETE FROM cache_index")
        conn.execute("DELETE FROM fingerprint_index")
        conn.commit()
        conn.close()
        
        return count
    
    def stats(self) -> Dict:
        """
        获取缓存统计
        
        Returns:
            统计字典
        """
        conn = sqlite3.connect(str(self.db_path))
        
        cursor = conn.execute("SELECT COUNT(*) FROM cache_index")
        total_entries = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT SUM(access_count) FROM cache_index")
        total_access = cursor.fetchone()[0] or 0
        
        cursor = conn.execute("SELECT COUNT(*) FROM fingerprint_index")
        total_fingerprints = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_entries": total_entries,
            "total_access_count": total_access,
            "total_fingerprints": total_fingerprints,
            "cache_dir": str(self.cache_dir),
        }