"""
媒体指纹模块

计算视频文件的指纹，用于缓存键生成。
"""
import os
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional


def compute_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """
    计算文件 SHA-256 哈希
    
    Args:
        file_path: 文件路径
        chunk_size: 块大小
        
    Returns:
        SHA-256 哈希值
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_media_fingerprint(video_path: str) -> Dict:
    """
    计算媒体指纹
    
    包含文件大小、mtime、SHA-256、时长、宽高、fps。
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        指纹字典
    """
    path = Path(video_path)
    stat = path.stat()
    
    # 基本文件信息
    fingerprint = {
        "file_size": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": compute_file_hash(video_path),
    }
    
    # 媒体信息
    try:
        import subprocess
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            info = json.loads(result.stdout)
            
            # 时长
            duration = float(info.get("format", {}).get("duration", 0))
            fingerprint["duration_s"] = duration
            
            # 流信息
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "video":
                    fingerprint["width"] = int(stream.get("width", 0))
                    fingerprint["height"] = int(stream.get("height", 0))
                    
                    # fps
                    fps_str = stream.get("r_frame_rate", "30/1")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        fingerprint["fps"] = float(num) / float(den)
                    else:
                        fingerprint["fps"] = float(fps_str)
                    
                    break
    except Exception:
        pass
    
    return fingerprint


def compute_request_hash(request: Dict) -> str:
    """
    计算请求哈希
    
    用于缓存键生成。
    
    Args:
        request: 请求字典
        
    Returns:
        请求哈希值
    """
    # 排序键以确保一致性
    sorted_request = json.dumps(request, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(sorted_request.encode()).hexdigest()[:16]


def generate_cache_key(fingerprint: Dict, request_hash: str) -> str:
    """
    生成缓存键
    
    Args:
        fingerprint: 媒体指纹
        request_hash: 请求哈希
        
    Returns:
        缓存键
    """
    # 使用文件 SHA-256 和请求哈希生成缓存键
    file_hash = fingerprint.get("sha256", "unknown")[:16]
    return f"{file_hash}_{request_hash}"


def get_cache_path(cache_dir: str, cache_key: str, filename: str) -> Path:
    """
    获取缓存文件路径
    
    Args:
        cache_dir: 缓存目录
        cache_key: 缓存键
        filename: 文件名
        
    Returns:
        缓存文件路径
    """
    cache_path = Path(cache_dir) / "objects" / cache_key
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path / filename