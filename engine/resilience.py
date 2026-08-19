"""
部署韧性模块

处理超时、错误恢复和资源清理。
按照升级规划 v2.1 Task 1.6 实现。
"""
import os
import signal
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Callable, Any
from contextlib import contextmanager
from functools import wraps
import time


class TimeoutError(Exception):
    """超时错误"""
    pass


class ResourceCleanup:
    """资源清理器"""
    
    def __init__(self):
        self._cleanup_hooks = []
    
    def register(self, hook: Callable):
        """注册清理钩子"""
        self._cleanup_hooks.append(hook)
    
    def cleanup(self):
        """执行所有清理钩子"""
        for hook in reversed(self._cleanup_hooks):
            try:
                hook()
            except Exception:
                pass
        self._cleanup_hooks.clear()


@contextmanager
def temporary_directory(prefix: str = "video_understand_", 
                       suffix: str = "",
                       cleanup: bool = True):
    """
    临时目录上下文管理器
    
    自动清理临时目录。
    """
    tmp_dir = tempfile.mkdtemp(prefix=prefix, suffix=suffix)
    try:
        yield tmp_dir
    finally:
        if cleanup and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


@contextmanager
def timeout(seconds: int, error_message: str = "操作超时"):
    """
    超时上下文管理器
    
    Args:
        seconds: 超时秒数
        error_message: 错误消息
    """
    def signal_handler(signum, frame):
        raise TimeoutError(error_message)
    
    # 设置超时信号（仅 Unix）
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows 不支持 SIGALRM，直接 yield
        yield


def retry(max_retries: int = 3, 
          delay: float = 1.0,
          backoff: float = 2.0,
          exceptions: tuple = (Exception,)):
    """
    重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟（秒）
        backoff: 退避因子
        exceptions: 需要重试的异常类型
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        return wrapper
    return decorator


def safe_run(cmd: list, 
             timeout: int = 300,
             cwd: Optional[str] = None,
             env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """
    安全运行子进程
    
    Args:
        cmd: 命令列表
        timeout: 超时秒数
        cwd: 工作目录
        env: 环境变量
        
    Returns:
        CompletedProcess 对象
    """
    # 合并环境变量
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=run_env
        )
        return result
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            cmd=cmd,
            returncode=-1,
            stdout="",
            stderr=f"命令超时 ({timeout}s): {' '.join(cmd)}"
        )
    except Exception as e:
        return subprocess.CompletedProcess(
            cmd=cmd,
            returncode=-1,
            stdout="",
            stderr=f"命令执行失败: {str(e)}"
        )


def ensure_directory(path: str) -> Path:
    """
    确保目录存在
    
    Args:
        path: 目录路径
        
    Returns:
        Path 对象
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_file_write(path: str, content: str, encoding: str = "utf-8") -> bool:
    """
    安全写入文件
    
    使用临时文件 + rename 实现原子写入。
    
    Args:
        path: 文件路径
        content: 文件内容
        encoding: 编码
        
    Returns:
        是否成功
    """
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入临时文件
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent),
            suffix=".tmp"
        )
        
        try:
            with os.fdopen(tmp_fd, "w", encoding=encoding) as f:
                f.write(content)
            
            # 原子替换
            os.replace(tmp_path, str(target))
            return True
        except Exception:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
            
    except Exception as e:
        print(f"文件写入失败: {path}, {e}")
        return False


def get_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """
    计算文件哈希
    
    Args:
        file_path: 文件路径
        chunk_size: 块大小
        
    Returns:
        SHA-256 哈希值
    """
    import hashlib
    
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_video_file(video_path: str) -> tuple[bool, str]:
    """
    验证视频文件
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        (是否有效, 错误消息)
    """
    path = Path(video_path)
    
    if not path.exists():
        return False, f"文件不存在: {video_path}"
    
    if not path.is_file():
        return False, f"不是文件: {video_path}"
    
    # 检查文件大小
    size = path.stat().st_size
    if size == 0:
        return False, f"文件为空: {video_path}"
    
    if size > 10 * 1024 * 1024 * 1024:  # 10GB
        return False, f"文件过大: {size / (1024**3):.1f}GB"
    
    # 检查文件扩展名
    valid_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv'}
    if path.suffix.lower() not in valid_extensions:
        return False, f"不支持的视频格式: {path.suffix}"
    
    return True, ""


def cleanup_old_files(directory: str, max_age_days: int = 7) -> int:
    """
    清理旧文件
    
    Args:
        directory: 目录路径
        max_age_days: 最大保留天数
        
    Returns:
        删除的文件数
    """
    import time
    
    if not os.path.exists(directory):
        return 0
    
    count = 0
    now = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                mtime = os.path.getmtime(file_path)
                if now - mtime > max_age_seconds:
                    os.unlink(file_path)
                    count += 1
            except Exception:
                pass
    
    return count