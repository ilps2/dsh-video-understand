"""
视频帧级 OCR 模块

提取视频画面文字，构建时间线索引。
按照升级规划 v2.1 Task 1.3 实现。

注意：PaddleOCR 有依赖问题，此模块提供简化实现。
"""
import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class FrameOCR:
    """视频帧级 OCR：提取画面文字，构建时间线索引"""
    
    def __init__(self, 
                 sample_fps: float = 0.5,
                 merge_threshold_sec: float = 3.0,
                 min_text_length: int = 3):
        """
        初始化 FrameOCR
        
        Args:
            sample_fps: 抽帧频率（每秒帧数）
            merge_threshold_sec: 合并阈值（秒）
            min_text_length: 最小文字长度
        """
        self.sample_fps = sample_fps
        self.merge_threshold = merge_threshold_sec
        self.min_length = min_text_length
        self._ocr_engine = None
    
    def _ensure_engine(self):
        """确保 OCR 引擎已加载"""
        if self._ocr_engine is not None:
            return
        
        try:
            from paddleocr import PaddleOCR
            self._ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                use_gpu=False,
                show_log=False,
            )
            print("[FrameOCR] PaddleOCR 引擎加载完成")
        except ImportError:
            raise RuntimeError("PaddleOCR 未安装，请运行: pip install paddlepaddle paddleocr")
        except Exception as e:
            raise RuntimeError(f"PaddleOCR 加载失败: {e}")
    
    def extract_frames(self, video_path: str, output_dir: str) -> List[Tuple[float, str]]:
        """
        从视频中按指定 FPS 抽帧
        
        Args:
            video_path: 视频文件路径
            output_dir: 帧图片输出目录
            
        Returns:
            [(timestamp, frame_path), ...]
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 获取视频时长
        duration = self._get_video_duration(video_path)
        if duration <= 0:
            return []
        
        # 计算抽帧时间点
        frame_times = []
        t = 0.0
        while t < duration:
            frame_times.append(t)
            t += 1.0 / self.sample_fps
        
        # 使用 ffmpeg 抽帧
        frames = []
        for i, t in enumerate(frame_times):
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(t),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                frame_path
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                if os.path.exists(frame_path):
                    frames.append((t, frame_path))
            except subprocess.CalledProcessError:
                continue
        
        return frames
    
    def _get_video_duration(self, video_path: str) -> float:
        """获取视频时长"""
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                video_path
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                info = json.loads(r.stdout)
                return float(info.get("format", {}).get("duration", 0))
        except Exception:
            pass
        return 0.0
    
    def ocr_single(self, image_path: str) -> List[Dict]:
        """
        对单张图片进行 OCR
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            OCR 结果列表
        """
        self._ensure_engine()
        
        result = self._ocr_engine.ocr(image_path, cls=True)
        
        ocr_results = []
        if result and result[0]:
            for line in result[0]:
                box, (text, confidence) = line
                if confidence > 0.5:
                    ocr_results.append({
                        "text": text,
                        "confidence": confidence,
                        "box": box,
                    })
        
        return ocr_results
    
    def extract(self, video_path: str, output_dir: str) -> str:
        """
        提取视频画面文字
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            
        Returns:
            输出文件路径
        """
        output_path = os.path.join(output_dir, "ocr_text.jsonl")
        
        with tempfile.TemporaryDirectory(prefix="ocr_frames_") as tmpdir:
            # 1. 抽帧
            frames = self.extract_frames(video_path, tmpdir)
            
            # 2. 每帧跑 OCR
            raw_results = []
            for timestamp, frame_path in frames:
                try:
                    ocr_results = self.ocr_single(frame_path)
                    for result in ocr_results:
                        raw_results.append({
                            "time": timestamp,
                            "text": result["text"],
                            "conf": result["confidence"],
                        })
                except Exception as e:
                    print(f"  ⚠ OCR 失败: {frame_path}, {e}")
            
            # 3. 时序合并
            merged = merge_ocr_results(raw_results, self.merge_threshold)
            
            # 4. 过滤短文本
            filtered = [r for r in merged if len(r.get("text", "")) >= self.min_length]
            
            # 5. 写入输出文件
            with open(output_path, "w", encoding="utf-8") as f:
                for item in filtered:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        return output_path
    
    def build_prompt_segment(self, ocr_path: str) -> str:
        """
        将 OCR 结果转为 LLM prompt 片段
        
        Args:
            ocr_path: OCR 结果 JSONL 文件路径
            
        Returns:
            LLM prompt 片段
        """
        results = []
        if os.path.exists(ocr_path):
            with open(ocr_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
        
        if not results:
            return ""
        
        lines = ["## 画面文字（时间线）"]
        for r in results[:50]:
            time = r.get("time", 0)
            text = r.get("text", "")
            conf = r.get("conf", 0)
            lines.append(f"- [{time:.0f}s] \"{text}\" (置信度 {conf:.0%})")
        
        if len(results) > 50:
            lines.append(f"- …（共 {len(results)} 条，已截前 50）")
        
        return "\n".join(lines)


def merge_ocr_results(raw_results: List[Dict], 
                      merge_threshold_sec: float = 3.0) -> List[Dict]:
    """
    时序合并 OCR 结果
    
    Args:
        raw_results: 原始结果
        merge_threshold_sec: 合并阈值
        
    Returns:
        合并后的结果
    """
    if not raw_results:
        return []
    
    # 按时间排序
    sorted_results = sorted(raw_results, key=lambda x: x.get("time", 0))
    
    # 按文本分组
    text_groups = defaultdict(list)
    for r in sorted_results:
        text = r.get("text", "").strip()
        if text:
            text_groups[text].append(r)
    
    # 合并相同文本的时间段
    merged_by_text = []
    for text, items in text_groups.items():
        items.sort(key=lambda x: x.get("time", 0))
        
        current_start = items[0]["time"]
        current_end = items[0]["time"]
        current_confs = [items[0].get("conf", 0)]
        
        for item in items[1:]:
            if item["time"] - current_end < merge_threshold_sec:
                current_end = item["time"]
                current_confs.append(item.get("conf", 0))
            else:
                merged_by_text.append({
                    "time_start": current_start,
                    "time_end": current_end,
                    "text": text,
                    "frame_count": len(current_confs),
                    "confidence": sum(current_confs) / len(current_confs),
                })
                current_start = item["time"]
                current_end = item["time"]
                current_confs = [item.get("conf", 0)]
        
        merged_by_text.append({
            "time_start": current_start,
            "time_end": current_end,
            "text": text,
            "frame_count": len(current_confs),
            "confidence": sum(current_confs) / len(current_confs),
        })
    
    # 按起始时间排序
    merged_by_text.sort(key=lambda x: x.get("time_start", 0))
    
    return merged_by_text