"""
视频理解 pipeline 编排模块

实现 pipeline 的编排逻辑，串联各个阶段。
"""
import time
from datetime import datetime
from typing import Dict, Optional

from .context import ProcessingContext
from .stages import (
    resolve_target,
    probe_media,
    extract_audio,
    transcribe,
    build_avis,
    select_visual_evidence,
    answer_questions,
    assemble_result,
    focused_answer_loop,
)
from .result_schema import ErrorCode


def run_pipeline(ctx: ProcessingContext) -> Dict:
    """
    运行 video understanding pipeline
    
    Args:
        ctx: 处理上下文
        
    Returns:
        结果字典
    """
    ctx.started_at = datetime.now()
    
    try:
        # 阶段 1: 解析目标
        if not resolve_target(ctx):
            ctx.finished_at = datetime.now()
            return assemble_result(ctx)
        
        # 阶段 2: 探测媒体
        if not probe_media(ctx):
            ctx.finished_at = datetime.now()
            return assemble_result(ctx)
        
        # 阶段 3: 提取音频
        if not extract_audio(ctx):
            ctx.finished_at = datetime.now()
            return assemble_result(ctx)
        
        # 阶段 4: ASR 转写
        if not transcribe(ctx):
            ctx.finished_at = datetime.now()
            return assemble_result(ctx)
        
        # 阶段 5: 构建 AVIS
        if not build_avis(ctx):
            ctx.finished_at = datetime.now()
            return assemble_result(ctx)
        
        # 阶段 6: 语义层路由 —— 已有完整层直接查层回答（中间件复用）；
        #        无层走多轮聚焦回答；request 要求建层时先建层再查层
        from .stages import (
            build_semantic_layer,
            answer_from_layer,
            _has_full_layer_cache,
        )

        build_layer = bool(getattr(ctx, "build_layer", False) or getattr(ctx, "ask_layer", False))
        if build_layer and not _has_full_layer_cache(ctx):
            layer = build_semantic_layer(ctx)
            if layer:
                ctx.cache_hit = True  # 建层本身已复用信息层
                if not answer_from_layer(ctx):
                    ctx.finished_at = datetime.now()
                    return assemble_result(ctx)
                ctx.finished_at = datetime.now()
                return assemble_result(ctx)
            # 建层失败 → 降级走普通路径
            ctx.add_warning("LAYER_FALLBACK", "语义层构建失败，降级到普通多轮问答", stage="layer")

        if _has_full_layer_cache(ctx):
            if answer_from_layer(ctx):
                ctx.finished_at = datetime.now()
                return assemble_result(ctx)
            # 层回答失败 → 降级走普通路径

        max_rounds = getattr(ctx, "max_rounds", 3) or 3
        if not focused_answer_loop(ctx, max_rounds=max_rounds):
            ctx.finished_at = datetime.now()
            return assemble_result(ctx)
        
        ctx.finished_at = datetime.now()
        return assemble_result(ctx)
        
    except Exception as e:
        ctx.add_error(
            ErrorCode.INTERNAL_ERROR.value,
            f"内部错误: {str(e)}",
            stage="runtime"
        )
        ctx.finished_at = datetime.now()
        return assemble_result(ctx)


def run_pipeline_with_timeout(ctx: ProcessingContext, timeout_s: int = 1800) -> Dict:
    """
    带超时的 pipeline 运行
    
    Args:
        ctx: 处理上下文
        timeout_s: 超时时间（秒）
        
    Returns:
        结果字典
    """
    import threading
    
    result = [None]
    error = [None]
    
    def target():
        try:
            result[0] = run_pipeline(ctx)
        except Exception as e:
            error[0] = e
    
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=timeout_s)
    
    if thread.is_alive():
        ctx.cancel()
        ctx.add_error(
            ErrorCode.LLM_TIMEOUT.value,
            f"处理超时 ({timeout_s}s)",
            stage="runtime"
        )
        return assemble_result(ctx)
    
    if error[0]:
        ctx.add_error(
            ErrorCode.INTERNAL_ERROR.value,
            f"内部错误: {str(error[0])}",
            stage="runtime"
        )
        return assemble_result(ctx)
    
    return result[0]