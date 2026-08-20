"""
问题路由分层引擎（v0.4 · 问题优先动态路由）

视频类型做先验，问题意图决定入口，证据充足度决定是否升级，预算/隐私决定上限。

核心改动（相对 v0.3 静态分层）：
  - classify_question(): 问题意图分类（规则优先 + LLM 兜底）
  - route_question():     问题 → {initial_layer, required_capability, escalation_path}
  - evidence_score():     已有证据充足度评估（决定是否升级）
  - choose():             最终路由（视频先验 + 问题意图 + 证据 + 隐私/预算）
  - split_question():     复杂问题拆解为子任务

层级语义（L1 不只是一个结果层，还是 L2 的定位器）：
  L0: 回答"说了什么"（ASR/OCR/字幕/标题）
  L1: 回答"什么时候、哪里动了、对象如何运动"（obj_tracks 轨迹）→ 定位窗口
  L2: 回答"画面具体长什么样、对象之间是什么关系"（关键帧 VLM 确认）
"""

from typing import Dict, List, Optional, Tuple

# ── 问题意图枚举 ─────────────────────────────────────────────
INTENTS = (
    "summary",              # 内容摘要
    "transcript_fact",      # 语音事实（谁说了什么/提到什么）
    "scene_fact",           # 场景结构事实
    "ocr_fact",             # 画面文字/字幕
    "temporal_event",       # 时间事件（第几分钟发生什么）
    "object_presence",      # 对象是否存在
    "object_count",         # 对象数量
    "motion_event",         # 运动事件（拿起/移动/走向）
    "appearance",           # 外观描述
    "clothing",             # 衣着
    "color",                # 颜色
    "pose",                 # 姿态
    "spatial_relation",     # 空间关系（左边/旁边/上方）
    "fine_visual_detail",   # 精细视觉细节
    "unsupported",          # 无法支持
)

# 问题意图 → 关键词（规则优先，零模型成本）
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "summary": ["概括", "总结", "讲了什么", "主要内容", "核心内容", "内容是什么", "说的是什么",
                "summar", "overview", "what is this video", "main content", "主题", "大致内容",
                "讲了些什么", "大概讲了"],
    "transcript_fact": ["说了", "提到", "说些什么", "讲了哪些", "观点", "台词", "表达",
                        "提到哪些", "说什么", "提到什么", "谁说的", "讲解", "介绍", "说话",
                        "讲话", "台词说了", "能力", "武器", "有什么能力", "什么能力",
                        "有什么武器", "什么武器", "技能", "超能力", "本领", "特长"],
    "scene_fact": ["场景", "画面切换", "镜头", "结构", "几个场景", "开头", "结尾"],
    "ocr_fact": ["字幕", "文字", "标题", "写着", "水印", "标注", "屏幕上的字", "ocr"],
    "temporal_event": ["什么时候", "第几", "几分", "多少秒", "几点", "何时", "多长", "几秒",
                       "开始时间", "结束时间", "时间点", "timeline", "when did"],
    "object_presence": ["有没有", "是否有", "是否出现", "有没有出现", "存在", "出现了什么",
                        "有什么物体", "presence", "appear"],
    "object_count": ["几个", "多少", "数量", "几个对象", "多少人", "几个人", "几辆车",
                     "几台", "几件", "count", "how many"],
    "motion_event": ["拿起", "放下", "移动", "走向", "走到", "运动", "动作", "走了", "跑",
                     "转身", "拿起杯子", "走进", "motion", "move", "walk", "pick up", "拿"],
    "appearance": ["长什么样", "长相", "外表", "样子", "外观", "长得", "什么样", "look like",
                   "appearance", "looks", "外形"],
    "clothing": ["穿", "衣服", "服装", "戴着", "穿着", "戴", "帽子", "裙子", "外套", "衬衫",
                 "裤子", "鞋", "clothing", "wear", "wearing", "dress", "衣"],
    "color": ["颜色", "什么色", "色", "color", "colour", "白色", "黑色", "红色", "蓝色", "绿色",
              "粉色", "黄色", "紫色", "橙色", "灰色"],
    "pose": ["姿态", "姿势", "pose", "posture", "动作姿态", "坐", "站", "躺"],
    "spatial_relation": ["左边", "右边", "旁边", "上方", "下方", "前面", "后面", "中间", "位置",
                         "在.*的", "left", "right", "beside", "above", "below", "之间", "附近",
                         "哪个方向", "空间"],
    "fine_visual_detail": ["细节", "特写", "具体", "什么牌子", "什么型号", "什么图案", "什么标志",
                           "detail", "logo", "brand", "型号", "上面写了什么"],
}

# 问题意图 → 路由（首选层 / 备用升级层 / 所需能力）
ROUTE_TABLE: Dict[str, Dict] = {
    "summary":            {"initial": "l0", "upgrade": "l1", "capability": "transcript+structure"},
    "transcript_fact":    {"initial": "l0", "upgrade": None,  "capability": "transcript"},
    "scene_fact":         {"initial": "l0", "upgrade": "l1", "capability": "transcript+scenes"},
    "ocr_fact":           {"initial": "l0", "upgrade": "l2", "capability": "ocr"},
    "temporal_event":     {"initial": "l0", "upgrade": "l1", "capability": "transcript+temporal"},
    "object_presence":    {"initial": "l1", "upgrade": "l2", "capability": "tracks+visual"},
    "object_count":       {"initial": "l1", "upgrade": "l2", "capability": "tracks+visual"},
    "motion_event":       {"initial": "l1", "upgrade": "l2", "capability": "tracks+visual"},
    "appearance":         {"initial": "l2", "upgrade": None,  "capability": "visual_detail"},
    "clothing":           {"initial": "l2", "upgrade": None,  "capability": "visual_detail"},
    "color":              {"initial": "l2", "upgrade": None,  "capability": "visual_detail"},
    "pose":               {"initial": "l2", "upgrade": None,  "capability": "visual_detail"},
    "spatial_relation":   {"initial": "l1", "upgrade": "l2", "capability": "tracks+visual"},
    "fine_visual_detail": {"initial": "l2", "upgrade": None,  "capability": "visual_detail"},
    "unsupported":        {"initial": "l0", "upgrade": None,  "capability": "none"},
}

# 证据源 → 质量（路由时评估已有证据）
SOURCE_CONFIDENCE = {
    "asr": 0.80,
    "ocr": 0.70,
    "obj_tracks": 0.75,
    "scenes": 0.65,
    "visual_l1": 0.55,
    "visual_l2": 0.85,
}

# 成本估算（元）：VLM 每帧 ≈ 253 tok in + ~60 tok out（MiMo v2.5 近似）
VLM_COST_PER_FRAME_CNY = 0.0002
LLM_COST_PER_ANSWER_CNY = 0.001


def classify_question(question: str) -> str:
    """
    问题意图分类：规则关键词优先（零模型成本），返回意图。

    规则无法判定时返回 "summary"（默认兜底，最宽松）。
    """
    q = (question or "").strip()
    if not q:
        return "summary"
    ql = q.lower()

    # 命中优先意图（遍历顺序即优先级：颜色/衣着/运动等具体意图优先于宽泛意图）
    priority = ("clothing", "color", "motion_event", "object_presence", "object_count",
                "temporal_event", "ocr_fact", "fine_visual_detail", "appearance", "pose",
                "spatial_relation", "transcript_fact", "scene_fact", "summary")
    for intent in priority:
        for kw in INTENT_KEYWORDS.get(intent, []):
            if kw.lower() in ql:
                return intent

    return "summary"


def route_question(question: str, level_hint: Optional[str] = None) -> Dict:
    """
    问题 → 路由决策（不考虑证据，纯意图路由）。

    Args:
        question: 用户问题
        level_hint: 用户显式指定的 level（l0/l1/l2），覆盖意图首选层

    Returns:
        {
            "intent": "clothing",
            "required_capability": "visual_detail",
            "initial_layer": "l2",
            "upgrade_layer": None,
            "visual_required": True,
        }
    """
    intent = classify_question(question)
    row = ROUTE_TABLE.get(intent, ROUTE_TABLE["summary"])

    initial = row["initial"]
    if level_hint in ("l0", "l1", "l2"):
        # 用户显式指定级别：作为初始层（意图仍决定是否需要视觉）
        initial = level_hint

    visual_required = row["capability"] in ("visual_detail", "ocr") or row["upgrade"] == "l2"

    return {
        "intent": intent,
        "required_capability": row["capability"],
        "initial_layer": initial,
        "upgrade_layer": row["upgrade"],
        "visual_required": visual_required,
    }


def evidence_score(question: str, intent: str, available: Dict[str, bool]) -> Dict:
    """
    已有证据充足度评估：source_relevance × source_confidence × temporal_overlap。

    Args:
        question: 用户问题
        intent: 问题意图
        available: {source: bool} 已有证据源（asr/ocr/obj_tracks/scenes/visual_l1/visual_l2）

    Returns:
        {"score": 0-1, "sufficient": bool, "missing": [source...], "sources": [...]}
    """
    row = ROUTE_TABLE.get(intent, ROUTE_TABLE["summary"])
    needed = row["capability"]

    # 该意图需要的证据源
    needed_sources = []
    if needed in ("transcript", "transcript+structure", "transcript+scenes", "transcript+temporal"):
        needed_sources = ["asr"]
        if "scenes" in needed:
            needed_sources.append("scenes")
    elif needed == "ocr":
        needed_sources = ["ocr", "visual_l2"]  # OCR 可能来自画面
    elif needed in ("tracks+visual",):
        needed_sources = ["obj_tracks", "visual_l2"]
    elif needed == "visual_detail":
        needed_sources = ["visual_l2", "visual_l1"]
    else:
        needed_sources = ["asr"]

    # 计算分数：命中源加权 / 需求源加权
    have = [s for s in needed_sources if available.get(s)]
    score = 0.0
    for s in needed_sources:
        if available.get(s):
            score += SOURCE_CONFIDENCE.get(s, 0.5)
    denom = sum(SOURCE_CONFIDENCE.get(s, 0.5) for s in needed_sources) or 1.0
    score = round(score / denom, 3)

    # 核心源判定：visual_detail 类意图任一视觉源命中即满足（L1/L2 可互相替代）
    or_sources = ("visual_detail", "tracks+visual", "ocr")
    if needed in or_sources:
        missing = []
        if needed == "visual_detail":
            if not (available.get("visual_l2") or available.get("visual_l1")):
                missing = ["visual_l2"]
        elif needed == "ocr":
            if not (available.get("ocr") or available.get("visual_l2")):
                missing = ["visual_l2"]
        else:  # tracks+visual
            if not available.get("obj_tracks"):
                missing.append("obj_tracks")
            if not (available.get("visual_l2") or available.get("visual_l1")):
                missing.append("visual_l2")
        # or 型源：无缺失即足够（不必满足全源加权阈值）
        sufficient = not missing
    else:
        missing = [s for s in needed_sources if not available.get(s)]
        sufficient = score >= 0.75 and not missing  # 核心源缺失即不足

    return {
        "score": score,
        "sufficient": sufficient,
        "missing": missing,
        "sources": have,
    }


def split_question(question: str) -> List[Dict]:
    """
    复杂问题拆解：识别多意图 → 子任务列表。
    简单问题返回单个子任务。当前基于关键词组合的启发式拆解。

    Returns:
        [{"intent": ..., "initial_layer": ..., "question": ...}, ...]
    """
    intent = classify_question(question)
    ql = question.lower()

    # 组合意图检测：含有运动/时间 + 视觉描述词 → 拆成定位 + 确认
    has_motion = any(k in ql for k in INTENT_KEYWORDS["motion_event"])
    has_visual = any(k in ql for k in
                     INTENT_KEYWORDS["clothing"] + INTENT_KEYWORDS["color"] +
                     INTENT_KEYWORDS["appearance"] + INTENT_KEYWORDS["pose"] +
                     INTENT_KEYWORDS["fine_visual_detail"])
    has_temporal = any(k in ql for k in INTENT_KEYWORDS["temporal_event"])
    has_object = any(k in ql for k in
                     INTENT_KEYWORDS["object_presence"] + INTENT_KEYWORDS["object_count"] +
                     INTENT_KEYWORDS["spatial_relation"])

    subtasks = []
    # 定位类子任务（时间/运动/对象）
    if has_temporal or has_motion or has_object:
        sub_intent = "motion_event" if has_motion else ("temporal_event" if has_temporal else "object_presence")
        subtasks.append({
            "intent": sub_intent,
            "layer": ROUTE_TABLE[sub_intent]["initial"],
            "question": question,
            "role": "locate",
        })
    # 确认类子任务（视觉细节）
    if has_visual:
        subtasks.append({
            "intent": intent if intent in ("clothing", "color", "appearance", "pose", "fine_visual_detail") else "appearance",
            "layer": "l2",
            "question": question,
            "role": "confirm",
        })
    # 兜底：无拆解 → 单任务
    if not subtasks:
        subtasks.append({
            "intent": intent,
            "layer": ROUTE_TABLE.get(intent, ROUTE_TABLE["summary"])["initial"],
            "question": question,
            "role": "answer",
        })
    return subtasks


def choose(video_profile: Dict, question: str, available: Dict[str, bool],
           level_hint: Optional[str] = None, privacy_mode: str = "remote_answer",
           max_frames: int = 6, budget_cny: Optional[float] = None) -> Dict:
    """
    最终动态路由：视频先验 + 问题意图 + 已有证据 + 隐私 + 预算。

    Args:
        video_profile: {"video_type", "asr_coverage", "motion_density", "track_count", ...}
        question: 用户问题
        available: 已有证据源 {asr/ocr/obj_tracks/scenes/visual_l1/visual_l2: bool}
        level_hint: 用户显式级别（可选）
        privacy_mode: 隐私模式（fully_local/local_extract 禁止远程 VLM）
        max_frames: 视觉层最大抽帧数
        budget_cny: 单次问题预算上限（元）。视觉成本估算超预算时拦截 L2 升级
                    （降级到 L0/L1），reason 记录 budget_blocked。

    Returns:
        完整路由决策：
        {
            "intent", "required_capability", "initial_layer", "effective_layer",
            "upgrade_layer", "escalation_reason": [...], "evidence_score",
            "frames", "visual_allowed", "subtasks": [...],
            "estimated_cost_cny", "budget_cny", "budget_blocked": bool,
        }
    """
    intent = classify_question(question)
    route = route_question(question, level_hint)

    video_type = video_profile.get("video_type", "mixed")
    visual_allowed = privacy_mode not in ("fully_local", "local_extract")

    # 视觉成本估算：需要抽帧时 ≈ 帧数 × 每帧成本 + LLM 回答成本
    frames_planned = max_frames if route["visual_required"] else 0
    estimated_cost = frames_planned * VLM_COST_PER_FRAME_CNY + LLM_COST_PER_ANSWER_CNY

    # 预算拦截：预算不足且意图需要视觉 → 禁止 L2，降级到 L0/L1
    budget_blocked = False
    if budget_cny is not None and budget_cny >= 0 and estimated_cost > budget_cny and route["visual_required"]:
        budget_blocked = True
        frames_planned = 0
        estimated_cost = LLM_COST_PER_ANSWER_CNY

    # 意图已强制需要视觉 → effective 直接 L2（若允许且预算足够）
    escalation = []
    effective = route["initial_layer"]
    if route["visual_required"]:
        if visual_allowed and not budget_blocked:
            effective = "l2"
            escalation.append(f"{intent} 问题需要视觉能力（{route['required_capability']}）")
        elif budget_blocked:
            effective = "l0"
            escalation.append(f"{intent} 问题需要视觉，但预算 {budget_cny} 元不足（估算 {estimated_cost} 元）→ 拦截 L2")
        else:
            effective = "l0"
            escalation.append(f"{intent} 问题需要视觉，但隐私模式 {privacy_mode} 禁止帧上传")
    else:
        # 视频类型只调整默认路线，不限制能力
        if video_type == "speech_dense" and effective == "l0":
            escalation.append(f"视频类型 {video_type} → ASR 优先（语音覆盖 {video_profile.get('asr_coverage', '?')}）")
        elif video_type == "motion_dense" and effective in ("l0",):
            # 运动密集视频：即使摘要也先看轨迹（低成本）
            effective = "l1"
            escalation.append(f"视频类型 {video_type} → 轨迹层优先")

    # 证据充足度：已有证据足够则不必升级
    ev = evidence_score(question, intent, available)
    if ev["sufficient"] and effective != "l2":
        escalation.append(f"已有证据足够（score={ev['score']}），不升级")
    elif ev["missing"] and effective != "l2" and route["upgrade_layer"] == "l2" and visual_allowed and not budget_blocked:
        effective = route["upgrade_layer"]
        frames_planned = max_frames
        estimated_cost = frames_planned * VLM_COST_PER_FRAME_CNY + LLM_COST_PER_ANSWER_CNY
        escalation.append(f"缺证据 {ev['missing']} → 升级到 {route['upgrade_layer']}")

    # ── ASR 稀疏自动升级：ASR 覆盖率 < 40% 且意图需要画面信息 ──
    # 当语音信息不足（BGM 剪辑/纯画面视频），即使 intent=summary 也应
    # 升级到 L2 全局扫描（均匀抽帧拼接 + VLM 批量分析），因为 ASR 无法回答。
    ASR_SPARSE_THRESHOLD = 0.40
    asr_cov = float(video_profile.get("asr_coverage", 1.0) or 1.0)
    if (asr_cov < ASR_SPARSE_THRESHOLD
            and effective in ("l0", "l1")
            and visual_allowed
            and not budget_blocked):
        effective = "l2"
        mode = "global_scan"
        escalation.append(
            f"ASR 覆盖率仅 {asr_cov:.0%}（<{ASR_SPARSE_THRESHOLD:.0%}），"
            f"语音信息不足 → 升级 L2 全局扫描模式（均匀抽帧拼接+VLM）"
        )

    # 子任务拆解
    subtasks = split_question(question)

    return {
        "intent": intent,
        "required_capability": route["required_capability"],
        "initial_layer": route["initial_layer"],
        "effective_layer": effective,
        "upgrade_layer": route["upgrade_layer"],
        "escalation_reason": escalation,
        "evidence_score": ev["score"],
        "evidence_sources": ev["sources"],
        "missing_evidence": ev["missing"],
        "frames": frames_planned,
        "visual_required": route["visual_required"],
        "visual_allowed": visual_allowed,
        "budget_blocked": budget_blocked,
        "estimated_cost_cny": round(estimated_cost, 5),
        "budget_cny": budget_cny,
        "scan_mode": "global_scan" if (effective == "l2" and asr_cov < ASR_SPARSE_THRESHOLD
                                        and not route.get("visual_required")) else None,
        "video_type": video_type,
        "subtasks": subtasks,
    }
