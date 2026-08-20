"""
问题路由分层引擎测试（engine/router.py）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.router import (
    classify_question,
    route_question,
    evidence_score,
    split_question,
    choose,
    INTENTS,
    ROUTE_TABLE,
)


class TestClassifyQuestion:
    """问题意图分类测试"""

    def test_clothing(self):
        assert classify_question("夏娃出场时穿的什么衣服？") == "clothing"
        assert classify_question("他穿什么颜色的外套？") == "clothing"

    def test_color(self):
        assert classify_question("画面主要是什么颜色？") == "color"
        assert classify_question("这个视频的主色调是什么？") == "color"

    def test_clothing_with_color_word(self):
        # "粉色衣服"：衣服词优先（clothing 优先于 color）
        assert classify_question("视频里的粉色衣服出现在什么时候？") == "clothing"

    def test_summary(self):
        assert classify_question("这段视频讲了什么？") == "summary"
        assert classify_question("概括一下视频内容") == "summary"

    def test_transcript_fact(self):
        assert classify_question("博主提到了哪些观点？") == "transcript_fact"
        assert classify_question("他说了什么台词？") == "transcript_fact"

    def test_ability_weapon(self):
        # "能力/武器" 是角色设定事实（语音可答），归 transcript_fact 而非 summary
        assert classify_question("密客的能力和武器是什么？") == "transcript_fact"
        assert classify_question("主角有什么超能力？") == "transcript_fact"

    def test_motion_event(self):
        assert classify_question("这个人什么时候拿起杯子？") == "motion_event"
        assert classify_question("主角走向哪里？") == "motion_event"

    def test_temporal_event(self):
        assert classify_question("第几分钟开始进入正片？") == "temporal_event"

    def test_object_count(self):
        assert classify_question("画面里有几个人？") == "object_count"

    def test_ocr_fact(self):
        assert classify_question("字幕上写着什么？") == "ocr_fact"

    def test_empty(self):
        assert classify_question("") == "summary"


class TestRouteQuestion:
    """问题路由测试"""

    def test_clothing_routes_l2(self):
        r = route_question("夏娃穿什么衣服？")
        assert r["intent"] == "clothing"
        assert r["initial_layer"] == "l2"
        assert r["visual_required"] is True

    def test_summary_routes_l0(self):
        r = route_question("视频讲了什么？")
        assert r["intent"] == "summary"
        assert r["initial_layer"] == "l0"
        assert r["visual_required"] is False

    def test_motion_routes_l1(self):
        r = route_question("什么时候拿起杯子？")
        assert r["intent"] == "motion_event"
        assert r["initial_layer"] == "l1"

    def test_level_hint_overrides(self):
        r = route_question("视频讲了什么？", level_hint="l2")
        assert r["initial_layer"] == "l2"

    def test_all_intents_have_routes(self):
        for intent in INTENTS:
            assert intent in ROUTE_TABLE, f"missing route for {intent}"


class TestEvidenceScore:
    """证据评估测试"""

    def test_sufficient(self):
        ev = evidence_score("夏娃穿什么？", "clothing", {"visual_l2": True})
        assert ev["sufficient"] is True
        assert ev["missing"] == []  # visual_l2 命中即足够（or 源）

    def test_missing_visual(self):
        ev = evidence_score("夏娃穿什么？", "clothing", {"asr": True})
        assert ev["sufficient"] is False
        assert "visual_l2" in ev["missing"]

    def test_asr_sufficient_for_summary(self):
        ev = evidence_score("视频讲了什么？", "summary", {"asr": True})
        assert ev["sufficient"] is True


class TestSplitQuestion:
    """复杂问题拆解测试"""

    def test_motion_plus_visual(self):
        subs = split_question("穿粉色衣服的人什么时候走到桌子旁边？")
        assert len(subs) >= 2
        roles = [s["role"] for s in subs]
        assert "locate" in roles and "confirm" in roles

    def test_simple_question_single(self):
        subs = split_question("视频讲了什么？")
        assert len(subs) == 1


class TestChoose:
    """动态路由测试"""

    def _profile(self, vtype="speech_dense"):
        return {"video_type": vtype, "asr_coverage": 0.91}

    def test_speech_dense_visual_question_upgrades_to_l2(self):
        """speech_dense 不限制能力：衣着问题照样升 L2"""
        d = choose(
            video_profile=self._profile("speech_dense"),
            question="夏娃穿什么衣服？",
            available={"asr": True, "obj_tracks": True},
            privacy_mode="remote_answer",
        )
        assert d["intent"] == "clothing"
        assert d["effective_layer"] == "l2"
        assert any("视觉" in r for r in d["escalation_reason"])

    def test_speech_dense_summary_stays_l0(self):
        """摘要问题 speech_dense 走 L0"""
        d = choose(
            video_profile=self._profile("speech_dense"),
            question="视频讲了什么？",
            available={"asr": True},
            privacy_mode="remote_answer",
        )
        assert d["effective_layer"] == "l0"

    def test_privacy_blocks_visual(self):
        """fully_local 禁止视觉 → 视觉问题降级 L0"""
        d = choose(
            video_profile=self._profile(),
            question="夏娃穿什么衣服？",
            available={"asr": True},
            privacy_mode="fully_local",
        )
        assert d["visual_allowed"] is False
        assert d["effective_layer"] == "l0"

    def test_motion_dense_prefers_l1(self):
        """运动密集视频摘要 → L1 轨迹优先"""
        d = choose(
            video_profile={"video_type": "motion_dense"},
            question="视频讲了什么？",
            available={"asr": True, "obj_tracks": True},
            privacy_mode="remote_answer",
        )
        assert d["effective_layer"] == "l1"

    def test_evidence_sufficient_no_upgrade(self):
        """已有足够证据 → 保持意图所需层级，理由记录证据足够"""
        d = choose(
            video_profile=self._profile(),
            question="夏娃穿什么衣服？",
            available={"asr": True, "visual_l2": True},
            privacy_mode="remote_answer",
        )
        assert d["effective_layer"] == "l2"   # 意图强制 L2
        assert d["missing_evidence"] == []    # 证据已足够


class TestBudget:
    """预算上限拦截测试"""

    def _profile(self):
        return {"video_type": "speech_dense", "asr_coverage": 0.91}

    def test_budget_blocks_visual(self):
        """预算不足拦截 L2：衣着问题降级 L0"""
        d = choose(
            video_profile=self._profile(),
            question="夏娃穿什么衣服？",
            available={"asr": True},
            privacy_mode="remote_answer",
            max_frames=6,
            budget_cny=0.0001,  # 极低预算：6帧VLM ≈ 0.0012元 > 0.0001
        )
        assert d["budget_blocked"] is True
        assert d["effective_layer"] == "l0"
        assert d["visual_required"] is True
        # 拦截后不抽帧：估算只含 LLM 回答成本（远小于完整视觉成本 0.0012）
        assert d["estimated_cost_cny"] < 0.0012

    def test_budget_allows_visual(self):
        """预算充足允许 L2"""
        d = choose(
            video_profile=self._profile(),
            question="夏娃穿什么衣服？",
            available={"asr": True},
            privacy_mode="remote_answer",
            max_frames=6,
            budget_cny=0.01,  # 足够：6帧VLM ≈ 0.0012元
        )
        assert d["budget_blocked"] is False
        assert d["effective_layer"] == "l2"

    def test_budget_no_effect_on_text_question(self):
        """纯文本问题不受预算影响（本就不走视觉）"""
        d = choose(
            video_profile=self._profile(),
            question="视频讲了什么？",
            available={"asr": True},
            privacy_mode="remote_answer",
            budget_cny=0.0001,
        )
        assert d["budget_blocked"] is False
        assert d["effective_layer"] in ("l0", "l1")

    def test_budget_none_no_block(self):
        """未设置预算不拦截"""
        d = choose(
            video_profile=self._profile(),
            question="夏娃穿什么衣服？",
            available={"asr": True},
            privacy_mode="remote_answer",
        )
        assert d["budget_blocked"] is False
        assert d["effective_layer"] == "l2"
