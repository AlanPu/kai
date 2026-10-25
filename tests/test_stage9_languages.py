"""
练习语言：英语 / 韩语。

这里不调模型，只验证「语言换了，提示词和缓存键有没有跟着换」。
"""

from __future__ import annotations

from app.core.languages import LANGUAGES, get_language
from app.services.corrector import build_prompt as build_correction_prompt
from app.services.plancache import PlanCache
from app.services.planner import (Plan, Topic, build_system_prompt,
                                  build_tutor_instructions, validate_plan)


def test_language_lookup_and_fallback():
    assert get_language("ko").native == "한국어"
    assert get_language("en").label == "英语"
    # 未知/空值不能抛错 —— 前端少传一个参数不该让备课直接挂掉
    assert get_language("").code == "en"
    assert get_language(None).code == "en"
    assert get_language("KO").code == "ko"
    assert set(LANGUAGES) == {"en", "ko"}


def test_planner_prompt_follows_language():
    ko = build_system_prompt(get_language("ko"))
    assert "韩语口语陪练" in ko
    assert "한국어" in ko
    assert "英语口语陪练" not in ko

    en = build_system_prompt(get_language("en"))
    assert "英语口语陪练" in en
    assert "English" in en


def test_tutor_instructions_follow_language():
    ko = build_tutor_instructions(
        Plan(opening="안녕하세요", topics=[Topic(title="여행", prompt="어디에 가고 싶어요?")],
             lang="ko"),
        profile_summary="喜欢旅行", minutes=30)
    assert "Korean speaking coach" in ko
    assert "한국어" in ko
    assert "해요체" in ko

    en = build_tutor_instructions(
        Plan(opening="Hi", topics=[Topic(title="travel", prompt="Where to?")],
             lang="en"))
    assert "English speaking coach" in en


def test_corrector_prompt_follows_language():
    ko = build_correction_prompt(get_language("ko"))
    assert "韩语口语纠错助手" in ko
    assert "助词" in ko
    # 韩语示例不该再出现英语纠错例子
    assert "I go to park yesterday" not in ko

    en = build_correction_prompt(get_language("en"))
    assert "英语口语纠错助手" in en
    assert "I go to park yesterday" in en


def test_plan_roundtrip_keeps_language():
    plan = Plan(opening="안녕", vocabulary=["여행"], lang="ko")
    assert plan.to_dict()["lang"] == "ko"
    assert Plan.from_dict(plan.to_dict()).lang == "ko"


def test_plan_cache_does_not_mix_languages():
    cache = PlanCache()
    en_plan = Plan(opening="English", lang="en")
    cache.put(1, "같은 입력", en_plan, "en")

    # 同一段素材、同一个人，但换成韩语：绝不能复用英文计划
    assert cache.get(1, "같은 입력", "ko") is None
    assert cache.get(1, "같은 입력", "en") is en_plan

    ko_plan = Plan(opening="안녕하세요", lang="ko")
    cache.put(1, "같은 입력", ko_plan, "ko")
    assert cache.get(1, "같은 입력", "ko") is ko_plan


def test_validate_plan_skips_english_heuristic_for_korean():
    plan = Plan(opening="안녕하세요", lang="ko", topics=[
        Topic(title="취미", prompt="요즘 어떤 게임을 좋아해요?"),
        Topic(title="여행", prompt="어디에 가 보고 싶어요?"),
        Topic(title="음악", prompt="어떤 음악을 자주 들어요?"),
    ])
    assert validate_plan(plan) == []
