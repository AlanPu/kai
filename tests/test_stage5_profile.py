"""
阶段 5 测试：个性化（个人画像）。

需求 6：你对我了解越多，就越清楚我的背景，
        从而更好地发起聊天，跟我聊我感兴趣的内容。
"""

import pytest

from app.services.profile import (CATEGORIES, INJECT_MIN_CONFIDENCE,
                                  MAX_INJECT_FACTS, ProfileExtractor,
                                  ProfileStore)
from app.storage.db import Database
from app.storage.models import ProfileFact


@pytest.fixture
def store(tmp_path):
    db = Database(tmp_path / "p.db")
    db.init_schema()
    yield ProfileStore(db, tmp_path / "profiles"), db
    db.close()


# ============================================================
#  抽取结果归一化（不调网络）
# ============================================================

def test_normalize_basic():
    facts = ProfileExtractor._normalize({"facts": [
        {"category": "interest", "key": "hobby", "value": "跑步",
         "confidence": 0.8}]})
    assert len(facts) == 1
    assert facts[0].category == "interest"
    assert facts[0].confidence == 0.8


def test_normalize_rejects_unknown_category():
    """分类必须在白名单内，否则数据会失控。"""
    facts = ProfileExtractor._normalize({"facts": [
        {"category": "随便编的", "key": "x", "value": "y"}]})
    assert facts == []


def test_normalize_rejects_missing_fields():
    facts = ProfileExtractor._normalize({"facts": [
        {"category": "interest", "key": "hobby"},          # 缺 value
        {"category": "interest", "value": "跑步"},          # 缺 key
        {"key": "hobby", "value": "跑步"},                  # 缺 category
    ]})
    assert facts == []


def test_normalize_normalizes_key_spacing():
    """'Hobby Time' 与 'hobby_time' 应视为同一条。"""
    facts = ProfileExtractor._normalize({"facts": [
        {"category": "interest", "key": "Hobby Time", "value": "a"},
        {"category": "interest", "key": "hobby_time", "value": "b"},
    ]})
    assert len(facts) == 1, "key 规范化后应去重"


def test_normalize_clamps_confidence():
    facts = ProfileExtractor._normalize({"facts": [
        {"category": "interest", "key": "a", "value": "x", "confidence": 5},
        {"category": "interest", "key": "b", "value": "y", "confidence": -3},
    ]})
    assert facts[0].confidence == 1.0
    assert facts[1].confidence == 0.0


def test_normalize_bad_confidence_defaults():
    facts = ProfileExtractor._normalize({"facts": [
        {"category": "interest", "key": "a", "value": "x",
         "confidence": "很高"}]})
    assert facts[0].confidence == 0.5


def test_normalize_handles_bare_list():
    facts = ProfileExtractor._normalize([
        {"category": "goal", "key": "g", "value": "v"}])
    assert len(facts) == 1


def test_extract_skips_short_text():
    """太短的输入直接短路，不调用模型。"""
    e = ProfileExtractor()
    assert e.extract(["yes"]) == []
    assert e.extract([""]) == []
    assert e.extract([]) == []


# ============================================================
#  累积
# ============================================================

def test_absorb_and_read(store):
    st, db = store
    sid = db.create_session("topic", "x")   # 外键要求会话真实存在
    n = st.absorb([
        ProfileFact(category="interest", key="hobby", value="跑步",
                    confidence=0.8),
        ProfileFact(category="background", key="job", value="工程师",
                    confidence=0.9),
    ], session_id=sid)
    assert n == 2
    facts = db.list_facts()
    assert len(facts) == 2
    assert facts[0].source_session_id == sid


def test_absorb_repeated_fact_raises_confidence(store):
    """同一事实被多次提到 → 更可信（这正是不重复问同样问题的价值）。"""
    st, db = store
    f = ProfileFact(category="interest", key="hobby", value="跑步",
                    confidence=0.5)
    st.absorb([f])
    st.absorb([f])
    facts = db.list_facts()
    assert len(facts) == 1
    assert facts[0].confidence > 0.5


def test_absorb_empty_is_noop(store):
    st, db = store
    assert st.absorb([]) == 0
    assert db.list_facts() == []


# ============================================================
#  注入（需求 6 的核心）
# ============================================================

def test_summary_empty_when_nothing_known(store):
    st, _ = store
    assert st.summary() == ""


def test_summary_excludes_low_confidence(store):
    """低置信度事实不能注入，否则会误导模型。"""
    st, _ = store
    st.absorb([
        ProfileFact(category="interest", key="a", value="确信的",
                    confidence=0.9),
        ProfileFact(category="interest", key="b", value="不确定的",
                    confidence=0.3),
    ])
    s = st.summary()
    assert "确信的" in s
    assert "不确定的" not in s


def test_summary_organized_by_category(store):
    st, _ = store
    st.absorb([
        ProfileFact(category="interest", key="hobby", value="跑步",
                    confidence=0.9),
        ProfileFact(category="goal", key="g", value="想开口流利",
                    confidence=0.9),
    ])
    s = st.summary()
    assert "跑步" in s and "想开口流利" in s
    assert s.count("\n") >= 1, "应按分类分行"


def test_summary_respects_max_facts(store):
    """注入总量要有上限，避免提示词无限膨胀。"""
    st, _ = store
    facts = [ProfileFact(category="interest", key=f"k{i}",
                         value=f"v{i}", confidence=0.9)
             for i in range(MAX_INJECT_FACTS + 15)]
    st.absorb(facts)
    s = st.summary()
    assert s.count("；") + s.count("\n") < MAX_INJECT_FACTS + 5


def test_summary_threshold_is_reasonable():
    assert 0.5 <= INJECT_MIN_CONFIDENCE <= 0.8


# ============================================================
#  Markdown 导出（用户可手工修正）
# ============================================================

def test_markdown_export(store, tmp_path):
    st, _ = store
    st.absorb([ProfileFact(category="interest", key="hobby", value="跑步",
                           confidence=0.9)])
    p = st.write_markdown()
    text = p.read_text(encoding="utf-8")
    assert "跑步" in text
    assert "画像" in text
    # 必须说明这是可编辑的，否则用户不知道能改
    assert "直接" in text or "修改" in text


def test_markdown_export_empty_does_not_crash(store):
    st, _ = store
    p = st.write_markdown()
    assert p.exists()


# ============================================================
#  语言问题归纳
# ============================================================

def test_language_issues_from_pronunciation_stats(store):
    """高频发音错误应被归纳出来（比单次纠错更有价值）。"""
    st, db = store
    sid = db.create_session("topic", "x")
    from app.storage.models import Correction
    for w in ["think", "think", "think", "three"]:
        db.add_correction(Correction(session_id=sid, kind="pronunciation",
                                     severity="critical", original=w,
                                     suggestion="?", word=w))
    issues = st.language_issues()
    assert any("think" in i for i in issues)


def test_language_issues_empty_when_clean(store):
    st, _ = store
    assert st.language_issues() == []


# ============================================================
#  分类白名单
# ============================================================

def test_categories_are_stable():
    """分类集合是对外契约，改动需同步文档。"""
    assert set(CATEGORIES) == {
        "background", "interest", "preference", "goal", "language", "life"}


# ============================================================
#  联网测试
# ============================================================

@pytest.mark.live
def test_live_extract_personal_facts():
    facts = ProfileExtractor().extract(
        ["I work as a software engineer in Shenzhen. "
         "I usually go running in the park on weekends."])
    assert facts, "明确陈述的个人信息应被抽出"
    cats = {f.category for f in facts}
    assert "background" in cats or "interest" in cats


@pytest.mark.live
def test_live_extract_silent_on_opinion():
    """纯观点不含个人信息 → 应抽不出来（宁缺毋滥）。"""
    facts = ProfileExtractor().extract(
        ["I think remote work is good because it saves commuting time."])
    assert facts == [], f"不该凭空抽出画像: {facts}"
