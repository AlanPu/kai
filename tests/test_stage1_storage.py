"""
阶段 1 验收测试：数据层。

验收标准（来自开发计划）：能写入并读回一次模拟会话。
"""

import json

import pytest

from app.storage.db import Database
from app.storage.models import Correction, ProfileFact, Session, Turn


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.init_schema()
    yield d
    d.close()


@pytest.fixture
def uid(db):
    """一个测试用户。会话和画像都必须挂在用户下。"""
    return db.create_user("测试用户")


# ============================================================
#  验收：完整往返
# ============================================================

def test_full_session_roundtrip(db, uid):
    """写入一次模拟会话，再完整读回。"""
    sid = db.create_session(
        uid, input_kind="article", input_raw="https://example.com/ai",
        input_title="The Future of AI",
        input_content="Artificial intelligence is transforming...")
    assert sid > 0

    db.save_plan(sid, {"topics": ["AI 对工作的影响", "你用过哪些 AI 工具"],
                       "opening": "I read an article about AI today."})

    t1 = db.add_turn(Turn(session_id=sid, seq=1, role="assistant",
                          text="I read an article about AI today. "
                               "What do you think about it?"))
    t2 = db.add_turn(Turn(session_id=sid, seq=2, role="user",
                          text="I think AI is very useful. I go to work by AI.",
                          audio_ms=4200))

    db.add_correction(Correction(
        session_id=sid, turn_id=t2, kind="grammar", severity="critical",
        original="I go to work by AI", suggestion="I use AI at work",
        explanation="go to work 是「去上班」，表达「使用」应该用 use"))

    db.refresh_session_stats(sid)
    db.finish_session(sid, duration_sec=1800, usage={"tokens": 12345})

    # ---- 读回会话 ----
    s = db.get_session(sid)
    assert s is not None
    assert s.input_kind == "article"
    assert s.input_title == "The Future of AI"
    assert s.status == "finished"
    assert s.duration_sec == 1800
    assert s.turn_count == 2
    assert s.user_char_count == len("I think AI is very useful. I go to work by AI.")

    plan = json.loads(s.plan_json)
    assert len(plan["topics"]) == 2

    # ---- 读回轮次 ----
    turns = db.list_turns(sid)
    assert [t.role for t in turns] == ["assistant", "user"]
    assert turns[1].audio_ms == 4200

    # ---- 读回纠错 ----
    cs = db.list_corrections(sid)
    assert len(cs) == 1
    assert cs[0].kind == "grammar"
    assert cs[0].should_show()

    assert json.loads(s.usage_json)["tokens"] == 12345


def test_auto_sequence_numbering(db, uid):
    """seq 不传时应自动递增。"""
    sid = db.create_session(uid, "topic", "聊天气")
    for i in range(3):
        db.add_turn(Turn(session_id=sid, role="user", text=f"line {i}"))
    turns = db.list_turns(sid)
    assert [t.seq for t in turns] == [1, 2, 3]


# ============================================================
#  需求 4：发音分级规则
# ============================================================

@pytest.mark.parametrize("kind,severity,expected", [
    # 发音：只有明显错误才纠正（需求 4 的核心）
    ("pronunciation", "critical", True),
    ("pronunciation", "minor",    False),   # 口音问题不打扰
    ("pronunciation", "ignore",   False),
    # 语法：critical 和 minor 都值得提
    ("grammar",       "critical", True),
    ("grammar",       "minor",    True),
    ("grammar",       "ignore",   False),
    # 地道表达
    ("vocabulary",    "minor",    True),
    ("vocabulary",    "ignore",   False),
])
def test_should_show_rules(kind, severity, expected, uid):
    """发音的 minor 不展示 —— 避免频繁打断（需求 2 + 需求 4）。"""
    c = Correction(kind=kind, severity=severity)
    assert c.should_show() is expected


def test_only_visible_filter(db, uid):
    """只取该展示的纠错时，发音 minor 应被过滤掉。"""
    sid = db.create_session(uid, "topic", "测试")
    db.add_correction(Correction(session_id=sid, kind="pronunciation",
                                 severity="critical", original="think",
                                 suggestion="/θɪŋk/", word="think"))
    db.add_correction(Correction(session_id=sid, kind="pronunciation",
                                 severity="minor", original="again",
                                 suggestion="/əˈɡen/", word="again"))
    db.add_correction(Correction(session_id=sid, kind="grammar",
                                 severity="minor", original="I go",
                                 suggestion="I went"))

    all_c = db.list_corrections(sid)
    visible = db.list_corrections(sid, only_visible=True)
    assert len(all_c) == 3
    assert len(visible) == 2
    assert all(c.severity != "minor" or c.kind != "pronunciation"
               for c in visible)


def test_top_error_words(db, uid):
    """高频错误单词统计（阶段 5 用）。"""
    sid = db.create_session(uid, "topic", "测试")
    for word in ["think", "think", "think", "three", "three", "world"]:
        db.add_correction(Correction(
            session_id=sid, kind="pronunciation", severity="critical",
            original=word, suggestion="?", word=word))
    top = db.top_error_words()
    assert top[0] == ("think", 3)
    assert ("three", 2) in top
    assert dict(top)["world"] == 1


# ============================================================
#  需求 6：个人画像
# ============================================================

def test_profile_upsert_increases_confidence(db, uid):
    """同一事实多次出现 → 置信度上升。"""
    sid = db.create_session(uid, "topic", "聊爱好")
    f = ProfileFact(user_id=uid, category="interest", key="hobby", value="跑步",
                    confidence=0.5, source_session_id=sid)
    fid1 = db.upsert_fact(f)
    fid2 = db.upsert_fact(f)
    assert fid1 == fid2, "同一 key 应更新而非新增"

    facts = db.list_facts(user_id=uid)
    assert len(facts) == 1
    assert facts[0].confidence > 0.5
    assert facts[0].value == "跑步"


def test_profile_confidence_capped(db, uid):
    """置信度不超过 1.0。"""
    for _ in range(20):
        db.upsert_fact(ProfileFact(user_id=uid, category="interest", key="food",
                                   value="火锅", confidence=0.9))
    facts = db.list_facts(user_id=uid)
    assert facts[0].confidence <= 1.0


def test_profile_min_confidence_filter(db, uid):
    """画像可按置信度过滤。"""
    db.upsert_fact(ProfileFact(user_id=uid, category="interest", key="a", value="1",
                               confidence=0.2))
    db.upsert_fact(ProfileFact(user_id=uid, category="interest", key="b", value="2",
                               confidence=0.9))
    assert len(db.list_facts(min_confidence=0.5, user_id=uid)) == 1


# ============================================================
#  约束与完整性
# ============================================================

def test_input_kind_constraint(db, uid):
    """非法输入类型被数据库拒绝。"""
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        db.create_session(uid, "invalid_kind", "x")


def test_cascade_delete(db, uid):
    """删除会话时，轮次和纠错一并删除。"""
    sid = db.create_session(uid, "topic", "x")
    t = db.add_turn(Turn(session_id=sid, role="user", text="hi"))
    db.add_correction(Correction(session_id=sid, turn_id=t,
                                 original="hi", suggestion="hello"))
    db.conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
    db.conn.commit()
    assert db.list_turns(sid) == []
    assert db.list_corrections(sid) == []


def test_list_sessions_ordering(db, uid):
    """会话按时间倒序。"""
    ids = [db.create_session(uid, "topic", f"t{i}") for i in range(3)]
    got = [s.id for s in db.list_sessions()]
    assert got == list(reversed(ids))
