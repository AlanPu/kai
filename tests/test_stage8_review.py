"""
复习功能测试：把重复犯的错归纳成可复习的清单。

用户的原话：
  「历史只记录了我的对话主题，但我更想记录的是在值得注意的地方里
    重复犯的一些问题，比如常用的固定说法、固定搭配。」

所以这里测的核心不是"能不能存"，而是：
  · 同一句话重复出现，是不是被合并成一条（而不是刷屏）
  · 偶发一次的错误，是不是被排除在"反复犯的问题"之外
  · 分类是不是稳定（同样的输入永远同一类）
  · 固定搭配有没有被正确识别出来
"""

import pytest

from app.services.review import (ReviewBuilder, bucketize, classify,
                                 dedup_key_for, guess_collocation,
                                 normalize_text, summarize, top_items)
from app.storage.db import Database
from app.storage.models import Correction, ReviewHabitRow, ReviewItem


def mk(kind="grammar", severity="critical", original="", suggestion="",
       explanation="", word=None, session_id=1, created_at=""):
    """造一条纠错。参数多，集中在这里免得每个测试都写一长串。"""
    return Correction(session_id=session_id, kind=kind, severity=severity,
                      original=original, suggestion=suggestion,
                      explanation=explanation, word=word,
                      created_at=created_at)


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "r.db")
    d.init_schema()
    d.create_user("测试")
    yield d
    d.close()


@pytest.fixture
def uid(db):
    return db.list_users()[0].id


# ============================================================
#  文本规范化与去重键
# ============================================================

def test_normalize_ignores_case_and_punctuation():
    """同一句话的不同转写形式必须归到同一条，否则去重失效。"""
    a = normalize_text("I went to the.")
    b = normalize_text("i went to the")
    c = normalize_text("  I went to the  ")
    assert a == b == c


def test_dedup_key_ignores_suggestion():
    """
    去重键只看原句：同一句错话即使模型给了不同改法，
    对用户来说也是"同一个问题"，应该合并。
    """
    k1 = dedup_key_for("I went to the.", "I went to the park.")
    k2 = dedup_key_for("I went to the.", "I went to the mall.")
    assert k1 == k2


# ============================================================
#  分类：规则、稳定、可预期
# ============================================================

def test_classify_fragment_short_and_truncated():
    """悬空收尾（停在冠词/介词上）就是"没说完"的典型特征。"""
    assert classify(mk(original="I went to the.")) == "fragment"
    assert classify(mk(original="The train will arrive at")) == "fragment"
    assert classify(mk(original="I don't need to think about")) == "fragment"


def test_classify_short_noun_phrase_is_not_fragment():
    """
    回归测试：曾经用「≤4 词且无动词」当兜底，
    结果把短名词短语全判成了 fragment。
    这些其实是用词/冠词/介词问题。
    """
    assert classify(mk(original="an old man",
                       kind="vocabulary")) != "fragment"
    assert classify(mk(original="good idea",
                       kind="vocabulary")) != "fragment"
    assert classify(mk(original="in the morning",
                       kind="grammar")) != "fragment"


def test_classify_duplication():
    """相邻重复词是典型的"边想边说"。"""
    assert classify(mk(original="Make make me feel tired")) == "duplication"
    assert classify(mk(original="helps me to use use, to, use the")) \
        == "duplication"


def test_classify_is_deterministic():
    """
    同样的输入必须永远得到同样的分类 ——
    分类是聚合的分组键，飘了的话计数就永远停在 1。
    """
    c = mk(original="I went to the.", explanation="句子不完整")
    assert len({classify(c) for _ in range(20)}) == 1


def test_classify_pronunciation_needs_word():
    """发音问题没有具体单词就无法定位，不归入复习。"""
    assert classify(mk(kind="pronunciation", severity="critical",
                       original="sink", word="think")) == "pronunciation"
    assert classify(mk(kind="pronunciation", severity="critical",
                       original="sink")) is None


def test_classify_ignores_non_visible():
    """ignore 级的纠错不该进复习（corrections.should_show 已表达此规则）。"""
    c = mk(severity="ignore", original="I went to the.")
    assert not c.should_show()


# ============================================================
#  固定搭配识别
# ============================================================

def test_collocation_keeps_real_phrase():
    """真正的固定说法要留下来。"""
    assert guess_collocation("one page by one one page",
                             "one page at a time") == "one page at a time"
    assert guess_collocation("tennis game", "tennis") == "tennis"
    assert guess_collocation("on the regular weekends",
                             "on weekends") == "on weekends"


def test_collocation_rejects_full_sentence_rewrite():
    """
    整句重写不是搭配 —— 收进来会让「搭配」列变成 suggestion 的复读机。
    反例：一个碎句被重写成完整句子。
    """
    assert guess_collocation(
        "The meeting, and then.",
        "For me, the meeting was important because...") is None
    assert guess_collocation(
        "I went to the.", "I went to the park.") is None
    assert guess_collocation(
        "The plan, results.",
        "It will affect the results.") is None


# ============================================================
#  归纳：只留"反复犯的"
# ============================================================

def test_bucketize_needs_repetition():
    """
    核心规则：出现 2 次及以上才算"反复犯的问题"。
    偶发一次的错误混进来只会淹没真正的老毛病。
    """
    corr = [
        mk(original="I went to the."),          # fragment ×2
        mk(original="I went to the."),
        mk(original="an old man", kind="vocabulary"),  # 只出现一次
    ]
    buckets = bucketize(corr)
    assert "fragment" in buckets
    assert buckets["fragment"].occurrences == 2
    assert "word_choice" not in buckets


def test_summarize_sorted_by_occurrences():
    """最该复习的（次数最多）排最前面。"""
    corr = [mk(original="I went to the.") for _ in range(5)]
    corr += [mk(original="Make make me tired") for _ in range(2)]
    habits = summarize(corr)
    assert habits[0].habit == "fragment"
    assert habits[0].occurrences == 5
    assert habits[0].label == "句子说完整"
    assert habits[0].advice            # 每类都该有"怎么办"


def test_summarize_tracks_first_and_last_seen():
    """要能回答"这个问题跟了我多久"。"""
    corr = [
        mk(original="I went to the.", created_at="2026-01-01T10:00:00"),
        mk(original="I went to the.", created_at="2026-03-01T10:00:00"),
    ]
    h = summarize(corr)[0]
    assert h.first_seen == "2026-01-01T10:00:00"
    assert h.last_seen == "2026-03-01T10:00:00"


# ============================================================
#  条目合并：同一句话不能刷屏
# ============================================================

def test_top_items_merges_duplicates():
    """
    同一句话重复 12 次时，复习清单里必须是 **一行**（次数 12），
    而不是 12 行。
    """
    corr = [mk(original="I went to the.",
               suggestion="I went to the park.") for _ in range(12)]
    items = top_items(corr)
    assert len(items) == 1
    _, it = items[0]
    assert it.occurrences == 12
    assert it.original == "I went to the."


def test_top_items_drops_noop_corrections():
    """改了等于没改的（建议与原句相同）没有复习价值。"""
    corr = [mk(original="after I get home",
               suggestion="after I get home") for _ in range(3)]
    assert top_items(corr) == []


def test_top_items_picks_majority_suggestion():
    """同一句有多个改法时，取出现最多的那个（多数派更稳）。"""
    corr = [mk(original="I went to the.", suggestion="I went to the park."),
            mk(original="I went to the.", suggestion="I went to the park."),
            mk(original="I went to the.", suggestion="I went to the mall.")]
    _, it = top_items(corr)[0]
    assert it.suggestion == "I went to the park."


# ============================================================
#  写库：幂等与累加
# ============================================================

def seed(db, uid, n_frag=3):
    """造一次带纠错的会话。"""
    sid = db.create_session(uid, "topic", "测试")
    for i in range(n_frag):
        db.add_correction(Correction(
            session_id=sid, kind="grammar", severity="critical",
            original="I went to the.", suggestion="I went to the park.",
            explanation="句子不完整"))
    return sid


def test_build_creates_habits_and_items(db, uid):
    seed(db, uid, 3)
    result = ReviewBuilder(db).build(uid)
    assert result["habits"] >= 1
    habits = db.list_review_habits(uid)
    assert habits[0].habit == "fragment"
    items = db.list_review_items(uid)
    assert len(items) == 1              # 合并成一条
    assert items[0].occurrences == 3


def test_build_reset_is_idempotent(db, uid):
    """
    reset 重建必须可重复执行而计数不虚高 ——
    次数是复习优先级，虚高会误导。
    """
    seed(db, uid, 4)
    b = ReviewBuilder(db)
    b.build(uid, reset=True)
    first = db.list_review_habits(uid)[0].occurrences
    b.build(uid, reset=True)
    second = db.list_review_habits(uid)[0].occurrences
    assert first == second == 4
    assert len(db.list_review_items(uid)) == 1


def test_build_min_occurrences_filters(db, uid):
    """调高门槛只看老毛病。"""
    seed(db, uid, 4)
    result = ReviewBuilder(db).build(uid, reset=True, min_occurrences=10)
    assert result["habits"] == 0
    assert db.list_review_habits(uid) == []


def test_build_only_uses_own_corrections(db, uid):
    """
    归纳必须按用户隔离 —— 把别人的错误算到你头上，
    复习清单会莫名其妙多出没犯过的毛病。
    """
    other = db.create_user("另一个人")
    seed(db, other, 5)
    result = ReviewBuilder(db).build(uid)
    assert result["habits"] == 0
    assert db.list_review_habits(uid) == []


def test_incremental_build_picks_up_new_corrections(db, uid):
    """
    会话结束时走的是增量归纳（reset=False）——
    新练出来的重复问题必须能进来。
    """
    seed(db, uid, 2)
    b = ReviewBuilder(db)
    b.build(uid, reset=True)
    assert [h.habit for h in db.list_review_habits(uid)] == ["fragment"]

    # 下一次练习又犯了同一件事
    seed(db, uid, 3)
    b.build(uid, reset=False)
    h = db.list_review_habits(uid)[0]
    assert h.habit == "fragment"
    assert h.occurrences >= 2          # 计数累加，没有丢


# ============================================================
#  已掌握：不能被后续归纳覆盖
# ============================================================

def test_mastered_survives_incremental_build(db, uid):
    """
    "已改掉"打过勾之后，后面再归纳不能把它重新变成待复习 ——
    否则用户的复习成果每次练习都被清空。
    """
    seed(db, uid, 3)
    b = ReviewBuilder(db)
    b.build(uid, reset=True)
    hid = db.list_review_habits(uid)[0].id
    db.set_review_mastered(uid, habit_id=hid, mastered=True)
    assert db.list_review_habits(uid) == []

    # 增量归纳（不是 reset）
    b.build(uid, reset=False)
    assert db.list_review_habits(uid) == []
    assert len(db.list_review_habits(uid, include_mastered=True)) == 1


def test_set_mastered_respects_user_scope(db, uid):
    """不能通过传别人的 id 改到别人的记录。"""
    other = db.create_user("另一个人")
    seed(db, other, 3)
    ReviewBuilder(db).build(other, reset=True)
    hid = db.list_review_habits(other)[0].id

    # 用 uid 去标记 other 的 habit → 不该成功
    assert db.set_review_mastered(uid, habit_id=hid, mastered=True) == 0
    assert len(db.list_review_habits(other)) == 1


def test_set_mastered_missing_id_returns_zero(db, uid):
    assert db.set_review_mastered(uid, habit_id=999999, mastered=True) == 0
    assert db.set_review_mastered(uid, item_id=999999, mastered=True) == 0


def test_count_review(db, uid):
    seed(db, uid, 3)
    ReviewBuilder(db).build(uid, reset=True)
    c = db.count_review(uid)
    assert c["habits"] == 1
    assert c["items"] == 1
    assert c["occurrences"] == 3
    assert c["mastered"] == 0


# ============================================================
#  存储层：去重与累加
# ============================================================

def test_upsert_habit_accumulates(db, uid):
    h1 = db.upsert_review_habit(ReviewHabitRow(
        user_id=uid, habit="fragment", title="句子说完整",
        occurrences=2, last_seen="2026-01-01"))
    h2 = db.upsert_review_habit(ReviewHabitRow(
        user_id=uid, habit="fragment", title="句子说完整",
        occurrences=3, last_seen="2026-02-01"))
    assert h1 == h2                     # 同一类只有一行
    h = db.list_review_habits(uid)[0]
    assert h.occurrences == 5
    assert h.last_seen == "2026-02-01"


def test_upsert_item_dedups_by_key(db, uid):
    i1 = db.upsert_review_item(ReviewItem(
        user_id=uid, dedup_key="i went to the",
        original="I went to the.", suggestion="I went to the park."))
    i2 = db.upsert_review_item(ReviewItem(
        user_id=uid, dedup_key="i went to the",
        original="I went to the.", suggestion="I went to the mall."))
    assert i1 == i2
    items = db.list_review_items(uid)
    assert len(items) == 1
    assert items[0].occurrences == 2
    # 建议更新为最新一次
    assert items[0].suggestion == "I went to the mall."


def test_clear_review_only_clears_own(db, uid):
    other = db.create_user("另一个人")
    db.upsert_review_habit(ReviewHabitRow(
        user_id=uid, habit="fragment", title="x", occurrences=2))
    db.upsert_review_habit(ReviewHabitRow(
        user_id=other, habit="fragment", title="x", occurrences=2))
    db.clear_review(uid)
    assert db.list_review_habits(uid) == []
    assert len(db.list_review_habits(other)) == 1


def test_delete_user_cascades_review(db, uid):
    """删用户要连复习数据一起删（schema 里写了 ON DELETE CASCADE）。"""
    db.upsert_review_habit(ReviewHabitRow(
        user_id=uid, habit="fragment", title="x", occurrences=2))
    db.upsert_review_item(ReviewItem(
        user_id=uid, dedup_key="k", original="a", suggestion="b"))
    assert db.count_user_data(uid)["sessions"] == 0
    db.delete_user(uid)
    assert db.list_review_habits(uid, include_mastered=True) == []
    assert db.list_review_items(uid, include_mastered=True) == []


# ============================================================
#  list_corrections_for_user
# ============================================================

def test_list_corrections_for_user_spans_sessions(db, uid):
    """跨会话取全部纠错 —— 复习归纳要"跨会话看重复"。"""
    seed(db, uid, 2)
    seed(db, uid, 3)
    assert len(db.list_corrections_for_user(uid)) == 5
    assert len(db.list_corrections_for_user(uid, session_id=1)) == 2
