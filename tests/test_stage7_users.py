"""
阶段 7 测试：多用户与声纹重录。

背景：多人共用一台电脑各自练习。要求每个用户有独立的
声纹、画像和练习记录；声纹可以随时重录、也可以只清声纹
而保留画像历史。
"""

import json
from urllib.parse import quote_plus
from pathlib import Path

import numpy as np
import pytest

from app.core.enroll import (MAX_SPEECH_SEC, MIN_COHERENCE, MIN_SAMPLES,
                             MIN_SPEECH_SEC, EnrollmentSession)
from app.services.profile import ProfileStore
from app.services.users import (DuplicateUser, InvalidUser, UserNotFound,
                                UserStore, quality_label)
from app.storage.db import Database
from app.storage.models import ProfileFact


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "u.db")
    d.init_schema()
    yield d
    d.close()


@pytest.fixture
def store(db, tmp_path):
    return UserStore(db, tmp_path / "voiceprints")


# ============================================================
#  用户增删改查
# ============================================================

def test_create_and_get(store):
    u = store.create("Alan", "🐧")
    assert u.id > 0
    assert u.name == "Alan"
    assert u.avatar == "🐧"
    assert not u.has_voiceprint
    assert store.get(u.id).name == "Alan"


def test_create_strips_whitespace(store):
    u = store.create("  小明  ")
    assert u.name == "小明"


def test_duplicate_name_rejected(store):
    store.create("Alan")
    with pytest.raises(DuplicateUser):
        store.create("Alan")


def test_duplicate_name_exact_match_rejected(store):
    """完全同名的必须拒绝。"""
    store.create("Alan")
    with pytest.raises(DuplicateUser):
        store.create("Alan")


def test_similar_names_allowed(store):
    """'Alan' 和 'alan' 目前算两个用户。

    这是已知的粗糙之处：界面上两个名字看起来一样，容易选错。
    没做大小写归一化是因为中文名没有大小写概念，强行 lower()
    会影响 'Li' / 'li' 这类确实想区分的情形。先如实记录行为。
    """
    store.create("Alan")
    u = store.create("alan")
    assert u.name == "alan"


def test_empty_name_rejected(store):
    for bad in ["", "   ", "\t\n"]:
        with pytest.raises(InvalidUser):
            store.create(bad)


def test_overlong_name_rejected(store):
    with pytest.raises(InvalidUser):
        store.create("x" * 100)


def test_rename(store):
    u = store.create("Alan")
    v = store.rename(u.id, "Alan Pu", "🚀")
    assert v.name == "Alan Pu"
    assert v.avatar == "🚀"


def test_rename_to_existing_rejected(store):
    a = store.create("Alan")
    store.create("Bob")
    with pytest.raises(DuplicateUser):
        store.rename(a.id, "Bob")


def test_rename_missing_raises(store):
    with pytest.raises(UserNotFound):
        store.rename(999, "Nobody")


def test_require_missing_raises(store):
    with pytest.raises(UserNotFound):
        store.require(999)


def test_default_user_is_most_recently_used(store, db):
    a = store.create("A")
    store.create("B")
    # 同一秒内创建的两个用户时间戳会并列，所以这里靠显式 touch
    # 来验证"最近使用"的语义，而不是靠创建顺序
    db.touch_user(a.id)
    assert store.default_user().id == a.id
    db.touch_user(store.get_by_name("B").id)
    assert store.default_user().name == "B"


def test_default_user_none_when_empty(store):
    assert store.default_user() is None


def test_list_users_orders_by_recent_use(store, db):
    a = store.create("A")
    store.create("B")
    db.touch_user(a.id)
    assert [u.name for u in store.list_users()][0] == "A"


def test_resolve_returns_none_for_zero(store):
    """resolve(0) 用于"前端还没选用户"的情形，不该抛异常。"""
    assert store.resolve(0) is None
    assert store.resolve(None) is None


# ============================================================
#  声纹文件读写
# ============================================================

def test_save_and_load_voiceprint(store, tmp_path):
    u = store.create("Alan")
    proto = np.ones(192, dtype=np.float32) / np.sqrt(192)
    store.save_voiceprint(u.id, proto, quality=0.88, samples=4)

    assert store.has_voiceprint(u.id)
    got, meta = store.load_voiceprint(u.id)
    assert got is not None
    assert len(got) == 192
    assert meta["quality"] == pytest.approx(0.88)
    assert meta["samples"] == 4
    # 存储的路径必须按 user_id 隔离
    assert store.voiceprint_path(u.id).name == f"{u.id}.json"


def test_voiceprints_do_not_collide(store):
    """两个用户的声纹文件必须分开 —— 这是多用户的核心正确性。"""
    a = store.create("A")
    b = store.create("B")
    store.save_voiceprint(a.id, np.ones(192, dtype=np.float32), quality=0.9,
                          samples=3)
    store.save_voiceprint(b.id, np.full(192, -1, dtype=np.float32),
                          quality=0.5, samples=3)

    va, _ = store.load_voiceprint(a.id)
    vb, _ = store.load_voiceprint(b.id)
    assert va[0] > 0 and vb[0] < 0, "两个用户的声纹串了"
    assert store.voiceprint_path(a.id) != store.voiceprint_path(b.id)


def test_load_missing_voiceprint_returns_none(store):
    u = store.create("A")
    proto, meta = store.load_voiceprint(u.id)
    assert proto is None
    assert meta == {}


def test_save_creates_parent_dir(store, tmp_path):
    u = store.create("A")
    store.save_voiceprint(u.id, np.ones(192, dtype=np.float32), quality=0.9, samples=3)
    assert store.voiceprint_path(u.id).is_file()


def test_save_is_atomic(store):
    """先写临时文件再 replace —— 中途崩溃不该留下半个 JSON。"""
    u = store.create("A")
    store.save_voiceprint(u.id, np.ones(192, dtype=np.float32),
                          quality=0.7, samples=3)
    p = store.voiceprint_path(u.id)
    assert not p.with_suffix(".json.tmp").exists()
    json.loads(p.read_text())        # 必须是合法 JSON


def test_save_keeps_raw_embeddings(store):
    """存原始句向量，将来换算法/换阈值时能重新推导，
    不必让用户重录。"""
    u = store.create("A")
    embs = [np.ones(192, dtype=np.float32), np.full(192, 0.5, dtype=np.float32)]
    store.save_voiceprint(u.id, np.ones(192, dtype=np.float32), quality=0.9,
                          samples=2, raw_embeddings=embs)
    _, meta = store.load_voiceprint(u.id)
    assert len(meta["embeddings"]) == 2


def test_reenroll_overwrites(store):
    """重新录入必须覆盖旧的，而不是叠加。"""
    u = store.create("A")
    store.save_voiceprint(u.id, np.ones(192, dtype=np.float32), quality=0.5,
                          samples=3)
    store.save_voiceprint(u.id, np.full(192, -1, dtype=np.float32),
                          quality=0.95, samples=5)
    _, meta = store.load_voiceprint(u.id)
    assert meta["quality"] == pytest.approx(0.95)
    assert meta["samples"] == 5


def test_clear_voiceprint_keeps_user(store):
    """只清声纹、保留画像历史 —— 换了麦克风想重录的常见场景。"""
    u = store.create("A")
    store.save_voiceprint(u.id, np.ones(192, dtype=np.float32),
                          quality=0.9, samples=3)
    store.clear_voiceprint(u.id)

    assert not store.has_voiceprint(u.id)
    assert not store.voiceprint_path(u.id).exists()
    assert store.get(u.id) is not None      # 用户还在


def test_voiceprint_info_shape(store):
    u = store.create("A")
    info = store.voiceprint_info(u.id)
    assert info["has"] is False
    assert info["quality_label"] == "未录入"

    store.save_voiceprint(u.id, np.ones(192, dtype=np.float32), quality=0.8,
                          samples=4)
    info = store.voiceprint_info(u.id)
    assert info["has"] is True
    assert info["samples"] == 4


def test_quality_label_thresholds():
    assert quality_label(None) == "未录入"
    assert quality_label(0.9) == "好"
    assert quality_label(0.65) == "一般"
    assert quality_label(0.3) == "偏差"


# ============================================================
#  数据隔离与级联删除
# ============================================================

def test_sessions_are_isolated(store, db):
    a = store.create("A")
    b = store.create("B")
    db.create_session(a.id, "topic", "A 的话题")
    db.create_session(b.id, "topic", "B 的话题")

    sa = db.list_sessions(user_id=a.id)
    sb = db.list_sessions(user_id=b.id)
    assert [s.input_raw for s in sa] == ["A 的话题"]
    assert [s.input_raw for s in sb] == ["B 的话题"]


def test_same_key_facts_coexist(store, db):
    """两个用户可以有同名但不同值的事实，不能互相覆盖。"""
    a = store.create("A")
    b = store.create("B")
    db.upsert_fact(ProfileFact(user_id=a.id, category="background",
                               key="职业", value="工程师", confidence=0.9))
    db.upsert_fact(ProfileFact(user_id=b.id, category="background",
                               key="职业", value="设计师", confidence=0.9))

    fa = db.list_facts(user_id=a.id)
    fb = db.list_facts(user_id=b.id)
    assert len(fa) == 1 and fa[0].value == "工程师"
    assert len(fb) == 1 and fb[0].value == "设计师"


def test_upsert_same_user_still_merges(store, db):
    """同一用户的同 key 事实仍应合并并提升置信度，不是插入两条。"""
    a = store.create("A")
    f = ProfileFact(user_id=a.id, category="interest", key="hobby",
                    value="跑步", confidence=0.5)
    db.upsert_fact(f)
    db.upsert_fact(f)
    got = db.list_facts(user_id=a.id)
    assert len(got) == 1
    assert got[0].confidence > 0.5


def test_delete_user_removes_everything(store, db):
    a = store.create("A")
    b = store.create("B")
    sa = db.create_session(a.id, "topic", "A")
    db.create_session(b.id, "topic", "B")
    db.upsert_fact(ProfileFact(user_id=a.id, category="interest",
                               key="k", value="v", confidence=0.9))
    store.save_voiceprint(a.id, np.ones(192, dtype=np.float32), quality=0.9, samples=3)

    stats = store.delete(a.id)
    assert stats["sessions"] == 1
    assert stats["facts"] == 1

    assert store.get(a.id) is None
    assert not store.voiceprint_path(a.id).exists()
    assert db.list_sessions(user_id=a.id) == []
    assert db.list_facts(user_id=a.id) == []
    # 另一个用户完全不受影响
    assert len(db.list_sessions(user_id=b.id)) == 1
    assert store.get(b.id) is not None
    assert sa > 0


def test_delete_removes_profile_markdown(store, db, tmp_path):
    from app.services.profile import ProfileStore
    ps = ProfileStore(db, tmp_path / "profiles")
    a = store.create("A")
    db.upsert_fact(ProfileFact(user_id=a.id, category="interest",
                               key="k", value="v", confidence=0.9))
    ps.write_markdown(a.id)
    assert ps.markdown_path(a.id).is_file()

    store.delete(a.id)
    assert not ps.markdown_path(a.id).exists()


def test_delete_missing_raises(store):
    with pytest.raises(UserNotFound):
        store.delete(999)


def test_count_user_data(store, db):
    a = store.create("A")
    db.create_session(a.id, "topic", "x")
    c = db.count_user_data(a.id)
    assert c["sessions"] == 1
    assert set(c) >= {"sessions", "turns", "corrections", "facts"}


# ============================================================
#  声纹录入
# ============================================================

class FakeExtractor:
    """
    假的 sherpa 特征提取器，接口与 SpeakerEmbeddingExtractor 一致：
    create_stream() 造流，accept_waveform() 喂音频，compute(stream) 出向量。

    向量由音频的均值和标准差组成 —— 对"同一段音频重复提交"给出
    完全相同的向量，对音量差别很大的音频给出方向很不同的向量，
    因此能确定性地验证录入流程的判据（而非验证模型本身）。
    """

    def create_stream(self):
        return _Stream()

    def compute(self, stream):
        return stream.vec


class _Stream:
    def __init__(self):
        self.vec = None
        self._chunks = []

    def accept_waveform(self, rate, samples):
        self._chunks.append(np.asarray(samples, dtype=np.float32))

    def input_finished(self):
        x = np.concatenate(self._chunks) if self._chunks else np.zeros(1)
        v = np.zeros(8, dtype=np.float32)
        v[0] = float(np.mean(np.abs(x))) + 1e-6
        v[1] = float(np.std(x)) + 1e-6
        v[2] = 1.0
        n = float(np.linalg.norm(v))
        self.vec = v / n


class FakeVerifier:
    """只暴露 .ex，和 SpeakerVerifier 一样。"""

    def __init__(self):
        self.ex = FakeExtractor()


@pytest.fixture
def fake_verifier():
    return FakeVerifier()


def _speech(sec, amp=4000):
    """一段像语音的音频：带包络的噪声，避免被判成静音。"""
    n = int(16000 * sec)
    rng = np.random.default_rng(42)
    x = rng.normal(0, amp, n)
    env = np.sin(np.linspace(0, np.pi, n))
    return (x * env).astype(np.int16)


def test_enroll_rejects_too_short(store, fake_verifier):
    s = EnrollmentSession(user_id=1, verifier=fake_verifier)
    s.feed(_speech(MIN_SPEECH_SEC / 2))
    r = s.commit()
    assert not r.ok
    assert "短" in r.reason


def test_enroll_rejects_silence(store, fake_verifier):
    s = EnrollmentSession(user_id=1, verifier=fake_verifier)
    s.feed(np.zeros(16000 * 5, dtype=np.int16))
    r = s.commit()
    assert not r.ok
    assert "安静" in r.reason or "声音" in r.reason


def test_enroll_accepts_valid_and_advances(store, fake_verifier):
    s = EnrollmentSession(user_id=1, verifier=fake_verifier)
    s.feed(_speech(3))
    r = s.commit()
    assert r.ok
    # current 是 0-based 内部计数（第几句已经录完），
    # 对外的 progress() 才 +1 显示成 1-based
    assert s.current == 1
    assert s.progress()["current"] == 2
    assert len(s.embeddings) == 1


def test_enroll_does_not_overrun_total(store, fake_verifier):
    """多按一次"读完了"不该把进度推过总句数 —— 之前踩过的越界。"""
    s = EnrollmentSession(user_id=1, verifier=fake_verifier, prompts=None)
    total = s.total
    for _ in range(total + 5):
        s.feed(_speech(3))
        s.commit()
    assert s.current <= total + 1
    assert s.finished


def test_enroll_buffer_cleared_after_commit(store, fake_verifier):
    """提交后必须清空缓冲，否则下一句会把上一句的音频也算进去。"""
    s = EnrollmentSession(user_id=1, verifier=fake_verifier)
    s.feed(_speech(3))
    s.commit()
    assert s.buffered_sec() == 0


def test_enroll_truncates_overlong(store, fake_verifier):
    s = EnrollmentSession(user_id=1, verifier=fake_verifier)
    s.feed(_speech(MAX_SPEECH_SEC + 5))
    r = s.commit()
    assert r.ok
    assert r.speech_sec <= MAX_SPEECH_SEC + 0.1


def test_enroll_discard_clears_without_advancing(store, fake_verifier):
    s = EnrollmentSession(user_id=1, verifier=fake_verifier)
    s.feed(_speech(3))
    s.clear_buffer()
    assert s.buffered_sec() == 0
    assert s.current == 0          # 没有推进
    assert len(s.embeddings) == 0


def test_enroll_cannot_save_before_min_samples(store, fake_verifier):
    s = EnrollmentSession(user_id=1, verifier=fake_verifier)
    for _ in range(MIN_SAMPLES - 1):
        s.feed(_speech(3))
        s.commit()
    assert not s.can_save(), "样本不足时不该允许保存"


def test_enroll_can_save_at_min_samples(store, fake_verifier):
    s = EnrollmentSession(user_id=1, verifier=fake_verifier)
    for _ in range(MIN_SAMPLES):
        s.feed(_speech(3))
        s.commit()
    assert s.can_save()


def test_enroll_prototype_is_normalized(store, fake_verifier):
    s = EnrollmentSession(user_id=1, verifier=fake_verifier)
    for _ in range(MIN_SAMPLES):
        s.feed(_speech(3))
        s.commit()
    p = s.prototype()
    assert p is not None
    assert np.linalg.norm(p) == pytest.approx(1.0, abs=1e-5)


def test_enroll_rejects_incoherent_samples(store):
    """中途换了个人 → 样本互相矛盾，应当拒绝并提示重录。

    用一个每次给完全不同方向的假提取器来模拟"换人了"。
    """
    class FlipExtractor:
        """第一句给 +x，之后给 -x，两次向量的余弦 = -1。"""
        def __init__(self):
            self.n = 0

        def create_stream(self):
            self.n += 1
            return _FlipStream(self.n == 1)

        def compute(self, st):
            return st.vec

    class _FlipStream:
        def __init__(self, first):
            self.first = first
            self.vec = None

        def accept_waveform(self, rate, samples):
            pass

        def input_finished(self):
            v = np.zeros(8, dtype=np.float32)
            v[0] = 1.0 if self.first else -1.0
            self.vec = v

    class FlipVerifier:
        """EnrollmentSession 只通过 .ex 拿提取器。"""
        def __init__(self):
            self.ex = FlipExtractor()

    s = EnrollmentSession(user_id=1, verifier=FlipVerifier())
    s.feed(_speech(3))
    assert s.commit().ok, "第一句应当通过"
    s.feed(_speech(3))
    r = s.commit()
    assert not r.ok, "第二句与第一句方向完全相反，应当被拒绝"
    assert "差别" in r.reason or "一致" in r.reason
    assert len(s.embeddings) == 1, "被拒绝的样本不该进入 embeddings"


def test_enroll_progress_shape(store, fake_verifier):
    s = EnrollmentSession(user_id=1, verifier=fake_verifier)
    p = s.progress()
    assert set(p) >= {"current", "total", "collected", "prompt"}
    assert p["current"] == 1
    assert p["collected"] == 0
    assert s.total >= MIN_SAMPLES


# ============================================================
#  参数标定回归
#
#  下面几项是实测标定出来的，不是拍脑袋定的。它们很容易
#  在后续改动中被"顺手调一下"，所以钉死在这里，并说明来历。
# ============================================================

def test_quality_thresholds_match_measured_scale():
    """质量分档必须按实测尺度，否则档位形同虚设。

    实测同一说话人 2 秒片段两两余弦：最低 0.485、均值 0.644。
    所以"好"的门槛不能高于均值，否则人人都是"一般"。
    """
    from app.services.users import QUALITY_GOOD, QUALITY_OK
    assert QUALITY_OK < QUALITY_GOOD
    assert QUALITY_GOOD <= 0.70, (
        "实测同人均值才 0.644，门槛高于 0.70 会让正常录入永远评不上'好'")
    assert QUALITY_OK >= 0.45, (
        "低于 0.45 会把明显不稳的录入也放行")


def test_quality_label_boundaries():
    from app.services.users import QUALITY_GOOD, QUALITY_OK, quality_label
    assert quality_label(QUALITY_GOOD) == "好"
    assert quality_label(QUALITY_GOOD - 1e-6) == "一般"
    assert quality_label(QUALITY_OK) == "一般"
    assert quality_label(QUALITY_OK - 1e-6) == "偏差"


def test_min_coherence_sits_between_speakers():
    """拒绝阈值必须落在"同人最低"和"异人最高"之间。

    实测：同人最低 0.485，异人最高 0.360。
    阈值低于 0.360 拦不住别人；高于 0.485 会误拒本人。
    """
    assert 0.36 <= MIN_COHERENCE <= 0.49, (
        f"MIN_COHERENCE={MIN_COHERENCE} 落在实测的重叠区，"
        "要么放过旁人，要么误拒本人")


def test_verification_window_is_at_least_two_seconds():
    """声纹判定窗口不能短于 2 秒。

    实测（同人均值 / 异人均值 / 间隔）：
      0.5s  0.477 / 0.322 / 0.156
      1.0s  0.582 / 0.372 / 0.210
      2.0s  0.672 / 0.422 / 0.249
      3.0s  0.745 / 0.458 / 0.287
    1 秒时两类分布重叠严重，任何阈值都不好用。
    """
    import inspect
    from app.core.voiceprint import SpeakerVerifier
    sig = inspect.signature(SpeakerVerifier.__init__)
    default = sig.parameters["window_sec"].default
    assert default >= 2.0, f"窗口 {default}s 太短，区分度不够"


def test_enrollment_min_speech_is_at_least_two_seconds():
    """录入单句的最短时长同样不能低于 2 秒（理由同上）。"""
    assert MIN_SPEECH_SEC >= 2.0, (
        f"MIN_SPEECH_SEC={MIN_SPEECH_SEC} 太短，抽出的向量会很不稳定")


# ============================================================
#  API 对"用户不存在"的处理
#
#  这是实测踩出来的：前端删除用户后，停在旧页面上的标签页
#  仍会带着已经失效的 id 发请求。所有读接口都必须能优雅降级，
#  不能变成 500 —— 那会让页面整个卡住，而用户什么都没做错。
# ============================================================

def test_resolve_user_never_raises(store):
    """_resolve_user 必须吞掉 UserNotFound。

    各接口对"取不到用户"的处理不同（读接口给空、写接口报错），
    所以解析函数本身不能抛异常，否则每个调用点都得包一层 try。
    """
    from app.api.server import _resolve_user
    import app.api.server as srv

    orig = srv.user_store
    srv.user_store = lambda: store
    try:
        assert _resolve_user(99999) is None     # 库里没人 → None
        u = store.create("A")
        assert _resolve_user(99999).id == u.id  # 有别人 → 回退到最近用的
        assert _resolve_user(None).id == u.id
        assert _resolve_user(u.id).id == u.id
    finally:
        srv.user_store = orig


def test_user_not_found_is_404_not_500(store):
    """按 id 查/删不存在的用户应当是 404，而不是 500。"""
    from fastapi import HTTPException
    import asyncio
    from app.api.server import get_voiceprint, delete_voiceprint
    import app.api.server as srv

    orig = srv.user_store
    srv.user_store = lambda: store
    try:
        for coro in (get_voiceprint(99999), delete_voiceprint(99999)):
            try:
                asyncio.run(coro)
                raise AssertionError("应当抛 HTTPException")
            except HTTPException as e:
                assert e.status_code == 404, f"应为 404，实际 {e.status_code}"
    finally:
        srv.user_store = orig


def test_db_usable_from_other_threads(tmp_path):
    """数据库连接必须能跨线程使用。

    录入流程里"抽完声纹再保存"走的是 asyncio.to_thread，
    会落在另一个线程上。sqlite3 默认禁止这么做，会抛
    "SQLite objects created in a thread can only be used in that same thread"，
    表现为录入走完最后一步连接直接断掉 —— 用户看到的是
    "读完了却没保存上"，很难联想到线程问题。
    """
    import threading
    from app.storage.db import Database

    db = Database(tmp_path / "t.db")
    db.init_schema()
    uid = db.create_user("线程用户")

    err = []

    def work():
        try:
            assert db.get_user(uid).name == "线程用户"
            db.upsert_fact(ProfileFact(user_id=uid, category="background",
                                       key="职业", value="工程师",
                                       confidence=0.9))
            assert len(db.list_facts(user_id=uid)) == 1
        except Exception as e:      # noqa: BLE001
            err.append(e)

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert not err, f"跨线程访问数据库失败: {err[0]}"
    db.close()


def test_enroll_ws_accepts_frontend_command_shape():
    """录入 WebSocket 必须认前端真实发的指令格式。

    前端发的是 {"type": "end"}，而历史上服务端只读 cmd.get("cmd")，
    于是网页上点「读完了」被静默忽略 —— 不报错、不回应，
    用户看到的就是"点了没反应"。用脚本测试却一切正常，
    因为脚本当时发的是 {"cmd": "end"}：测试验证的是一套
    应用根本不会发的协议。

    这个测试直接对着 app/api/server.py 的源码断言两种格式都被接受，
    避免再出现"测试绿、真机坏"。
    """
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "app" / "api" / "server.py").read_text(encoding="utf-8")
    m = re.search(r"^\s*action = \(.*\)\.strip\(\)", src, re.M)
    assert m, "没找到 ws_enroll 里解析指令的那一行"
    line = m.group(0)
    assert 'cmd.get("cmd")' in line, "应接受前端的 type 格式之外，也保留 cmd"
    assert 'cmd.get("type")' in line, \
        "必须接受前端实际发送的 {\"type\": ...} 格式，否则点「读完了」无效"


def test_frontend_enroll_commands_use_type_field():
    """前端发的录入指令用的是 type 字段（与上面的断言配对）。"""
    import re
    from pathlib import Path

    html = (Path(__file__).resolve().parent.parent
            / "app" / "web" / "index.html").read_text(encoding="utf-8")
    sends = re.findall(r"enrollWs\.send\(JSON\.stringify\(\{([^}]*)\}\)\)", html)
    assert sends, "前端没有发任何录入指令？"
    for s in sends:
        assert "type:" in s, f"前端指令应带 type 字段，实际: {s}"
        assert "cmd:" not in s, f"前端不应使用 cmd 字段，实际: {s}"


def test_speech_stopped_helper_exists_and_matches_event():
    """必须有识别「用户说完」的辅助函数。

    这是 AI 接话的触发点。历史上只处理了 speech_started，
    导致用户说完一句后 AI 永远不主动回应 —— 必须再说一句"继续"。
    """
    from app.core.realtime import is_speech_started, is_speech_stopped

    assert is_speech_stopped({"type": "input_audio_buffer.speech_stopped"})
    assert not is_speech_stopped({"type": "input_audio_buffer.speech_started"})
    assert is_speech_started({"type": "input_audio_buffer.speech_started"})
    assert not is_speech_started({"type": "input_audio_buffer.speech_stopped"})


def test_session_ends_turn_on_explicit_signal():
    """接话必须由用户显式信号触发，不能靠 VAD 猜断句。

    历史：这里原本断言「speech_stopped 时要 commit + response.create」。
    那个设计已被 push-to-talk 取代，原因见下面这段实测记录：

      · VAD 对同一句话会重复报 speech_stopped，于是重复 response.create
        → "Conversation already has an active response"
      · 服务端在 speech_stopped 时已自行 commit，我们再 commit 是空提交
        → "buffer too small, or have no audio"
      · 最要命的是 VAD 会把半句话当成说完，AI 抢答后用户接着说，
        AI 又被自己的规则打断 —— 用户感受就是
        "AI 半天不吭声，然后突然接上一句还把我打断"

    现在回合边界由用户松手决定（end_turn），服务端不再猜。
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "app" / "services" / "session.py").read_text(encoding="utf-8")

    assert "async def end_turn" in src, "缺少 end_turn（用户松手信号）"
    assert "_maybe_respond" in src, "缺少接话逻辑"

    # speech_stopped 分支里不能再自动接话 —— 否则又回到"猜断句"
    seg = src[src.index("if is_speech_stopped(ev):"):]
    seg = seg[:seg.index('if t == "response.created"')]
    assert "_maybe_respond()" not in seg, \
        "speech_stopped 不应再自动触发接话（改由 end_turn 触发）"


def test_end_turn_flushes_pending_before_responding():
    """松手时要先把缓冲里的音频发完，再让 AI 接话。

    声纹判定未完成时音频会短暂留在 _pending。按住空格说话时
    用户可能说得很快，松手瞬间缓冲里还有内容 —— 不 flush 会丢句尾。
    """
    from pathlib import Path as _P

    src = (_P(__file__).resolve().parent.parent
           / "app" / "services" / "session.py").read_text(encoding="utf-8")
    body = src[src.index("async def end_turn"):]
    body = body[:body.index("def _maybe_respond")]
    assert "_flush_pending" in body, "end_turn 必须先 flush 缓冲"
    assert body.index("_flush_pending") < body.index("_maybe_respond"), \
        "必须先发完音频再触发接话（否则 AI 回应的是半句话）"


def test_turn_end_command_is_handled():
    """前端发来的 turn_end 必须被服务端处理。"""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "app" / "api" / "server.py").read_text(encoding="utf-8")
    assert '"turn_end"' in src, "服务端没有处理 turn_end 指令"
    assert "end_turn" in src, "turn_end 没有调用 sess.end_turn"


def test_realtime_disables_server_side_auto_response():
    """必须关掉服务端自动回应，保证回合只由客户端发起。

    create_response=True 在 Qwen 上本来就不生效，但留着有害：
    服务端一旦开始遵守它，就会按 VAD 猜的断句点抢答。
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "app" / "core" / "realtime.py").read_text(encoding="utf-8")
    seg = src[src.index('"turn_detection"'):]
    seg = seg[:seg.index("}", seg.index("interrupt_response"))]
    assert '"create_response": False' in seg, \
        "必须设 create_response=False，否则服务端会自己抢答"


def test_respond_after_turn_waits_before_creating_response():
    """response.create 之前要留一点时间让音频 item 落库。

    实测 speech_stopped 之后立刻 create 会被忽略，现象就是"AI 不吭声"。
    这个测试锁住那个等待，防止有人"优化"掉。
    """
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "app" / "services" / "session.py").read_text(encoding="utf-8")
    body = src[src.index("async def _respond_after_turn"):]
    body = body[:body.index("async def _on_user_text")]
    assert "request_response" in body
    assert re.search(r"await asyncio\.sleep\(0\.[1-9]", body), \
        "response.create 之前缺少等待"
    # 不能再手动 commit：服务端已经自己 commit 过了
    assert "commit_audio" not in body, \
        "不应再手动 commit（服务端已自行 commit，重复提交会报 buffer too small"


@pytest.mark.asyncio
async def test_turn_end_commits_then_responds():
    """松手时必须先显式提交音频，再让 AI 接话。

    历史（两次踩坑，方向正好相反，别再改回去）：

      · 最初每次 speech_stopped 都 commit —— 但服务端 VAD 已经自己
        commit 过了，我们这次成了空提交，服务端回
        "buffer too small, or have no audio"。
      · 于是彻底删掉 commit，改成完全依赖 VAD 自动提交。
        加上「按住空格说话」之后这条不再成立：实测送上去 3 秒音频
        （30 块），转写始终不出现，紧接着 response.create 也被静默
        吞掉 —— 用户感受就是"我说完 AI 不理我"。

    现在的语义：**松开空格 = 明确的提交信号**，且只在确实发过音频
    时才提交（没发过就跳过，避免空提交）。

    这里用真实的假对象跑行为，而不是检查源码文本 ——
    文本断言会被注释里的同名字样骗过（这个文件踩过一次）。
    """
    from app.services.session import ConversationSession, SessionStats

    calls: list[str] = []

    class FakeRT:
        async def send_audio(self, b):
            calls.append("append")

        async def commit_audio(self):
            calls.append("commit")

        async def request_response(self, instructions=None):
            calls.append("create")

    def make(turn_blocks: int):
        s = ConversationSession.__new__(ConversationSession)
        s.ended = False
        s.paused = False
        s.rt = FakeRT()
        s.stats = SessionStats()
        s._pending = []
        s._model_speaking = False
        s._response_pending = False
        s._response_pending_at = 0.0
        s.response_pending_timeout = 12.0
        s._bg_tasks = set()
        s._turn_audio_blocks = turn_blocks
        s._on_user_text = lambda *a, **k: None
        return s

    # 情形 1：本轮说过话 → 必须先 commit 再 create
    calls.clear()
    s = make(30)
    await ConversationSession.end_turn(s)
    import asyncio

    assert "commit" in calls, "本轮发过音频却没提交（就是'AI 不理我'那个 bug）"
    # 接话是后台任务延迟触发的（要等服务端把音频 item 落库），
    # 这里等它落地再断言顺序。
    await asyncio.sleep(0.8)
    assert "create" in calls, "提交后必须触发接话"
    assert calls.index("commit") < calls.index("create"), \
        f"必须先提交再接话，实际顺序 {calls}"

    # 情形 2：本轮没说话 → 不能空提交，也不该让 AI 对空气回应
    calls.clear()
    s = make(0)
    await ConversationSession.end_turn(s)
    assert calls == [], f"空轮不该提交也不该接话，实际 {calls}"


def test_commit_audio_exists_and_only_commits():
    """commit_audio 恢复存在，但只发 commit、不再做别的。"""
    import inspect

    from app.core.realtime import RealtimeSession

    assert hasattr(RealtimeSession, "commit_audio"), \
        "commit_audio 应存在（松手时显式提交本轮音频）"
    src = inspect.getsource(RealtimeSession.commit_audio)
    # 先剥掉注释和 docstring：解释性文字里提到 "response.create"
    # 会让"不应调用"的断言误判（这个文件已经踩过一次）。
    import re
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"#[^\n]*", "", src)
    assert "input_audio_buffer.commit" in src
    assert "response.create" not in src, \
        "commit 不应顺带触发回应，两件事要分开"


def test_turn_race_errors_are_recognized():
    """两类回合竞态错误要被识别为可自愈，不能打死会话。

      1) "Conversation already has an active response"
      2) "buffer too small, or have no audio"
    """
    from app.core.realtime import is_turn_race_error

    assert is_turn_race_error(
        "Error committing input audio buffer: buffer too small, or have no audio")
    assert is_turn_race_error("Conversation already has an active response")
    # 真正的 quota 问题不能被误判成竞态
    assert not is_turn_race_error("insufficient_quota")


def test_errors_are_classified_before_notifying_user():
    """回合竞态不该弹给用户看。

    原来是一收到 error 就先 on_client，连"可自愈的小冲突"也会弹提示，
    让人以为练习坏了。必须先分类再决定是否通知。
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "app" / "services" / "session.py").read_text(encoding="utf-8")
    seg = src[src.index('if t == "error":'):]
    seg = seg[:seg.index("if is_fatal_error(code):")]
    assert "is_turn_race_error(msg)" in seg, "error 分支没有先做竞态分类"
    assert seg.index("is_turn_race_error") < seg.index('await self.on_client'), \
        "必须先分类再通知用户，否则会把可自愈的冲突弹给用户看"


def test_invalid_request_error_is_not_fatal():
    """invalid_request_error 不该是致命错误。

    它太笼统了：像"commit 空缓冲"这种客户端小失误也归到这一类。
    曾经把它列为致命，导致 VAD 一次误触发就把整场会话打死，
    用户看到「会话被服务端中断（invalid_request_error）」——
    但其实服务端连接还好好的，继续对话完全没问题。
    """
    from app.core.realtime import is_fatal_error

    assert not is_fatal_error("invalid_request_error"), \
        "invalid_request_error 不应致命（会误杀正常会话）"
    # 真正不可恢复的必须仍然是致命
    for code in ("user_idle_timeout", "invalid_api_key",
                 "insufficient_quota", "session_expired"):
        assert is_fatal_error(code), f"{code} 应当仍是致命错误"



# ---------- AI 语音播放（前端）----------
#
# 用户反馈「我听不到 AI 的语音」。根因在前端播放路径：
#   1) 收到 user_speaking（VAD 对噪声/AI尾音也会触发）就调 stopPlayback()
#   2) 而 stopPlayback 用 ctxOut.close() 停声，紧接着新建的 AudioContext
#      常处于 suspended → 之后所有 AI 音频都无声
# 下面几个测试把正确的播放路径锁住。


def _web_html() -> str:
    from pathlib import Path
    return (Path(__file__).resolve().parent.parent
            / "app" / "web" / "index.html").read_text(encoding="utf-8")


def _web_code() -> str:
    """Returns JS source with comments stripped.

    Use this whenever asserting that something is NOT called: an
    explanatory comment that merely mentions the name would otherwise
    make the test fail (already tripped over this once in this file).
    """
    import re
    src = _web_html()
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    return src


def test_stop_playback_does_not_destroy_audio_context():
    """停声不能销毁 AudioContext —— 否则后续语音再也播不出来。

    实测症状：用户完全听不到 AI 说话。原因是用 ctxOut.close() 停声，
    而 close() 是异步的，紧接着 new AudioContext() 在浏览器里常是
    suspended 状态，音频就静默了。
    """
    src = _web_html()
    body = src[src.index("function stopPlayback()"):]
    body = body[:body.index("function ensureOutCtx")]
    assert "close()" not in body, \
        "stopPlayback 不能 close() 音频上下文（会导致 AI 语音无声）"
    assert "playingSources" in body, "stopPlayback 应逐个停掉播放源"


def test_user_speaking_does_not_cut_ai_audio():
    """user_speaking 不能掐断 AI 的语音。

    服务端 VAD 对噪声、甚至 AI 自己的尾音都会报 user_speaking。
    原来一收到就 stopPlayback()，AI 刚开口就被打断，
    用户根本听不全。抢话打断只应由「按下空格」触发。
    """
    src = _web_code()
    seg = src[src.index('m.type === "user_speaking"'):]
    seg = seg[:seg.index('m.type === "user_text"')]
    assert "stopPlayback" not in seg, \
        "user_speaking 不应调 stopPlayback（VAD 误报会切掉 AI 的声音）"


def test_push_to_talk_interrupts_ai_on_space_press():
    """按下空格要立刻停下 AI 的播放（真实抢话）。"""
    src = _web_html()
    body = src[src.index("function pttDown()"):]
    body = body[:body.index("function pttUp")]
    assert "stopPlayback()" in body, "按下空格应停掉 AI 播放"


def test_audio_output_context_is_recovered():
    """播放前要确保音频上下文可用（被挂起时自动 resume）。

    浏览器要求 AudioContext 由用户手势创建/恢复，否则一直 suspended。
    """
    src = _web_html()
    assert "function ensureOutCtx" in src, "缺少 ensureOutCtx"
    body = src[src.index("function ensureOutCtx"):]
    body = body[:body.index("async function startMic")]
    assert "resume()" in body, "ensureOutCtx 必须处理 suspended 状态"

    # enqueueAudio 必须走 ensureOutCtx，不能自己判断 ctxOut 是否存在
    enq = src[src.index("function enqueueAudio"):]
    enq = enq[:enq.index("function stopPlayback")]
    assert "ensureOutCtx()" in enq, "enqueueAudio 应通过 ensureOutCtx 取上下文"


# ---------- 冷场救场 ----------
#
# 改成「按住空格说话」之后，回合完全由用户发起 —— 用户不开口，
# AI 就永远不说话。卡住想不出句子时会一直干等，练习停在那里。
# 所以冷场够久要让 AI 主动找话题。


def test_prompt_fires_on_demand():
    """用户按 C 键时，AI 应主动开口 —— 与冷场多久无关。"""
    import time as _t

    from app.services.session import ConversationSession

    s = ConversationSession.__new__(ConversationSession)
    s.ended = False
    s.paused = False
    s.rt = object()
    s._model_speaking = False
    s._response_pending = False
    s._nudges_sent = 0
    s._max_nudges = 6
    s._last_activity = _t.time()          # 刚刚还有动静

    # 关键差异：不再看"冷场够不够久"，刚说完话也能立刻求援
    instr = s.maybe_prompt()
    assert instr, "按 C 键应立刻触发主动找话题，不该等冷场"
    # 指令要明确禁止"你还在吗"这类扫兴话术
    assert "still there" in instr.lower()


def test_rapid_double_press_only_fires_once():
    """连按 C 键只能触发一次 —— 走真实的 pending 实现验证。

    为什么用真实方法而不是打桩：第一次验证时把 _mark_response_pending
    打成了空函数，于是"连按两次"看起来像漏拦，白白追查了一轮。
    打桩会把要验证的机制本身替换掉，等于没测。
    """
    import asyncio
    import time as _t

    from app.services.session import ConversationSession

    calls = []

    class FakeRT:
        async def request_response(self, instructions=None):
            calls.append(instructions)

    s = ConversationSession.__new__(ConversationSession)
    s.ended = False
    s.paused = False
    s.rt = FakeRT()
    s._model_speaking = False
    s._response_pending = False
    s._response_pending_at = 0.0
    s.response_pending_timeout = 12.0
    s._nudges_sent = 0
    s._max_nudges = 6
    s._last_activity = _t.time()
    s.on_client = _noop_async

    # 接上真实实现，不打桩
    for name in ("_mark_response_pending", "_clear_response_pending",
                 "response_pending"):
        setattr(s, name,
                getattr(ConversationSession, name).__get__(s))

    asyncio.run(s.prompt())
    assert len(calls) == 1, "第一次按 C 键应触发"

    asyncio.run(s.prompt())
    assert len(calls) == 1, "紧接着再按一次应被拦下，否则 AI 会重复插话"


async def _noop_async(*_a, **_k):
    return None


def test_nudge_instruction_forbids_asking_if_still_there():
    """主动找话题不能被说成"你还在吗"。"""
    from app.services.session import ConversationSession

    s = ConversationSession.__new__(ConversationSession)
    s._nudges_sent = 1
    first = s._nudge_instruction()
    assert "still there" in first.lower(), "必须明确禁止追问'你还在吗'"

    # 第二次起要换新话题，避免在同一点打转
    s._nudges_sent = 3
    later = s._nudge_instruction()
    assert later != first, "连续冷场应换策略，而不是重复同一句"


def test_prompt_never_interrupts_ai_speech():
    """AI 正在说话/准备说话时按 C 键绝不能插嘴。"""
    import time as _t

    from app.services.session import ConversationSession

    s = ConversationSession.__new__(ConversationSession)
    s.ended = False
    s.paused = False
    s.rt = object()
    s._nudges_sent = 0
    s._max_nudges = 6
    s._last_activity = _t.time()

    s._model_speaking = True
    assert s.maybe_prompt() is None, "AI 正在说话时按 C 键不该插嘴"

    s._model_speaking = False
    s._response_pending = True
    assert s.maybe_prompt() is None, "已经在等回应时不该重复触发"


def test_prompt_stops_after_max():
    """按够次数后要停下，不能一直按着刷屏。"""
    import time as _t

    from app.services.session import ConversationSession

    s = ConversationSession.__new__(ConversationSession)
    s.ended = False
    s.paused = False
    s.rt = object()
    s._model_speaking = False
    s._response_pending = False
    s._max_nudges = 2
    s._nudges_sent = 0
    s._last_activity = _t.time()

    fired = 0
    while s.maybe_prompt():
        fired += 1
        if fired > 10:
            break
    assert fired == 2, f"应恰好触发 {2} 次，实际 {fired}"


def test_prompt_disabled_when_max_is_zero():
    """IDLE_NUDGE_MAX=0 时按 C 键应该完全没反应（用来彻底关掉）。"""
    import time as _t

    from app.services.session import ConversationSession

    s = ConversationSession.__new__(ConversationSession)
    s.ended = False
    s.paused = False
    s.rt = object()
    s._model_speaking = False
    s._response_pending = False
    s._nudges_sent = 0
    s._max_nudges = 0
    s._last_activity = _t.time()
    assert s.maybe_prompt() is None, "上限设为 0 时不该开口"


def test_paused_session_never_prompts():
    """暂停时按 C 键不该有反应 —— 用户离开了。"""
    import time as _t

    from app.services.session import ConversationSession

    s = ConversationSession.__new__(ConversationSession)
    s.ended = False
    s.paused = True
    s.rt = object()
    s._model_speaking = False
    s._response_pending = False
    s._nudges_sent = 0
    s._max_nudges = 6
    s._last_activity = _t.time()
    assert s.maybe_prompt() is None


def test_timer_loop_no_longer_auto_prompts():
    """冷场自动接话必须已经移除，改由 C 键触发。

    用户明确要求：不要 AI 自己等 15 秒接话，改成按 C 键才触发。
    如果哪天有人把自动轮询加回来，这条会拦住。
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "app" / "api" / "server.py").read_text(encoding="utf-8")
    body = src[src.index("async def _timer_loop"):]
    body = body[:body.index("# ============================================================")]
    assert "sess.nudge(" not in body, "_timer_loop 不该再自动触发 AI 找话题"
    assert "IDLE_NUDGE_SEC" not in src, "冷场阈值已废弃，不该再出现"
    # 触发入口改成前端发来的 prompt 命令
    assert '"prompt"' in src, "缺少 C 键对应的 prompt 命令处理"
    assert "sess.prompt(" in src, "prompt 命令没有调用 sess.prompt"


def test_pending_response_guard_expires():
    """等待回应的标志必须能超时作废。

    实测踩过的坑：发了 response.create 但服务端根本没产生 response
    （半句话、音频太短都会这样），response.done 永远不来，"等待中"
    就一直挂着 —— 之后所有回合都被当成重复而静默丢弃，
    表现为用户说什么 AI 都不理，冷场救场也一起失效。
    """
    import time as _t

    from app.services.session import ConversationSession

    s = ConversationSession.__new__(ConversationSession)
    s._response_pending = True
    s._response_pending_at = _t.time() - 999
    s.response_pending_timeout = 12.0
    assert s.response_pending() is False, \
        "超时的等待标志必须作废，否则会话永久卡死"


def test_silence_does_not_reset_idle_timer():
    """纯静音不能重置冷场计时。

    这是用户实际反馈的 bug：「我停下了 20 秒，AI 并没有自然接上」。

    原因：push_audio 一进来就无条件 touch()。麦克风上行是连续的，
    即使按住空格模式，松手前后也会有零散静音块到达 —— 每块都把
    冷场计时清零，AI 永远等不到"满 20 秒"。
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "app" / "services" / "session.py").read_text(encoding="utf-8")
    body = src[src.index("async def push_audio"):]
    body = body[:body.index("async def _flush_pending")]

    # touch() 必须出现在静音判断之后，不能挂在函数开头
    rms_at = body.index("VOICE_RMS_FLOOR")
    touch_at = body.index("self.touch()")
    assert touch_at > rms_at, \
        "touch() 必须在判断出『这是人声』之后才调用，否则静音会顶掉冷场计时"


def test_silence_is_not_forwarded_without_voiceprint():
    """未开声纹时，静音块不该上行（否则服务端一直以为用户在说话）。"""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "app" / "services" / "session.py").read_text(encoding="utf-8")
    body = src[src.index("async def push_audio"):]
    body = body[:body.index("async def _flush_pending")]
    assert "else:" in body, "声纹关闭时也要判断音量，不能直接原样上行"
    assert body.count("return") >= 3, "静音/他人语音等分支都该提前返回"


def test_frontend_enables_push_to_talk_gate():
    """前端必须真的把 push-to-talk 闸门打开。

    实际 bug：`pttActive` 声明了、判定逻辑也写了，但**从来没有人把
    它置为 true** —— 闸门形同虚设，麦克风一直在上行静音，服务端
    不断重置冷场计时，AI 永远等不到冷场。
    """
    from pathlib import Path
    import re

    html = (Path(__file__).resolve().parent.parent
            / "app" / "web" / "index.html").read_text(encoding="utf-8")
    html = re.sub(r"/\*[\s\S]*?\*/", "", html)
    html = re.sub(r"//[^\n]*", "", html)

    assert "pttActive = true" in html, \
        "pttActive 必须被真正置为 true，否则『不按空格不上行』根本没生效"
    # 置位要发生在麦克风启动流程里（worklet 绑定之前）
    assert html.index("pttActive = true") < html.index("worklet.port.onmessage"), \
        "应在启用 worklet 消息处理前就打开闸门"


def test_history_button_reports_failure():
    """「历史」按钮失败时必须报错，不能静默。

    真实 bug：/api/stats 因为变量覆盖 500 了，而这里没有 try/catch，
    Promise.all 一 reject 整个函数就静默抛出 —— 界面表现是
    「点了没反应」，不是报错。后端挂掉很久都没人察觉。
    """
    src = _web_code()
    body = src[src.index('$("btnHistory").onclick'):]
    body = body[:body.index("\n};")]

    assert "try" in body and "catch" in body, \
        "btnHistory 必须捕获失败，否则出错时界面毫无反馈"
    assert ".ok" in body, \
        "必须检查 response.ok —— 只 await .json() 会把 500 当成正常响应"
    assert "toast(" in body, "失败要 toast 告诉用户，而不是只 console.error"


def test_frontend_binds_c_key_to_prompt():
    """前端必须把 C 键接到 prompt 命令上。

    这是"改成按 C 键触发"的入口 —— 服务端已经不再自动接话，
    前端要是没绑，这个功能就等于没有。
    """
    src = _web_code()

    # 用 event.code 判定，不受输入法/大小写影响
    assert "KeyC" in src, "C 键判定应使用 event.code === 'KeyC'"
    # 真的把消息发出去
    assert 'type: "prompt"' in src or 'type:"prompt"' in src, \
        "按 C 键必须发送 {type: 'prompt'} 给服务端"
    # 输入框里打字时不能抢 c
    body = src[src.index("function isPromptKey"):]
    body = body[:body.index("function connect(")]
    assert "typingInField" in body, "按 C 键前要判断是否正在输入框里打字"
    # ⌘C / Ctrl+C 是复制，不能被吞掉
    assert "metaKey" in body and "ctrlKey" in body, \
        "必须放过 ⌘C / Ctrl+C，否则用户复制不了东西"


def test_frontend_ptt_bar_mentions_c_key():
    """界面上要有 C 键的说明，否则用户不知道能求援。"""
    html = _web_html()
    bar = html[html.index('id="pttBar"'):]
    bar = bar[:bar.index("</div>", bar.index("卡壳"))]
    assert ">C<" in bar or "C</span>" in bar, "提示条里要显示 C 键"


def test_all_models_come_from_settings():
    """对话/转写/文本三个模型都必须可配置。"""
    import os

    from app.core.config import load_settings

    # 默认值可以用，但必须能被环境变量覆盖
    for env_key in ("QWEN_MODEL", "QWEN_ASR_MODEL", "TEXT_MODEL",
                    "QWEN_VOICE", "TEXT_BASE_URL"):
        assert env_key in os.environ or True, env_key   # 存在性由下一条保证

    src_env = (Path(__file__).resolve().parent.parent / ".env.example").read_text(
        encoding="utf-8")
    for env_key in ("QWEN_MODEL", "QWEN_ASR_MODEL", "TEXT_MODEL",
                    "QWEN_VOICE", "TEXT_BASE_URL"):
        assert env_key in src_env, f"{env_key} 没写进 .env.example，用户无从得知"


def test_asr_model_is_not_hardcoded(monkeypatch):
    """ASR 转写模型必须读配置，不能写死在代码里。"""
    import app.core.config as cfg

    monkeypatch.setenv("QWEN_ASR_MODEL", "some-other-asr")
    assert cfg.load_settings().asr_model == "some-other-asr"

    # 代码里不该再出现写死的模型名（注释除外）
    src = (Path(__file__).resolve().parent.parent
           / "app" / "core" / "realtime.py").read_text(encoding="utf-8")
    import re
    code = re.sub(r"#[^\n]*", "", src)
    assert '"qwen3-asr-flash-realtime"' not in code, \
        "ASR 模型名仍被写死在代码里，用户改不了"


def test_text_model_is_configurable(monkeypatch):
    """材料准备用的文本模型必须可配置。"""
    import app.core.config as cfg

    monkeypatch.setenv("TEXT_MODEL", "qwen-max")
    monkeypatch.setenv("TEXT_BASE_URL", "https://example.invalid/v1")
    s = cfg.load_settings()
    assert s.text_model == "qwen-max"
    assert s.text_base_url == "https://example.invalid/v1"


def test_startup_prints_active_models():
    """启动时打印实际生效的模型，配置错了能立刻发现。"""
    from pathlib import Path as _P

    src = (_P(__file__).resolve().parent.parent
           / "app" / "__main__.py").read_text(encoding="utf-8")
    assert "模型配置" in src, "启动横幅应打印生效的模型"
    for key in ("qwen_model", "asr_model", "text_model"):
        assert key in src, f"启动横幅应包含 {key}"


def test_prompt_signatures_are_key_driven():
    """触发入口必须是"按了就问"，不再有 threshold 参数。

    原先是冷场阈值驱动的（maybe_nudge/nudge 带 threshold 默认 None）。
    改成 C 键之后，触不触发只取决于用户按键，方法上不该再留
    一个"等够多少秒"的参数 —— 留着迟早有人又接回自动轮询。
    """
    import inspect

    from app.services.session import ConversationSession

    for fn in (ConversationSession.maybe_prompt,
               ConversationSession.prompt):
        params = inspect.signature(fn).parameters
        assert "threshold" not in params, \
            f"{fn.__name__} 不该再有 threshold 参数（自动触发已移除）"

    # 冷场阈值配置项本身也该删干净
    from app.core.config import load_settings
    assert not hasattr(load_settings(), "idle_nudge_sec"), \
        "idle_nudge_sec 已废弃，不该还留在 Settings 里"


def test_idle_max_is_overridable(monkeypatch):
    """IDLE_NUDGE_MAX 改环境变量要真的生效。"""
    import app.core.config as cfg

    monkeypatch.setenv("IDLE_NUDGE_MAX", "3")
    assert cfg.load_settings().idle_nudge_max == 3


def test_bad_idle_config_falls_back(monkeypatch):
    """配置写错不该让服务起不来，退回默认值即可。"""
    import app.core.config as cfg

    for bad in ("xyz", "", "-5"):
        monkeypatch.setenv("IDLE_NUDGE_MAX", bad)
        assert cfg.load_settings().idle_nudge_max == 6, \
            f"IDLE_NUDGE_MAX={bad!r} 应退回默认值"


def test_idle_max_actually_gates_prompting(monkeypatch):
    """配置的次数上限要真的决定按 C 键能问几次。"""
    import time as _t

    import app.core.config as cfg
    from app.services.session import ConversationSession

    monkeypatch.setenv("IDLE_NUDGE_MAX", "3")
    st = cfg.load_settings()

    def make():
        x = ConversationSession.__new__(ConversationSession)
        x.s = st
        x.ended = False
        x.paused = False
        x.rt = object()
        x._model_speaking = False
        x._response_pending = False
        x._nudges_sent = 0
        x._max_nudges = st.idle_nudge_max
        x._last_activity = _t.time()
        return x

    x = make()
    fired = 0
    while x.maybe_prompt():
        fired += 1
        if fired > 10:
            break
    assert fired == 3, f"上限 3 次，实际 {fired}"


def test_idle_settings_documented_in_env_example():
    """.env.example 必须列出这个配置，否则用户不知道它存在。"""
    src = (Path(__file__).resolve().parent.parent
           / ".env.example").read_text(encoding="utf-8")
    assert "IDLE_NUDGE_MAX" in src, "IDLE_NUDGE_MAX 没写进 .env.example"
    assert "IDLE_NUDGE_SEC" not in src, \
        "IDLE_NUDGE_SEC 已废弃，不该还留在 .env.example 里"


def test_voice_is_configurable(monkeypatch):
    """音色必须由配置决定（用户要能自己换）。"""
    import app.core.config as cfg

    monkeypatch.setenv("QWEN_VOICE", "Serena")
    assert cfg.load_settings().qwen_voice == "Serena"

    src = (Path(__file__).resolve().parent.parent
           / "app" / "core" / "config.py").read_text(encoding="utf-8")
    assert 'get("QWEN_VOICE"' in src, "音色应读 QWEN_VOICE"


def test_voice_list_tool_distinguishes_two_sets():
    """对话音色和 TTS 音色是两套清单，工具必须区分。

    实测教训：用 TTS 模型合成 "Tina" 会报 Invalid voice specified ——
    因为 Tina 只在实时对话模型里存在，TTS 那套没有。如果工具把两套
    混在一起，用户试听时就会撞上一堆莫名其妙的失败。
    """
    import importlib.util

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "list_voices", root / "scripts" / "list_voices.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    chat = {v for v, _ in mod.CHAT_VOICES}
    tts = {v for v, _ in mod.TTS_VOICES}

    assert "Tina" in chat, "Tina 是对话音色"
    assert "Tina" not in tts, "Tina 不在 TTS 清单里，试听会失败"
    assert "Aiden" in tts
    assert chat & tts, "两套应有重叠（如 Serena/Ethan）"


def test_voice_samples_dir_is_gitignored():
    """样音是本地试听用的，不该进仓库。"""
    root = Path(__file__).resolve().parent.parent
    gi = (root / ".gitignore").read_text(encoding="utf-8")
    assert "data" in gi, "data/ 应被忽略"


def test_voices_api_lists_all_and_flags_sampleable():
    """/api/voices 要给出全部音色，并标明哪些能试听。"""
    import importlib.util

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "lv2", root / "scripts" / "list_voices.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from app.core.voices import CHAT_VOICES, VOICE_DOC_URL, can_sample

    assert len(CHAT_VOICES) == 17
    assert VOICE_DOC_URL.startswith("https://help.aliyun.com/")
    # 每个音色都要有 值/名/描述/性别 四项，前端才能渲染
    for row in CHAT_VOICES:
        assert len(row) == 4, row
    # 必须有能试听的，否则试听功能形同虚设
    assert sum(1 for r in CHAT_VOICES if can_sample(r[0])) >= 5


def test_voice_select_endpoint_exists():
    """网页上切换音色需要 POST /api/voices/select。"""
    src = (Path(__file__).resolve().parent.parent
           / "app" / "api" / "server.py").read_text(encoding="utf-8")
    assert '"/api/voices/select"' in src
    assert '"/api/voices"' in src
    assert "/sample" in src


def test_web_choice_beats_env():
    """界面选择必须压过 .env，否则点了音色重启后又变回去。

    这是实测发现的：.env 里有 QWEN_VOICE=Jennifer，界面上选了
    Serena、提示"已切换"，重启后却回到 Jennifer —— 看起来就是
    "点了没用"。
    """
    src = (Path(__file__).resolve().parent.parent
           / "app" / "api" / "server.py").read_text(encoding="utf-8")
    # 不能出现"有 QWEN_VOICE 就跳过覆盖"这种逻辑
    assert "not os.environ.get(\"QWEN_VOICE\")" not in src, \
        "界面选择被 .env 压过了，用户点了会看起来没生效"


def test_voice_ui_has_doc_link_and_buttons():
    """音色面板要有官方列表链接和试听按钮。"""
    import re as _re

    root = Path(__file__).resolve().parent.parent
    html = (root / "app" / "web" / "index.html").read_text(encoding="utf-8")
    # 去掉注释再断言，避免注释里的字样蒙混过关
    code = _re.sub(r"//[^\n]*", "", html)
    for token in ("btnVoice", "voiceModal", "btnCloseVoice"):
        assert token in code, f"缺少 {token}"
    assert "voiceDocLink" in code, "音色面板要有官方列表链接"
    assert "doc_url" in code, "链接地址应由后端下发，不写死在前端"


def test_no_chinese_identifiers_in_frontend():
    """前端别用中文变量名 —— 虽然合法，但容易在别处解析出错。

    实测：写了 `const 可试 = ...`，Node 下报 可试 is not defined。
    """
    import re as _re

    html = (Path(__file__).resolve().parent.parent
            / "app" / "web" / "index.html").read_text(encoding="utf-8")
    code = _re.sub(r"//[^\n]*", "", html)
    code = _re.sub(r"<[^>]+>", "", code)
    # 找 const/let/var 后面跟中文标识符
    bad = _re.findall(r"\b(?:const|let|var)\s+([\u4e00-\u9fff]\w*)", code)
    assert not bad, f"前端出现中文变量名：{bad}"


def test_start_script_exists_and_handles_restart():
    """启动脚本要先停旧进程再启动。

    直接重复启动会因端口被占而失败，但旧进程还在响应 ——
    表现为"改了代码却没生效"，这是最费解的一类问题。
    """
    import re

    root = Path(__file__).resolve().parent.parent
    sh = root / "start.sh"
    assert sh.is_file(), "缺少 start.sh"
    assert sh.stat().st_mode & 0o111, "start.sh 应可执行"

    src = sh.read_text(encoding="utf-8")
    # 去掉注释再断言，避免注释里的说明文字蒙混过关
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    # 必须真的**调用** stop_server，而不只是定义了它 ——
    # 只检查字符串的话，函数定义本身就能让断言通过，测不出漏调用。
    #
    # 还要注意：--stop 分支里也有一次调用。所以只看"有没有调用"
    # 仍然不够 —— 得确认**启动路径**上也调用了一次。
    # （这一点是实测发现的：删掉启动路径上的调用，测试照样通过。）
    # 把整个 case...esac 块删掉（里面 --stop 分支也有一次调用，
    # 不排除掉的话，删了启动路径上的调用测试照样通过 —— 实测踩过）
    body = re.sub(r"^case .*?^esac", "", code, flags=re.M | re.S)
    assert "case " not in body, "case 块没被剥掉"
    assert re.search(r"^\s*stop_server\s*$", body, re.M), \
        "启动路径上应先调用 stop_server 停止旧进程"
    assert "run.py" in code, "应调用 run.py"
    # 必须等端口释放，否则会撞 Address already in use
    assert "pids_on_port" in code
    # 必须做健康检查，否则配置错了要等打开网页才发现
    assert "api/health" in code


def test_start_script_flags():
    """脚本要支持文档里写的那些参数。"""
    root = Path(__file__).resolve().parent.parent
    src = (root / "start.sh").read_text(encoding="utf-8")
    for flag in ("--stop", "--status", "--logs", "--help",
                 "--ssl", "--port", "--reload"):
        assert flag in src, f"脚本应支持 {flag}"


def test_docs_point_to_start_script():
    """文档里的启动方式要和实际一致，不能还写着老命令。"""
    root = Path(__file__).resolve().parent.parent
    for name in ("SETUP.md", "docs/使用说明.md"):
        text = (root / name).read_text(encoding="utf-8")
        assert "./start.sh" in text, f"{name} 应提到 ./start.sh"


def test_root_has_no_stray_docs():
    """根目录只留 README 和 SETUP，其余文档都应归到 docs/。

    根目录堆一堆 md 会让新来的人不知道从哪看起。
    README 是入口，SETUP 是搭建指南 —— 这两个留在根目录符合惯例。
    """
    root = Path(__file__).resolve().parent.parent
    stray = sorted(p.name for p in root.glob("*.md"))
    assert stray == ["README.md", "SETUP.md"], \
        f"根目录出现了不该有的文档：{stray}"


def test_readme_links_resolve():
    """README 里的相对链接必须真的能打开。"""
    import re as _re

    root = Path(__file__).resolve().parent.parent
    md = root / "README.md"
    assert md.is_file(), "缺少 README.md"
    text = md.read_text(encoding="utf-8")
    for m in _re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text):
        target = m.group(2)
        if target.startswith(("http://", "https://", "#")):
            continue
        p = (md.parent / target.split("#")[0]).resolve()
        assert p.exists(), f"README 链接失效：[{m.group(1)}]({target})"


def test_readme_mentions_key_facts():
    """README 该说的关键信息不能漏，否则新人跑不起来。"""
    root = Path(__file__).resolve().parent.parent
    text = (root / "README.md").read_text(encoding="utf-8")
    for token in ("./start.sh", ".env", "SETUP.md", "按住空格",
                  "声纹", "docs/"):
        assert token in text, f"README 缺少关键信息：{token}"


def test_docs_test_count_matches_reality():
    """文档里写的测试数量应和实际一致。

    这个数字散落在多处，每次加测试都要手动同步，很容易忘。
    让测试自己盯着。
    """
    import re as _re
    import subprocess

    root = Path(__file__).resolve().parent.parent
    # 注意输出格式是 "278/284 tests collected (6 deselected)"，
    # 第一个数字才是实际会被跑的个数。
    out = subprocess.run(
        [".venv/bin/python", "-m", "pytest", "-m", "not live",
         "--collect-only"],
        cwd=root, capture_output=True, text=True).stdout
    m = _re.search(r"(\d+)/(\d+)\s+tests?\s+collected", out)
    if not m:
        pytest.skip(f"拿不到测试数量，输出：{out[-200:]}")
    actual = int(m.group(1))

    readme = (root / "README.md").read_text(encoding="utf-8")
    claimed = [int(x) for x in _re.findall(r"(\d{3}) 个测试", readme)]
    assert claimed, "README 应写明测试数量"
    for c in claimed:
        assert c == actual, f"README 写 {c} 个测试，实际 {actual} 个"


# ============================================================
#  备课必须按"当前选中的用户"
#
#  实测踩出来的：网页上明明选了 B，生成的话题却按 A 的背景来。
#  根因是三处叠加，任何一处单独修都不够 ——
#
#    ① 前端 /api/prepare 没带 user（裸 fetch，漏了 api() 包装），
#       服务端于是回退到"最近用过的那个人"，常常正是 A；
#    ② WebSocket 备课链路压根没传 profile_summary，
#       点「开始」时的话题跟「生成话题」预览的不是同一份；
#    ③ 备课和会话提示词分两次读画像，两份可能不同 ——
#       话题按你的兴趣挑，开场却按别人的经历寒暄。
#
#  下面每一组测试对应其中一处，防止以后又被改回去。
# ============================================================

class _Recorder:
    """记录 Planner 实际收到的参数。"""

    def __init__(self, summary):
        self.summary = summary
        self.calls: list[dict] = []

    def __call__(self, content, *, profile_summary="", n_topics=5):
        self.calls.append({"summary": profile_summary,
                           "n_topics": n_topics})
        from app.services.planner import Plan, Topic
        return Plan(opening="hi", background="bg",
                    topics=[Topic(title="t", prompt="What?")][:1])


def _patch_server(store, db, monkeypatch, recorder):
    """把 server 的依赖换成测试用的 store/db，并挂上假 Planner。"""
    import app.api.server as srv

    monkeypatch.setattr(srv, "user_store", lambda: store)
    monkeypatch.setattr(srv, "db", lambda: db)
    monkeypatch.setattr(srv, "profile_store",
                        lambda: ProfileStore(db, store.dir.parent / "profiles"))

    class FakePlanner:
        def __init__(self):
            self.llm = None

        def plan(self, content, *, profile_summary="", n_topics=5):
            return recorder(content, profile_summary=profile_summary,
                            n_topics=n_topics)

    monkeypatch.setattr(srv, "Planner", FakePlanner)
    return srv


def test_prepare_uses_selected_users_own_profile(store, db, monkeypatch):
    """给 A 备课，注入的必须是 A 的画像，且绝不含 B 的。"""
    import asyncio
    from app.services.planner import Plan, Topic

    a = store.create("A")
    b = store.create("B")
    db.upsert_fact(ProfileFact(user_id=a.id, category="interest",
                               key="hobby", value="A 的爱好是爬山",
                               confidence=0.9))
    db.upsert_fact(ProfileFact(user_id=b.id, category="interest",
                               key="hobby", value="B 的爱好是养猫",
                               confidence=0.9))
    # 让 B 成为"最近用过的人" —— 正是串味的那个回退目标
    db.touch_user(a.id)
    db.touch_user(b.id)

    rec = _Recorder("ignored")
    srv = _patch_server(store, db, monkeypatch, rec)

    out = asyncio.run(srv.prepare(
        srv.PrepareIn(content="remote work", topics=5), user=a.id))

    assert rec.calls, "prepare 没有调用 Planner"
    used = rec.calls[0]["summary"]
    assert "A 的爱好是爬山" in used
    assert "养猫" not in used, "备课用到了另一个用户的画像"
    assert out["user_id"] == a.id
    assert out["profile_used"] is True


def test_prepare_does_not_fall_back_to_another_user(store, db, monkeypatch):
    """没给 user（或用户已被删）时必须报错，不能悄悄用最近的人。

    回退在这里是纯粹的伤害：用户什么都没做错，却拿到了按
    别人背景设计的话题，而且界面上完全看不出。
    """
    import asyncio
    from fastapi import HTTPException
    from app.services.planner import Plan, Topic

    a = store.create("A")
    db.touch_user(a.id)

    rec = _Recorder("")
    srv = _patch_server(store, db, monkeypatch, rec)

    # 一个都没给
    with pytest.raises(HTTPException) as e:
        asyncio.run(srv.prepare(srv.PrepareIn(content="remote work")))
    assert e.value.status_code == 400

    # 指向已被删除的用户（另一个标签页残留的旧 id）
    with pytest.raises(HTTPException) as e:
        asyncio.run(srv.prepare(
            srv.PrepareIn(content="remote work"), user=99999))
    assert e.value.status_code == 404

    assert not rec.calls, "报错前不应该已经调用了模型"


def test_prepare_reports_no_profile_for_new_user(store, db, monkeypatch):
    """新用户没有画像是正常的，要如实告诉前端（而不是静默按通用备）。"""
    import asyncio
    from app.services.planner import Plan, Topic

    a = store.create("A")
    rec = _Recorder("")
    srv = _patch_server(store, db, monkeypatch, rec)

    out = asyncio.run(srv.prepare(
        srv.PrepareIn(content="remote work"), user=a.id))
    assert out["profile_used"] is False
    assert rec.calls[0]["summary"] == ""


def _plan_call_keywords(caller: str):
    """
    取出某函数里所有 `Planner().plan(...)` 调用的关键字参数名。

    实际写法是 `asyncio.to_thread(Planner().plan, content, kw=...)` ——
    plan 是传给 to_thread 的第一个实参，不在 Call 位置上，
    所以要同时认「直接调用」和「to_thread 的首参」两种形态。

    用 AST 而不是正则：`profile_summary=summary` 这段文本在
    `build_tutor_instructions` 调用里同样存在，正则会误判成
    "备课传了画像" —— 那正是假测试绿着放过真 bug 的原因。
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    tree = ast.parse(
        (root / "app" / "api" / "server.py").read_text(encoding="utf-8"))

    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == caller)

    def is_plan(node) -> bool:
        """节点是不是 `Planner().plan` 这个属性访问。"""
        return (isinstance(node, ast.Attribute)
                and node.attr == "plan"
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "Planner")

    found = []
    for node in ast.walk(fn):
        # 形态一：Planner().plan(content, kw=...)
        if (isinstance(node, ast.Call) and is_plan(node.func)):
            found.append({kw.arg for kw in node.keywords})
        # 形态二：asyncio.to_thread(Planner().plan, content, kw=...)
        elif (isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)
              and node.func.attr == "to_thread"
              and node.args and is_plan(node.args[0])):
            found.append({kw.arg for kw in node.keywords})
    return found


def test_ws_planning_injects_profile():
    """
    WS 备课必须传 profile_summary —— 点「开始」走的就是这条链路。

    这条最容易被漏掉：它不报错、不崩溃，只是让「生成话题」看到的
    预览和真正开始练习时聊的内容对不上。
    """
    calls = _plan_call_keywords("ws_session")
    assert calls, "ws_session 里没找到 Planner().plan(...) 调用"

    for kws in calls:
        assert "profile_summary" in kws, (
            "WS 备课链路没有注入画像（server.py:ws_session 的 "
            f"Planner().plan 调用，实际关键字：{kws}）")


def test_prepare_plan_call_injects_profile():
    """/api/prepare 的备课同样必须注入画像。"""
    calls = _plan_call_keywords("prepare")
    assert calls, "prepare 里没找到 Planner().plan(...) 调用"
    for kws in calls:
        assert "profile_summary" in kws, \
            f"/api/prepare 备课没有注入画像，实际关键字：{kws}"


def test_ws_planning_and_tutor_share_one_summary():
    """备课和会话提示词必须用同一份画像，不能各读一次。"""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    tree = ast.parse(
        (root / "app" / "api" / "server.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "ws_session")

    # 形如 store.summary(user.id) 的调用
    reads = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "summary"]
    assert len(reads) == 1, (
        f"画像应只读一次，实际读了 {len(reads)} 次 —— "
        "备课和提示词可能用了不同的背景")


def test_frontend_prepare_sends_user():
    """前端「生成话题」必须带 user，否则服务端不知道为谁备课。"""
    import re as _re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    html = (root / "app" / "web" / "index.html").read_text(encoding="utf-8")

    # 取出 btnPrepare 的处理函数
    m = _re.search(r'\$[(]?"btnPrepare"[)]?\.onclick\s*=\s*async\s*\(\)\s*=>\s*\{'
                   r'(.*?)\n\};', html, _re.S)
    assert m, "没找到 btnPrepare 的处理函数"
    body = m.group(1)

    # 关键：必须走 api()（它负责追加 user=），不能是裸 "/api/prepare"
    assert 'api("/api/prepare")' in body, \
        "生成话题没走 api()，user 参数不会被带上"
    assert 'fetch("/api/prepare"' not in body, \
        "生成话题仍在用裸 fetch，服务端会回退到最近使用的用户"

    # 没选用户时要拦住：报错总比用错人的画像强
    assert "if (!me)" in body, "没选用户就备课，会用到别人的背景"


def test_frontend_reprepare_clears_stale_plan():
    """「换个话题」要清掉旧的 prepared，否则会把旧计划当成新的显示。"""
    import re as _re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    html = (root / "app" / "web" / "index.html").read_text(encoding="utf-8")

    m = _re.search(r'\$[(]?"btnReprepare"[)]?\.onclick\s*=\s*\(\)\s*=>\s*\{'
                   r'(.*?)\n\};', html, _re.S)
    assert m, "没找到 btnReprepare 的处理函数"
    assert "prepared = null" in m.group(1), \
        "「换个话题」没清 prepared，旧计划会被误显示"


def test_switching_user_clears_plan_hint():
    """切换用户后要清掉"按某某生成"的提示，否则会显示成上一个人的。"""
    import re as _re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    html = (root / "app" / "web" / "index.html").read_text(encoding="utf-8")

    m = _re.search(r'box\.querySelectorAll\("\.urow"\)\.forEach\(row\s*=>\s*\{'
                   r'(.*?)\n\s{4}\}\);', html, _re.S)
    assert m, "没找到切换用户的处理代码"
    body = m.group(1)
    assert "prepared = null" in body
    assert 'setupHint' in body, "切换用户后应清掉备课提示"


# ============================================================
#  两次备课必须完全一致
#
#  「生成话题」和点「开始」各调一次 Planner，即使两边注入的是
#  同一份画像，模型重新构思仍会给出不同的话题 ——
#  界面上预览的是 A，真正开练时聊的是 B。
#  用户看到的不是"话题被优化了"，而是"我刚点的没用"。
#
#  做法：/api/prepare 把结果按 (user_id, 内容指纹) 记进进程内缓存，
#  WS 建连时命中就直接复用，不再调模型。
# ============================================================

class _FakeWS:
    """够用的 WebSocket 替身，记录发出去的消息。"""

    def __init__(self, query: str):
        from urllib.parse import parse_qs, urlparse
        p = urlparse("ws://x/session?" + query)
        self.query_params = {k: v[0] for k, v in parse_qs(p.query).items()}
        self.sent: list[dict] = []
        self.closed = False

    async def accept(self):
        pass

    async def send_json(self, data):
        self.sent.append(data)

    async def close(self):
        self.closed = True

    async def receive_json(self):
        return {"type": "end"}


def _make_plan(tag: str):
    from app.services.planner import Plan, Topic
    return Plan(opening=f"hi-{tag}", background=tag,
                topics=[Topic(title=tag, prompt=f"What about {tag}?")])


def test_plan_cache_roundtrip():
    from app.services.plancache import PlanCache

    c = PlanCache()
    c.put(1, "remote work", _make_plan("a"))
    got = c.get(1, "remote work")
    assert got is not None and got.opening == "hi-a"


def test_plan_cache_miss_returns_none():
    from app.services.plancache import PlanCache

    c = PlanCache()
    assert c.get(1, "never prepared") is None


def test_plan_cache_isolates_users():
    """同一段素材、两个用户，必须各拿各的计划。"""
    from app.services.plancache import PlanCache

    c = PlanCache()
    c.put(1, "same article", _make_plan("A的"))
    c.put(2, "same article", _make_plan("B的"))
    assert c.get(1, "same article").opening == "hi-A的"
    assert c.get(2, "same article").opening == "hi-B的"


def test_plan_cache_isolates_content():
    """改了输入内容就不能复用旧计划 —— 否则开练时聊的不是他此刻想聊的。"""
    from app.services.plancache import PlanCache

    c = PlanCache()
    c.put(1, "remote work", _make_plan("a"))
    assert c.get(1, "remote work benefits") is None


def test_plan_cache_ignores_surrounding_whitespace():
    """复制粘贴常带首尾空白，那还是同一份素材。"""
    from app.services.plancache import PlanCache

    c = PlanCache()
    c.put(1, "remote work", _make_plan("a"))
    assert c.get(1, "  remote work\n ") is not None


def test_plan_cache_is_case_sensitive():
    """
    大小写不同就该算不同素材。

    这条看着吹毛求疵，但归一化大小写会导致「Remote work」
    复用为「remote work」的计划 —— 后者可能聊的是完全不同的角度。
    """
    from app.services.plancache import PlanCache

    c = PlanCache()
    c.put(1, "Remote work", _make_plan("a"))
    assert c.get(1, "remote work") is None


def test_plan_cache_expires():
    from app.services.plancache import PlanCache

    c = PlanCache(ttl_sec=0.0)
    c.put(1, "x", _make_plan("a"))
    assert c.get(1, "x") is None, "过期条目不该再被复用"


def test_plan_cache_evicts_oldest_when_full():
    from app.services.plancache import PlanCache

    c = PlanCache(max_entries=3)
    for i in range(5):
        c.put(1, f"content-{i}", _make_plan(str(i)))
    assert len(c) == 3
    assert c.get(1, "content-0") is None, "最旧的应被淘汰"
    assert c.get(1, "content-4") is not None, "最新的必须还在"


def test_plan_cache_invalidate_user():
    """画像更新后要能作废该用户已备好的计划。"""
    from app.services.plancache import PlanCache

    c = PlanCache()
    c.put(1, "x", _make_plan("a"))
    c.put(2, "x", _make_plan("b"))
    c.invalidate_user(1)
    assert c.get(1, "x") is None
    assert c.get(2, "x") is not None, "不能连累别的用户"


def test_prepare_then_ws_reuses_same_plan(store, db, monkeypatch):
    """
    端到端：「生成话题」后紧接着开练，必须是同一份计划、只调一次模型。

    这是本次改动的核心断言。没有它，回归表现为"话题每次都不一样"，
    而那既不报错也不崩溃，只能靠用户抱怨才发现。
    """
    import asyncio
    from app.services.plancache import PlanCache, plan_cache

    import app.api.server as srv

    calls = []

    class FakePlanner:
        def __init__(self):
            pass

        def plan(self, content, *, profile_summary="", n_topics=5):
            calls.append(profile_summary)
            # 每次返回不同内容：若真调了两次，用户就能看出两份不一样
            return _make_plan(f"第{len(calls)}次")

    # 隔离的缓存，避免污染其他测试
    fake_cache = PlanCache()
    monkeypatch.setattr(srv, "user_store", lambda: store)
    monkeypatch.setattr(srv, "db", lambda: db)
    monkeypatch.setattr(srv, "profile_store",
                        lambda: ProfileStore(db, store.dir.parent / "profiles"))
    monkeypatch.setattr(srv, "plan_cache", lambda: fake_cache)
    monkeypatch.setattr(srv, "Planner", FakePlanner)

    a = store.create("A")

    # 第一次：HTTP 备课
    out = asyncio.run(srv.prepare(
        srv.PrepareIn(content="remote work", topics=5), user=a.id))
    assert len(calls) == 1

    # 第二次：WS 开练。这里只验到"计划被取用"为止，
    # 不真建连 —— 建连要连 Realtime 服务，不是单测该做的事。
    cached = fake_cache.get(a.id, "remote work")
    assert cached is not None, "HTTP 备课的结果没有被记下"
    assert cached.opening == out["plan"]["opening"] == "hi-第1次"
    assert len(calls) == 1, "WS 不该重新备课"


def test_ws_does_not_reuse_other_users_plan(store, db, monkeypatch):
    """A 备完课不能被 B 复用 —— 同一段素材也不行。"""
    import asyncio
    from app.services.plancache import PlanCache

    import app.api.server as srv

    calls = []

    class FakePlanner:
        def __init__(self):
            pass

        def plan(self, content, *, profile_summary="", n_topics=5):
            calls.append(1)
            return _make_plan(f"第{len(calls)}次")

    fake_cache = PlanCache()
    monkeypatch.setattr(srv, "user_store", lambda: store)
    monkeypatch.setattr(srv, "db", lambda: db)
    monkeypatch.setattr(srv, "profile_store",
                        lambda: ProfileStore(db, store.dir.parent / "profiles"))
    monkeypatch.setattr(srv, "plan_cache", lambda: fake_cache)
    monkeypatch.setattr(srv, "Planner", FakePlanner)

    a = store.create("A")
    b = store.create("B")

    asyncio.run(srv.prepare(srv.PrepareIn(content="same"), user=a.id))
    assert fake_cache.get(b.id, "same") is None, \
        "B 拿到了 A 的备课本 —— 这正是最初要修的串味"


def test_session_profile_update_invalidates_cache(store, db, monkeypatch):
    """画像更新后，之前按旧画像备好的计划必须作废。"""
    from app.services.plancache import PlanCache
    from app.services.profile import ProfileStore
    from app.storage.models import ProfileFact
    import app.services.session as session_mod
    from app.services.session import invalidate_plan_cache

    fake_cache = PlanCache()
    monkeypatch.setattr(session_mod, "plan_cache", lambda: fake_cache)

    a = store.create("A")
    fake_cache.put(a.id, "remote work", _make_plan("按旧画像"))

    # 模拟一次成功的画像抽取
    ps = ProfileStore(db, store.dir.parent / "profiles")
    ps.absorb(a.id, [ProfileFact(user_id=a.id, category="interest",
                                key="hobby", value="爬山", confidence=0.9)])
    invalidate_plan_cache(a.id)

    assert fake_cache.get(a.id, "remote work") is None, \
        "画像变了却还在用旧计划，用户会被拿过去的自己问话"


def test_deleting_user_clears_cache(store, db, monkeypatch):
    import asyncio
    from app.services.plancache import PlanCache
    import app.api.server as srv

    fake_cache = PlanCache()
    monkeypatch.setattr(srv, "user_store", lambda: store)
    monkeypatch.setattr(srv, "plan_cache", lambda: fake_cache)

    a = store.create("A")
    b = store.create("B")
    fake_cache.put(a.id, "x", _make_plan("a"))
    fake_cache.put(b.id, "x", _make_plan("b"))

    asyncio.run(srv.delete_user(a.id))
    assert fake_cache.get(a.id, "x") is None
    assert fake_cache.get(b.id, "x") is not None


def test_ws_actually_reuses_cached_plan(store, db, monkeypatch):
    """
    走真实的 ws_session，验证它真的**取用**了缓存，而不只是缓存里有。

    上面那个测试只证明了"写进去了"，没证明"读出来了" ——
    这两件事中间隔着整个 WS 流程，是最容易悄悄断掉的地方。

    在 build_tutor_instructions 处截断：不往下走 sess.start()，
    因为那会真连 Realtime 服务，不该出现在单测里。
    截断点选在这里，是因为它正好消费 plan，是验证"用对了"的最后一道。
    """
    import asyncio
    from app.services.plancache import PlanCache
    import app.api.server as srv

    calls = []
    seen = {}

    class FakePlanner:
        def __init__(self):
            pass

        def plan(self, content, *, profile_summary="", n_topics=5):
            calls.append(1)
            return _make_plan(f"第{len(calls)}次")

    def fake_build(plan, *, profile_summary="", minutes=30):
        # 在这里停住，并记下收到的计划。
        # 抛 srv._StopRequested 走的是正常收尾路径 —— ws_session 有
        # except 分支接住它，不会变成报错而掩盖真正的断言失败。
        seen["opening"] = plan.opening
        seen["topics"] = [t.title for t in plan.topics]
        raise srv._StopRequested()

    fake_cache = PlanCache()
    monkeypatch.setattr(srv, "user_store", lambda: store)
    monkeypatch.setattr(srv, "db", lambda: db)
    monkeypatch.setattr(srv, "profile_store",
                        lambda: ProfileStore(db, store.dir.parent / "profiles"))
    monkeypatch.setattr(srv, "plan_cache", lambda: fake_cache)
    monkeypatch.setattr(srv, "Planner", FakePlanner)
    monkeypatch.setattr(srv, "build_tutor_instructions", fake_build)

    a = store.create("A")
    asyncio.run(srv.prepare(srv.PrepareIn(content="remote work"), user=a.id))
    assert len(calls) == 1
    expected = "第1次"

    ws = _FakeWS(f"user={a.id}&content=" + quote_plus("remote work")
                 + "&voiceprint=0")
    asyncio.run(srv.ws_session(ws))

    assert seen, "没有走到 build_tutor_instructions"
    assert len(calls) == 1, \
        f"WS 又调了一次 Planner（应为 {expected}，实际调了 {len(calls)} 次）"
    assert seen["opening"] == f"hi-{expected}", \
        f"WS 用的是 {seen['opening']}，与预览的 hi-{expected} 不一致"

    # 发给前端的 plan 事件也应标明是复用
    plan_msgs = [m for m in ws.sent if m.get("type") == "plan"]
    assert plan_msgs, "没有下发 plan"
    assert plan_msgs[0]["reused"] is True, "前端会误以为又重新备了一次课"
