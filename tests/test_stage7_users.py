"""
阶段 7 测试：多用户与声纹重录。

背景：多人共用一台电脑各自练习。要求每个用户有独立的
声纹、画像和练习记录；声纹可以随时重录、也可以只清声纹
而保留画像历史。
"""

import json

import numpy as np
import pytest

from app.core.enroll import (MAX_SPEECH_SEC, MIN_COHERENCE, MIN_SAMPLES,
                             MIN_SPEECH_SEC, EnrollmentSession)
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
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
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


def test_realtime_has_no_commit_audio():
    """commit_audio 已废弃：服务端在 speech_stopped 时自己 commit。

    我们再做一次就是空提交，服务端回
    "buffer too small, or have no audio"。
    """
    from app.core.realtime import RealtimeSession

    assert not hasattr(RealtimeSession, "commit_audio"), \
        "commit_audio 应已删除，否则会诱导再次写出空提交"


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

