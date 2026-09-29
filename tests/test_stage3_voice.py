"""
阶段 3 测试：语音核心。
  · 音频工具（单位换算、重采样）
  · SSRF 防护（含误拦回归）
  · 声纹过滤逻辑（不加载真模型，用打桩）
  · 会话编排（打桩 Realtime，验证待定缓冲与纠错推送）
"""

import asyncio
import base64
import json

import numpy as np
import pytest

from app.core.audio import IN_RATE, resample_linear, rms, to_int16
from app.core.config import load_settings
from app.core.content import _assert_public_host, classify
from app.services.session import (MAX_PENDING, VOICE_RMS_FLOOR,
                                  ConversationSession)


# ============================================================
#  音频工具
# ============================================================

def test_rms_consistent_between_dtypes():
    """float32 与 int16 的 RMS 必须一致（曾经的单位 bug）。"""
    x = (np.sin(np.linspace(0, 100, 16000)) * 0.3).astype(np.float32)
    assert abs(rms(x) - rms(to_int16(x))) < 50


def test_rms_empty_is_zero():
    assert rms(np.zeros(0, dtype=np.float32)) == 0.0


def test_to_int16_scales_float():
    x = np.array([0.5, -0.5, 1.0], dtype=np.float32)
    out = to_int16(x)
    assert out.dtype == np.int16
    assert out[0] == 16384
    assert out[2] == 32767


def test_resample_length():
    a = np.arange(48000, dtype=np.float32)
    assert len(resample_linear(a, 48000, IN_RATE)) == IN_RATE


def test_resample_noop_when_same_rate():
    a = np.arange(100, dtype=np.float32)
    assert len(resample_linear(a, IN_RATE, IN_RATE)) == 100


# ============================================================
#  SSRF 防护（含真实踩到的误拦）
# ============================================================

@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8000/x",
    "http://localhost/x",
    "http://10.0.0.5",
    "http://192.168.3.1",
    "http://172.16.0.1",
    "http://169.254.169.254/latest/meta-data",   # 云元数据端点
    "http://[::1]/",
    "http://0.0.0.0",
    "http://100.64.0.1",                          # 运营商级 NAT
])
def test_ssrf_blocks_internal(url):
    with pytest.raises(ValueError):
        _assert_public_host(url)


@pytest.mark.parametrize("url", [
    "https://en.wikipedia.org/wiki/Remote_work",
    "https://example.com",
])
def test_ssrf_allows_public(url):
    """回归：en.wikipedia.org 曾因 IPv6 判定被误拦。"""
    _assert_public_host(url)   # 不应抛异常


def test_ssrf_rejects_unresolvable():
    with pytest.raises(ValueError):
        _assert_public_host("http://this-domain-does-not-exist-xyz123.invalid")


# ============================================================
#  声纹过滤（打桩）
# ============================================================

class FakeVerifier:
    """可编排返回值的声纹打桩。"""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.reset_count = 0
        self.last_score = 0.9

    def feed(self, samples):
        return self.verdicts.pop(0) if self.verdicts else None

    def reset(self):
        self.reset_count += 1


class FakeRT:
    """Realtime 打桩，记录发出的音频。"""

    def __init__(self, **kw):
        self.sent = []
        self.last_usage = None
        self.closed = False
        self.cancelled = 0

    async def connect(self):
        pass

    async def send_audio(self, pcm):
        self.sent.append(pcm)

    async def cancel_response(self):
        self.cancelled += 1

    async def close(self):
        self.closed = True


@pytest.fixture
def db(tmp_path):
    from app.storage.db import Database
    d = Database(tmp_path / "t.db")
    d.init_schema()
    yield d
    d.close()


def make_session(db, verifier=None, on_client=None, monkeypatch=None):
    """构造会话，Realtime 已打桩。"""
    settings = load_settings({"DASHSCOPE_API_KEY": "x",
                              "QWEN_WORKSPACE_ID": "ws-test"})
    msgs = []

    async def default_client(m):
        msgs.append(m)

    # 必须先建会话行：turns/corrections 有外键约束
    sid = db.create_session("topic", "test input")
    sess = ConversationSession(
        settings, db, session_id=sid, instructions="test",
        on_client=on_client or default_client,
        voiceprint=verifier, minutes=30)
    fake = FakeRT()
    if monkeypatch is not None:
        monkeypatch.setattr("app.services.session.RealtimeSession",
                            lambda *a, **k: fake)
    sess.rt = fake
    return sess, fake, msgs, sid


def loud_block():
    """一块足够响的音频（超过静音门槛）。"""
    return (np.sin(np.linspace(0, 500, 320)) * 20000).astype(np.int16).tobytes()


def quiet_block():
    return np.zeros(320, dtype=np.int16).tobytes()


@pytest.mark.asyncio
async def test_voiceprint_unknown_does_not_pass_through(db, monkeypatch):
    """关键回归：判定未知期间音频必须攒着，不能直接放行。

    若"未知就放行"，旁人一句开场白就能触发模型。
    """
    v = FakeVerifier([None, None])
    sess, fake, _, sid = make_session(db, v, monkeypatch=monkeypatch)

    await sess.push_audio(loud_block())
    await sess.push_audio(loud_block())

    assert fake.sent == [], "判定未知时不应向模型发送音频"
    assert len(sess._pending) == 2


@pytest.mark.asyncio
async def test_voiceprint_true_flushes_pending(db, monkeypatch):
    """判定为本人后，之前攒的音频要一起送出去，不能丢。"""
    v = FakeVerifier([None, True])
    sess, fake, _, sid = make_session(db, v, monkeypatch=monkeypatch)

    await sess.push_audio(loud_block())
    await sess.push_audio(loud_block())

    assert len(fake.sent) == 2, f"应补发 2 块，实际 {len(fake.sent)}"


@pytest.mark.asyncio
async def test_voiceprint_false_drops_pending(db, monkeypatch):
    """判定非本人：丢弃待定缓冲，并通知前端。"""
    v = FakeVerifier([None, False])
    sess, fake, msgs, sid = make_session(db, v, monkeypatch=monkeypatch)

    await sess.push_audio(loud_block())
    await sess.push_audio(loud_block())

    assert fake.sent == [], "非本人的音频不应送出"
    assert any(m["type"] == "voiceprint_skip" for m in msgs)
    assert sess.stats.blocked_chunks == 1


@pytest.mark.asyncio
async def test_silence_does_not_clear_verifier_buffer(db, monkeypatch):
    """关键回归：静音必须攒进待定缓冲，不能清空声纹缓冲。

    浏览器降噪会把句间停顿压得很低，一旦清空就永远攒不满 1 秒。
    """
    v = FakeVerifier([])
    sess, fake, _, sid = make_session(db, v, monkeypatch=monkeypatch)

    for _ in range(5):
        await sess.push_audio(quiet_block())

    assert len(sess._pending) == 5, "静音块应被保留"
    assert fake.sent == []


@pytest.mark.asyncio
async def test_pending_overflow_flushes(db, monkeypatch):
    """长时间判不出声纹时要兜底放行，否则对话彻底哑掉。"""
    v = FakeVerifier([None] * (MAX_PENDING + 5))
    sess, fake, msgs, sid = make_session(db, v, monkeypatch=monkeypatch)

    for _ in range(MAX_PENDING + 1):
        await sess.push_audio(loud_block())

    assert fake.sent, "溢出后应兜底放行"
    assert any(m["type"] == "voiceprint_stuck" for m in msgs)


@pytest.mark.asyncio
async def test_no_verifier_passes_through(db, monkeypatch):
    """不开声纹时，音频直接送模型。"""
    sess, fake, _, sid = make_session(db, None, monkeypatch=monkeypatch)
    await sess.push_audio(loud_block())
    assert len(fake.sent) == 1


@pytest.mark.asyncio
async def test_audio_ignored_while_model_speaking(db, monkeypatch):
    """AI 说话期间不处理输入，避免自我对话。"""
    sess, fake, _, sid = make_session(db, None, monkeypatch=monkeypatch)
    sess._model_speaking = True
    await sess.push_audio(loud_block())
    assert fake.sent == []


# ============================================================
#  会话编排
# ============================================================

@pytest.mark.asyncio
async def test_model_speaking_flag_toggles(db, monkeypatch):
    sess, _, _, sid = make_session(db, None, monkeypatch=monkeypatch)
    await sess._on_qwen_event({"type": "response.created"})
    assert sess._model_speaking is True
    await sess._on_qwen_event({"type": "response.done"})
    assert sess._model_speaking is False


@pytest.mark.asyncio
async def test_verifier_reset_after_response(db, monkeypatch):
    """AI 说完要清声纹缓冲，避免把自己的尾音算进去。"""
    v = FakeVerifier([])
    sess, _, _, sid = make_session(db, v, monkeypatch=monkeypatch)
    await sess._on_qwen_event({"type": "response.done"})
    assert v.reset_count == 1


@pytest.mark.asyncio
async def test_audio_delta_forwarded_as_base64(db, monkeypatch):
    sess, _, msgs, sid = make_session(db, None, monkeypatch=monkeypatch)
    pcm = b"\x01\x02\x03\x04"
    await sess._on_qwen_event({
        "type": "response.audio.delta",
        "delta": base64.b64encode(pcm).decode()})
    got = [m for m in msgs if m["type"] == "audio"]
    assert got and base64.b64decode(got[0]["pcm"]) == pcm


@pytest.mark.asyncio
async def test_user_text_persisted(db, monkeypatch):
    sess, _, msgs, sid = make_session(db, None, monkeypatch=monkeypatch)
    await sess._on_user_text("I went to the park yesterday.")

    turns = db.list_turns(sid)
    assert len(turns) == 1
    assert turns[0].role == "user"
    assert any(m["type"] == "user_text" for m in msgs)


@pytest.mark.asyncio
async def test_ai_text_persisted(db, monkeypatch):
    sess, _, msgs, sid = make_session(db, None, monkeypatch=monkeypatch)
    await sess._on_ai_text("That sounds fun! What did you do there?")

    turns = db.list_turns(sid)
    assert turns[0].role == "assistant"
    assert sess.stats.ai_chars > 0


@pytest.mark.asyncio
async def test_correction_pushed_and_filtered(db, monkeypatch):
    """发音 minor 入库但不推前端（需求 4）。"""
    sess, _, msgs, sid = make_session(db, None, monkeypatch=monkeypatch)

    class FakeCorrector:
        def check(self, text, **kw):
            from app.services.corrector import CorrectionItem
            return [
                CorrectionItem(kind="grammar", severity="critical",
                               original="I go", suggestion="I went"),
                CorrectionItem(kind="pronunciation", severity="minor",
                               original="again", suggestion="/əˈɡen/",
                               word="again"),
            ]

    sess.corrector = FakeCorrector()
    await sess._on_user_text("I go to park.")
    await asyncio.sleep(0.2)      # 等异步纠错任务

    stored = db.list_corrections(sid)
    assert len(stored) == 2, "两条都应入库"
    pushed = [m for m in msgs if m["type"] == "correction"]
    assert len(pushed) == 1, "只应推语法那条"
    assert pushed[0]["kind"] == "grammar"


@pytest.mark.asyncio
async def test_correction_failure_does_not_crash(db, monkeypatch):
    """纠错模型报错不能影响对话。"""
    sess, _, _, sid = make_session(db, None, monkeypatch=monkeypatch)

    class BrokenCorrector:
        def check(self, text, **kw):
            raise RuntimeError("模型挂了")

    sess.corrector = BrokenCorrector()
    await sess._on_user_text("hello there")   # 不应抛出
    await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_short_input_skips_corrector(db, monkeypatch):
    """太短的输入不该触发纠错调用。"""
    sess, _, _, sid = make_session(db, None, monkeypatch=monkeypatch)
    from app.services.corrector import Corrector
    sess.corrector = Corrector()
    # 直接调 check，应短路返回空
    assert sess.corrector.check("yes") == []
    assert sess.corrector.check("") == []


# ============================================================
#  计时与统计
# ============================================================

@pytest.mark.asyncio
async def test_remaining_and_expiry(db, monkeypatch):
    sess, _, _, sid = make_session(db, None, monkeypatch=monkeypatch)
    sess.duration_limit = 0
    assert sess.remaining_sec() == 0
    assert sess.is_expired() is True


@pytest.mark.asyncio
async def test_snapshot_shape(db, monkeypatch):
    sess, _, _, sid = make_session(db, None, monkeypatch=monkeypatch)
    snap = sess.snapshot()
    for k in ["elapsed_sec", "remaining_sec", "user_chars", "ai_chars",
              "blocked_sec"]:
        assert k in snap


@pytest.mark.asyncio
async def test_stop_is_idempotent(db, monkeypatch):
    sess, fake, _, sid = make_session(db, None, monkeypatch=monkeypatch)
    await sess.stop()
    await sess.stop()          # 第二次不应报错
    assert fake.closed
