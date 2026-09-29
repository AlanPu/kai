"""
阶段 6 测试：费用统计与长期稳定性。

阶段 6 解决三件遗留问题：
  · 费用不可见 —— 代码收集 usage 但从不展示
  · 断线无处理 —— 只是弹个提示就丢弃整个会话
  · 长时间稳定性从未验证（原型遗留债务）
"""

import asyncio
import json

import pytest

from app.services.cost import (DEFAULT_PRICE, TEXT_PRICE, Usage,
                               estimate_cost, format_usage,
                               parse_realtime_usage)

# Qwen Realtime 实测返回的真实结构 ——
# 字段名与 OpenAI 不同，照抄文档会静默归零
REAL_USAGE = {
    "total_tokens": 712,
    "input_tokens": 695,
    "output_tokens": 17,
    "input_tokens_details": {"text_tokens": 695},
    "output_tokens_details": {"text_tokens": 5, "audio_tokens": 12},
}


# ============================================================
#  解析
# ============================================================

def test_parse_real_structure():
    """回归：必须能解析实测的真实结构。

    早期版本按 OpenAI 的字段名找 input_text_tokens，
    在这份数据上全部落空 —— 用量永远显示 0，
    看起来像"免费"，实际是解析失败。
    """
    u = parse_realtime_usage(REAL_USAGE)
    assert u.text_in_tokens == 695
    assert u.audio_out_tokens == 12
    assert u.text_out_tokens == 5
    assert u.total_tokens == 712


def test_parse_total_matches_input_plus_output():
    """解析出的各部分之和应等于 total_tokens，不多不少。"""
    u = parse_realtime_usage(REAL_USAGE)
    assert u.total_tokens == REAL_USAGE["total_tokens"]


def test_parse_none_and_empty():
    for bad in (None, {}, {"foo": "bar"}):
        u = parse_realtime_usage(bad)
        assert u.total_tokens == 0


def test_parse_falls_back_to_toplevel():
    """细节字段缺失时用顶层总数兜底，不能记成 0。"""
    u = parse_realtime_usage({"input_tokens": 500, "output_tokens": 80})
    assert u.total_tokens == 580


def test_parse_ignores_non_numeric():
    u = parse_realtime_usage({
        "input_tokens_details": {"text_tokens": "很多"},
        "input_tokens": 300})
    assert u.text_in_tokens == 300


def test_parse_handles_wrong_detail_type():
    """细节字段类型异常时不能崩。"""
    u = parse_realtime_usage({
        "input_tokens_details": "not a dict",
        "input_tokens": 100})
    assert u.text_in_tokens == 100


def test_parse_cached_tokens():
    u = parse_realtime_usage({"input_tokens_details": {"cached_tokens": 64},
                              "input_tokens": 200})
    assert u.cached_tokens == 64


# ============================================================
#  计价
# ============================================================

def test_estimate_cost_is_positive():
    assert estimate_cost(parse_realtime_usage(REAL_USAGE)) > 0


def test_estimate_cost_zero_when_no_usage():
    assert estimate_cost(Usage()) == 0


def test_estimate_cost_scales_linearly():
    """用量翻倍，费用应翻倍。"""
    a = estimate_cost(Usage(text_in_tokens=1000))
    b = estimate_cost(Usage(text_in_tokens=2000))
    assert abs(b - 2 * a) < 1e-6


def test_estimate_cost_accepts_custom_price():
    """价格参数要真的生效，不能名字叫 price 却只管一部分。"""
    free = {"input_per_1k": 0.0, "output_per_1k": 0.0}
    u = Usage(text_in_tokens=9999, audio_out_tokens=9999)
    assert estimate_cost(u, price=free, text_price=free) == 0
    assert estimate_cost(u, price=free) > 0, "只覆盖音频时文本仍计费"


def test_prices_are_sane():
    """单价是量级参考，防手滑写成 800 倍。"""
    for p in (DEFAULT_PRICE, TEXT_PRICE):
        assert 0 < p["input_per_1k"] < 1
        assert 0 < p["output_per_1k"] < 1


def test_cost_is_small_for_single_session():
    """
    单次半小时会话的费用应在几毛钱量级。

    这条断言的意义：如果哪天单价改错或 token 解析错位，
    费用会突然变成几块或几千块，这里会立刻发现。
    """
    u = Usage(text_in_tokens=200_000, text_out_tokens=40_000,
              audio_out_tokens=30_000)
    assert 0 < estimate_cost(u) < 2.0


# ============================================================
#  展示
# ============================================================

def test_format_usage_lists_parts():
    s = format_usage(parse_realtime_usage(REAL_USAGE))
    assert "695" in s and "12" in s


def test_format_usage_empty():
    assert "无" in format_usage(Usage())


# ============================================================
#  usage_json 往返（数据库 → 报告）
# ============================================================

def test_usage_survives_json_roundtrip(tmp_path):
    """用量存进库再读出来，费用必须一致。"""
    from app.storage.db import Database

    db = Database(tmp_path / "u.db")
    db.init_schema()
    sid = db.create_session("topic", "x")
    db.finish_session(sid, duration_sec=1800, usage=REAL_USAGE)
    s = db.get_session(sid)
    db.close()

    assert s.usage_json
    u = parse_realtime_usage(json.loads(s.usage_json))
    assert u.total_tokens == 712
    assert estimate_cost(u) > 0


# ============================================================
#  稳定性测试脚本本身
# ============================================================

def test_soak_script_exists():
    """30 分钟稳定性测试必须可复现，不能只靠一次性手工验证。"""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "soak_test.py").is_file()
    assert (root / "scripts" / "make_sample_speech.py").is_file()


def test_soak_uses_real_speech():
    """
    回归：稳定性测试必须用真人声。

    早期用谐波合成音，Qwen 的 VAD 不认，
    测试报告"从未识别出用户语音" —— 看起来像 bug，
    实际是素材不像人话，白白浪费一次 30 分钟的排查。
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "soak_test.py").read_text(encoding="utf-8")
    assert "sample_speech" in src
    assert "speech_like" not in src, "不该再用合成音"


# ============================================================
#  致命错误处理（30 分钟稳定性测试的真实发现）
# ============================================================

def test_fatal_error_detection():
    """
    回归：30 分钟稳定性测试暴露的真实问题。

    Qwen 在长时间没有「被识别为用户说话」的音频后，
    会直接关闭会话（user_idle_timeout，300 秒）。
    但客户端一直在发音频字节，所以程序不知道连接已经死了。

    最初的表现：报错后服务器又空转 24 分钟，
    用户对着一个死连接说话，以为还在练。
    """
    from app.core.realtime import is_fatal_error
    assert is_fatal_error("user_idle_timeout")
    assert is_fatal_error("session_expired")
    assert is_fatal_error("invalid_api_key")
    assert is_fatal_error("insufficient_quota")
    assert is_fatal_error("connection_closed")


def test_non_fatal_errors_do_not_end_session():
    """这些错误只影响一次回复，不该终止整场对话。"""
    from app.core.realtime import is_fatal_error
    assert not is_fatal_error("content_filter")
    assert not is_fatal_error("response_failed")
    assert not is_fatal_error("")
    assert not is_fatal_error(None)


def test_session_fatal_defined_in_core():
    """
    分层：SessionFatal 必须定义在 core。

    它最初定义在 services，然后 core.realtime 反过来导入它 ——
    底层依赖上层，会形成循环，只是碰巧导入顺序对才没炸。
    """
    from app.core.realtime import SessionFatal
    assert SessionFatal.__module__ == "app.core.realtime"


@pytest.mark.asyncio
async def test_fatal_error_raises_and_marks():
    """收到致命错误必须抛出，并记下原因。"""
    from app.core.realtime import SessionFatal
    from app.services.session import ConversationSession, SessionStats

    sent = []

    async def on_client(m):
        sent.append(m)

    s = ConversationSession.__new__(ConversationSession)
    s.on_client = on_client
    s.ended = False
    s.verifier = None
    s.stats = SessionStats()
    s.fatal_error = None
    s._model_speaking = False

    with pytest.raises(SessionFatal):
        await ConversationSession._on_qwen_event(
            s, {"type": "error",
                "error": {"code": "user_idle_timeout"}})

    assert s.fatal_error == "user_idle_timeout"
    assert any(m["type"] == "error" for m in sent), "要先告知客户端"


@pytest.mark.asyncio
async def test_non_fatal_error_continues():
    """非致命错误不应打断对话。"""
    from app.services.session import ConversationSession, SessionStats

    async def on_client(m):
        pass

    s = ConversationSession.__new__(ConversationSession)
    s.on_client = on_client
    s.ended = False
    s.verifier = None
    s.stats = SessionStats()
    s.fatal_error = None
    s._model_speaking = False

    # 不应抛出
    await ConversationSession._on_qwen_event(
        s, {"type": "error", "error": {"code": "content_filter"}})
    assert s.fatal_error is None


def test_fatal_message_is_human_readable():
    """错误码要翻译成人话，不能把 code 直接丢给用户看。"""
    from app.api.server import _fatal_message
    idle = _fatal_message("user_idle_timeout")
    assert "说话" in idle, "要说明原因"
    assert "报告" in idle, "要告诉用户已有内容还在"
    assert _fatal_message("invalid_api_key")
    assert _fatal_message("insufficient_quota")
    # 未知错误也要能给出可读文本
    assert _fatal_message("some_unknown_thing")


# ============================================================
#  暂停 / 恢复
# ============================================================

def _bare_session():
    """构造一个不连网的最小会话，只测计时逻辑。"""
    import time
    from app.services.session import ConversationSession, SessionStats
    s = ConversationSession.__new__(ConversationSession)
    s.stats = SessionStats()
    s.started_at = time.time()
    s.paused = False
    s.paused_total = 0.0
    s._paused_at = None
    s.ended = False
    s.duration_limit = 1800
    return s


def test_pause_stops_the_clock():
    """暂停期间不该计时 —— 中途去倒杯水不该算进练习时长。"""
    import time
    s = _bare_session()
    time.sleep(0.2)
    before = s.elapsed()
    s.pause()
    time.sleep(0.5)
    assert abs(s.elapsed() - before) < 0.1, "暂停后 elapsed 不该增长"


def test_unpause_resumes_the_clock():
    import time
    s = _bare_session()
    s.pause()
    time.sleep(0.3)
    s.unpause()
    time.sleep(0.2)
    assert s.elapsed() > 0.15, "恢复后应继续计时"
    assert s.paused_total >= 0.25


def test_remaining_sec_excludes_pause():
    """剩余时间也要排除暂停，否则用户歇一会儿就被判超时。"""
    import time
    s = _bare_session()
    s.duration_limit = 10
    s.pause()
    time.sleep(0.5)
    s.unpause()
    assert s.remaining_sec() >= 9, "暂停不该消耗配额时间"


def test_pause_twice_is_idempotent():
    """重复暂停不能把暂停起点刷新，否则暂停时长会算错。"""
    import time
    s = _bare_session()
    s.pause()
    first = s._paused_at
    time.sleep(0.15)
    s.pause()
    assert s._paused_at == first


def test_unpause_without_pause_is_safe():
    s = _bare_session()
    s.unpause()          # 不应抛出
    assert not s.paused
    assert s.paused_total == 0.0


@pytest.mark.asyncio
async def test_paused_session_drops_audio():
    """
    暂停期间必须丢弃音频。

    否则麦克风收到的环境音会照常上传，
    既可能误触发 Qwen 的说话检测，也让 AI 对着空房间回话。
    """
    from app.services.session import ConversationSession, SessionStats

    s = ConversationSession.__new__(ConversationSession)
    s.paused = True
    s.ended = False
    s.stats = SessionStats()
    s.rt = object()                       # 非 None，确保是 paused 拦下的
    s.verifier = None
    s._model_speaking = False

    sent = []
    s.stats.audio_in_blocks = 0
    await ConversationSession.push_audio(s, b"\x00" * 640)
    assert s.stats.audio_in_blocks == 0, "暂停时不该处理音频"


def test_snapshot_reports_paused_state():
    """前端要靠 snapshot 知道当前是否暂停。"""
    s = _bare_session()
    s.pause()
    snap = s.snapshot()
    assert snap["paused"] is True
    assert "paused_sec" in snap


# ============================================================
#  严重回归：用户只能说话一次
# ============================================================

@pytest.mark.asyncio
async def test_user_speech_start_does_not_lock_out_user():
    """
    回归：本项目最严重的一个逻辑错误。

    is_speech_started 表示「用户开始说话」，
    但代码当时把它当成「AI 开始说话」，置 _model_speaking = True。

    而 push_audio 在 _model_speaking 为真时丢弃所有音频，
    该标志又只在 response.done 复位 ——
    于是用户说完第一句后再也发不出声音，音频全被丢掉，
    不产生 response，也就永远等不到 response.done，形成死锁。

    实测后果：一整场练习里用户只能被识别一次。
    """
    from app.services.session import ConversationSession, SessionStats

    async def on_client(m):
        pass

    s = ConversationSession.__new__(ConversationSession)
    s.on_client = on_client
    s.ended = False
    s.verifier = None
    s.stats = SessionStats()
    s._model_speaking = False

    await ConversationSession._on_qwen_event(
        s, {"type": "input_audio_buffer.speech_started"})

    assert s._model_speaking is False, \
        "用户开始说话时必须解除 AI 说话状态，否则用户会被永久静音"
    assert s.stats.user_speech_starts == 1


@pytest.mark.asyncio
async def test_audio_flows_after_user_speech_started():
    """用户开始说话后，音频必须能继续送上去。"""
    from app.services.session import (ConversationSession, SessionStats,
                                      VOICE_RMS_FLOOR)

    async def on_client(m):
        pass

    s = ConversationSession.__new__(ConversationSession)
    s.on_client = on_client
    s.ended = False
    s.paused = False
    s.stats = SessionStats()
    s._model_speaking = False
    s.verifier = None

    sent = []

    class FakeRT:
        last_usage = None

        async def send_audio(self, b):
            sent.append(b)

    s.rt = FakeRT()

    # 模拟用户开始说话
    await ConversationSession._on_qwen_event(
        s, {"type": "input_audio_buffer.speech_started"})

    # 再发一块有声音的音频，必须被送出
    loud = (b"\x10\x20" * 320)          # 非静音
    await ConversationSession.push_audio(s, loud)
    assert sent, "用户说话后音频不该被丢弃"
    assert s.stats.audio_in_blocks == 1


@pytest.mark.asyncio
async def test_response_done_clears_speaking_flag():
    """AI 说完后要复位标志，轮到用户。"""
    from app.services.session import ConversationSession, SessionStats

    async def on_client(m):
        pass

    s = ConversationSession.__new__(ConversationSession)
    s.on_client = on_client
    s.ended = False
    s.verifier = None
    s.stats = SessionStats()
    s._model_speaking = True

    await ConversationSession._on_qwen_event(
        s, {"type": "response.done"})
    assert s._model_speaking is False


def test_stats_has_speech_start_counter():
    """
    这个计数器是排查上面那个 bug 的关键手段：
    它只涨到 1 就说明用户说完第一句后再也发不出声音。
    """
    from app.services.session import SessionStats
    assert SessionStats().user_speech_starts == 0


# ============================================================
#  上下文轮转（Qwen 的 320 条 audio item 上限）
# ============================================================

def test_audio_limit_constants_are_conservative():
    """
    轮转阈值必须比真实上限保守。

    我数的是 input_audio_buffer.committed（只在用户侧），
    而服务端计的是全部 audio item（还含 AI 的输出）。
    实测每轮对话我数到约 2 条、服务端约 2.6 条，
    所以数到 240 就该轮转，等到 320 就来不及了。
    """
    from app.services.session import (AUDIO_ITEM_LIMIT,
                                      AUDIO_ITEM_ROTATE_AT)
    assert AUDIO_ITEM_LIMIT == 320, "Qwen 的硬上限"
    assert AUDIO_ITEM_ROTATE_AT < AUDIO_ITEM_LIMIT
    # 留足余量：至少 15%（约 15 轮对话）
    assert AUDIO_ITEM_ROTATE_AT <= AUDIO_ITEM_LIMIT * 0.85


def test_recoverable_error_detection():
    """Too many audios 必须是可恢复的，不能当致命错误结束会话。"""
    from app.core.realtime import is_recoverable_error
    assert is_recoverable_error(
        "InvalidParameter: Too many audios. The maximum allowed is 320.")
    assert is_recoverable_error("too_many_audios")
    # 真正的致命错误不能被误判成可恢复
    assert not is_recoverable_error("user_idle_timeout")
    assert not is_recoverable_error("invalid_api_key")
    assert not is_recoverable_error("")


def test_recoverable_is_checked_before_fatal():
    """
    Too many audios 的 code 是 InvalidParameter，
    消息里才带 Too many audios。若先判 is_fatal_error，
    会被当成致命错误直接结束会话 —— 顺序不能反。
    """
    import pathlib
    src = pathlib.Path(
        "./app/services/session.py"
    ).read_text()
    i_rec = src.index("is_recoverable_error(msg)")
    i_fat = src.index("is_fatal_error(code)")
    assert i_rec < i_fat, "可恢复判断必须先于致命判断"


@pytest.mark.asyncio
async def test_committed_event_counts_audio_items():
    """每个 committed 都要计数，这是轮转的触发依据。"""
    from app.services.session import ConversationSession, SessionStats

    async def on_client(m):
        pass

    s = ConversationSession.__new__(ConversationSession)
    s.on_client = on_client
    s.ended = False
    s.verifier = None
    s.stats = SessionStats()
    s._model_speaking = False
    s._rotate_at = 3
    s._rotating = False
    s._want_rotate = False

    for expect in (1, 2):
        await ConversationSession._on_qwen_event(
            s, {"type": "input_audio_buffer.committed"})
        assert s.stats.audio_items == expect
    assert s.wants_rotate() is False, "还没到阈值不该请求轮转"

    await ConversationSession._on_qwen_event(
        s, {"type": "input_audio_buffer.committed"})
    assert s.wants_rotate() is True, "到阈值应请求轮转"


@pytest.mark.asyncio
async def test_rotation_is_not_triggered_inside_recv_callback():
    """
    轮转只能在 recv 回调之外执行。

    曾经的实现直接在回调里 await rotate_context()，
    而它会关掉正在跑这个回调的连接 —— 接收循环被掐断，
    异常冒泡出去整个会话被判异常结束。
    所以回调里只能置标志。
    """
    from app.services.session import ConversationSession, SessionStats

    async def on_client(m):
        pass

    s = ConversationSession.__new__(ConversationSession)
    s.on_client = on_client
    s.ended = False
    s.verifier = None
    s.stats = SessionStats()
    s._model_speaking = False
    s._rotate_at = 1
    s._rotating = False
    s._want_rotate = False
    s._correction_tasks = set()

    called = []

    async def fake_rotate():
        called.append(1)
        return True

    s.rotate_context = fake_rotate

    await ConversationSession._on_qwen_event(
        s, {"type": "input_audio_buffer.committed"})
    await asyncio.sleep(0)          # 让可能被创建的任务跑一下

    assert s.wants_rotate() is True
    assert called == [], "回调里不能直接执行轮转"


def test_wants_rotate_is_consumed_once():
    """
    轮转标志必须只被消费一次。

    否则 timer 循环每 5 秒重试，一次会话里会反复重连。
    """
    import pathlib
    src = pathlib.Path(
        "./app/services/session.py"
    ).read_text()
    i = src.index("async def rotate_context")
    body = src[i:i + 900]
    assert "_want_rotate = False" in body, "进入轮转时应先清标志"
