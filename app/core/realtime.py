"""
Qwen Realtime 会话封装。

协议要点（实测确定，改动会直接导致失败）：
  · 上行 16 kHz / 下行 24 kHz —— 上下行采样率**不同**
  · modalities 必须同时含 "text" 和 "audio"，只给 "audio" 会 invalid_value
  · 转写模型固定为 qwen3-asr-flash-realtime
  · WebSocket 协议**无 AEC/降噪**，需客户端自行处理

本模块只负责协议，不含业务逻辑（声纹过滤、纠错在别处）。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Optional

import websockets

from .config import Settings
from .llm import LLMError

log = logging.getLogger(__name__)


class SessionFatal(Exception):
    """
    服务端已判定会话不可继续（如长时间无有效输入）。

    定义在 core 而不是 services —— core 是底层模块，
    反过来依赖 services 会形成循环，且违反分层。
    """


# 这些错误意味着连接已经没用了，继续等只是浪费时间
# 「audio item 数量超限」是可恢复的：重建连接即可继续，
# 不该当成致命错误直接结束会话。
RECOVERABLE_CODES = {
    "too_many_audios",
}


def is_recoverable_error(code: str) -> bool:
    """audio item 超限等错误可以通过重连恢复。"""
    if not code:
        return False
    c = code.lower()
    if c in RECOVERABLE_CODES:
        return True
    return "too many audio" in c or "too_many_audio" in c


# 额度类错误：重连也不会好，必须如实告诉用户去充值/改计费方式，
# 而不是让他对着一个永远连不上的会话反复重试。
QUOTA_CODES = {
    "AllocationQuota.FreeTierOnly",
    "AllocationQuota.FreeTierExceeded",
    "Throttling.AllocationQuota",
    "insufficient_quota",
}

FATAL_CODES = {
    "user_idle_timeout",     # 用户长时间没说话，服务端关闭
    "session_expired",
    "session_closed",
    "connection_closed",
    "internal_error",
    "invalid_api_key",
    "insufficient_quota",
    "rate_limit_exceeded",
    "invalid_request_error",
}


def is_fatal_error(code) -> bool:
    c = str(code or "").lower()
    return any(f in c for f in FATAL_CODES) or is_quota_error(code)


def is_quota_error(code) -> bool:
    """
    是否是额度/计费问题。

    单独拎出来是因为要给不同的提示：致命错误说「连接不可用了」，
    额度问题要说「去充值或关掉仅用免费额度」——
    后者用户自己能解决，前者只能重试。
    """
    c = str(code or "").lower()
    return any(q.lower() in c for q in QUOTA_CODES)

# 事件处理器：收到事件时调用，可 await
EventHandler = Callable[[dict], Awaitable[None]]


class RealtimeSession:
    """
    一条与 Qwen Realtime 的 WebSocket 会话。

    用法：
        s = RealtimeSession(settings, instructions="...")
        await s.connect()
        await s.send_audio(pcm_int16_bytes)
        ...
        await s.close()
    """

    def __init__(self, settings: Settings, *, instructions: str = "",
                 voice: Optional[str] = None,
                 vad_threshold: float = 0.5,
                 silence_ms: int = 800,
                 on_event: Optional[EventHandler] = None):
        self.s = settings
        self.instructions = instructions
        self.voice = voice or settings.qwen_voice
        self.vad_threshold = vad_threshold
        self.silence_ms = silence_ms
        self.on_event = on_event

        self.ws: Optional[Any] = None
        self.ready = False
        self.closed = False
        self.session_id: Optional[str] = None
        self._recv_task: Optional[asyncio.Task] = None
        # 由接收任务结束时写入，供上层判断「连接是不是因为
        # 致命错误而死的」（区别于正常关闭）
        self.fatal_code: Optional[str] = None
        self._last_error: Optional[str] = None
        self.last_usage: Optional[dict] = None

    # ---------- 连接 ----------

    async def connect(self) -> None:
        if not self.s.has_voice_credentials():
            raise LLMError("未配置语音凭据（DASHSCOPE_API_KEY / QWEN_WORKSPACE_ID）")

        import certifi
        import ssl
        ctx = ssl.create_default_context(cafile=certifi.where())

        url = self.s.voice_ws_url()
        log.info("连接 Qwen Realtime: %s", url.split("?")[0])
        self.ws = await websockets.connect(
            url,
            additional_headers={"Authorization": f"Bearer {self.s.dashscope_api_key}"},
            ssl=ctx,
            max_size=None,
            open_timeout=20,
            ping_interval=20,
            ping_timeout=20)

        # 服务端会先发 session.created
        first = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=20))
        if first.get("type") == "session.created":
            self.session_id = (first.get("session") or {}).get("id")
        self.ready = True

        await self._configure()
        self._recv_task = asyncio.create_task(self._recv_loop())
        # 接收任务是 fire-and-forget 的，没人 await 它。
        # 不给它挂回调的话，_recv_loop 里冒出来的异常（尤其是
        # SessionFatal）会被静静存在 task 对象里，永远没人发现 ——
        # 上层以为连接还好着，会话一直挂着发 tick，
        # 用户看到「报错了但画面卡住不动」。
        # 实测这个坑让一次会话在服务端已死之后又空转了 24 分钟。
        self._recv_task.add_done_callback(self._on_recv_done)

    async def _configure(self) -> None:
        """发送 session.update，配置语音、提示词、VAD。"""
        cfg: dict[str, Any] = {
            # 必须同时含 text 和 audio
            "modalities": ["text", "audio"],
            "voice": self.voice,
            "instructions": self.instructions,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm24",
            "input_audio_transcription": {"model": "qwen3-asr-flash-realtime"},
            "turn_detection": {
                "type": "server_vad",
                "threshold": self.vad_threshold,
                "silence_duration_ms": self.silence_ms,
                "prefix_padding_ms": 300,
                "create_response": True,
                "interrupt_response": True,
            },
        }
        await self._send({"type": "session.update", "session": cfg})
        log.info("已发送 session.update（voice=%s）", self.voice)

    # ---------- 收发 ----------

    async def _send(self, obj: dict) -> None:
        if not self.ws or self.closed:
            raise LLMError("连接已关闭")
        await self.ws.send(json.dumps(obj, ensure_ascii=False))

    async def send_audio(self, pcm: bytes) -> None:
        """发送一块 16 kHz / int16 / 单声道 PCM。"""
        import base64
        await self._send({"type": "input_audio_buffer.append",
                          "audio": base64.b64encode(pcm).decode()})

    async def cancel_response(self) -> None:
        """打断 AI 当前回复（用户抢话时用）。"""
        await self._send({"type": "response.cancel"})

    async def request_response(self, instructions: Optional[str] = None) -> None:
        """
        主动要求模型开口。

        Realtime API 不会自己先说话 —— 必须显式触发，
        否则开场白永远不会出现，用户对着沉默发呆。
        """
        msg: dict[str, Any] = {"type": "response.create"}
        if instructions:
            msg["response"] = {"instructions": instructions}
        await self._send(msg)

    async def _recv_loop(self) -> None:
        try:
            async for raw in self.ws:
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._handle(ev)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # 致命错误要冒泡出去，让上层知道会话已经不可用。
            # 全部吞掉的话，连接死了上层还在傻等。
            if isinstance(e, SessionFatal):
                raise
            if not self.closed:
                log.warning("接收循环结束: %s", e)
                self._last_error = str(e)
        finally:
            self.ready = False

    def _on_recv_done(self, task: "asyncio.Task") -> None:
        """
        接收任务结束时把异常暴露出来。

        CancelledError 是我们自己 close() 时取消的，正常，不记。
        其它异常都要记下来并通过 _last_error 让上层能看到 ——
        否则连接死了而没人知道。
        """
        self.ready = False
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            if not self.closed:
                log.warning("接收循环已结束（连接被对端关闭）")
            return
        if isinstance(exc, SessionFatal):
            log.warning("接收循环因致命错误终止: %s", exc)
            self.fatal_code = str(exc)
        else:
            log.warning("接收循环异常终止: %r", exc)
        self._last_error = f"{type(exc).__name__}: {exc}"

    async def _handle(self, ev: dict) -> None:
        t = ev.get("type", "")

        if t == "error":
            err = ev.get("error") or {}
            self._last_error = f"{err.get('code')}: {err.get('message')}"
            log.warning("Qwen 返回错误: %s", self._last_error)

        # 判据收紧到「必须有 code」：真实的服务端错误帧一定带 code，
        # 而正常帧偶尔也会带无关的 message 字段 ——
        # 只凭 message 判断会把正常帧误报成错误。
        elif not t and ev.get("code"):
            # 服务端有些错误不带 type，只有 code/message 两个字段。
            # 典型：额度耗尽时回
            #   {"code": "AllocationQuota.FreeTierOnly", "message": "..."}
            # 然后直接掐断连接。
            #
            # 不识别这种情况的后果：接收循环只看到「连接意外关闭」，
            # 会话被当成正常结束，用户看到的是「莫名其妙就结束了」，
            # 而真正的原因（额度用完）完全没显示出来。
            code = str(ev.get("code") or "")
            msg = str(ev.get("message") or "")
            self._last_error = f"{code}: {msg}" if code else msg
            log.warning("Qwen 返回无 type 错误帧: %s", self._last_error)
            if self.on_event:
                try:
                    await self.on_event({
                        "type": "error",
                        "error": {"code": code, "message": msg},
                    })
                except SessionFatal:
                    # 致命信号必须透传，不能降级成 warning ——
                    # 吞掉的话接收循环不会终止，会话一直挂着，
                    # 用户看到的是「报错了但画面卡住不动」。
                    raise
                except Exception as e:
                    log.warning("错误事件处理失败: %s", e)
            return

        elif t == "response.done":
            resp = ev.get("response") or {}
            if resp.get("usage"):
                self.last_usage = resp["usage"]

        if self.on_event:
            try:
                await self.on_event(ev)
            except SessionFatal:
                # 致命信号必须透传。
                # 上面那三个分支都可能让 on_event 抛 SessionFatal
                # （比如识别出额度耗尽），被这里吞掉的话
                # 接收循环不会终止，会话就一直挂着不结束 ——
                # 用户那边表现为「报错了但画面卡住不动」。
                raise
            except Exception as e:
                log.warning("事件处理失败 (%s): %s", t, e)

    # ---------- 关闭 ----------

    async def close(self, timeout: float = 5.0) -> None:
        """
        关闭连接。

        注意：ws.close() 会挂住（实测 >8s 不返回）——
        它要等对端回 close 帧，而对端此时可能已经不理我们了。
        所以必须限时，超时就强制断开，否则整个会话收不了尾、
        报告发不出去、前端一直转圈。
        """
        self.closed = True

        if self._recv_task:
            self._recv_task.cancel()
            try:
                await asyncio.wait_for(self._recv_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
            self._recv_task = None

        if self.ws:
            try:
                await asyncio.wait_for(self.ws.close(), timeout=timeout)
            except (asyncio.TimeoutError, Exception):
                # 优雅关闭失败 → 直接掐断底层连接
                log.debug("ws.close() 超时，强制断开")
                transport = getattr(self.ws, "transport", None)
                if transport is not None:
                    try:
                        transport.abort()
                    except Exception:
                        pass
            self.ws = None

    async def __aenter__(self) -> "RealtimeSession":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()


# ============================================================
#  事件解析辅助
# ============================================================

def extract_audio_delta(ev: dict) -> Optional[bytes]:
    """response.audio.delta → 24 kHz PCM 字节。"""
    if ev.get("type") != "response.audio.delta":
        return None
    import base64
    d = ev.get("delta")
    if not d:
        return None
    try:
        return base64.b64decode(d)
    except Exception:
        return None


def extract_transcript(ev: dict) -> Optional[str]:
    """用户语音的转写完成。"""
    if ev.get("type") != "conversation.item.input_audio_transcription.completed":
        return None
    return (ev.get("transcript") or "").strip() or None


def extract_ai_text(ev: dict) -> Optional[str]:
    """AI 回复的文本完成。"""
    if ev.get("type") not in ("response.audio_transcript.done",
                              "response.text.done"):
        return None
    return (ev.get("transcript") or ev.get("text") or "").strip() or None


def is_speech_started(ev: dict) -> bool:
    return ev.get("type") == "input_audio_buffer.speech_started"


def is_response_done(ev: dict) -> bool:
    return ev.get("type") == "response.done"
