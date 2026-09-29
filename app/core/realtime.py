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
    return any(f in c for f in FATAL_CODES)

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

    async def _handle(self, ev: dict) -> None:
        t = ev.get("type", "")

        if t == "error":
            err = ev.get("error") or {}
            self._last_error = f"{err.get('code')}: {err.get('message')}"
            log.warning("Qwen 返回错误: %s", self._last_error)

        elif t == "response.done":
            resp = ev.get("response") or {}
            if resp.get("usage"):
                self.last_usage = resp["usage"]

        if self.on_event:
            try:
                await self.on_event(ev)
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
