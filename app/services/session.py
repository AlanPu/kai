"""
会话编排：把 Realtime、声纹过滤、纠错、持久化串起来。

这是整个应用的核心 —— 一条音频从浏览器进来，
要经过：声纹判定 → 送模型 → 转写入库 → 纠错 → 回推前端。
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import numpy as np

from ..core.audio import rms
from ..core.config import Settings
from ..core.realtime import (RealtimeSession, extract_ai_text,
                             extract_audio_delta, extract_transcript,
                             is_speech_started)
from ..core.voiceprint import SpeakerVerifier
from ..storage.db import Database
from ..storage.models import Correction, Turn
from .corrector import Corrector

log = logging.getLogger(__name__)

SendToClient = Callable[[dict], Awaitable[None]]

# 静音门槛（int16 量级）：低于此值不送声纹判定，但仍积累在待定缓冲
VOICE_RMS_FLOOR = 120
# 待定缓冲上限（约 4 秒）：防止声纹长时间判不出来导致对话哑掉
MAX_PENDING = 100


@dataclass
class SessionStats:
    audio_in_blocks: int = 0
    audio_out_blocks: int = 0
    blocked_chunks: int = 0
    blocked_sec: float = 0.0
    user_chars: int = 0
    ai_chars: int = 0


class ConversationSession:
    """
    一次完整的练习会话。

    生命周期：start() → push_audio() … → stop()
    """

    def __init__(self, settings: Settings, db: Database, *,
                 session_id: int, instructions: str,
                 on_client: SendToClient,
                 voiceprint: Optional[SpeakerVerifier] = None,
                 verifier_threshold: float = 0.5,
                 corrector: Optional[Corrector] = None,
                 minutes: Optional[int] = None):
        self.s = settings
        self.db = db
        self.session_id = session_id
        self.instructions = instructions
        self.on_client = on_client
        self.verifier = voiceprint
        self.corrector = corrector

        self.stats = SessionStats()
        self.started_at = time.time()
        self.duration_limit = (minutes or settings.session_minutes) * 60

        self.rt: Optional[RealtimeSession] = None
        self.ended = False

        # 声纹待定缓冲：判定未知期间的音频先攒着，判定后才决定放行/丢弃。
        # 若"未知就放行"，旁人一句开场白就能触发模型（真实踩过的坑）。
        self._pending: list[bytes] = []
        # AI 正在说话时暂停声纹判定（否则会把 AI 自己的声音当输入）
        self._model_speaking = False
        self._user_buf: list[str] = []
        self._correction_tasks: set[asyncio.Task] = set()

    # ---------- 生命周期 ----------

    async def start(self) -> None:
        self.rt = RealtimeSession(self.s, instructions=self.instructions,
                                  on_event=self._on_qwen_event)
        await self.rt.connect()

        # 让 AI 主动开场。Realtime API 自己不会先说话，
        # 不显式触发的话用户会对着沉默等待。
        try:
            await self.rt.request_response(
                "Greet the student warmly in one or two sentences, "
                "briefly introduce today's topic, then ask them one open "
                "question to get them talking. Keep it short.")
        except Exception as e:
            log.warning("开场触发失败: %s", e)

        await self.on_client({
            "type": "ready",
            "model": self.s.qwen_model,
            "voice": self.s.qwen_voice,
            "voiceprint": self.verifier is not None,
        })

    async def stop(self, status: str = "finished") -> None:
        if self.ended:
            return
        self.ended = True

        for t in list(self._correction_tasks):
            t.cancel()

        if self.rt:
            await self.rt.close()

        elapsed = int(time.time() - self.started_at)
        usage = self.rt.last_usage if self.rt else None
        self.db.refresh_session_stats(self.session_id)
        self.db.finish_session(self.session_id, status=status,
                               duration_sec=elapsed, usage=usage)

    # ---------- 音频输入 ----------

    async def push_audio(self, pcm: bytes) -> None:
        """
        处理一块来自浏览器的 16 kHz PCM。

        声纹开启时的三种情形：
          · 静音        → 攒入待定缓冲
          · 判定未知    → 攒入待定缓冲（不放行！）
          · 判定为本人  → 放行（含之前攒的）
          · 判定非本人  → 丢弃缓冲并通知前端
        """
        if self.ended or not self.rt:
            return

        # AI 说话期间不处理输入，避免自我对话
        if self._model_speaking:
            return

        if self.verifier is not None:
            arr = np.frombuffer(pcm, dtype=np.int16)
            if arr.size == 0:
                return

            if rms(arr) <= VOICE_RMS_FLOOR:
                # 静音：不能清空声纹缓冲！浏览器降噪会把句间停顿压得很低，
                # 一旦清空就永远攒不满 1 秒（真实踩过的坑）。
                self._pending.append(pcm)
                self._trim_pending()
                return

            verdict = self.verifier.feed(arr.astype(np.float32))

            if verdict is None:
                self._pending.append(pcm)
                if len(self._pending) >= MAX_PENDING:
                    # 兜底：长时间判不出来就放行，否则对话会彻底哑掉
                    await self._flush_pending()
                    await self.on_client({
                        "type": "voiceprint_stuck",
                        "sec": round(len(self._pending) * 0.02, 1)})
                else:
                    self._trim_pending()
                return

            if verdict is False:
                # 不是本人 → 丢弃全部待定音频
                dropped = len(self._pending) * 0.02 + 0.02
                self._pending.clear()
                self.stats.blocked_chunks += 1
                self.stats.blocked_sec += dropped
                await self.on_client({
                    "type": "voiceprint_skip",
                    "score": round(self.verifier.last_score or 0.0, 3),
                    "blocked_sec": round(self.stats.blocked_sec, 1),
                })
                return

            # 是本人 → 连同待定的音频一起送
            await self._flush_pending()

        await self._send_audio(pcm)

    async def _flush_pending(self) -> None:
        pending, self._pending = self._pending, []
        for chunk in pending:
            await self._send_audio(chunk)

    def _trim_pending(self) -> None:
        while len(self._pending) > MAX_PENDING:
            self._pending.pop(0)

    async def _send_audio(self, pcm: bytes) -> None:
        assert self.rt
        await self.rt.send_audio(pcm)
        self.stats.audio_in_blocks += 1

    # ---------- 事件处理 ----------

    async def _on_qwen_event(self, ev: dict) -> None:
        t = ev.get("type", "")

        if t == "error":
            await self.on_client({"type": "error", "error": ev.get("error")})
            return

        audio = extract_audio_delta(ev)
        if audio:
            self.stats.audio_out_blocks += 1
            await self.on_client({"type": "audio", "pcm": _b64(audio)})
            return

        if is_speech_started(ev):
            # 用户抢话：AI 应停止说话
            self._model_speaking = True
            await self.on_client({"type": "user_speaking"})
            return

        if t == "response.created":
            self._model_speaking = True
            return

        if t == "response.done":
            self._model_speaking = False
            # AI 说完后清空声纹缓冲，避免把自己的尾音算进判定
            if self.verifier:
                self.verifier.reset()
            return

        text = extract_transcript(ev)
        if text:
            await self._on_user_text(text)
            return

        ai = extract_ai_text(ev)
        if ai:
            await self._on_ai_text(ai)
            return

    async def _on_user_text(self, text: str) -> None:
        """用户说完一句：入库、回推、异步纠错。"""
        self.stats.user_chars += len(text)
        turn_id = self.db.add_turn(Turn(session_id=self.session_id,
                                        role="user", text=text))
        await self.on_client({"type": "user_text", "text": text})

        # 纠错异步进行，不阻塞对话
        if self.corrector is not None:
            task = asyncio.create_task(self._correct(turn_id, text))
            self._correction_tasks.add(task)
            task.add_done_callback(self._correction_tasks.discard)

    async def _on_ai_text(self, text: str) -> None:
        self.stats.ai_chars += len(text)
        self.db.add_turn(Turn(session_id=self.session_id,
                              role="assistant", text=text))
        await self.on_client({"type": "ai_text", "text": text})

    async def _correct(self, turn_id: int, text: str) -> None:
        """调用纠错模型，把结果入库并推送前端。"""
        try:
            items = await asyncio.to_thread(self.corrector.check, text)
        except Exception as e:
            log.warning("纠错失败: %s", e)
            return
        if not items or self.ended:
            return

        for it in items:
            c = Correction(session_id=self.session_id, turn_id=turn_id,
                           kind=it.kind, severity=it.severity,
                           original=it.original, suggestion=it.suggestion,
                           explanation=it.explanation, word=it.word,
                           phonetic=it.phonetic)
            cid = self.db.add_correction(c)
            # 只有该展示的才推给前端（发音 minor 被过滤，见 should_show）
            if c.should_show():
                c.id = cid
                await self.on_client({
                    "type": "correction",
                    "kind": c.kind, "severity": c.severity,
                    "original": c.original, "suggestion": c.suggestion,
                    "explanation": c.explanation,
                    "word": c.word, "phonetic": c.phonetic,
                })

    # ---------- 计时 ----------

    def remaining_sec(self) -> int:
        return max(0, int(self.duration_limit - (time.time() - self.started_at)))

    def is_expired(self) -> bool:
        return self.remaining_sec() <= 0

    def snapshot(self) -> dict:
        return {
            "elapsed_sec": int(time.time() - self.started_at),
            "remaining_sec": self.remaining_sec(),
            "audio_in_blocks": self.stats.audio_in_blocks,
            "audio_out_blocks": self.stats.audio_out_blocks,
            "blocked_chunks": self.stats.blocked_chunks,
            "blocked_sec": round(self.stats.blocked_sec, 1),
            "user_chars": self.stats.user_chars,
            "ai_chars": self.stats.ai_chars,
        }


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()
