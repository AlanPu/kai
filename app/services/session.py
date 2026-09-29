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
from ..core.realtime import (RealtimeSession, SessionFatal, extract_ai_text,
                             is_recoverable_error,
                             extract_audio_delta, extract_transcript,
                             is_fatal_error, is_speech_started)
from ..core.voiceprint import SpeakerVerifier
from ..storage.db import Database
from ..storage.models import Correction, Turn
from .corrector import Corrector
from .profile import ProfileExtractor, ProfileStore

log = logging.getLogger(__name__)



SendToClient = Callable[[dict], Awaitable[None]]

# Qwen 每条会话最多 320 条 audio item，超出会报
# InvalidParameter: Too many audios 并断开连接。
# 实测：连续对话约 12.5 分钟撞到上限。
AUDIO_ITEM_LIMIT = 320
# 提前轮转的阈值。
#
# 我数的是 input_audio_buffer.committed，而服务端计的是全部
# audio item（还含 AI 的输出 item），两者不是一回事，
# 实测我数到的明显偏少（30 分钟实测：服务端已 320，我还没到 240）。
#
# 所以阈值取「远低于观测比例」的安全值。
# 提前轮转的代价只是 1~2 秒停顿，撞上限的代价是报错，
# 两者不对等 —— 宁可早转。
AUDIO_ITEM_ROTATE_AT = 150

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
    # 用户开始说话的次数。用来识别"每句都只说一次就没了"这类问题 ——
    # 如果它只涨到 1，说明用户说完第一句后再也发不出声音。
    user_speech_starts: int = 0
    # 已消耗的 audio item 数。Qwen 上限是每条会话 320 条，
    # 撞上会直接报 InvalidParameter 并断开。
    audio_items: int = 0


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
                 profile_store: Optional[ProfileStore] = None,
                 profile_extractor: Optional[ProfileExtractor] = None,
                 minutes: Optional[int] = None):
        self.s = settings
        self.db = db
        self.session_id = session_id
        self.instructions = instructions
        self.on_client = on_client
        self.verifier = voiceprint
        self.corrector = corrector
        self.profile_store = profile_store
        self.profile_extractor = profile_extractor

        # 攒够一定量的用户发言才做一次画像抽取（省钱、少打扰）
        self._user_since_extract: list[str] = []
        self._extract_threshold = 5

        self.stats = SessionStats()
        self.fatal_error: Optional[str] = None

        # 暂停：用户中途离开时不计时。
        # 起因：Qwen 在 300 秒无有效语音输入后会关闭会话，
        # 而一次 30 分钟的练习中途离开几分钟很正常。
        self.paused = False
        self.paused_total = 0.0
        self._paused_at: Optional[float] = None

        self._rotate_at = AUDIO_ITEM_ROTATE_AT
        self._rotating = False
        # 由外部循环轮询：轮转必须在 recv 循环之外执行
        self._want_rotate = False
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

        # 结束前把剩下的发言也抽一次画像
        try:
            await self.flush_profile()
        except Exception as e:
            log.warning("结束时画像冲刷失败: %s", e)

        # 用 elapsed()：暂停的时间不算练习时长
        elapsed = int(self.elapsed())
        usage = self.rt.last_usage if self.rt else None
        self.db.refresh_session_stats(self.session_id)
        self.db.finish_session(self.session_id, status=status,
                               duration_sec=elapsed, usage=usage)

    # ---------- 音频输入 ----------

    async def push_audio(self, pcm: bytes) -> None:
        """
        处理一块来自浏览器的 16 kHz PCM。

        暂停期间直接丢弃：用户离开了，麦克风收到什么与我们无关，
        送上去只会让 Qwen 的空闲计时被误重置、也让 AI 对空气说话。

        声纹开启时的三种情形：
          · 静音        → 攒入待定缓冲
          · 判定未知    → 攒入待定缓冲（不放行！）
          · 判定为本人  → 放行（含之前攒的）
          · 判定非本人  → 丢弃缓冲并通知前端
        """

        if self.paused or self.ended or not self.rt:
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
            err = ev.get("error") or {}
            code = str(err.get("code") or err.get("type") or "")
            await self.on_client({"type": "error", "error": err})

            # audio item 超限是可恢复的 —— 重建连接就能继续。
            # 必须先于 is_fatal_error 判断：这个错误的 code 是
            # InvalidParameter，而消息里才带 Too many audios。
            msg = str(err.get("message", "")) + str(code)
            if is_recoverable_error(msg):
                # 只置标志，真正的重连由外部循环执行。
                # 不能在这里直接关连接 —— 会掐断正在跑本回调的
                # _recv_loop，异常冒泡出去整个会话就被判异常结束。
                log.warning("audio item 超限（我数到 %d 条），请求轮转",
                            self.stats.audio_items)
                self._want_rotate = True
                return

            if is_fatal_error(code):
                # 服务端已经关掉会话了。必须如实结束，
                # 否则会继续假装正常运行 —— 实测见过报错后
                # 又空转 24 分钟，用户对着一个死连接说话。
                log.warning("会话致命错误，主动结束: %s", code)
                self.fatal_error = code
                # 抛出以终止 recv 循环，让上层走正常收尾
                raise SessionFatal(code)
            return

        audio = extract_audio_delta(ev)
        if audio:
            self.stats.audio_out_blocks += 1
            await self.on_client({"type": "audio", "pcm": _b64(audio)})
            return

        # 每个 input_audio_buffer.committed 都产生一条 audio item，
        # 累加用于在撞上 320 上限前主动轮转
        if t == "input_audio_buffer.committed":
            self.stats.audio_items += 1
            if (self.stats.audio_items >= self._rotate_at
                    and not self._rotating):
                # 同样只置标志，交给外部循环执行
                self._want_rotate = True
            return

        if is_speech_started(ev):
            # 用户开始说话 = 抢话。
            #
            # 这里曾经写成 _model_speaking = True，是个严重后果的逻辑错误：
            # push_audio 在 _model_speaking 为真时丢弃所有音频，
            # 而该标志只在 response.done 复位。
            # 于是用户说第一句后就再也不能说话 —— 音频全被丢掉，
            # 不产生 response，也就永远等不到 response.done，形成死锁。
            #
            # 正确语义：speech_started 意味着「轮到用户」，AI 要闭嘴。
            self._model_speaking = False
            self.stats.user_speech_starts += 1
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

        # 画像抽取：攒够若干句做一次，避免每句都调模型
        if self.profile_extractor is not None:
            self._user_since_extract.append(text)
            if len(self._user_since_extract) >= self._extract_threshold:
                batch, self._user_since_extract = self._user_since_extract, []
                task = asyncio.create_task(self._extract_profile(batch))
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

    async def _extract_profile(self, texts: list[str]) -> None:
        """从一批发言中抽取画像事实并累积（需求 6）。"""
        if not self.profile_extractor or not self.profile_store:
            return
        try:
            facts = await asyncio.to_thread(self.profile_extractor.extract, texts)
        except Exception as e:
            log.warning("画像抽取失败: %s", e)
            return
        if not facts or self.ended:
            return
        try:
            n = self.profile_store.absorb(facts, session_id=self.session_id)
            log.info("画像更新 %d 条: %s", n,
                     ", ".join(f"{f.key}={f.value}" for f in facts[:4]))
            await self.on_client({
                "type": "profile_learned",
                "facts": [{"category": f.category, "value": f.value}
                          for f in facts],
            })
        except Exception as e:
            log.warning("画像入库失败: %s", e)

    async def flush_profile(self) -> None:
        """会话结束时把剩余的发言也抽一次，别浪费。"""
        if self._user_since_extract:
            batch, self._user_since_extract = self._user_since_extract, []
            await self._extract_profile(batch)

    # ---------- 计时 ----------

    def wants_rotate(self) -> bool:
        """是否有待处理的轮转请求（由外部循环消费）。"""
        return self._want_rotate

    # ---------- 上下文轮转 ----------

    async def rotate_context(self) -> bool:
        """
        撞上 audio item 上限前重建连接，保留对话记忆。

        Qwen 每条会话最多 320 条 audio item（实测约 12.5 分钟到顶），
        超出会直接报错断开。这里在接近上限时主动重连，
        并把最近几轮对话写成摘要带进新的 instructions，
        用户感受是一句话的停顿，而不是突然掉线。

        返回是否轮转成功。
        """
        # 无论成败都先清标志，避免外部循环每 5 秒重试一次
        self._want_rotate = False
        if self.ended or self._rotating or not self.rt:
            return False
        self._rotating = True

        log.info("上下文轮转：已用 %d/%d audio item",
                 self.stats.audio_items, AUDIO_ITEM_LIMIT)

        # 用最近的对话生成承接语，让新一轮知道刚才聊到哪
        recent = self.db.list_turns(self.session_id)[-12:]
        if recent:
            lines = []
            for t in recent:
                who = "Student" if t.role == "user" else "You"
                lines.append(f"{who}: {t.text[:200]}")
            carry = ("\n\n## 刚才的对话（你要接着聊，不要重新自我介绍）\n"
                     + "\n".join(lines))
        else:
            carry = ""

        try:
            await self.rt.close()
        except Exception as e:
            log.warning("轮转时关闭旧连接失败: %s", e)

        try:
            self.rt = RealtimeSession(
                self.s,
                instructions=self.instructions + carry,
                voice=self.s.qwen_voice,
                on_event=self._on_qwen_event,
            )
            await self.rt.connect()
        except Exception as e:
            log.error("轮转重连失败: %s", e)
            self._rotating = False
            return False
        self._rotating = False

        self.stats.audio_items = 0
        self._rotate_at = AUDIO_ITEM_ROTATE_AT
        self._model_speaking = False
        if self.verifier:
            self.verifier.reset()

        # 新连接必须重新触发说话，否则轮转完就是一片沉默。
        # Realtime API 永远不会自己开口 —— 这一点在开场白那里
        # 已经踩过一次，重连后同理。
        try:
            await self.rt.request_response(
                "Continue the conversation naturally. Do NOT introduce "
                "yourself again and do NOT restart the topic. Pick up "
                "where you left off with one short reaction and one "
                "open question.")
        except Exception as e:
            log.warning("轮转后触发说话失败: %s", e)

        await self.on_client({
            "type": "context_rotated",
            "message": "对话记忆已整理，继续聊。",
        })
        return True

    # ---------- 暂停 ----------

    def pause(self) -> None:
        """用户中途离开。暂停期间不计时、不收音频。"""
        if self.paused or self.ended:
            return
        self.paused = True
        self._paused_at = time.time()

    def unpause(self) -> None:
        if not self.paused:
            return
        self.paused = False
        if self._paused_at:
            self.paused_total += time.time() - self._paused_at
        self._paused_at = None

    def elapsed(self) -> float:
        """已进行秒数，不含暂停时间。"""
        if not self.started_at:
            return 0.0
        e = time.time() - self.started_at - self.paused_total
        if self.paused and self._paused_at:
            e -= (time.time() - self._paused_at)
        return max(0.0, e)

    def remaining_sec(self) -> int:
        # 用 elapsed() 而非直接减 started_at —— 暂停的时间不该计时
        return max(0, int(self.duration_limit - self.elapsed()))

    def is_expired(self) -> bool:
        return self.remaining_sec() <= 0

    def snapshot(self) -> dict:
        return {
            "elapsed_sec": int(self.elapsed()),
            "paused": self.paused,
            "paused_sec": int(self.paused_total),
            "remaining_sec": self.remaining_sec(),
            "audio_in_blocks": self.stats.audio_in_blocks,
            "audio_out_blocks": self.stats.audio_out_blocks,
            "blocked_chunks": self.stats.blocked_chunks,
            "user_speech_starts": self.stats.user_speech_starts,
            "audio_items": self.stats.audio_items,
            "blocked_sec": round(self.stats.blocked_sec, 1),
            "user_chars": self.stats.user_chars,
            "ai_chars": self.stats.ai_chars,
        }


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()
