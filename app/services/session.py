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
                             is_fatal_error, is_speech_started,
                             is_speech_stopped, is_turn_race_error)
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
# 我数的是 input_audio_buffer.committed，服务端计的是全部
# audio item（还含 AI 的输出 item、以及 VAD 切碎的内部片段）。
#
# 30 分钟实测标定：服务端报「已到 320」时，我这边只数到 91 条 ——
# 比例约 3.5:1。按这个比例，阈值必须远低于 91，
# 否则等服务端到顶时我还差得远（这正是 150 那次失败的原因）。
#
# 取 60：约在服务端 210 条时轮转，留 110 条余量。
# 提前轮转的代价只是 1~2 秒停顿，撞上限的代价是断线，
# 两者不对等 —— 宁可早转。
AUDIO_ITEM_ROTATE_AT = 60

# 静音门槛（int16 量级）：低于此值不送声纹判定，但仍积累在待定缓冲
VOICE_RMS_FLOOR = 120
# 浏览器 AudioWorklet 每块 128 采样 @16kHz = 8ms。
# 用于把"块数"换算成秒数来统计丢弃时长。
CHUNK_SEC = 128 / 16000
# 待定缓冲上限（约 4 秒）。只用于静音段的裁剪：
# 判不出来的音频现在会立即放行，不再靠这个阈值兜底。
# 按 8ms/块算，500 块 ≈ 4 秒。
MAX_PENDING = 500


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
                 session_id: int, user_id: int, instructions: str,
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
        # 画像读写必须知道是谁的 —— 多人共用时写错用户
        # 会把甲的偏好记到乙头上，且很难发现
        self.user_id = user_id
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

        # 请求 AI 主动找话题：卡壳时用户按 C 键，AI 换个角度提问。
        #
        # 为什么需要：改成「按住空格说话」之后，回合完全由用户发起 ——
        # 用户不开口，AI 就永远不会说话。安静本身没错，但用户
        # 卡住想不出句子时会一直干等，练习就停在那里了。
        #
        # 为什么不再是「冷场 15 秒自动接话」：那样 AI 会在用户
        # 正组织句子的时候突然插话，把思考时间抢走；而且用户
        # 根本没法预判它什么时候开口。改成按键触发之后，
        # 什么时候需要帮忙完全由用户决定。
        #
        # 每轮上限由配置决定（.env 里的 IDLE_NUDGE_MAX），
        # 代码不替用户拍板。
        self._last_activity = time.time()
        self._nudges_sent = 0
        self._max_nudges = getattr(self.s, "idle_nudge_max", 6)

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
        # 用户当前是否正在说话（由服务端 VAD 的 started/stopped 维护）
        self._user_speaking = False
        # 是否已经请求过回应但还没等到 response.created。
        # 用来去重：VAD 对同一句话会重复报 speech_stopped，
        # 重复发 response.create 会被服务端拒绝
        # （"Conversation already has an active response"）。
        #
        # ⚠️ 这个标志必须能超时复位。
        # 实测踩过的坑：发了 response.create 但服务端**根本没产生
        # response**（半句话、音频太短、空缓冲都会这样），于是
        # response.done 永远不来，"等待中"就一直挂着 —— 之后所有回合
        # 都被当成重复而静默丢弃，表现为用户说什么 AI 都不理，
        # 冷场救场也一起失效。所以记录发起时刻，超时即作废。
        self._response_pending = False
        self._response_pending_at = 0.0
        # 本轮（本次按住空格）已经送出去的音频块数。
        # end_turn 靠它判断"要不要 commit"：发过才提交，
        # 没发过就不提交（空提交会被服务端拒绝）。
        self._turn_audio_blocks = 0
        # 请求回应后等多久算"服务端根本没理我"。
        # 正常 response.created 在 1 秒内就回；给到 12 秒是留足
        # 网络抖动和模型首字延迟的余量，又能保证卡死能自愈。
        self.response_pending_timeout = 12.0
        # 后台小任务的引用集合（触发接话等），避免被 GC 提前回收
        self._bg_tasks: set[asyncio.Task] = set()
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

        # 关连接本身可能失败或卡住（Qwen 不一定会回 close 帧）。
        # 这里必须兜住：一旦抛出去，后面的报告就发不出去，
        # 前端会永远停在「正在生成报告…」转圈。
        if self.rt:
            try:
                await self.rt.close()
            except Exception as e:
                log.warning("关闭语音连接失败: %s", e)

        # 结束前把剩下的发言也抽一次画像
        try:
            await self.flush_profile()
        except Exception as e:
            log.warning("结束时画像冲刷失败: %s", e)

        # 用 elapsed()：暂停的时间不算练习时长
        elapsed = int(self.elapsed())
        usage = self.rt.last_usage if self.rt else None
        # 同样兜住：落库失败不该挡住用户看报告（前面已经存了逐条发言，
        # 丢的只是汇总统计）。
        try:
            self.db.refresh_session_stats(self.session_id)
            self.db.finish_session(self.session_id, status=status,
                                   duration_sec=elapsed, usage=usage)
        except Exception as e:
            log.error("会话收尾落库失败: %s", e)

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

        # ⚠️ 注意：这里**不能**无条件 touch()。
        #
        # 踩过的坑：麦克风上行是连续的，即使按住空格模式，松手前后
        # 也会有零散的静音块到达。如果每块都算"用户有动静"，
        # 冷场计时就永远被重置，AI 永远等不到"冷场"
        # ——用户反馈"我停下 20 秒，AI 并没有接上"。
        #
        # 所以只有**检测到真实人声**才清零计时。静音和极低音量的
        # 块只当背景，不算用户说话。

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

            # 确实有人声 → 才算"用户有动静"，冷场计时清零
            self.touch()

            verdict = self.verifier.feed(arr.astype(np.float32))

            if verdict is None:
                # ⚠️ 判定还没出来 —— 这是最常见的状态，不是异常。
                #
                # 声纹要攒满 window_sec（2 秒）才能算一次，所以每句话的
                # 前 2 秒必然落在"未知"里。此处曾经把音频压着不发（攒够
                # MAX_PENDING 才兜底），后果很严重：
                #
                #   · 用户开口后最多 2 秒，Qwen 完全收不到声音
                #   · 攒够块数后一次性成批补发，音频变成一段段带空洞的
                #     碎片送给 Qwen
                #   · Qwen 的 VAD 把每个碎片末尾当成"说完了" → 生成回应
                #   · 下一个碎片又被当成"又开始说" → 打断刚生成的回应
                #
                # 用户看到的就是：「声纹无法识别」+ AI 半天不吭声 +
                # 突然回应上一句并把我打断。全都来自这一处。
                #
                # 顺带一提，原来的兜底阈值也算错了：MAX_PENDING=100 块
                # × 8ms(浏览器块大小) = 0.8 秒 < 2 秒声纹窗口，所以
                # "攒够就放行"永远发生在 verdict 仍为 None 的时候。
                #
                # 现在改成：判不出来就直接放行。声纹只用来否决
                # （明确判定为他人时才丢），而不是用来放行 ——
                # 宁可放过，也不要让本人说的话卡住。
                await self._flush_pending()
                await self._send_audio(pcm)
                return

            if verdict is False:
                # 不是本人 → 丢弃全部待定音频
                dropped = len(self._pending) * CHUNK_SEC + CHUNK_SEC
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

        else:
            # 未开启声纹：没有 verifier 帮我们判断人声，
            # 只能自己按音量判断 —— 否则静音块也会一直重置
            # 冷场计时，AI 永远等不到"冷场"。
            arr = np.frombuffer(pcm, dtype=np.int16)
            if arr.size and rms(arr) <= VOICE_RMS_FLOOR:
                return                      # 纯静音，不上行也不计时
            self.touch()

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
        # 用 getattr 兜底：部分测试用 __new__ 手工构造会话对象、不走
        # __init__，属性可能不存在。
        self._turn_audio_blocks = getattr(self, "_turn_audio_blocks", 0) + 1

    # ---------- 事件处理 ----------

    async def _on_qwen_event(self, ev: dict) -> None:
        t = ev.get("type", "")
        if t == "error":
            err = ev.get("error") or {}
            code = str(err.get("code") or err.get("type") or "")
            msg = str(err.get("message", "")) + str(code)

            # 先分类，再决定要不要打扰用户。
            # 原来是一进来就 on_client({"type":"error"})，
            # 结果连"回合竞态"这种可自愈的小冲突也会弹给用户看，
            # 让人以为练习出了问题。
            if is_turn_race_error(msg):
                log.info("回合竞态（可自愈，已忽略）: %s",
                         str(err.get("message", ""))[:80])
                return

            await self.on_client({"type": "error", "error": err})

            # audio item 超限是可恢复的 —— 重建连接就能继续。
            # 必须先于 is_fatal_error 判断：这个错误的 code 是
            # InvalidParameter，而消息里才带 Too many audios。
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
            self._user_speaking = True
            self.touch()
            # 用户又开口了 —— 取消「等待回应」的挂起状态。
            # 否则上一句的 pending 会一直挂着，这一句说完时被误判为重复
            # 而不再触发接话（表现为"说了新的一句但 AI 没反应"）。
            self._clear_response_pending()
            self.stats.user_speech_starts += 1
            await self.on_client({"type": "user_speaking"})
            return

        if is_speech_stopped(ev):
            # 用户说完一句 —— 该 AI 接话了。
            #
            # 为什么必须显式触发，而不能靠 turn_detection 的
            # create_response=True：
            #
            # 实测（Qwen cn-beijing，qwen3.8-omni-flash-realtime）：
            # 即使 create_response=True，服务端也只在 speech_stopped 时
            # 发 committed，并不会自己生成回复。结果是用户说完一句、
            # 停在那儿等，AI 一直不吭声 —— 必须再说一句"继续"才动。
            # 这正是用户反馈的"不像真人对话"。
            #
            # ⚠️ 这里有两种都会踩坑的写法，都实测过：
            #
            #   1) 手动 input_audio_buffer.commit
            #      服务端在 speech_stopped 时**已经自己 commit 过了**，
            #      我们再 commit 一次就是空提交，服务端回：
            #        "buffer too small, or have no audio"
            #
            #   2) 只发 response.create
            #      实测 AI 不会回应（音频 item 落库尚需时间）
            #
            # 现在改用「按住空格说话」：断句由用户的松手动作决定
            # （见 end_turn）。所以这里**不再**自动触发接话 ——
            # VAD 的 speech_stopped 只当作参考信息（用于前端指示），
            # 否则它又会抢在用户真正说完之前把半句话送出去。
            self._user_speaking = False
            return

        if t == "response.created":
            self._model_speaking = True
            self.touch()
            self._clear_response_pending()      # 回应已开始，允许下一次触发
            return

        if t == "response.done":
            self._model_speaking = False
            self._clear_response_pending()
            # AI 刚说完，从这一刻开始算冷场：用户若一直不开口，
            # 20 秒后由 _timer_loop 触发主动找话题。
            self.touch()
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

    async def end_turn(self) -> None:
        """用户明确表示「这一轮说完了」（松开空格）。

        和 VAD 自动断句的区别：这是用户主动给的信号，不需要猜。

        为什么要先 flush：声纹判定未完成时音频会短暂留在 _pending，
        用户松手时缓冲里可能还有内容。不 flush 就会丢掉句尾几个字。
        """
        if self.ended or not self.rt:
            return
        await self._flush_pending()

        # 显式提交这一轮的音频。
        #
        # 必须做：前端只在按住空格时上行音频，服务端 VAD 的自动
        # commit 变得不可靠 —— 实测送上去 3 秒音频（30 块）却始终
        # 不出现转写，紧接着 response.create 被静默吞掉，
        # 用户感受就是"我说完 AI 不理我"。
        #
        # 只在**确实发过音频**时提交：空提交会让服务端回
        # "buffer too small, or have no audio" —— 这正是当初删掉
        # commit_audio 的原因，不能重蹈覆辙。
        #
        # 判据是"本轮开始后有没有送过音频"（_turn_audio_blocks 在
        # push_audio 里累加），而不是对比 flush 前后的计数 ——
        # 声纹关闭时音频在 push_audio 阶段就已经直接送出去了，
        # flush 阶段一块都没有，用差值判断会永远得 0（踩过）。
        if getattr(self, "_turn_audio_blocks", 0) > 0:
            try:
                await self.rt.commit_audio()
            except Exception as e:               # noqa: BLE001
                log.warning("提交音频失败: %s", e)
        else:
            # 这一轮一个字节都没上来（按了空格但没说话）。
            # 直接接话会让 AI 对着空气回应，所以跳过。
            log.info("本轮没有音频，跳过接话")
            return

        self._turn_audio_blocks = 0
        self._maybe_respond()

    def _maybe_respond(self) -> None:
        """触发 AI 接话，并保证一轮只触发一次。

        去重理由：VAD 可能对同一句话重复报 speech_stopped，
        每次都发 response.create 会撞上
        "Conversation already has an active response"。

        另外**不要**手动 commit：服务端在 speech_stopped 时已经自己
        commit 过了，我们再 commit 是空提交，会报 "buffer too small"。
        """
        if self.response_pending():
            return                       # 已经在等回应了，忽略重复触发
        if self.ended or not self.rt:
            return
        self._mark_response_pending()
        task = asyncio.create_task(self._respond_after_turn())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _respond_after_turn(self) -> None:
        """让 AI 接话。只发 response.create，不碰音频缓冲。

        延迟是必要的：speech_stopped 之后服务端还要把音频 item 落库，
        太早 create 会被忽略（现象就是 AI 不吭声）。0.5 秒实测够用。
        """
        try:
            await asyncio.sleep(0.5)
            if self._model_speaking or self.ended or not self.rt:
                return
            await self.rt.request_response()
        except Exception as e:                   # noqa: BLE001
            # 接话失败不该让整场练习崩掉：用户还能自己继续说
            log.warning("触发 AI 接话失败: %s", e)

    async def _on_user_text(self, text: str) -> None:
        """用户说完一句：入库、回推、异步纠错。"""
        self.touch()
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
            n = self.profile_store.absorb(self.user_id, facts,
                                         session_id=self.session_id)
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

    def is_connection_lost(self) -> bool:
        """
        语音连接是不是已经死了。

        判据：接收任务已结束、且我们没在主动关闭。
        不看 rt.ready —— 那个标志在 close() 里也会被清掉，
        用它判断会把「正常收尾」误判成「异常掉线」。

        为什么需要这个：接收任务抛出的 SessionFatal 没人 await，
        异常只存在 task 对象里。没有这层看护，连接死了上层
        完全不知道，会话一直挂着发 tick。
        """
        if self.ended or not self.rt:
            return False
        task = getattr(self.rt, "_recv_task", None)
        if task is None:
            return False
        if not task.done() or task.cancelled():
            return False
        # 主动关闭时 closed 已置位，属于正常收尾
        return not getattr(self.rt, "closed", False)

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

    # ---------- 回合状态 ----------

    def response_pending(self) -> bool:
        """是否还在等上一次回应的 response.created（带超时作废）。

        超时后会把标志清掉，这样卡死的回合能自愈 ——
        否则用户后面说什么都会被当成重复而丢弃。
        """
        if not self._response_pending:
            return False
        timeout = getattr(self, "response_pending_timeout", 12.0)
        since = getattr(self, "_response_pending_at", 0.0)
        # since 为 0 说明标志是外部直接置上的（没有时间戳），
        # 这种情况按"刚刚发起"处理，不能当成已超时 ——
        # 否则会把正在等待的回合误判成卡死。
        if since and time.time() - since > timeout:
            log.warning("等待 AI 回应超时（%.0f 秒无响应），作废并允许下一轮",
                        timeout)
            self._response_pending = False
            return False
        return True

    def _mark_response_pending(self) -> None:
        self._response_pending = True
        self._response_pending_at = time.time()

    def _clear_response_pending(self) -> None:
        self._response_pending = False
        self._response_pending_at = 0.0

    # ---------- 按 C 键让 AI 找话题 ----------

    def touch(self) -> None:
        """记录"有事发生"。

        在收到用户音频、用户转写、AI 开始/结束说话时都要调用。
        """
        self._last_activity = time.time()

    def idle_seconds(self) -> float:
        """距离上一次"有事发生"过了多久（暂停期间不计）。"""
        if self.paused:
            return 0.0
        return time.time() - self._last_activity

    def maybe_prompt(self) -> Optional[str]:
        """判断此刻能不能让 AI 主动找话题。

        返回要发给模型的提示语；不该开口时返回 None。

        这里不判断"冷场够不够久" —— 触不触发完全由用户按 C 键决定。
        守卫只拦那些"开口就会出问题"的情况：
        AI 正在说话或已经在等回应时插嘴，会变成自己跟自己抢话。

        为什么用 instructions 而不是让 AI 自由发挥：
        直接在 response.create 里带 instructions，可以指定
        "换个角度追问、不要把话题聊死"，否则模型容易重复上一句，
        或者说出"你还在吗？"这种扫兴的话。
        """
        if self.ended or self.paused or not self.rt:
            return None
        if self._model_speaking or self.response_pending():
            return None                      # AI 正在说话/准备说话，别插嘴
        if self._nudges_sent >= self._max_nudges:
            return None                      # 这一轮问够了，剩下的交给用户

        self._nudges_sent += 1
        self.touch()
        return self._nudge_instruction()

    def _nudge_instruction(self) -> str:
        """请求 AI 找话题时给模型的指令。

        第 1 次温和地换个角度追问；第 2 次起主动引入新话题，
        避免在同一个点上反复打转（那会让冷场更尴尬）。
        """
        common = ("The user asked you to help them keep going because they "
                  "are stuck. Do NOT ask whether they are still there, and do "
                  "not mention the silence or apologize. ")
        if self._nudges_sent <= 1:
            return (common +
                    "Gently continue the conversation: react to what they "
                    "said last, then ask ONE easy follow-up question from a "
                    "slightly different angle. Keep it to one short sentence.")
        return (common +
                "Start a fresh but related topic that fits what you know "
                "about them, and invite them to speak with ONE open question. "
                "Keep it to one short sentence.")

    async def prompt(self) -> bool:
        """用户按 C 键：让 AI 主动开口。返回是否真的开口了。

        由前端的 prompt 命令调用。回合完全由用户发起
        （按住空格说话），所以卡壳时得有个求援的入口。
        """
        idle = self.idle_seconds()
        instr = self.maybe_prompt()
        if not instr:
            return False
        try:
            self._mark_response_pending()
            await self.rt.request_response(instructions=instr)
            await self.on_client({"type": "ai_prompted", "idle_sec": int(idle)})
            log.info("用户按 C 键，AI 主动找话题（第 %d 次，已静默 %.0f 秒）",
                     self._nudges_sent, idle)
            return True
        except Exception as e:                   # noqa: BLE001
            self._clear_response_pending()
            log.warning("主动找话题失败: %s", e)
            return False

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
