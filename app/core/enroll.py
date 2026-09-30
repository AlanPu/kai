"""
声纹录入：引导用户跟读数句，从中提取一份可靠的声纹。

为什么不是"边聊边采集"：
    聊天时声音状态变化大（远近、情绪、被打断），混进来的旁人声音
    也分不清，采到的原型会"四不像"，反而更容易误判。
    跟读几句话只有几十秒，但质量可控得多。

为什么每句都要校验：
    录到一半麦克风没声音、或者用户没跟上，都会产出一个坏样本。
    等录完再发现就白录了，所以每句当场判定，不合格就重来。

判定思路（全部基于能量和向量相似度，不依赖模型之外的假设）：
    · 太安静            → 没收到声音，重录
    · 太短              → 不够一秒，模型算不准，重录
    · 和你已有的样本差太远 → 可能换了位置/设备，或混进别人声音，重录
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .audio import IN_RATE, rms
from .voiceprint import SILENCE_RMS

log = logging.getLogger(__name__)

# 每句至少要这么多有效语音，才够模型算出一个稳定的向量。
#
# 实测（Cherry vs Ethan/Chelsie 两个不同音色，各取多段算余弦）：
#   窗口   同人均值  异人均值  间隔
#   0.5s    0.477    0.322   0.156
#   1.0s    0.582    0.372   0.210
#   2.0s    0.672    0.422   0.249
#   3.0s    0.745    0.458   0.287
# 1 秒时同人分数低到 0.36、异人高到 0.51，分布重叠严重，
# 任何阈值都会要么频繁误拒本人、要么频繁放进旁人。
# 2 秒以上才把两类分开。所以这里取 2 秒，不是 1 秒。
MIN_SPEECH_SEC = 2.0
# 超过这么长就截断：再长也不会更准，只是浪费用户时间
MAX_SPEECH_SEC = 12.0

# 低于此能量认为没说话（沿用声纹模块的门槛，保持一致）
SPEECH_RMS_FLOOR = SILENCE_RMS

# 和已有样本的平均相似度低于此值 → 怀疑换人了或换了环境，拒绝这句。
#
# 实测（CAM++ / 2 秒窗口）：
#   同一说话人内部两两：最低 0.485，均值 0.644
#   不同说话人两两：    最高 0.360，均值 0.200
# 两类之间有一条真空带（0.36 ~ 0.49），阈值取在带内最稳。
# 之前写 0.45 也能用，但对本人偏严 —— 本人最低 0.485 离它只差一点，
# 环境稍有波动就会误拒真人。取 0.42 留出余量，仍远高于异人最高分。
MIN_COHERENCE = 0.42

# 至少要有这么多句才允许保存。3 句是最低要求 ——
# 少于 3 句就没法做多数表决，单次录音的噪声会直接进原型。
MIN_SAMPLES = 3


@dataclass
class Prompt:
    """一句跟读文本。"""

    text: str
    hint: str = ""


# 选句原则：
#   · 覆盖足够多的音素（英语的全部元音、常见辅音）
#   · 长度 3 秒左右，太长会累
#   · 内容中性，不含隐私
# 这组句子覆盖了 /iː/ /æ/ /ɑː/ /uː/ /əʊ/ /aɪ/ 等主要元音
# 和 /θ/ /ð/ /r/ /l/ /v/ /w/ 等中文里没有或易混的辅音。
DEFAULT_PROMPTS_LIST: list[Prompt] = [
    Prompt("The weather is really nice today, isn't it?",
           "注意 weather 的 /ð/ 和 really 的 /r/"),
    Prompt("I usually walk to work in the morning.",
           "注意 usually 的 /ʒ/ 和 walk 的 /w/"),
    Prompt("Could you please show me the way to the station?",
           "注意 could 的 /ʊ/ 和 station 的 /ʃ/"),
    Prompt("I think this book is more interesting than that one.",
           "注意 think 的 /θ/ 和 interesting 的重音"),
    Prompt("We should probably leave before it gets too late.",
           "注意 probably 的连读和 late 的 /eɪ/"),
]


@dataclass
class SampleResult:
    """一句录音的校验结果。"""

    ok: bool
    reason: str = ""                 # 不合格的原因（给用户看）
    embedding: Optional[np.ndarray] = None
    speech_sec: float = 0.0
    score: Optional[float] = None    # 与已有样本的相似度


@dataclass
class EnrollmentSession:
    """
    一次录入过程的状态。

    设计成有状态对象，是因为录入要跨多句、跨多个 WebSocket 消息：
    每句录完校验一次，通过了才进入下一句。状态放在这里，
    WebSocket 那层只管转发音频和回消息。
    """

    user_id: int
    verifier: "object"               # SpeakerVerifier，用于抽向量
    prompts: list[Prompt] = field(default_factory=list)
    current: int = 0                 # 正在录第几句（从 0 开始）
    embeddings: list[np.ndarray] = field(default_factory=list)
    buf: list[np.ndarray] = field(default_factory=list)

    # 已收集样本的质量，随样本增多而更新
    quality: Optional[float] = None

    finished: bool = False
    error: str = ""

    def __post_init__(self):
        if not self.prompts:
            self.prompts = list(DEFAULT_PROMPTS_LIST)

    # ---------- 进度 ----------

    @property
    def total(self) -> int:
        return len(self.prompts)

    @property
    def prompt(self) -> Optional[Prompt]:
        if self.current >= self.total:
            return None
        return self.prompts[self.current]

    def progress(self) -> dict:
        """给前端渲染进度条的状态。"""
        return {
            "current": self.current + 1 if not self.finished else self.total,
            "total": self.total,
            "collected": len(self.embeddings),
            "needed": MIN_SAMPLES,
            "prompt": self.prompt.text if self.prompt else "",
            "hint": self.prompt.hint if self.prompt else "",
            "quality": self.quality,
            "finished": self.finished,
        }

    # ---------- 音频 ----------

    def feed(self, samples) -> None:
        """累积音频块（int16 或 float32）。"""
        x = np.asarray(samples)
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if x.size:
            self.buf.append(x)

    def buffered_sec(self) -> float:
        return sum(len(b) for b in self.buf) / IN_RATE

    def clear_buffer(self) -> None:
        self.buf = []

    # ---------- 校验与提交 ----------

    def commit(self) -> SampleResult:
        """
        结束当前这句，校验并抽取声纹向量。

        无论成功失败都会清空缓冲 —— 失败时用户要重录这一句，
        旧音频留着会和新录的混在一起。
        """
        # 已经录够还继续调用（前端重复点、消息重复投递）时直接返回，
        # 否则 current 会越过 total，进度条和句子索引全乱。
        if self.finished:
            return SampleResult(True, reason="已经录完了")

        if not self.buf:
            return SampleResult(False, "没有收到声音，请靠近麦克风再试一次")

        audio = np.concatenate(self.buf)
        self.buf = []

        # 太长就截断：取前 MAX_SPEECH_SEC
        limit = int(IN_RATE * MAX_SPEECH_SEC)
        if audio.size > limit:
            audio = audio[:limit]

        speech_sec = audio.size / IN_RATE
        if rms(audio) < SPEECH_RMS_FLOOR:
            return SampleResult(
                False, f"这段太安静了（{speech_sec:.1f} 秒），请正常音量朗读")
        if speech_sec < MIN_SPEECH_SEC:
            return SampleResult(
                False, f"只录到 {speech_sec:.1f} 秒，太短了，请完整读完这一句")

        emb = self._embed(audio)
        if emb is None:
            return SampleResult(False, "这段声音没能识别，请重新朗读这一句")

        # 和已有样本比对：差太多说明环境或人变了
        score = None
        if self.embeddings:
            sims = [float(np.dot(emb, e)) for e in self.embeddings]
            score = float(np.mean(sims))
            if score < MIN_COHERENCE:
                return SampleResult(
                    False,
                    f"这句和前面几句差别较大（{score:.2f}）。"
                    "请保持同样的位置和音量再读一次",
                    speech_sec=speech_sec, score=score)

        self.embeddings.append(emb)
        self.current += 1
        self.quality = self._calc_quality()

        if self.current >= self.total:
            self.finished = True

        return SampleResult(True, embedding=emb, speech_sec=speech_sec,
                            score=score)

    def _embed(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """用声纹模型抽一个单位向量。"""
        try:
            ex = self.verifier.ex          # SpeakerVerifier 内部的提取器
            s = ex.create_stream()
            s.accept_waveform(IN_RATE, audio.astype(np.float32))
            s.input_finished()
            v = np.array(ex.compute(s), dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n <= 0:
                return None
            return v / n
        except Exception as e:
            log.warning("录入时抽取声纹失败: %s", e)
            return None

    def _calc_quality(self) -> Optional[float]:
        """
        质量 = 样本内部的平均两两相似度。

        含义：几次录音彼此越像，说明特征稳定、原型可信。
        这是"自洽性"，不是"和别人的区分度" ——
        单次录入拿不到别人的声音，没法算后者。
        所以文案上说"一般/偏差"是提示可能不够稳，不是断言不准。
        """
        if len(self.embeddings) < 2:
            return None
        sims = []
        for i in range(len(self.embeddings)):
            for j in range(i + 1, len(self.embeddings)):
                sims.append(float(np.dot(self.embeddings[i],
                                         self.embeddings[j])))
        return float(np.mean(sims)) if sims else None

    # ---------- 产出 ----------

    def prototype(self) -> Optional[np.ndarray]:
        """
        合成最终原型：所有样本归一化后取平均，再归一化。

        平均而不是取第一条：单条录音的噪声会被抵消一部分，
        多次录音的共同成分（也就是你的音色）被强化。
        """
        if not self.embeddings:
            return None
        m = np.mean(np.stack(self.embeddings), axis=0)
        n = float(np.linalg.norm(m))
        if n <= 0:
            return None
        return (m / n).astype(np.float32)

    def can_save(self) -> bool:
        """至少要有 MIN_SAMPLES 个样本才允许保存。"""
        return len(self.embeddings) >= MIN_SAMPLES
