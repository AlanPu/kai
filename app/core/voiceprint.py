"""
声纹校验：判断"这段音频是不是本人"。

从原型移植，逻辑经过实测验证：
  · float32/int16 单位问题已修（见 core.audio.rms）
  · 1 秒窗口 + 3 次多数表决，降低单次波动误判
  · 太安静不判断（返回 None），避免噪声给出随机结果

用途：过滤旁人说话和外部噪音，让 AI 只回应本人。
局限：不是安全级认证；换麦克风或感冒时准确率下降。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .audio import IN_RATE, rms

log = logging.getLogger(__name__)

# 静音门槛（int16 量级）。低于此值不参与判定。
SILENCE_RMS = 200


def load_voiceprint(path: str | Path) -> tuple[Optional[np.ndarray], dict]:
    """读取声纹档案。返回 (原型向量, 元数据)；不存在则 (None, {})。"""
    p = Path(path)
    if not p.is_file():
        return None, {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        proto = np.array(d["prototype"], dtype=np.float32)
        n = float(np.linalg.norm(proto))
        if n > 0:
            proto /= n        # 归一化，余弦相似度才有意义
        return proto, d
    except Exception as e:
        log.warning("声纹档案读取失败: %s", e)
        return None, {}


class SpeakerVerifier:
    """
    逐块喂入音频，判断是否为本人。

    feed() 返回：
        True  = 是本人
        False = 不是本人
        None  = 样本不足或太安静（暂不判断）
    """

    def __init__(self, model_path: str | Path, prototype: np.ndarray, *,
                 threshold: float = 0.5, window_sec: float = 2.0,
                 history: int = 3):
        import sherpa_onnx as so

        cfg = so.SpeakerEmbeddingExtractorConfig(
            model=str(model_path), num_threads=1, debug=False, provider="cpu")
        self.ex = so.SpeakerEmbeddingExtractor(cfg)
        self.proto = prototype
        self.threshold = threshold
        self.window_sec = window_sec
        self.history_len = history

        self.buf: list[np.ndarray] = []
        self.history: list[bool] = []
        self.last_score: Optional[float] = None
        self.judged = 0        # 判定次数，便于诊断

    def reset(self) -> None:
        """清空缓冲与表决历史（例如 AI 说话期间）。"""
        self.buf = []
        self.history = []

    def feed(self, samples) -> Optional[bool]:
        """喂入一小段音频（float32 或 int16 数组）。"""
        x = np.asarray(samples)
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if x.size == 0:
            return None
        self.buf.append(x)

        need = int(IN_RATE * self.window_sec)
        if sum(len(b) for b in self.buf) < need:
            return None

        audio = np.concatenate(self.buf)
        self.buf = []

        if rms(audio) < SILENCE_RMS:
            return None

        try:
            s = self.ex.create_stream()
            s.accept_waveform(IN_RATE, audio)
            s.input_finished()
            v = np.array(self.ex.compute(s), dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n <= 0:
                return None
            v /= n
            score = float(np.dot(self.proto, v))
        except Exception as e:
            log.debug("声纹计算失败: %s", e)
            return None

        self.last_score = score
        self.judged += 1
        is_me = score >= self.threshold
        self.history.append(is_me)
        if len(self.history) > self.history_len:
            self.history.pop(0)
        # 多数表决：3 次里过半才算
        return sum(self.history) > len(self.history) / 2
