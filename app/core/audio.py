"""
音频基础常量与工具。

这些常量是实测确定的，改动会导致协议错误：
  · Qwen Realtime 上行必须 16 kHz，下行是 24 kHz（**上下行不同**）
  · 20 ms 一块是官方示例的推荐值
"""

from __future__ import annotations

import numpy as np

# ---- 采样率（不可随意改动）----
IN_RATE = 16000
OUT_RATE = 24000

# ---- 分块 ----
BLOCK_MS = 20
IN_BLOCK = IN_RATE * BLOCK_MS // 1000      # 320 样本
IN_BLOCK_BYTES = IN_BLOCK * 2              # int16


def rms(x: np.ndarray) -> float:
    """均方根音量，统一换算到 int16 量级（±32768）。

    踩过的坑：门槛按 int16 定（如 200），但 sounddevice / numpy
    常给出 float32（±1.0）音频，RMS 相差 32768 倍，
    导致明明有声音却被判成"太安静"。
    这里按峰值自动判定量级，两种输入都能正确工作。
    """
    if x is None or len(x) == 0:
        return 0.0
    a = np.asarray(x, dtype=np.float64)
    r = float(np.sqrt(np.mean(a ** 2)))
    peak = float(np.max(np.abs(a)))
    if peak <= 1.5:          # float32 (±1.0) 量级
        r *= 32768.0
    return r


def to_int16(x: np.ndarray) -> np.ndarray:
    """把任意量级的音频转成 int16。"""
    a = np.asarray(x, dtype=np.float32)
    if a.size and float(np.max(np.abs(a))) <= 1.5:
        a = a * 32768.0
    return np.clip(a, -32768, 32767).astype(np.int16)


def resample_linear(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """线性重采样。仅在采样率不符时兜底，质量足够声纹使用。"""
    if src == dst or len(x) == 0:
        return x
    n = int(len(x) * dst / src)
    if n <= 0:
        return np.zeros(0, dtype=x.dtype)
    idx = np.linspace(0, len(x) - 1, n)
    return np.interp(idx, np.arange(len(x)), x).astype(x.dtype)
