"""
生成测试用的人声样本（调用阿里云 TTS）。

为什么需要它：
  稳定性测试要模拟"用户一直在说话"，但纯合成的正弦/谐波音
  会被 Qwen 的 VAD 拒绝，导致测试报告"从未识别出用户语音"，
  看起来像程序坏了，其实是素材不像人话。

产物：scripts/sample_speech.pcm（16kHz / int16 / 单声道）
"""

from __future__ import annotations

import io
import json
import ssl
import sys
import urllib.request
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import certifi
import numpy as np

from app.core.audio import resample_linear, to_int16
from app.core.config import load_settings

TTS_URL = ("https://dashscope.aliyuncs.com/api/v1/services/aigc/"
           "multimodal-generation/generation")

LINES = [
    "Hello, I went to the park yesterday with my friend.",
    "I usually work from home on Fridays and I really like it.",
    "My job is a software engineer and I live in Shenzhen.",
    "I think learning English is important for my career.",
    "On weekends I like running and eating spicy food.",
]


def _opener():
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(
            context=ssl.create_default_context(cafile=certifi.where())))


def tts_url(text: str, voice: str, key: str) -> str:
    body = json.dumps({"model": "qwen3-tts-flash",
                       "input": {"text": text, "voice": voice}}).encode()
    req = urllib.request.Request(TTS_URL, data=body, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"})
    d = json.load(_opener().open(req, timeout=60))
    return d["output"]["audio"]["url"]


def fetch(url: str) -> bytes:
    return _opener().open(url, timeout=60).read()


def main() -> int:
    s = load_settings()
    if not s.dashscope_api_key:
        print("缺少 DASHSCOPE_API_KEY")
        return 1

    voice = sys.argv[1] if len(sys.argv) > 1 else "Cherry"
    parts: list[np.ndarray] = []

    for i, text in enumerate(LINES, 1):
        raw = fetch(tts_url(text, voice, s.dashscope_api_key))
        w = wave.open(io.BytesIO(raw))
        rate = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() == 2:
            data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)
        if rate != 16000:
            data = to_int16(resample_linear(data.astype(np.float32),
                                            rate, 16000))
        parts.append(data)
        parts.append(np.zeros(8000, dtype=np.int16))     # 0.5s 停顿
        print(f"  [{i}] {text[:46]}  ({len(data)/16000:.1f}s)")

    out = np.concatenate(parts)
    dest = Path(__file__).with_name("sample_speech.pcm")
    out.tofile(dest)
    print(f"\n  {len(out)/16000:.1f}s → {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
