"""
长稳定性测试：模拟真实使用时长的连续会话。

原型阶段从未验证过 30 分钟连续运行 —— 这是最大的未知风险。
本脚本持续发送真实音频、记录事件与内存，验证：
  · 连接是否会被服务端断开
  · 内存是否持续增长（泄漏）
  · 音频往返是否一直正常
  · 长时间静默后是否还能唤醒

用法：
    python scripts/soak_test.py --minutes 30
    python scripts/soak_test.py --minutes 3        # 快速检查
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import sys
import time
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import websockets


def mem_mb() -> float:
    """当前进程内存占用（MB）。"""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 / 1024 if sys.platform == "darwin" else r / 1024


SPEECH_FILE = Path(__file__).with_name("sample_speech.pcm")


def load_speech() -> bytes:
    """
    读取真实人声样本（16kHz int16 单声道）。

    为什么要用真人声音而不是合成音：
    Qwen 的 VAD 会拒绝纯谐波合成音，测出来"从未识别出用户语音"，
    看起来像 bug，其实是测试素材不像人话。
    scripts/make_sample_speech.py 可重新生成。
    """
    if not SPEECH_FILE.is_file():
        raise SystemExit(
            f"缺少人声样本 {SPEECH_FILE}\n"
            "请先运行：python scripts/make_sample_speech.py")
    return SPEECH_FILE.read_bytes()


class Watch:
    def __init__(self):
        self.t0 = time.time()
        self.audio_in = 0
        self.audio_out = 0
        self.user_texts = 0
        self.ai_texts = 0
        self.errors: list[str] = []
        self.mem_samples: list[tuple[float, float]] = []

    def elapsed(self) -> float:
        return time.time() - self.t0


async def run(minutes: float, url: str, *, talk_every: float = 25.0):
    w = Watch()
    speech = load_speech()                   # 真实人声
    silence = (np.zeros(16000 * 1, dtype=np.int16)).tobytes()

    print(f"  时长上限 {minutes:.1f} 分钟，URL={url[:60]}…")
    print(f"  起始内存 {mem_mb():.0f} MB\n")

    async with websockets.connect(url, max_size=None,
                                  ping_interval=None) as ws:
        ready = False
        last_talk = 0.0
        deadline = time.time() + minutes * 60

        while time.time() < deadline:
            # --- 发音频 ---
            now = time.time()
            if now - last_talk > talk_every:
                last_talk = now
                # 分成 20ms 块发，与浏览器行为一致（320 样本 = 640 字节）
                for i in range(0, len(speech), 640):
                    await ws.send(speech[i:i + 640])
                    w.audio_in += 1
                    await asyncio.sleep(0.002)      # 略快于实时
                # 句尾补静音，帮助 VAD 判定说完
                for i in range(0, len(silence), 640):
                    await ws.send(silence[i:i + 640])
                    w.audio_in += 1
                    await asyncio.sleep(0.002)

            # --- 收消息 ---
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            except websockets.exceptions.ConnectionClosed as e:
                print(f"  ❌ 连接被断开 @ {w.elapsed():.0f}s: {e}")
                w.errors.append(f"closed@{w.elapsed():.0f}s")
                break
            else:
                if isinstance(raw, str):
                    m = json.loads(raw)
                    t = m.get("type")
                    if t == "audio":
                        w.audio_out += 1
                    elif t == "ready":
                        ready = True
                    elif t == "user_text":
                        w.user_texts += 1
                        print(f"  [{w.elapsed():6.0f}s] 我: "
                              f"{m['text'][:60]}")
                    elif t == "ai_text":
                        w.ai_texts += 1
                        print(f"  [{w.elapsed():6.0f}s] AI: "
                              f"{m['text'][:60]}")
                    elif t == "error":
                        e = m.get("error") or {}
                        msg = f"{e.get('code')}: {e.get('message')}"
                        w.errors.append(msg)
                        print(f"  [{w.elapsed():6.0f}s] ⚠️ 错误 {msg[:90]}")

            # --- 每 60 秒记录一次内存 ---
            if not w.mem_samples or w.elapsed() - w.mem_samples[-1][0] > 60:
                w.mem_samples.append((w.elapsed(), mem_mb()))
                print(f"  [{w.elapsed():6.0f}s] 内存 {mem_mb():.0f} MB  "
                      f"(入 {w.audio_in} 出 {w.audio_out} "
                      f"我 {w.user_texts} AI {w.ai_texts})")

        try:
            await ws.send(json.dumps({"type": "stop"}))
            # 等报告
            t_end = time.time() + 15
            while time.time() < t_end:
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                if isinstance(raw, str):
                    m = json.loads(raw)
                    if m.get("type") == "finished":
                        r = m["report"]
                        print(f"\n  ✅ 正常收尾：{r['turn_count']} 轮, "
                              f"{r['duration_sec']}s, "
                              f"用户占比 {r['user_ratio']:.0%}")
                        break
        except Exception as e:
            print(f"\n  ⚠️ 收尾异常: {e}")

    # ---- 结论 ----
    print(f"\n{'='*56}")
    print(f"  运行 {w.elapsed()/60:.1f} 分钟")
    print(f"  音频 入 {w.audio_in} 块（{w.audio_in*20/1000/60:.1f} 分钟）/ "
          f"出 {w.audio_out} 块")
    print(f"  转写 我 {w.user_texts} 句 / AI {w.ai_texts} 句")
    print(f"  内存 {w.mem_samples[0][1]:.0f} → {mem_mb():.0f} MB")
    if len(w.mem_samples) > 2:
        growth = mem_mb() - w.mem_samples[1][1]
        print(f"  内存增长 {growth:+.0f} MB "
              f"{'（正常波动）' if abs(growth) < 150 else '⚠️ 疑似泄漏'}")
    if w.errors:
        print(f"  ⚠️ 错误 {len(w.errors)} 次:")
        for e in w.errors[:5]:
            print(f"     {e[:100]}")
    else:
        print("  ✅ 全程无错误、无断线")

    if w.user_texts == 0:
        print("  ⚠️ 从未识别出用户语音 —— VAD 或音频未被接受")
    return len(w.errors) == 0 and w.user_texts > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--host", default="127.0.0.1:8011")
    ap.add_argument("--content", default="稳定性测试 聊一聊我的工作")
    ap.add_argument("--talk-every", type=float, default=25.0)
    a = ap.parse_args()

    url = (f"ws://{a.host}/ws/session?content={quote(a.content)}"
           f"&voiceprint=0")
    ok = asyncio.run(run(a.minutes, url, talk_every=a.talk_every))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
