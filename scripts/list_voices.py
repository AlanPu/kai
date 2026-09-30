#!/usr/bin/env python3
"""
列出可用音色，并可生成样音试听。

用法：
    .venv/bin/python scripts/list_voices.py              # 列出全部音色
    .venv/bin/python scripts/list_voices.py --sample     # 每个都生成样音
    .venv/bin/python scripts/list_voices.py --sample Tina Serena Ryan

为什么需要这个脚本：
    音色在官方文档里是**两套不同的清单** ——
      · 实时对话模型（qwen*-omni-realtime）用一套
      · 语音合成模型（qwen3-tts-flash）用另一套
    两套只有部分重叠。用 TTS 生成样音时，只能用 TTS 支持的名字，
    否则会报 "Invalid voice specified"。这个脚本已经帮你分好组了。

    名字本身也看不出效果（比如 Wil、Qiao、Nofish），所以直接
    合成一段英文样音来听，满意了再把名字填进 .env 的 QWEN_VOICE。
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import load_settings, ssl_context   # noqa: E402


def _opener() -> urllib.request.OpenerDirector:
    """国内服务直连：绕过代理，否则可能失败。"""
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl_context()))


from app.core.voices import (CHAT_VOICES as _CHAT, TTS_VOICES as _TTS,
                             VOICE_DOC_URL)                 # noqa: E402

CHAT_VOICES = [(v, f"{name} — {desc}") for v, name, desc, _g in _CHAT]
TTS_VOICES = [(v, "") for v in _TTS]

SAMPLE_TEXT = ("Hey! I'm your English speaking partner. "
               "Let's talk about how your week has been going. "
               "Don't worry about making mistakes — that's how we learn.")


def synth(text: str, voice: str) -> Path:
    """调 TTS 合成样音，返回文件路径。"""
    s = load_settings()
    payload = {"model": "qwen3-tts-flash",
               "input": {"text": text, "voice": voice}}
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
        "multimodal-generation/generation",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {s.dashscope_api_key}",
                 "Content-Type": "application/json"})
    with _opener().open(req, timeout=60) as r:
        url = json.load(r)["output"]["audio"]["url"]

    out = ROOT / "data" / "voice_samples"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{voice.replace(' ', '_')}.wav"
    with _opener().open(url, timeout=60) as r:
        dest.write_bytes(r.read())
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="列出/试听音色")
    ap.add_argument("--sample", nargs="*", metavar="VOICE",
                    help="生成试听样音；不带值则生成全部 TTS 音色")
    ap.add_argument("--all-chat", action="store_true",
                    help="连同对话音色清单一起显示")
    a = ap.parse_args()

    if a.sample is None:
        s = load_settings()
        print(f"当前模型：{s.qwen_model}")
        print(f"当前音色：{s.qwen_voice}\n")
        print(f"【对话音色】{len(CHAT_VOICES)} 个 —— 这是 QWEN_VOICE 能填的值")
        for i, (n, d) in enumerate(CHAT_VOICES, 1):
            cur = "   ← 当前" if n == s.qwen_voice else ""
            print(f"  {i:2d}. {n:14s} {d}{cur}")
        print(f"\n【可试听音色】{len(TTS_VOICES)} 个 —— 能合成样音的名"
              f"（与上面部分重叠）")
        print("      这些是 TTS 模型支持的，用它来听效果。")
        print(f"\n官方音色列表：{VOICE_DOC_URL}")
        print("网页上也可以直接选：点右上角「音色」按钮（可试听）。")
        print("命令行切换：改 .env 里的 QWEN_VOICE，改完重启服务。")
        print("先听一下：.venv/bin/python scripts/list_voices.py --sample Tina Serena Ryan")
        return

    targets = a.sample or [v for v, _ in TTS_VOICES]
    ok = 0
    for v in targets:
        try:
            p = synth(SAMPLE_TEXT, v)
            print(f"  ✅ {v:14s} → {p.relative_to(ROOT)}")
            ok += 1
        except Exception as e:                       # noqa: BLE001
            print(f"  ❌ {v:14s} 失败：{str(e)[:70]}")
    print(f"\n  成功 {ok}/{len(targets)}，样音目录：data/voice_samples")
    print("  打开：open data/voice_samples")


if __name__ == "__main__":
    main()
