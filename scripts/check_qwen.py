#!/usr/bin/env python3
"""
连通性自检：确认配置能用，且几乎不花钱。

只做两件事：
  1. 检查 .env 里的密钥/工作区是否填了
  2. 真的连一次 Qwen Realtime，收到 session.created/updated 就算通过

**不推送任何音频**，所以成本接近零（只有一次握手）。
出问题时先跑这个，能把"配置错"和"代码错"分开。

用法：
    .venv/bin/python scripts/check_qwen.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import certifi                                     # noqa: E402
import ssl                                         # noqa: E402
import websockets                                  # noqa: E402
from app.core.config import load_settings          # noqa: E402

OK = "  ✅"
BAD = "  ❌"


async def check_realtime(s) -> bool:
    """连一次 Realtime：握手 + 发 session.update，确认配置被接受。

    不发音频，所以成本接近零。

    注意：`session.created` 只是握手回执，不代表配置能用。
    服务端只有在收到 session.update 之后才会回 session.updated，
    所以这里必须真的发一次配置 —— 否则会误报"超时失败"。
    """
    url = s.voice_ws_url()
    print(f"  连接 {url.split('?')[0]}")

    ctx = ssl.create_default_context(cafile=certifi.where())
    try:
        async with websockets.connect(
            url,
            additional_headers={"Authorization": f"Bearer {s.dashscope_api_key}"},
            max_size=None,
            ssl=ctx,
            open_timeout=20,
        ) as ws:
            # 1) 握手
            try:
                first = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
            except asyncio.TimeoutError:
                print(f"{BAD} 连上了但没收到 session.created（超时）")
                return False
            if first.get("type") == "error" or first.get("code"):
                print(f"{BAD} 鉴权/参数被拒: {first}")
                return False
            print(f"{OK} 鉴权通过（{first.get('type')}）")

            # 2) 真的发一次配置，看服务端认不认
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "voice": s.qwen_voice,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm24",
                    "input_audio_transcription": {
                        "model": "qwen3-asr-flash-realtime"},
                    "turn_detection": {
                        "type": "server_vad", "threshold": 0.5,
                        "silence_duration_ms": 800, "prefix_padding_ms": 300,
                        "create_response": False, "interrupt_response": True},
                },
            }))

            deadline = asyncio.get_event_loop().time() + 25
            while asyncio.get_event_loop().time() < deadline:
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                except asyncio.TimeoutError:
                    break
                t = msg.get("type", "")
                if t == "session.updated":
                    print(f"{OK} 会话配置已接受（session.updated）")
                    return True
                if t == "error" or msg.get("code"):
                    print(f"{BAD} 配置被拒: {msg}")
                    return False
    except Exception as e:                          # noqa: BLE001
        print(f"{BAD} 连接失败: {type(e).__name__}: {e}")
        return False
    print(f"{BAD} 发了 session.update 但没收到 session.updated（超时）")
    return False


def main() -> int:
    print("口语陪练 —— 连通性自检\n")

    try:
        s = load_settings()
    except Exception as e:                          # noqa: BLE001
        print(f"{BAD} 读配置失败: {e}")
        print("   → 是不是忘了 cp .env.example .env 并填写？")
        return 1

    ok = True

    # 1) 配置
    if s.dashscope_api_key and s.dashscope_api_key.startswith("sk-") \
            and "your-key" not in s.dashscope_api_key:
        print(f"{OK} DASHSCOPE_API_KEY 已填写")
    else:
        print(f"{BAD} DASHSCOPE_API_KEY 没填或仍是占位符")
        ok = False

    if s.qwen_workspace_id and "your" not in s.qwen_workspace_id:
        print(f"{OK} QWEN_WORKSPACE_ID = {s.qwen_workspace_id}")
    else:
        print(f"{BAD} QWEN_WORKSPACE_ID 没填")
        ok = False

    print(f"  区域 {s.qwen_region} / 模型 {s.qwen_model} / 音色 {s.qwen_voice}")

    # 2) 声纹模型（非必需，但没它就不能过滤旁人说话）
    if s.voiceprint_model.is_file():
        size = s.voiceprint_model.stat().st_size
        print(f"{OK} 声纹模型已就位（{size // 1024 // 1024} MB）")
    else:
        print(f"  ⚠️  声纹模型缺失: {s.voiceprint_model}")
        print("     练习仍可进行，但不会过滤旁人说话。下载方法见 SETUP.md")

    if not ok:
        print("\n配置不完整，先补齐再跑。")
        return 1

    # 3) 真连一次
    print("\n  测试连接（不发送音频，几乎零成本）…")
    if not asyncio.run(check_realtime(s)):
        print("\n连接失败。排查顺序：")
        print("   1. 密钥是否过期 / 工作区 ID 是否对应")
        print("   2. 区域是否写错（cn-beijing / 其他）")
        print("   3. 额度是否用完")
        return 1

    print("\n✅ 全部通过，可以开始了：.venv/bin/python -m app")
    return 0


if __name__ == "__main__":
    sys.exit(main())
