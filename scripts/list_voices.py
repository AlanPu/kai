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


# ---------- 实时对话模型可选音色 ----------
# 来源：百炼「非实时（Qwen-Omni）和实时（Qwen-Omni-Realtime）支持的音色列表」
# 这些是**对话时** AI 说话的音色，即 .env 里 QWEN_VOICE 的取值。
CHAT_VOICES: list[tuple[str, str]] = [
    ("Tina",       "甜甜 Tina — 温热奶茶般甜暖（默认）"),
    ("Cindy",      "林欣宜 Cindy — 台湾口音，嗲嗲的小姐姐"),
    ("Liora Mira", "清欢 Liora Mira — 温柔，用声音织就烟火人间"),
    ("Raymond",    "林川野 Raymond — 声音清亮的宅男"),
    ("Zane",       "泽恩 Zane — 磁性迷人"),
    ("Katerina",   "卡捷琳娜 Katerina — 御姐音，韵律回味十足"),
    ("Ryan",       "甜茶 Ryan — 节奏拉满，戏感炸裂"),
    ("Mia",        "舒然 Mia — 温柔生活博主，慢生活美学"),
    ("Cici",       "绵绵 Cici — 邻家妹妹，声线软糯"),
    ("Theo Calm",  "予安 Theo Calm — 沉静，在静默处传递理解"),
    ("Serena",     "苏瑶 Serena — 温柔小姐姐"),
    ("Maia",       "四月 Maia — 知性与温柔的碰撞"),
    ("Evan",       "江晨 Evan — 男大学生"),
    ("Qiao",       "小乔妹 Qiao — 表面甜妹，个性十足（台湾口音）"),
    ("Momo",       "茉兔 Momo — 撒娇搞怪"),
    ("Wil",        "伟伦 Wil — 深圳长大的港台腔小哥哥"),
    ("Angel",      "安琪 Angel — 台式口音，很甜"),
]

# ---------- 语音合成（TTS）可用音色 ----------
# 来源：百炼「Qwen-TTS音色列表」中 qwen3-tts-flash 支持的部分。
# 试听样音只能用这里的名字合成。
TTS_VOICES: list[tuple[str, str]] = [
    ("Cherry",       "芊悦 — 阳光积极、亲切自然（女）"),
    ("Serena",       "苏瑶 — 温柔小姐姐（女）"),
    ("Ethan",        "晨煦 — 阳光温暖，带北方口音（男）"),
    ("Chelsie",      "千雪 — 二次元虚拟女友（女）"),
    ("Momo",         "茉兔 — 撒娇搞怪（女）"),
    ("Vivian",       "十三 — 拽拽的可爱小暴躁（女）"),
    ("Moon",         "月白 — 率性帅气（男）"),
    ("Maia",         "四月 — 知性与温柔（女）"),
    ("Kai",          "凯 — 耳朵的一场 SPA（男）"),
    ("Nofish",       "不吃鱼 — 不会翘舌音的设计师（男）"),
    ("Bella",        "萌宝 — 小萝莉（女）"),
    ("Jennifer",     "詹妮弗 — 品牌级、电影质感美语女声（女）"),
    ("Ryan",         "甜茶 — 节奏拉满，戏感炸裂（男）"),
    ("Katerina",     "卡捷琳娜 — 御姐音（女）"),
    ("Aiden",        "艾登 — 精通厨艺的美语大男孩（男）"),
    ("Eldric Sage",  "沧明子 — 沉稳睿智的老者（男）"),
    ("Mia",          "乖小妹 — 温顺乖巧（女）"),
    ("Mochi",        "沙小弥 — 聪明伶俐的小大人（男）"),
    ("Bellona",      "燕铮莺 — 声音洪亮，吐字清晰（女）"),
    ("Vincent",      "田叔 — 沙哑烟嗓（男）"),
    ("Bunny",        "萌小姬 — 萌属性爆棚（女）"),
    ("Neil",         "阿闻 — 字正腔圆的新闻主持人（男）"),
    ("Elias",        "墨讲师 — 把复杂知识讲清楚（女）"),
    ("Arthur",       "徐大爷 — 质朴的乡音（男）"),
    ("Nini",         "邻家妹妹 — 软糯（女）"),
    ("Seren",        "小婉 — 温和舒缓，助眠（女）"),
    ("Pip",          "顽屁小孩 — 调皮捣蛋（男）"),
    ("Stella",       "少女阿月 — 迷糊少女（女）"),
    ("Bodega",       "博德加 — 热情的西班牙大叔（男）"),
    ("Sonrisa",      "索尼莎 — 热情开朗的拉美大姐（女）"),
    ("Alek",         "阿列克 — 战斗民族的冷与暖（男）"),
    ("Dolce",        "多尔切 — 慵懒的意大利大叔（男）"),
    ("Sohee",        "素熙 — 温柔开朗的韩国欧尼（女）"),
    ("Ono Anna",     "小野杏 — 鬼灵精怪的青梅竹马（女）"),
    ("Lenn",         "莱恩 — 穿西装也听后朋克的德国青年（男）"),
    ("Emilien",      "埃米尔安 — 浪漫的法国大哥哥（男）"),
    ("Andre",        "安德雷 — 磁性沉稳（男）"),
    ("Jada",         "上海-阿珍 — 沪上阿姐（女）"),
    ("Dylan",        "北京-晓东 — 胡同少年（男）"),
    ("Sunny",        "四川-晴儿 — 川妹子（女）"),
    ("Eric",         "四川-程川 — 成都男子（男）"),
    ("Rocky",        "粤语-阿强 — 幽默风趣（男）"),
    ("Kiki",         "粤语-阿清 — 甜美港妹（女）"),
]

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
        print("\n改 .env 里的 QWEN_VOICE 即可切换（改完重启服务）。")
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
