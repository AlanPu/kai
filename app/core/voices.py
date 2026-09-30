"""可选音色清单。

为什么单独放一个文件：
    音色清单同时被三处用到 —— 命令行脚本（scripts/list_voices.py）、
    网页版的实时选择列表、以及 /api/voices。如果各写一份，
    改了一处忘另一处，用户就会看到互相矛盾的两份清单。

⚠️ 实测教训：音色有**两套不同的清单**，只有部分重叠。
    · 对话音色（CHAT_VOICES）—— .env 里 QWEN_VOICE 能填的值
    · TTS 音色（TTS_VOICES）  —— 只有这些能用语音合成生成试听样音
    拿 Tina（只在对话那套）去合成试听会报 Invalid voice specified，
    那不是名字写错，是那个模型根本不带这个音色。

来源：
    对话音色 https://help.aliyun.com/zh/model-studio/omni-voice-list
    TTS 音色 https://help.aliyun.com/zh/model-studio/qwen-tts-voice-list
    不同模型支持的音色可能不同，换模型后值得重新核对。
"""

# 官方音色列表页 —— 网页版会把链接放在选择列表旁边，
# 让用户能随时打开自己对比试听。
VOICE_DOC_URL = "https://help.aliyun.com/zh/model-studio/omni-voice-list"

# 实时对话模型可选音色：(参数值, 中文名, 描述, 性别)
CHAT_VOICES: list[tuple[str, str, str, str]] = [
    ("Tina",       "甜甜 Tina",       "温热奶茶般甜暖，解决问题不含糊", "女"),
    ("Cindy",      "林欣宜 Cindy",    "台湾口音，嗲嗲的小姐姐",         "女"),
    ("Liora Mira", "清欢 Liora Mira", "温柔，用声音织就烟火人间",       "女"),
    ("Raymond",    "林川野 Raymond",  "声音清亮，爱吃外卖的宅男",       "男"),
    ("Zane",       "泽恩 Zane",       "磁性迷人",                       "男"),
    ("Katerina",   "卡捷琳娜 Katerina", "御姐音，韵律回味十足",         "女"),
    ("Ryan",       "甜茶 Ryan",       "节奏拉满，戏感炸裂",             "男"),
    ("Mia",        "舒然 Mia",        "温柔生活博主，慢生活美学",       "女"),
    ("Cici",       "绵绵 Cici",       "邻家妹妹，声线软糯",             "女"),
    ("Theo Calm",  "予安 Theo Calm",  "沉静，在静默处传递理解",         "男"),
    ("Serena",     "苏瑶 Serena",     "温柔小姐姐",                     "女"),
    ("Maia",       "四月 Maia",       "知性与温柔的碰撞",               "女"),
    ("Evan",       "江晨 Evan",       "男大学生",                       "男"),
    ("Qiao",       "小乔妹 Qiao",     "表面甜妹，个性十足（台湾口音）", "女"),
    ("Momo",       "茉兔 Momo",       "撒娇搞怪，逗你开心",             "女"),
    ("Wil",        "伟伦 Wil",        "在深圳长大的港台腔小哥哥",       "男"),
    ("Angel",      "安琪 Angel",      "台式口音，超甜",                 "女"),
]

# TTS 模型可选音色（用于合成试听样音）
TTS_VOICES: list[str] = [
    "Cherry", "Serena", "Ethan", "Chelsie", "Momo", "Vivian", "Moon",
    "Maia", "Kai", "Nofish", "Bella", "Jennifer", "Ryan", "Katerina",
    "Aiden", "Eldric Sage", "Mia", "Mochi", "Bellona", "Vincent",
    "Bunny", "Neil", "Elias", "Arthur", "Nini", "Seren", "Pip",
    "Stella", "Bodega", "Sonrisa", "Alek", "Dolce", "Sohee",
    "Ono Anna", "Lenn", "Emilien", "Andre", "Jada", "Dylan", "Sunny",
    "Eric", "Rocky", "Kiki",
]


def chat_voice_names() -> list[str]:
    return [v[0] for v in CHAT_VOICES]


def can_sample(voice: str) -> bool:
    """这个音色能不能用 TTS 合成试听样音。"""
    return voice in TTS_VOICES
