"""
练习语言。

这个工具最初只为英语设计：备课要英文引导句、教练提示词写死
"English speaking coach"、纠错按英文语法规则来。想练韩语时，
这些地方全都对不上 —— 模型会拿韩语素材生成一套**英文**问题。

所以把「目标语言」抽成一个显式参数，所有提示词都从它派生。
语言不是全局开关：同一个人可能今天练英语、明天练韩语，
它跟着**这一次会话**走，不进 .env。

新增一种语言要做的事：
  1. 在 LANGUAGES 里加一条（名字 + 提示词里要用的语言名）；
  2. 如果它有特殊的语法/口语要求，在该语言的 note 里写清楚；
  3. 前端下拉框加一个选项（app/web/index.html 的 optLang）。
不需要改数据库：语言跟着会话的计划一起存进 plan_json。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    """一种练习语言。"""

    code: str        # 内部标识：en / ko
    label: str       # 中文名，用于界面：英语 / 韩语
    english: str     # 英文名，用于英文提示词：English / Korean
    native: str      # 母语写法：English / 한국어

    # 备课时的额外要求（中文写，模型看得懂）。
    # 不同语言的口语习惯差别很大，光换语言名不够。
    planner_note: str = ""

    # 纠错时要重点看什么（中文写，拼进纠错提示词）。
    correction_focus: str = ""


LANGUAGES: dict[str, Language] = {
    "en": Language(
        code="en",
        label="英语",
        english="English",
        native="English",
        planner_note=(
            "- 话题用日常口语，像朋友聊天；不要用考试腔。\n"
            "- 不要设计成只能回答 yes/no 的问题。"
        ),
        correction_focus=(
            "- grammar：时态、单复数、冠词、介词、语序\n"
            "- vocabulary：中式英语、过于书面的用词\n"
            "- fluency：啰嗦、填充词过多"
        ),
    ),
    "ko": Language(
        code="ko",
        label="韩语",
        english="Korean",
        native="한국어",
        planner_note=(
            "- 用日常口语的 해요체，不要用 합니다 体或 반말；\n"
            "  这是练口语，不是写作文。\n"
            "- 词汇给韩语原形或常用搭配，不要给汉字词堆砌。\n"
            "- 话题要能从「저는 …」这种简单句开始说，别一上来就问\n"
            "  需要复杂语法（-더라、-았/었더라면 之类）的问题。"
        ),
        correction_focus=(
            "- grammar：助词（은/는、이/가、을/를、에/에서）、\n"
            "  动词/形容词词尾、时态（-았/었-、-겠-）、敬语（높임법）、语序\n"
            "- vocabulary：从中文直译过去的不自然说法\n"
            "- fluency：重复、停顿填充词（그…、저…）过多"
        ),
    ),
}

DEFAULT_LANGUAGE = "en"


def get_language(code: str | None) -> Language:
    """按 code 取语言；未知/空值回退到英语，绝不抛错。"""
    return LANGUAGES.get((code or "").strip().lower(),
                         LANGUAGES[DEFAULT_LANGUAGE])


def language_codes() -> list[str]:
    return list(LANGUAGES)
