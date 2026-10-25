"""
纠错分析：把用户说的话变成分级纠错项。

需求 3：语法纠错 + 地道表达建议
需求 4：发音只在「明显单词错误」时纠正，口音问题默认不纠

本模块的 key 在于 **分级**：
模型必须为每条纠错给出 severity，数据库层再据此过滤
（见 storage/models.py 的 Correction.should_show）。
这样"不要频繁打断"就不是一句提示词嘱咐，而是代码约束。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ..core.languages import Language, get_language
from ..core.llm import LLMClient

log = logging.getLogger(__name__)


@dataclass
class CorrectionItem:
    """一条纠错建议。"""

    kind: str          # grammar / vocabulary / pronunciation / fluency
    severity: str      # critical / minor / ignore
    original: str      # 原句或原词
    suggestion: str    # 建议说法
    explanation: str = ""   # 中文解释
    word: Optional[str] = None      # 发音错误涉及的单词
    phonetic: Optional[str] = None  # 音标


def build_prompt(lang: Language | None = None) -> str:
    """
    纠错提示词。

    语言相关的部分只有三处：判断标准（该语言的重点语法）、
    JSON 里的示例、以及发音部分的注意事项。其余「别频繁打断」
    的约束对所有语言一样。
    """
    lang = lang or get_language("en")

    if lang.code == "ko":
        example = ('    {"kind": "grammar",\n'
                   '     "severity": "critical",\n'
                   '     "original": "저는 어제 학교를 갔어요",\n'
                   '     "suggestion": "저는 어제 학교에 갔어요",\n'
                   '     "explanation": "가다 表示去某地，地点后面用 에，不用 를",\n'
                   '     "word": null,\n'
                   '     "phonetic": null}')
        pronunciation_note = (
            "韩语的转写会自然按连音/紧音规则写出，不要把正常的连音\n"
            "（如 한국어 → 한구거）当成发音错误。")
    else:
        example = ('    {"kind": "grammar",\n'
                   '     "severity": "critical",\n'
                   '     "original": "I go to park yesterday",\n'
                   '     "suggestion": "I went to the park yesterday",\n'
                   '     "explanation": "yesterday 表示过去，动词要用过去式 went",\n'
                   '     "word": null,\n'
                   '     "phonetic": null}')
        pronunciation_note = (
            "不要把常见的口音差异（如 th/s、l/r、词尾辅音弱化）\n"
            "标成 critical —— 那些是 ignore 或 minor。")

    return f"""你是一位{lang.label}口语纠错助手，为中文母语的{lang.label}学习者服务。

学生会给你一段他说过的{lang.label}（语音转写）。请找出值得纠正的地方。

## 纠错的四类
- grammar：语法错误
- vocabulary：用词不地道（虽然能懂，但母语者不会这么说）
- pronunciation：发音问题（**只能依据转写文本推断**，见下方严格限制）
- fluency：表达不流畅（啰嗦、直译腔、停顿填充词过多）

## 这个语言要重点看
{lang.correction_focus}

## 严重度分级（非常重要）
- critical：**明显的错误**，会让听者困惑或听起来明显不对。
- minor：不够地道，但不影响理解。可以用更自然的说法。
- ignore：可以接受的说法，**不需要提醒**。宁可少提也不要多提。

## 严格限制
1. **发音纠错必须非常保守。** 你只能看到文字，看不到声音。
   只有当拼写/上下文强烈暗示明显发音错误时才标注。
   {pronunciation_note}
   拿不准就不要报发音问题。
2. 每次最多报 3 条纠错，按重要程度排序。
3. 如果这句话没有值得纠正的地方，就返回空数组 —— 这是完全可以的。
4. explanation 用中文，简明扼要（一句话），说清"为什么"。
5. original 必须是学生原话里的片段，suggestion 是改好的说法。

严格按 JSON 输出，不要任何额外文字：
{{
  "corrections": [
{example}
  ]
}}"""


class Corrector:
    """分析一句话，产出纠错项。"""

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def check(self, text: str, *, context: str = "",
              max_items: int = 3,
              lang: Language | str = "en") -> list[CorrectionItem]:
        """
        检查一段目标语言的话。

        text    ：用户说的话（转写文本）
        context ：可选的上下文（前几轮对话），帮助判断时态等
        lang    ：目标语言（en / ko），决定语法判断标准和示例
        """
        text = (text or "").strip()
        if not text:
            return []

        language = lang if isinstance(lang, Language) else get_language(lang)

        # 太短的输入没有纠错价值，省一次调用。
        # 韩语按词切分后通常比英语短（"저는 학생이에요" 只有两个词），
        # 用英语的阈值会把正常的短句全部跳过。
        min_words = 3 if language.code == "en" else 2
        if len(text.split()) < min_words:
            return []

        user = f"学生说的话：\n{text}"
        if context.strip():
            user = f"上文（仅供参考）：\n{context.strip()[:800]}\n\n{user}"

        messages = [
            {"role": "system", "content": build_prompt(language)},
            {"role": "user", "content": user},
        ]

        # 1200 对推理模型不够：思考本身就可能上千 token，
        # 正文被截断后 JSON 解析失败，纠错等于白做。
        data = self.llm.chat_json(messages, temperature=0.3, max_tokens=2500)
        items = self._normalize(data)
        return items[:max_items]

    @staticmethod
    def _normalize(data) -> list[CorrectionItem]:
        """容错解析，并强制合法枚举值。"""
        if isinstance(data, list):
            raw = data
        elif isinstance(data, dict):
            raw = data.get("corrections") or []
        else:
            return []

        kinds = {"grammar", "vocabulary", "pronunciation", "fluency"}
        sevs = {"critical", "minor", "ignore"}
        out: list[CorrectionItem] = []

        for c in raw:
            if not isinstance(c, dict):
                continue
            original = str(c.get("original") or "").strip()
            suggestion = str(c.get("suggestion") or "").strip()
            if not original or not suggestion:
                continue

            kind = str(c.get("kind") or "grammar").strip().lower()
            if kind not in kinds:
                kind = "grammar"
            sev = str(c.get("severity") or "minor").strip().lower()
            if sev not in sevs:
                sev = "minor"

            word = c.get("word")
            word = str(word).strip() if word else None
            phonetic = c.get("phonetic")
            phonetic = str(phonetic).strip() if phonetic else None

            # 非发音类不该带 word/phonetic，清掉避免污染统计
            if kind != "pronunciation":
                word = phonetic = None
            elif not word:
                # 发音问题但没有具体单词 → 无法定位，降级为 ignore
                sev = "ignore"

            out.append(CorrectionItem(
                kind=kind, severity=sev, original=original,
                suggestion=suggestion,
                explanation=str(c.get("explanation") or "").strip(),
                word=word, phonetic=phonetic))
        return out
