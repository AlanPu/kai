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

from ..core.llm import LLMClient, LLMError

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


SYSTEM_PROMPT = """你是一位英语口语纠错助手，为中文母语的英语学习者服务。

学生会给你一段他说过的英文（语音转写）。请找出值得纠正的地方。

## 纠错的四类
- grammar：语法错误（时态、单复数、介词、冠词、语序等）
- vocabulary：用词不地道（虽然能懂，但母语者不会这么说）
- pronunciation：发音问题（**只能依据转写文本推断**，见下方严格限制）
- fluency：表达不流畅（啰嗦、中式英语、停顿填充词过多）

## 严重度分级（非常重要）
- critical：**明显的错误**，会让听者困惑或听起来明显不对。
  例如：时态用错（"I go there yesterday"）、'think' 读成 'sink' 这种
  会引发误解的发音错误、严重的中式英语。
- minor：不够地道，但不影响理解。
  例如：用词偏书面（"I am very fond of it"）、轻微口音、
  可以用更自然的说法。
- ignore：可以接受的说法，**不需要提醒**。宁可少提也不要多提。

## 严格限制
1. **发音纠错必须非常保守。** 你只能看到文字，看不到声音。
   只有当某个词的拼写/上下文强烈暗示明显发音错误时才标注
   （例如学生写出 "sink" 但显然想说 "think"）。
   **绝对不要**把常见的口音差异（如 th/s、l/r、词尾辅音弱化）
   标成 critical —— 那些是 ignore 或 minor。
   拿不准就不要报发音问题。
2. 每次最多报 3 条纠错，按重要程度排序。
3. 如果这句话没有值得纠正的地方，就返回空数组 —— 这是完全可以的。
4. explanation 用中文，简明扼要（一句话），说清"为什么"。
5. original 必须是学生原话里的片段，suggestion 是改好的说法。

严格按 JSON 输出，不要任何额外文字：
{
  "corrections": [
    {"kind": "grammar",
     "severity": "critical",
     "original": "I go to park yesterday",
     "suggestion": "I went to the park yesterday",
     "explanation": "yesterday 表示过去，动词要用过去式 went",
     "word": null,
     "phonetic": null}
  ]
}"""


class Corrector:
    """分析一句话，产出纠错项。"""

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def check(self, text: str, *, context: str = "",
              max_items: int = 3) -> list[CorrectionItem]:
        """
        检查一段英文。

        text    ：用户说的话（转写文本）
        context ：可选的上下文（前几轮对话），帮助判断时态等
        """
        text = (text or "").strip()
        if not text:
            return []

        # 太短的输入（如 "yes"、"ok"）没有纠错价值，省一次调用
        if len(text.split()) < 3:
            return []

        user = f"学生说的话：\n{text}"
        if context.strip():
            user = f"上文（仅供参考）：\n{context.strip()[:800]}\n\n{user}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

        data = self.llm.chat_json(messages, temperature=0.3, max_tokens=1200)
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
