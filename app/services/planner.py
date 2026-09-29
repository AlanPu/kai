"""
话题规划：把输入内容变成一份「聊什么」的计划。

需求 2：我不太清楚该聊什么，所以你要主动引导我说，
        但要多给我说的机会。

因此产出的话题必须是**开放式、能让我多说**的，
而不是一问一答就结束的封闭问题。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..core.content import Content
from ..core.llm import LLMClient, LLMError

log = logging.getLogger(__name__)


@dataclass
class Topic:
    """一个可聊的话题。"""

    title: str                    # 话题标题（中文，便于我理解）
    prompt: str                   # 引导我开口的英文句子
    why: str = ""                 # 为什么问这个（中文说明）
    level: str = "open"           # open / guided / followup


@dataclass
class Plan:
    """一次会话的话题计划。"""

    opening: str = ""             # AI 的开场白（英文）
    topics: list[Topic] = field(default_factory=list)
    vocabulary: list[str] = field(default_factory=list)   # 可能用到的词
    background: str = ""          # 内容摘要（中文，供 AI 理解上下文）

    def to_dict(self) -> dict:
        return {
            "opening": self.opening,
            "background": self.background,
            "vocabulary": self.vocabulary,
            "topics": [asdict(t) for t in self.topics],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        return cls(
            opening=d.get("opening", ""),
            background=d.get("background", ""),
            vocabulary=list(d.get("vocabulary") or []),
            topics=[Topic(**{k: v for k, v in t.items()
                             if k in Topic.__dataclass_fields__})
                    for t in (d.get("topics") or [])],
        )


SYSTEM_PROMPT = """你是一位英语口语陪练，正在为一次真实的口语练习备课。

你的学生是中文母语的英语学习者。他的诉求是：
1. 他不太清楚该聊什么，所以你要主动引导他开口；
2. 但他更想多说话，所以你的引导要简短，把说话机会留给他；
3. 话题要贴近他的生活，让他有内容可说，而不是考试式提问。

给你一段素材（可能是主题、段落、文章或网页正文），
请设计一份对话计划。

要求：
- 产出 4~6 个话题，由浅入深；
- 每个话题的 prompt 是**一句英文**，用于引导他开口。
  必须是开放式的（用 what/why/how/tell me about），
  绝不要用只能回答 yes/no 的问题；
- 每个 prompt 要能让他说 3 句以上，而不是一个词；
- 优先设计「联系他自身经历」的问题
  （比如 "Have you ever...", "What do you usually do when..."）；
- 语言要口语化、自然，像朋友聊天，不要像考试；
- vocabulary 给出 5~8 个这个话题下他可能需要的英文词/短语。

严格按下面的 JSON 输出，不要有任何额外文字：
{
  "background": "用中文简述这段素材讲了什么（2-3句）",
  "opening": "AI 的开场白（英文，2-3句，自我介绍+引出话题+邀请他开口）",
  "vocabulary": ["phrase1", "phrase2", ...],
  "topics": [
    {"title": "话题的中文小标题",
     "prompt": "英文引导问句",
     "why": "中文说明为什么问这个",
     "level": "open"}
  ]
}"""


class Planner:
    """把输入内容转成话题计划。"""

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def plan(self, content: Content, *,
             profile_summary: str = "",
             n_topics: int = 5) -> Plan:
        """
        生成话题计划。

        profile_summary：来自历史画像的摘要（需求 6），
        有则注入，让话题更贴近用户本人。
        """
        if not content.ok:
            raise LLMError(f"输入内容不可用: {content.error}")

        user_parts = [f"素材类型：{content.kind}"]
        if content.title:
            user_parts.append(f"标题：{content.title}")
        user_parts.append(f"\n素材内容：\n{content.text[:8000]}")
        user_parts.append(f"\n请设计 {n_topics} 个话题。")

        if profile_summary.strip():
            user_parts.append(
                "\n\n【关于这位学生的已知信息】\n"
                f"{profile_summary}\n"
                "请把话题往他感兴趣的方向靠，让他更容易开口。")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

        data = self.llm.chat_json(messages, temperature=0.8, max_tokens=2500)
        plan = self._normalize(data, n_topics)

        if not plan.topics:
            raise LLMError("模型未产出任何话题")
        return plan

    @staticmethod
    def _normalize(data: dict, n_topics: int) -> Plan:
        """容错：模型偶尔会少字段或给错结构。"""
        if not isinstance(data, dict):
            raise LLMError(f"计划格式错误: {type(data).__name__}")

        raw_topics = data.get("topics") or []
        topics: list[Topic] = []
        for t in raw_topics:
            if isinstance(t, str):
                topics.append(Topic(title=t, prompt=t))
                continue
            if not isinstance(t, dict):
                continue
            prompt = (t.get("prompt") or t.get("question") or "").strip()
            if not prompt:
                continue
            topics.append(Topic(
                title=(t.get("title") or prompt)[:80],
                prompt=prompt,
                why=(t.get("why") or "").strip(),
                level=(t.get("level") or "open").strip() or "open"))

        vocab = [str(v).strip() for v in (data.get("vocabulary") or [])
                 if str(v).strip()]

        return Plan(
            opening=(data.get("opening") or "").strip(),
            background=(data.get("background") or "").strip(),
            vocabulary=vocab[:10],
            topics=topics[:n_topics],
        )


# ============================================================
#  话题质量校验
# ============================================================

# 开放式疑问词。含这些词的问题通常能让人展开说。
_OPEN_MARKERS = (
    "what", "why", "how", "tell me", "describe", "imagine",
    "which", "who", "where", "what's", "whats",
)
# 封闭式开头（只能答 yes/no），除非句中另有开放式疑问词。
_CLOSED_STARTERS = (
    "do you", "does", "did you", "is ", "are ", "was ", "were ",
    "have you", "has ", "can you", "could you", "would you",
    "will you", "should ",
)


def is_open_question(prompt: str) -> bool:
    """
    判断一句引导语是否「能让我多说」。

    只看首词会误判 —— 例如：
        "Have you ever felt tension ...? What happened?"
    以 Have 开头，但有 "What happened" 追问，实际是开放的。
    所以规则是：出现开放式疑问词即算开放。
    """
    p = (prompt or "").strip().lower()
    if not p:
        return False
    if any(m in p for m in _OPEN_MARKERS):
        return True
    return not any(p.startswith(s) for s in _CLOSED_STARTERS)


def validate_plan(plan: Plan) -> list[str]:
    """返回计划的问题列表（空列表表示没问题）。"""
    issues: list[str] = []
    if not plan.opening.strip():
        issues.append("缺少开场白")
    if len(plan.topics) < 3:
        issues.append(f"话题太少（{len(plan.topics)} 个，建议 4~6 个）")
    for i, t in enumerate(plan.topics, 1):
        if not t.prompt.strip():
            issues.append(f"话题 {i} 缺少引导语")
        elif not is_open_question(t.prompt):
            issues.append(f"话题 {i} 疑似封闭式（只能答 yes/no）: "
                          f"{t.prompt[:60]}")
    return issues


# ============================================================
#  注入对话的系统提示词
# ============================================================

def build_tutor_instructions(plan: Plan, *,
                             profile_summary: str = "",
                             minutes: int = 30) -> str:
    """
    把计划转成 Realtime 会话的系统提示词。

    这里集中体现需求 2/3/4/5：
      · 角色：陪练（需求 5）
      · 多让用户说（需求 2）
      · 纠错但不打断（需求 3）
      · 发音只纠明显错误（需求 4）
    """
    lines = [
        "You are a friendly one-on-one English speaking coach.",
        "Your student is a Chinese native speaker practising spoken English.",
        "",
        "## Your role",
        "- Be a patient practice partner, not an examiner.",
        "- Keep the conversation natural and warm, like talking with a friend.",
        "",
        "## Let the student talk (very important)",
        "- Your replies must be SHORT: 1-2 sentences at most.",
        "- Aim for the student to speak about 60% of the time.",
        "- Ask ONE question at a time. Never stack multiple questions.",
        "- Wait patiently. Do not fill silence with your own talking.",
        "- If the student gives a short answer, ask a gentle follow-up",
        "  instead of moving on immediately.",
        "",
        "## Correcting mistakes",
        "- Do NOT interrupt the student while they are speaking.",
        "- After they finish, you may naturally weave ONE correction into",
        "  your reply, like this:",
        "    \"Oh, you went to the park? Nice - by the way, we usually say",
        "     'I went', not 'I go'.\"",
        "- At most one correction per turn. Never lecture.",
        "- Prefer recasting (saying it correctly yourself) over explaining rules.",
        "- Suggest a more natural expression when the student sounds textbook-ish.",
        "",
        "## Pronunciation",
        "- Only correct CLEAR word-level mispronunciations that would confuse",
        "  a listener (e.g. 'think' said as 'sink').",
        "- Do NOT correct accents, slight mispronunciations, or intonation.",
        "- Never interrupt for pronunciation. Mention it at most once per",
        "  several turns, and only if it really matters.",
        "",
    ]

    if plan.background:
        lines += ["## Today's material", plan.background, ""]

    if plan.topics:
        lines.append("## Topics to explore (in rough order, stay flexible)")
        for i, t in enumerate(plan.topics, 1):
            lines.append(f"{i}. {t.prompt}")
        lines.append("")

    if plan.vocabulary:
        lines.append("## Vocabulary you may introduce when relevant")
        lines.append(", ".join(plan.vocabulary))
        lines.append("")

    if profile_summary.strip():
        lines += [
            "## What you already know about this student",
            profile_summary.strip(),
            "Use this to make the conversation personal. Do not recite it back.",
            "",
        ]

    lines += [
        f"## Session",
        f"- This session is about {minutes} minutes.",
        "- Start by greeting the student and introducing today's topic briefly,",
        "  then invite them to speak.",
        "- If the student is quiet or gives one-word answers, be encouraging",
        "  and offer a concrete example from your own (role-played) experience.",
    ]
    return "\n".join(lines)
