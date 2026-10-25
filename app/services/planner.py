"""
话题规划：把输入内容变成一份「聊什么」的计划。

需求 2：我不太清楚该聊什么，所以你要主动引导我说，
        但要多给我说的机会。

因此产出的话题必须是**开放式、能让我多说**的，
而不是一问一答就结束的封闭问题。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..core.content import Content
from ..core.languages import Language, get_language
from ..core.llm import LLMClient, LLMError

log = logging.getLogger(__name__)


@dataclass
class Topic:
    """一个可聊的话题。"""

    title: str                    # 话题标题（中文，便于我理解）
    prompt: str                   # 引导我开口的目标语言句子
    why: str = ""                 # 为什么问这个（中文说明）
    level: str = "open"           # open / guided / followup


@dataclass
class Plan:
    """一次会话的话题计划。"""

    opening: str = ""             # AI 的开场白（目标语言）
    topics: list[Topic] = field(default_factory=list)
    vocabulary: list[str] = field(default_factory=list)   # 可能用到的词
    background: str = ""          # 内容摘要（中文，供 AI 理解上下文）
    lang: str = "en"              # 练习语言：en / ko（见 core/languages.py）

    def to_dict(self) -> dict:
        return {
            "opening": self.opening,
            "background": self.background,
            "vocabulary": self.vocabulary,
            "topics": [asdict(t) for t in self.topics],
            "lang": self.lang,
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
            lang=d.get("lang", "en"),
        )


def build_system_prompt(lang: Language | None = None) -> str:
    """
    备课提示词。

    语言名、引导句语言、词汇语言都从 Language 派生 ——
    最初这里写死「英语」，导致输入韩语素材时模型仍生成英文问题。
    """
    lang = lang or get_language("en")
    return f"""你是一位{lang.label}口语陪练，正在为一次真实的口语练习备课。

你的学生是中文母语的{lang.label}学习者。他的诉求是：
1. 他不太清楚该聊什么，所以你要主动引导他开口；
2. 但他更想多说话，所以你的引导要简短，把说话机会留给他；
3. 话题要贴近他的生活，让他有内容可说，而不是考试式提问。

给你一段素材（可能是主题、段落、文章或网页正文），
请设计一份对话计划。

要求：
- 产出 4~6 个话题，由浅入深；
- 每个话题的 prompt 是**一句{lang.label}**（用 {lang.native} 写），
  用于引导他开口。
  必须是开放式的，绝不要用只能回答 yes/no 的问题；
- 每个 prompt 要能让他说 3 句以上，而不是一个词；
- 优先设计「联系他自身经历」的问题；
- 语言要口语化、自然，像朋友聊天，不要像考试；
- vocabulary 给出 5~8 个这个话题下他可能需要的{lang.label}词/短语。
{lang.planner_note}

严格按下面的 JSON 输出，不要有任何额外文字：
{{
  "background": "用中文简述这段素材讲了什么（2-3句）",
  "opening": "AI 的开场白（{lang.label}，2-3句，自我介绍+引出话题+邀请他开口）",
  "vocabulary": ["phrase1", "phrase2", ...],
  "topics": [
    {{"title": "话题的中文小标题",
     "prompt": "{lang.label}引导问句",
     "why": "中文说明为什么问这个",
     "level": "open"}}
  ]
}}"""


class Planner:
    """把输入内容转成话题计划。"""

    # 备课的 token 预算。
    #
    # 不能用「够写一份 JSON」来估：推理模型（比如 glm-5.3）会先写
    # reasoning_content，再写正文，两者共用同一个 max_tokens。
    # 实测 2500 的预算下，2498 个 token 花在思考上，finish_reason=length，
    # 正文只剩 "{\n" —— 解析失败，整个「生成话题」稳定报错。
    PLAN_MAX_TOKENS = 6000
    PLAN_RETRY_MAX_TOKENS = 12000

    def __init__(self, llm: Optional[LLMClient] = None):
        # 备课要一次写出一份完整计划，比纠错一句话重得多。
        # 默认 60 秒在推理模型上会读超时（实测 qwen3.8-flash 备课
        # 单次接近一分钟），表现为「生成话题」偶发 502：
        # 话题规划失败：TimeoutError: The read operation timed out。
        self.llm = llm or LLMClient(timeout=180)

    def plan(self, content: Content, *,
             profile_summary: str = "",
             n_topics: int = 5,
             lang: Language | str = "en") -> Plan:
        """
        生成话题计划。

        profile_summary：来自历史画像的摘要（需求 6），
        有则注入，让话题更贴近用户本人。
        lang：本次练习的目标语言（en / ko），决定引导句和词汇的语言。
        """
        if not content.ok:
            raise LLMError(f"输入内容不可用: {content.error}")

        language = lang if isinstance(lang, Language) else get_language(lang)

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
            {"role": "system", "content": build_system_prompt(language)},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

        data = self._chat_json(messages)
        plan = self._normalize(data, n_topics, language.code)

        if not plan.topics:
            raise LLMError("模型未产出任何话题")
        return plan

    def _chat_json(self, messages: list[dict]) -> dict:
        """要求模型返回 JSON；被截断时提高预算重试一次。"""
        try:
            return self.llm.chat_json(
                messages, temperature=0.8,
                max_tokens=self.PLAN_MAX_TOKENS)
        except LLMError as e:
            # 只有「解析不了 JSON」才值得重试；鉴权/配额错误重试也没用，
            # 反而多打一次接口。
            if "无法解析为 JSON" not in str(e):
                raise
            log.warning("备课 JSON 被截断，提高预算重试: %s", e)
            return self.llm.chat_json(
                messages, temperature=0.8,
                max_tokens=self.PLAN_RETRY_MAX_TOKENS)

    @staticmethod
    def _normalize(data: dict, n_topics: int, lang: str = "en") -> Plan:
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
            lang=lang,
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
        # is_open_question 是基于英文疑问词的启发式判断，
        # 拿它去判韩语只会把每一条都误报成「封闭式」。
        elif plan.lang == "en" and not is_open_question(t.prompt):
            issues.append(f"话题 {i} 疑似封闭式（只能答 yes/no）: "
                          f"{t.prompt[:60]}")
    return issues


# ============================================================
#  注入对话的系统提示词
# ============================================================

def build_tutor_instructions(plan: Plan, *,
                             profile_summary: str = "",
                             minutes: int = 30,
                             lang: Language | str | None = None) -> str:
    """
    把计划转成 Realtime 会话的系统提示词。

    这里集中体现需求 2/3/4/5：
      · 角色：陪练（需求 5）
      · 多让用户说（需求 2）
      · 纠错但不打断（需求 3）
      · 发音只纠明显错误（需求 4）

    lang 不传时用计划里的语言，避免「备课选韩语、开练却按英语提示词」
    这种前后不一致。
    """
    language = (lang if isinstance(lang, Language)
                else get_language(lang or plan.lang))
    lines = [
        f"You are a friendly one-on-one {language.english} speaking coach.",
        f"Your student is a Chinese native speaker practising spoken "
        f"{language.english}.",
        "",
        "## Language",
        f"- Speak ONLY {language.english} ({language.native}) unless the "
        f"student explicitly asks you to switch.",
        f"- If the student answers in Chinese, gently encourage them to try "
        f"in {language.english}.",
        "- Keep your wording natural and conversational, not textbook-style.",
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
        "  your reply: say the correct form in your own sentence, then add",
        "  a brief hint like \"by the way, we usually say ...\".",
        "- At most one correction per turn. Never lecture.",
        "- Prefer recasting (saying it correctly yourself) over explaining rules.",
        "- Suggest a more natural expression when the student sounds textbook-ish.",
        "",
        "## Pronunciation",
        "- Only correct CLEAR word-level mispronunciations that would confuse",
        "  a listener.",
        "- Do NOT correct accents, slight mispronunciations, or intonation.",
        "- Never interrupt for pronunciation. Mention it at most once per",
        "  several turns, and only if it really matters.",
        "- You only see transcripts, so be conservative: if the transcript",
        "  looks fine, do not invent a pronunciation problem.",
        "",
    ]

    if language.code == "ko":
        lines += [
            "## Korean specifics",
            "- Use polite casual speech (해요체) by default, like a friendly",
            "  tutor. Do not use 습니다体 or 반말 unless the student starts it.",
            "- If the student mixes up particles (은/는, 이/가, 을/를) or verb",
            "  endings, recast once, naturally. Do not explain grammar rules.",
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
