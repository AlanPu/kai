"""
复习：把散落在各次会话里的纠错，归纳成「反复犯的问题」。

为什么需要这个模块
------------------
`corrections` 表回答的是「**这一次**我说错了什么」。
用户真正要复习的是「**我老是**犯哪几类错」，以及「哪几个固定说法
我总说不对」。

真实数据说明了差距：同一句 `I went to the.` 在库里出现了 7 次，
是 7 条互不相干的纠错记录，散在 7 个会话里。单看任何一次都只是
「这次没说完整」，看不出它已经重复了 7 次 —— 而这恰恰是最该
被指出来的信息。

两层结构
--------
  ReviewHabit —— 问题类型（"句子说完整"），带累计次数
  ReviewItem  —— 挂在类型下的具体说法（"one ... at a time"）

分类用**规则**而不是让模型自由发挥
--------------------------------
reason: 分类结果是聚合的分组键。如果让模型每次自己起名字，
同一类"漏主语"这次叫 missing_subject、下次叫 no_subject，
计数就永远停在 1，整个复习页失去意义。
所以错误类型（kind/severity + 原句特征）走确定性规则，
模型只在「抽固定搭配」这件真正需要理解语义的事上出场。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from ..core.llm import LLMClient
from ..storage.db import Database
from ..storage.models import (HABIT_LABELS, Correction, ReviewHabitRow,
                              ReviewItem)

log = logging.getLogger(__name__)

# 每个类型给一句"怎么办"。规则判定出类型后直接用，
# 不必每次再问模型 —— 省调用，也保证说法一致。
HABIT_ADVICE: dict[str, str] = {
    "fragment": "开口前先想好主谓宾。哪怕说慢一点，也要把一句话说完，"
                "不要停在 the / and / because 这种地方。",
    "duplication": "起头之后别回头改。重复的词（use use、make make）"
                   "说明在边想边说，可以先用 well / let me think 占住，"
                   "想清楚再开口。",
    "missing_subject": "英语句子必须有主语。中文能省略『我』，英语不行 ——"
                       "Is a very... 要补成 It is a very...",
    "tense": "先定时间，再定时态。出现 yesterday / last week 就立刻用过去式。",
    "article": "单数可数名词前基本要有 a/an/the。泛指用 a，特指用 the。",
    "preposition": "介词搭配要整个短语一起记，不要按中文逐字翻。",
    "agreement": "注意第三人称单数和名词复数：he makes、seven days。",
    "word_choice": "换词之前先想母语者会不会这么说，而不是中文直译。",
    "collocation": "固定搭配整块记：one ... at a time、think about it。",
    "chinglish": "别逐字翻译中文句式。先想英语里这个意思怎么说，再开口。",
    "pronunciation": "把读错的词单独拎出来念几遍，直到不用想。",
    "fluency": "减少填充词，把零散短语连成完整句子。",
}

# 归纳出来的类型，按「先说什么」排序 —— 每次练习最该听到的提示
HABIT_PRIORITY = [
    "fragment", "duplication", "missing_subject", "chinglish",
    "collocation", "word_choice", "tense", "article", "preposition",
    "agreement", "pronunciation", "fluency",
]


# ============================================================
#  文本规范化 / 去重键
# ============================================================

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """
    归一化用于去重：小写、去标点、合并空白。

    "I went to the." 和 "i went to the" 必须归到同一条，
    否则同一句话的多次记录会各占一行，复习页被刷屏。
    """
    s = (s or "").strip().lower()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def dedup_key_for(original: str, suggestion: str = "") -> str:
    """
    一条复习条目的去重键。

    只用 original：同一句错话即使模型给了不同建议，
    对用户来说也是"同一个问题"，应该合并成一条。
    """
    return normalize_text(original)[:200]


# ============================================================
#  规则分类
# ============================================================

# 句子在这些词上断掉 —— 典型"说到一半"。
# 包括：冠词、连接词、介词、物主代词、助动词，
# 以及**及物动词后缺宾语**的介词（think about / talk about / depend on）。
_DANGLING_TAIL = re.compile(
    r"\b(the|a|an|and|but|or|because|so|that|to|of|in|on|at|for|with|"
    r"about|from|into|over|under|between|without|"
    r"my|your|his|her|its|our|their|is|are|was|were|will|can|i)\s*$",
    re.IGNORECASE)

# 明显的中式直译信号
_CHINGLISH_HINTS = (
    "i what i", "very like", "how to say", "i very", "open the light",
    "close the light", "i think so that", "very enjoy",
)


def _has_duplicate_word(text: str) -> bool:
    """相邻重复词（use use / make make / one one）。"""
    words = normalize_text(text).split()
    return any(a == b and len(a) > 1 for a, b in zip(words, words[1:]))


def _is_fragment(text: str) -> bool:
    """
    判断是不是"话没说完"。

    只认**真实的截断信号**，不靠"句子短"来猜：

      1. 以连接词/冠词/介词/物主代词/助动词收尾
         （"I don't need to think about" / "I went to the."）
      2. 以逗号收尾，且逗号后还有新起的片段
         （"The meeting, and then."）
      3. 两个及以上句号/逗号切出的碎块，整体没有主谓

    ⚠️ 曾经用「词数 ≤4 且无动词」当兜底，结果把
    "an old man"、"good idea"、"in the morning"
    这类**短名词短语**全判成了 fragment —— 它们其实是
    用词/冠词/介词问题。片段判断必须看截断特征，不能看长度。
    """
    t = (text or "").strip()
    if not t:
        return False

    # 信号 1：悬空收尾
    if _DANGLING_TAIL.search(t):
        return True

    words = normalize_text(t).split()
    if not words:
        return False

    # 信号 2：截断在末尾的标点上，且话明显没完
    #   "The train leaves at six." / "Let me think about it,"
    #   与 "The meeting, and then." —— 逗号切碎、没有主谓
    if t.endswith((".", "…", ",")) and len(words) >= 3:
        tail = words[-1]
        # 以 ... me / it / them / the 等收尾，说明宾语没跟上
        if tail in {"me", "it", "them", "us", "him", "her", "the", "a",
                    "an", "and", "but", "or", "because", "so", "that",
                    "to", "of", "in", "on", "at", "for", "with",
                    "my", "your", "his", "our", "their"}:
            return True

    # 信号 3：句中被切成一堆碎块，且没有像样的谓语
    #   "The plan, results." / "Fine. If I just get the ticket, but."
    chunks = [c for c in re.split(r"[.,;!?]+", t) if c.strip()]
    if len(chunks) >= 2:
        lowered = normalize_text(t)
        has_subject_verb = re.search(
            r"\b(i|you|he|she|we|they|it)\s+"
            r"(am|is|are|was|were|have|has|had|do|does|did|will|would|"
            r"can|could|should|may|might|must|'m|'s|'re|'ll|'ve|'d)\b",
            lowered)
        if not has_subject_verb and len(words) <= 8:
            return True

    return False


def classify(c: Correction) -> Optional[str]:
    """
    把一个纠错项判进某个问题类型。返回 None 表示不值得复习。

    这是**确定性规则**，同样的输入永远得到同样的分类 ——
    复习页的计数才有意义（见模块开头的说明）。
    """
    text = c.original or ""
    low = normalize_text(text)
    if not low:
        return None

    # 发音类直接归发音，不再往下判
    if c.kind == "pronunciation":
        return "pronunciation" if c.word else None

    # 重复词先于"没说完"判：一句重复重启的话
    # （"helps me to use use, to, use the"）往往同时带着悬空收尾，
    # 但用户真正该改的是**边想边说、反复重启**这个习惯 ——
    # 提示"把话说完整"对他没有帮助。
    if _has_duplicate_word(text):
        return "duplication"

    # 说话不完整：优先于其它语法点。一句话都没说完时，
    # 再纠冠词/时态没有意义，用户该先学会说完整。
    if _is_fragment(text):
        return "fragment"

    # 中式直译（先于普通用词问题）
    if any(h in low for h in _CHINGLISH_HINTS):
        return "chinglish"
    # 原句里混着中文，多半是逐字直译的痕迹
    if re.search(r"[\u4e00-\u9fff]", text):
        return "chinglish"

    # 缺主语：以 Is/Are/Will/Can + 动词开头，且没有主语
    if re.match(r"^(is|are|was|were|will|can|do|does|am)\b", low) and \
            not re.match(r"^(is|are|was|were)\s+(it|this|that|there)\b", low):
        return "missing_subject"

    if c.kind == "fluency":
        return "fluency"

    # 语法类：从解释里辨认具体语法点。
    # 解释是中文（提示词要求的），关键词稳定。
    #
    # 顺序有讲究：越具体的语法点越先判。比如"时态"的解释里
    # 常同时出现"主语"二字，先判主语就会把时态问题归错。
    expl = c.explanation or ""
    checks = (
        ("tense", ("时态", "过去式", "现在完成", "进行时")),
        ("article", ("冠词",)),
        ("agreement", ("单复数", "复数", "第三人称", "主谓一致")),
        ("preposition", ("介词", "搭配")),
        ("missing_subject", ("主语", "谓语")),
        ("fragment", ("不完整", "缺少", "语序", "片段")),
    )
    for habit, keywords in checks:
        if any(k in expl for k in keywords):
            return habit

    # 解释里没写关键词时，看原句表面特征
    if re.search(r"\ba\b|\ban\b|\bthe\b", low) and c.kind == "grammar":
        return "article"

    # 词汇类 → 用词 / 固定搭配
    if c.kind == "vocabulary":
        return "collocation" if _looks_like_collocation(c) else "word_choice"

    # 兜底：不强行归类，避免噪音污染统计
    return None


def _looks_like_collocation(c: Correction) -> bool:
    """
    判断一组 original→suggestion 是不是「固定说法」问题。

    特征：都很短（短语而非整句），说明改的是一个搭配块。
    """
    o = normalize_text(c.original).split()
    s = normalize_text(c.suggestion).split()
    return len(o) <= 6 and len(s) <= 8


# ============================================================
#  固定搭配抽取
# ============================================================

# 搭配是词汇级的「块」，不是句子。超过这个词数一定是整句，
# 收进来只会让「搭配」列变成 suggestion 的复读机。
MAX_COLLOCATION_WORDS = 5

# 一句完整的句子通常有这些标志；带这些的不能当搭配。
_CLAUSE_MARKERS = (
    " i ", " you ", " he ", " she ", " we ", " they ", " it ",
    " is ", " are ", " was ", " were ", " will ", " would ",
    " can ", " could ", " should ", " have ", " has ", " had ",
)


def guess_collocation(original: str, suggestion: str) -> Optional[str]:
    """
    从 original → suggestion 里猜出「该记的固定说法」。

    只有确认建议是**一个短语**（而不是重写出来的完整句子）时才返回。
    否则返回 None —— 宁可没有搭配，也不要把整句塞进这一列。

    反例（都必须被拒绝）：
      'The meeting, and then.'   → 'The meeting went well.'
      'I went to the.'           → 'I went to the park.'
    正例（应该通过）：
      'one page by one one page' → 'one page at a time'
      'tennis game'              → 'tennis'
    """
    o = normalize_text(original).split()
    s = normalize_text(suggestion).split()
    if not o or not s:
        return None

    # 长度上限：超过 5 个词是句子，不是搭配
    if len(s) > MAX_COLLOCATION_WORDS:
        return None
    if abs(len(o) - len(s)) > 3:
        return None

    # 含主谓结构的完整小句 → 不是搭配
    padded = f" {' '.join(s)} "
    if any(m in padded for m in _CLAUSE_MARKERS):
        return None

    # 至少一半的词相同，才算"同一个短语被修好"
    common = set(o) & set(s)
    if len(common) < max(1, min(len(o), len(s)) // 2):
        return None

    return suggestion.strip()[:80]


# ============================================================
#  归纳主流程
# ============================================================

@dataclass
class HabitBucket:
    """归类过程中的中间结果。"""

    habit: str
    corrections: list[Correction] = field(default_factory=list)

    @property
    def occurrences(self) -> int:
        return len(self.corrections)


def bucketize(corrections: list[Correction]) -> dict[str, HabitBucket]:
    """
    把纠错按问题类型分组。

    只有出现 **2 次及以上** 的才算"反复犯的问题" ——
    偶发一次的错误没有复习价值，混进来只会让清单变长、
    真正的老毛病被淹没。这是本功能的核心判断。
    """
    raw: dict[str, HabitBucket] = {}
    for c in corrections:
        if not c.should_show():
            continue
        h = classify(c)
        if not h:
            continue
        raw.setdefault(h, HabitBucket(habit=h)).corrections.append(c)

    return {h: b for h, b in raw.items() if b.occurrences >= 2}


def summarize(corrections: list[Correction]) -> list[ReviewHabitRow]:
    """
    归类并生成复习用的 habit 行（未落库）。

    时间取 first_seen / last_seen：让用户看到"这个问题跟了我多久"。
    """
    buckets = bucketize(corrections)
    out: list[ReviewHabitRow] = []
    for habit, b in buckets.items():
        dates = sorted(x.created_at for x in b.corrections if x.created_at)
        out.append(ReviewHabitRow(
            habit=habit,
            title=HABIT_LABELS.get(habit, habit),
            advice=HABIT_ADVICE.get(habit),
            occurrences=b.occurrences,
            first_seen=dates[0] if dates else None,
            last_seen=dates[-1] if dates else None))
    out.sort(key=lambda h: (-h.occurrences,
                            HABIT_PRIORITY.index(h.habit)
                            if h.habit in HABIT_PRIORITY else 99))
    return out


def top_items(corrections: list[Correction], limit: int = 200
              ) -> list[tuple[Optional[str], ReviewItem]]:
    """
    从纠错里挑出值得单独复习的**具体说法**。

    返回 (habit, item) 对 —— habit 让调用方把条目挂到对应类型下。

    规则：
      · 同一句（normalize 后）合并计数，只留一条
      · 出现 ≥2 次的优先（真的在重复）
      · 固定搭配优先于整句重写
    """
    groups: dict[str, list[Correction]] = {}
    for c in corrections:
        if not c.should_show():
            continue
        key = dedup_key_for(c.original)
        if not key:
            continue
        groups.setdefault(key, []).append(c)

    out: list[tuple[int, Optional[str], ReviewItem]] = []
    for key, group in groups.items():
        # 建议取出现最多的那个 —— 模型对同一句可能给出不同改法，
        # 多数派通常更稳定
        sugg: dict[str, int] = {}
        for c in group:
            sugg[c.suggestion] = sugg.get(c.suggestion, 0) + 1
        best = max(sugg.items(), key=lambda kv: kv[1])[0]

        # 改了等于没改（建议与原句完全相同）→ 没有复习价值。
        # 这类多半是模型复述原句，收进来只会让人觉得清单很水。
        if normalize_text(best) == key:
            continue

        # 太短或只剩标点差异的（'and.' → 'and'）也不值得占一行
        if len(key) < 3:
            continue

        dates = sorted(x.created_at for x in group if x.created_at)
        habit = classify(group[0])
        colloc = guess_collocation(group[0].original, best)
        note = next((c.explanation for c in group if c.explanation), None)

        out.append((len(group), habit, ReviewItem(
            dedup_key=key,
            original=group[0].original.strip()[:300],
            suggestion=best.strip()[:300],
            note=note,
            collocation=colloc,
            occurrences=len(group),
            first_seen=dates[0] if dates else None,
            last_seen=dates[-1] if dates else None,
            source_session_id=group[0].session_id,
        )))

    # 出现次数多的排前面；同次数时，带固定搭配的更值得看
    out.sort(key=lambda t: (-t[0], 0 if t[2].collocation else 1))
    return [(h, it) for _, h, it in out[:limit]]


class ReviewBuilder:
    """
    把某个用户的纠错历史归纳成复习清单。

    幂等：重复调用不会产生重复条目（靠 dedup_key 与 (user_id, habit)
    唯一约束），所以可以随时重建。
    """

    def __init__(self, db: Database, llm: Optional[LLMClient] = None):
        self.db = db
        # llm 目前保留给「把整句重写归纳得更自然」用；规则已能覆盖
        # 主要场景，所以默认不强制需要它。
        self.llm = llm

    def build(self, user_id: int, *, reset: bool = False,
              min_occurrences: int = 2) -> dict:
        """
        归纳某用户的复习清单。

        reset=True 时先清空重建（回填历史用）；
        否则增量累加（每次练习结束后调用）。
        返回统计，供接口/脚本显示。
        """
        if reset:
            self.db.clear_review(user_id)

        corrections = self.db.list_corrections_for_user(user_id)
        habits = summarize(corrections)
        # 允许调用方调高门槛（只看重复 ≥3 次的老毛病）
        habits = [h for h in habits if h.occurrences >= min_occurrences]

        habit_ids: dict[str, int] = {}
        for h in habits:
            h.user_id = user_id
            habit_ids[h.habit] = self.db.upsert_review_habit(h)

        items = top_items(corrections)
        kept = 0
        for h, it in items:
            if h not in habit_ids:
                continue          # 类型没进清单 → 条目也不单独列
            it.user_id = user_id
            it.habit_id = habit_ids[h]
            self.db.upsert_review_item(it)
            kept += 1

        return {
            "habits": len(habit_ids),
            "items": kept,
            "corrections": len(corrections),
            "counts": self.db.count_review(user_id),
        }
