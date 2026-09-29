"""
个人画像：从对话中不断了解用户。

需求 6：你对我了解越多，就越清楚我的背景，
        从而更好地发起聊天，跟我聊我感兴趣的内容。

两个部分：
  · ProfileExtractor —— 从一次会话中抽取新事实（调模型）
  · ProfileStore     —— 累积、去重、注入（本地文件 + 数据库）

为什么同时用数据库和 Markdown：
  · 数据库便于程序查询、按置信度过滤
  · Markdown 便于人读、手工编辑（用户随时能改掉错误画像）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..core.llm import LLMClient
from ..storage.db import Database
from ..storage.models import ProfileFact

log = logging.getLogger(__name__)

# 画像分类。限定范围避免模型自由发挥导致数据杂乱。
CATEGORIES = {
    "background": "职业、居住地、教育、家庭等客观背景",
    "interest": "兴趣爱好、喜欢的话题",
    "preference": "学习偏好、沟通风格偏好",
    "goal": "英语学习目标、想提升的方向",
    "language": "反复出现的语言问题、常犯错误",
    "life": "生活习惯、日常安排",
}

EXTRACT_PROMPT = """你是一位细心的英语口语老师，正在记录对学生的了解。

学生会给你一段他刚说的话（语音转写的英文）。请从中提取
**关于这个人的、值得长期记住的事实**。

## 分类（category 只能取以下之一）
- background：职业、居住地、教育背景、家庭情况
- interest：兴趣爱好、喜欢的话题、常做的事
- preference：学习偏好、喜欢的沟通方式
- goal：英语学习目标、想提升的方向
- language：反复出现的语言错误（如"总是漏掉冠词"）
- life：生活习惯、日常安排

## 要求
1. **只提取明确说出来的事实**，不要推测、不要脑补。
   学生说"我昨天去公园跑步" → 可以记 interest:跑步。
   但没有说他住在哪个城市，就**不要**填居住地。
2. 一件事实一条，key 用简短的英文小写下划线形式
   （如 hobby、job、city、weak_tense）。
3. value 用中文（便于用户阅读），一个短语或短句。
4. confidence 表示你的确信程度（0~1）：
   - 直接明确陈述（"我是程序员"）→ 0.9
   - 顺带提到、可能是一次性的（"我昨天去了公园"）→ 0.5
5. **宁缺毋滥。** 如果这段话没有任何值得长期记住的个人信息，
   返回空数组 —— 这是很常见的情况。
   绝大多数寒暄、观点表达都不含个人信息。

严格按 JSON 输出：
{"facts": [{"category": "interest", "key": "hobby",
            "value": "跑步", "confidence": 0.8}]}"""


class ProfileExtractor:
    """从一轮发言里抽取关于用户的事实。"""

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def extract(self, texts: list[str]) -> list[ProfileFact]:
        """从若干句用户发言中抽取事实。"""
        joined = "\n".join(t.strip() for t in texts if t and t.strip())
        if len(joined) < 15:      # 太短没有信息量，省一次调用
            return []

        messages = [
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": f"学生说的话：\n{joined[:3000]}"},
        ]
        try:
            data = self.llm.chat_json(messages, temperature=0.2,
                                      max_tokens=800)
        except Exception as e:
            log.warning("画像抽取失败: %s", e)
            return []
        return self._normalize(data)

    @staticmethod
    def _normalize(data) -> list[ProfileFact]:
        if isinstance(data, dict):
            raw = data.get("facts") or []
        elif isinstance(data, list):
            raw = data
        else:
            return []

        out: list[ProfileFact] = []
        seen: set[tuple[str, str]] = set()
        for f in raw:
            if not isinstance(f, dict):
                continue
            cat = str(f.get("category") or "").strip().lower()
            key = str(f.get("key") or "").strip().lower()
            val = str(f.get("value") or "").strip()
            if cat not in CATEGORIES or not key or not val:
                continue
            # key 规范化，避免 "Hobby" / "hobby " 被当成两条
            key = key.replace(" ", "_")[:40]
            if (cat, key) in seen:
                continue
            seen.add((cat, key))

            try:
                conf = float(f.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            conf = max(0.0, min(1.0, conf))

            out.append(ProfileFact(category=cat, key=key, value=val,
                                   confidence=conf))
        return out


# ============================================================
#  累积与注入
# ============================================================

# 只有够可信的事实才注入提示词，避免误导模型
INJECT_MIN_CONFIDENCE = 0.6
# 注入总量上限，避免提示词过长
MAX_INJECT_FACTS = 25


def _group_by_category(facts: list[ProfileFact],
                       ) -> dict[str, list[ProfileFact]]:
    """按分类分组，组内按置信度从高到低。

    summary() 和 write_markdown() 都要这个，
    抽出来避免两处各写一遍、后续改漏一处。
    """
    by_cat: dict[str, list[ProfileFact]] = {}
    for f in facts:
        by_cat.setdefault(f.category, []).append(f)
    for group in by_cat.values():
        group.sort(key=lambda x: -x.confidence)
    return by_cat


class ProfileStore:
    """画像的读写与注入。"""

    def __init__(self, db: Database, profiles_dir: Path):
        self.db = db
        self.dir = Path(profiles_dir)

    # ---------- 写入 ----------

    def absorb(self, facts: list[ProfileFact],
               session_id: Optional[int] = None) -> int:
        """把新抽取的事实并入画像。返回新增/更新的条数。"""
        if not facts:
            return 0
        for f in facts:
            f.source_session_id = session_id
            self.db.upsert_fact(f)
        return len(facts)

    # ---------- 读取 ----------

    def summary(self, min_confidence: float = INJECT_MIN_CONFIDENCE,
                max_facts: int = MAX_INJECT_FACTS) -> str:
        """
        生成给模型看的画像摘要。

        按分类组织，只保留置信度够高的，总量有上限。
        """
        facts = self.db.list_facts(min_confidence=min_confidence)
        if not facts:
            return ""

        by_cat = _group_by_category(facts)

        lines: list[str] = []
        used = 0
        for cat, label in CATEGORIES.items():
            group = by_cat.get(cat)
            if not group:
                continue
            items = []
            for f in group:
                if used >= max_facts:
                    break
                items.append(f.value)
                used += 1
            if items:
                lines.append(f"- {label}：{'；'.join(items)}")
        return "\n".join(lines)

    def write_markdown(self, path: Optional[Path] = None) -> Path:
        """
        导出为 Markdown，供人阅读和手工修改。

        这是"用户能纠正 AI 对自己的误解"的出口 ——
        模型记错了，直接改文件即可。
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        p = path or (self.dir / "profile.md")

        facts = self.db.list_facts()
        by_cat = _group_by_category(facts)

        lines = [
            "# 我的英语学习画像",
            "",
            "> 这份文件由程序自动生成，记录了 AI 对你的了解。",
            "> 它是**给人看的**：如果某条记错了，直接删掉或修改即可。",
            "> 也可以直接补充你希望 AI 知道的信息。",
            "",
            f"共 {len(facts)} 条。",
            "",
        ]
        for cat, label in CATEGORIES.items():
            group = by_cat.get(cat)
            if not group:
                continue
            lines.append(f"## {label}")
            lines.append("")
            for f in group:
                star = "●" if f.confidence >= 0.8 else \
                       "○" if f.confidence >= 0.6 else "·"
                lines.append(f"- {star} **{f.value}** "
                             f"`{f.key}` (置信 {f.confidence:.2f})")
            lines.append("")

        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    # ---------- 语言问题 ----------

    def language_issues(self, limit: int = 5) -> list[str]:
        """
        从历史纠错中归纳反复出现的问题。

        比单次纠错更有价值 —— "你总是漏冠词" 比
        "这一句漏了 the" 有用得多。
        """
        words = self.db.top_error_words(limit=limit)
        facts = [f for f in self.db.list_facts(min_confidence=0.6)
                 if f.category == "language"]
        out = [f"{f.value}" for f in facts[:limit]]
        if words:
            top = "、".join(f"{w}（{n}次）" for w, n in words[:3])
            out.append(f"发音上反复出现问题：{top}")
        return out
