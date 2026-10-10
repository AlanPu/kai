"""
数据模型：与数据库表一一对应的类型定义。

用 dataclass 而非裸 dict，好处是字段名写错时立刻报错，
而不是等到运行期才发现 KeyError。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

InputKind = Literal["topic", "passage", "article", "url"]
Role = Literal["user", "assistant"]
CorrectionKind = Literal["grammar", "vocabulary", "pronunciation", "fluency"]
Severity = Literal["critical", "minor", "ignore"]
SessionStatus = Literal["active", "finished", "aborted"]


@dataclass
class User:
    """
    一个练口语的人。

    多人共用一台机器时，声纹、画像、历史都按用户隔离。
    声纹单独存成文件（data/voiceprints/<id>.json）而不是塞进
    数据库：它是 192 维浮点数组，放文件里便于直接查看和备份。
    """

    id: Optional[int] = None
    name: str = ""
    avatar: Optional[str] = None            # 单个 emoji，纯显示
    has_voiceprint: bool = False
    voiceprint_quality: Optional[float] = None
    voiceprint_samples: Optional[int] = None
    created_at: str = ""
    last_used_at: Optional[str] = None

    def voiceprint_path(self, base_dir) -> "Path":  # noqa: F821
        """这个用户的声纹文件位置。用 id 命名，避免中文名变成文件名问题。"""
        from pathlib import Path
        return Path(base_dir) / f"{self.id}.json"


@dataclass
class Session:
    """一次完整的练习。"""

    id: Optional[int] = None
    user_id: int = 0
    started_at: str = ""
    ended_at: Optional[str] = None
    input_kind: InputKind = "topic"
    input_raw: str = ""
    input_title: Optional[str] = None
    input_content: Optional[str] = None
    plan_json: Optional[str] = None
    duration_sec: Optional[int] = None
    turn_count: int = 0
    user_char_count: int = 0
    ai_char_count: int = 0
    status: SessionStatus = "active"
    usage_json: Optional[str] = None


@dataclass
class Turn:
    """会话中的一轮发言。"""

    id: Optional[int] = None
    session_id: int = 0
    seq: int = 0
    role: Role = "user"
    text: str = ""
    audio_ms: Optional[int] = None
    created_at: str = ""


@dataclass
class Correction:
    """一条纠错建议。"""

    id: Optional[int] = None
    session_id: int = 0
    turn_id: Optional[int] = None
    kind: CorrectionKind = "grammar"
    severity: Severity = "minor"
    original: str = ""
    suggestion: str = ""
    explanation: Optional[str] = None
    word: Optional[str] = None
    phonetic: Optional[str] = None
    shown: bool = False
    created_at: str = ""

    def should_show(self) -> bool:
        """
        是否应该展示给用户。

        对应需求 4：发音问题只在 critical（明显错误）时纠正，
        minor（口音/不标准）默认不打扰。语法类则 critical/minor 都值得提。
        """
        if self.severity == "ignore":
            return False
        if self.kind == "pronunciation":
            return self.severity == "critical"
        return self.severity in ("critical", "minor")


@dataclass
class ProfileFact:
    """个人画像中的一条事实。"""

    id: Optional[int] = None
    user_id: int = 0
    category: str = ""
    key: str = ""
    value: str = ""
    confidence: float = 0.5
    source_session_id: Optional[int] = None
    created_at: str = ""
    updated_at: str = ""


# ============================================================
#  复习：反复犯的问题
# ============================================================

# 受控词表。加新类型时**必须**同步改这里和 schema.sql 的注释 ——
# 它是聚合的分组键，放任自由文本会让同类问题被拆散。
ReviewHabit = Literal[
    "fragment", "duplication", "missing_subject", "tense", "article",
    "preposition", "agreement", "word_choice", "collocation",
    "chinglish", "pronunciation", "fluency",
]

# 类型 → 中文标签。界面和导出都用它，避免各处硬编码中文。
HABIT_LABELS: dict[str, str] = {
    "fragment": "句子说完整",
    "duplication": "别重复、别重启",
    "missing_subject": "补上主语",
    "tense": "时态",
    "article": "冠词",
    "preposition": "介词",
    "agreement": "主谓一致 / 单复数",
    "word_choice": "用词更地道",
    "collocation": "固定搭配",
    "chinglish": "别逐字直译",
    "pronunciation": "明显发音错误",
    "fluency": "少啰嗦、少填充词",
}


@dataclass
class ReviewHabitRow:
    """归纳出的一类反复出现的问题。"""

    id: Optional[int] = None
    user_id: int = 0
    habit: str = "grammar"
    title: str = ""
    advice: Optional[str] = None
    occurrences: int = 0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    mastered: bool = False
    mastered_at: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    @property
    def label(self) -> str:
        """中文标签，未知类型回退到原始值而不是显示空白。"""
        return HABIT_LABELS.get(self.habit, self.habit)


@dataclass
class ReviewItem:
    """一条要复习的具体说法（通常是一组固定搭配）。"""

    id: Optional[int] = None
    user_id: int = 0
    habit_id: Optional[int] = None
    dedup_key: str = ""
    original: str = ""
    suggestion: str = ""
    note: Optional[str] = None
    collocation: Optional[str] = None
    occurrences: int = 0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    mastered: bool = False
    mastered_at: Optional[str] = None
    source_session_id: Optional[int] = None
    created_at: str = ""
    updated_at: str = ""
