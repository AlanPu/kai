"""
数据模型：与数据库表一一对应的类型定义。

用 dataclass 而非裸 dict，好处是字段名写错时立刻报错，
而不是等到运行期才发现 KeyError。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

InputKind = Literal["topic", "passage", "article", "url"]
Role = Literal["user", "assistant"]
CorrectionKind = Literal["grammar", "vocabulary", "pronunciation", "fluency"]
Severity = Literal["critical", "minor", "ignore"]
SessionStatus = Literal["active", "finished", "aborted"]


@dataclass
class Session:
    """一次完整的练习。"""

    id: Optional[int] = None
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
    category: str = ""
    key: str = ""
    value: str = ""
    confidence: float = 0.5
    source_session_id: Optional[int] = None
    created_at: str = ""
    updated_at: str = ""
