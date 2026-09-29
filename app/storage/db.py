"""
数据库访问层。

上层代码只使用本模块的函数，不直接写 SQL ——
这样 schema 变更时只需改这里，不会波及业务逻辑。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import (Correction, InputKind, ProfileFact, Session,
                     SessionStatus, Turn)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class Database:
    """SQLite 封装。用作上下文管理器时自动提交/回滚。"""

    def __init__(self, path: str | Path = "data/app.db"):
        self.path = Path(path)
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")

    # ---------- 生命周期 ----------

    def init_schema(self) -> None:
        """建表（幂等）。"""
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.close()

    # ---------- 会话 ----------

    def create_session(self, input_kind: InputKind, input_raw: str,
                       input_title: Optional[str] = None,
                       input_content: Optional[str] = None) -> int:
        cur = self.conn.execute(
            """INSERT INTO sessions
               (started_at, input_kind, input_raw, input_title, input_content)
               VALUES (?,?,?,?,?)""",
            (_now(), input_kind, input_raw, input_title, input_content))
        self.conn.commit()
        return int(cur.lastrowid)

    def get_session(self, session_id: int) -> Optional[Session]:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return _row_to_session(row) if row else None

    def list_sessions(self, limit: int = 50,
                      status: Optional[SessionStatus] = None
                      ) -> list[Session]:
        sql = "SELECT * FROM sessions"
        args: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            args.append(status)
        # 加 id DESC 兜底：同一秒内创建的会话 started_at 完全相同，
        # 只按时间排序结果不确定（真实踩过的坑）。
        sql += " ORDER BY started_at DESC, id DESC LIMIT ?"
        args.append(limit)
        return [_row_to_session(r)
                for r in self.conn.execute(sql, args).fetchall()]

    def save_plan(self, session_id: int, plan: dict) -> None:
        self.conn.execute(
            "UPDATE sessions SET plan_json = ? WHERE id = ?",
            (json.dumps(plan, ensure_ascii=False), session_id))
        self.conn.commit()

    def finish_session(self, session_id: int, status: SessionStatus = "finished",
                       duration_sec: Optional[int] = None,
                       usage: Optional[dict] = None) -> None:
        self.conn.execute(
            """UPDATE sessions
               SET ended_at = ?, status = ?, duration_sec = ?, usage_json = ?
               WHERE id = ?""",
            (_now(), status, duration_sec,
             json.dumps(usage, ensure_ascii=False) if usage else None,
             session_id))
        self.conn.commit()

    def refresh_session_stats(self, session_id: int) -> None:
        """重算轮次与字数统计（用于衡量"有没有多给我说的机会"）。"""
        row = self.conn.execute(
            """SELECT
                 COUNT(*) FILTER (WHERE role='user')      AS u_turns,
                 COUNT(*) FILTER (WHERE role='assistant') AS a_turns,
                 COALESCE(SUM(LENGTH(text)) FILTER (WHERE role='user'),0) AS u_chars,
                 COALESCE(SUM(LENGTH(text)) FILTER (WHERE role='assistant'),0) AS a_chars
               FROM turns WHERE session_id = ?""",
            (session_id,)).fetchone()
        self.conn.execute(
            """UPDATE sessions
               SET turn_count = ?, user_char_count = ?, ai_char_count = ?
               WHERE id = ?""",
            (row["u_turns"] + row["a_turns"], row["u_chars"], row["a_chars"],
             session_id))
        self.conn.commit()

    # ---------- 轮次 ----------

    def add_turn(self, turn: Turn) -> int:
        """追加一轮。seq 为空时自动取下一个序号。"""
        seq = turn.seq
        if not seq:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(seq),0)+1 AS n FROM turns WHERE session_id=?",
                (turn.session_id,)).fetchone()
            seq = int(row["n"])
        cur = self.conn.execute(
            """INSERT INTO turns (session_id, seq, role, text, audio_ms)
               VALUES (?,?,?,?,?)""",
            (turn.session_id, seq, turn.role, turn.text, turn.audio_ms))
        self.conn.commit()
        return int(cur.lastrowid)

    def list_turns(self, session_id: int) -> list[Turn]:
        rows = self.conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY seq",
            (session_id,)).fetchall()
        return [_row_to_turn(r) for r in rows]

    # ---------- 纠错 ----------

    def add_correction(self, c: Correction) -> int:
        cur = self.conn.execute(
            """INSERT INTO corrections
               (session_id, turn_id, kind, severity, original, suggestion,
                explanation, word, phonetic, shown)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (c.session_id, c.turn_id, c.kind, c.severity, c.original,
             c.suggestion, c.explanation, c.word, c.phonetic, int(c.shown)))
        self.conn.commit()
        return int(cur.lastrowid)

    def list_corrections(self, session_id: int,
                         only_visible: bool = False) -> list[Correction]:
        rows = self.conn.execute(
            "SELECT * FROM corrections WHERE session_id = ? ORDER BY id",
            (session_id,)).fetchall()
        out = [_row_to_correction(r) for r in rows]
        return [c for c in out if c.should_show()] if only_visible else out

    def top_error_words(self, limit: int = 20,
                        kind: CorrectionKind = "pronunciation"
                        ) -> list[tuple[str, int]]:
        """高频错误单词 —— 阶段 5 用来发现"总是读错的音"。"""
        rows = self.conn.execute(
            """SELECT word, COUNT(*) AS n FROM corrections
               WHERE word IS NOT NULL AND word <> '' AND kind = ?
               GROUP BY word ORDER BY n DESC, word LIMIT ?""",
            (kind, limit)).fetchall()
        return [(r["word"], int(r["n"])) for r in rows]

    # ---------- 个人画像 ----------

    def upsert_fact(self, fact: ProfileFact) -> int:
        """
        写入一条画像事实。

        已存在则更新 value/confidence，并把 confidence 略微上调
        （同一事实被多次提到 → 更可信）。
        """
        row = self.conn.execute(
            "SELECT id, confidence FROM profile_facts WHERE category=? AND key=?",
            (fact.category, fact.key)).fetchone()
        if row:
            new_conf = min(1.0, max(float(row["confidence"]),
                                    fact.confidence) + 0.1)
            self.conn.execute(
                """UPDATE profile_facts
                   SET value=?, confidence=?, updated_at=?,
                       source_session_id=COALESCE(?, source_session_id)
                   WHERE id=?""",
                (fact.value, new_conf, _now(), fact.source_session_id,
                 row["id"]))
            self.conn.commit()
            return int(row["id"])
        cur = self.conn.execute(
            """INSERT INTO profile_facts
               (category, key, value, confidence, source_session_id)
               VALUES (?,?,?,?,?)""",
            (fact.category, fact.key, fact.value, fact.confidence,
             fact.source_session_id))
        self.conn.commit()
        return int(cur.lastrowid)

    def list_facts(self, min_confidence: float = 0.0) -> list[ProfileFact]:
        rows = self.conn.execute(
            """SELECT * FROM profile_facts WHERE confidence >= ?
               ORDER BY category, confidence DESC""",
            (min_confidence,)).fetchall()
        return [_row_to_fact(r) for r in rows]


# ============================================================
#  行 → 模型
# ============================================================

def _row_to_session(r: sqlite3.Row) -> Session:
    return Session(
        id=r["id"], started_at=r["started_at"], ended_at=r["ended_at"],
        input_kind=r["input_kind"], input_raw=r["input_raw"],
        input_title=r["input_title"], input_content=r["input_content"],
        plan_json=r["plan_json"], duration_sec=r["duration_sec"],
        turn_count=r["turn_count"], user_char_count=r["user_char_count"],
        ai_char_count=r["ai_char_count"], status=r["status"],
        usage_json=r["usage_json"])


def _row_to_turn(r: sqlite3.Row) -> Turn:
    return Turn(id=r["id"], session_id=r["session_id"], seq=r["seq"],
                role=r["role"], text=r["text"], audio_ms=r["audio_ms"],
                created_at=r["created_at"])


def _row_to_correction(r: sqlite3.Row) -> Correction:
    return Correction(
        id=r["id"], session_id=r["session_id"], turn_id=r["turn_id"],
        kind=r["kind"], severity=r["severity"], original=r["original"],
        suggestion=r["suggestion"], explanation=r["explanation"],
        word=r["word"], phonetic=r["phonetic"], shown=bool(r["shown"]),
        created_at=r["created_at"])


def _row_to_fact(r: sqlite3.Row) -> ProfileFact:
    return ProfileFact(
        id=r["id"], category=r["category"], key=r["key"], value=r["value"],
        confidence=r["confidence"],
        source_session_id=r["source_session_id"],
        created_at=r["created_at"], updated_at=r["updated_at"])
