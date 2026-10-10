"""
数据库访问层。

上层代码只使用本模块的函数，不直接写 SQL ——
这样 schema 变更时只需改这里，不会波及业务逻辑。
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import (Correction, CorrectionKind, InputKind, ProfileFact,
                     ReviewHabitRow, ReviewItem, Session, SessionStatus, Turn,
                     User)

log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
MIGRATIONS_DIR = Path(__file__).with_name("migrations")

# 迁移记录表：记录已经跑过哪些迁移，避免重复执行。
# 不记录的话，第二次启动会因为「表已存在」而报错或静默跳过，
# 谁也说不清当前库到底是哪个版本。
META_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    name        TEXT PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _now_precise() -> str:
    """
    精确到微秒的时间戳，用于 last_used_at。

    秒级精度不够：连续切换两个用户常常落在同一秒内，
    时间戳相同后排序只能靠 id 兜底，"最近使用"的语义就失效了
    （刚切过去的人反而排在后面）。
    """
    return datetime.now(timezone.utc).astimezone().isoformat(
        timespec="microseconds")


class Database:
    """SQLite 封装。用作上下文管理器时自动提交/回滚。"""

    def __init__(self, path: str | Path = "data/app.db"):
        self.path = Path(path)
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：部分调用走 asyncio.to_thread（比如
        # 抽取声纹后保存），会落在别的线程上。默认限制会直接抛
        # "SQLite objects created in a thread can only be used in that same thread"。
        #
        # 这样是安全的，因为本项目所有写操作都是短事务、且都在同一个
        # 事件循环里串行发起；没有多线程并发写同一连接的场景。
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")

    # ---------- 生命周期 ----------

    def init_schema(self) -> None:
        """
        建表（幂等）+ 按需迁移。

        先跑迁移再建表：迁移负责把旧结构的库改成新结构，
        schema.sql 用 IF NOT EXISTS 建剩余的表。
        顺序反了的话，旧库会因为表已存在而跳过迁移，
        结果一直缺 users 表。
        """
        self._apply_migrations()
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.conn.commit()

    def _apply_migrations(self) -> None:
        """
        执行未应用过的迁移。

        检测「旧库」的判据：已经有业务表，但没有 schema_migrations 记录。
        一个全新的空库不需要迁移（schema.sql 会建出最新结构）。
        """
        self.conn.executescript(META_TABLE)
        self.conn.commit()

        applied = {
            r["name"] for r in
            self.conn.execute("SELECT name FROM schema_migrations").fetchall()
        }

        existing = {
            r["name"] for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }

        # 全新库：没有任何业务表 → 不需要迁移，schema.sql 直接建最新结构
        if not (existing & {"sessions", "turns", "corrections",
                            "profile_facts"}):
            return
        # 已经是新结构 → 不用迁移
        if "users" in existing:
            return

        for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if sql_file.name in applied:
                continue
            self._backup_before_migration(sql_file.name)
            log.warning("执行迁移 %s（旧数据将被重建，已备份）", sql_file.name)
            self.conn.executescript(sql_file.read_text(encoding="utf-8"))
            self.conn.execute(
                "INSERT OR REPLACE INTO schema_migrations (name) VALUES (?)",
                (sql_file.name,))
            self.conn.commit()
            log.warning("迁移 %s 完成", sql_file.name)

    def _backup_before_migration(self, name: str) -> None:
        """
        迁移前备份数据库文件。

        迁移涉及改表结构，万一中途失败，用户的历史就没了。
        备份成本极低（几十 KB），不做没有道理。
        """
        try:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = self.path.with_name(f"{self.path.name}.bak-{stamp}")
            self.conn.commit()
            shutil.copy2(self.path, backup)
            log.warning("迁移前已备份: %s", backup)
        except Exception as e:
            log.error("备份失败（继续迁移）: %s", e)

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

    # ---------- 用户 ----------

    def create_user(self, name: str, avatar: Optional[str] = None) -> int:
        """新建用户。名字重复会抛 sqlite3.IntegrityError，由上层转成友好提示。"""
        cur = self.conn.execute(
            "INSERT INTO users (name, avatar, last_used_at) VALUES (?,?,?)",
            (name.strip(), avatar, _now()))
        self.conn.commit()
        return int(cur.lastrowid)

    def get_user(self, user_id: int) -> Optional[User]:
        row = self.conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _row_to_user(row) if row else None

    def get_user_by_name(self, name: str) -> Optional[User]:
        row = self.conn.execute(
            "SELECT * FROM users WHERE name = ?", (name.strip(),)).fetchone()
        return _row_to_user(row) if row else None

    def list_users(self) -> list[User]:
        """全部用户，最近用过的排前面（切换列表的默认顺序）。"""
        rows = self.conn.execute(
            """SELECT * FROM users
               ORDER BY COALESCE(last_used_at, created_at) DESC, id""",
        ).fetchall()
        return [_row_to_user(r) for r in rows]

    def touch_user(self, user_id: int) -> None:
        # 用微秒精度：同一秒内切换多个用户时，"最近"必须仍然可分辨
        self.conn.execute("UPDATE users SET last_used_at = ? WHERE id = ?",
                          (_now_precise(), user_id))
        self.conn.commit()

    def rename_user(self, user_id: int, name: str,
                    avatar: Optional[str] = None) -> None:
        if avatar is None:
            self.conn.execute("UPDATE users SET name = ? WHERE id = ?",
                              (name.strip(), user_id))
        else:
            self.conn.execute(
                "UPDATE users SET name = ?, avatar = ? WHERE id = ?",
                (name.strip(), avatar, user_id))
        self.conn.commit()

    def set_voiceprint_info(self, user_id: int, *,
                            quality: Optional[float] = None,
                            samples: Optional[int] = None,
                            has: bool = True) -> None:
        """记录声纹录入结果，用于界面上显示质量。"""
        self.conn.execute(
            """UPDATE users
               SET has_voiceprint = ?, voiceprint_quality = ?,
                   voiceprint_samples = ?, last_used_at = ?
               WHERE id = ?""",
            (int(has), quality, samples, _now(), user_id))
        self.conn.commit()

    def delete_user(self, user_id: int) -> None:
        """
        删除用户及其全部数据。

        外键写了 ON DELETE CASCADE，所以 sessions/turns/corrections/
        profile_facts 会一起消失。声纹文件由调用方删除
        （数据库管不到文件系统）。
        """
        self.conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.conn.commit()

    def count_user_data(self, user_id: int) -> dict:
        """统计某用户有多少数据，删除前提示用。"""
        def one(sql: str) -> int:
            return int(self.conn.execute(sql, (user_id,)).fetchone()[0])

        return {
            "sessions": one(
                "SELECT COUNT(*) FROM sessions WHERE user_id = ?"),
            "turns": one(
                """SELECT COUNT(*) FROM turns WHERE session_id IN
                   (SELECT id FROM sessions WHERE user_id = ?)"""),
            "corrections": one(
                """SELECT COUNT(*) FROM corrections WHERE session_id IN
                   (SELECT id FROM sessions WHERE user_id = ?)"""),
            "facts": one(
                "SELECT COUNT(*) FROM profile_facts WHERE user_id = ?"),
        }

    # ---------- 会话 ----------

    def create_session(self, user_id: int, input_kind: InputKind,
                       input_raw: str,
                       input_title: Optional[str] = None,
                       input_content: Optional[str] = None) -> int:
        cur = self.conn.execute(
            """INSERT INTO sessions
               (user_id, started_at, input_kind, input_raw,
                input_title, input_content)
               VALUES (?,?,?,?,?,?)""",
            (user_id, _now(), input_kind, input_raw, input_title,
             input_content))
        self.conn.commit()
        self.touch_user(user_id)
        return int(cur.lastrowid)

    def get_session(self, session_id: int) -> Optional[Session]:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return _row_to_session(row) if row else None

    def list_sessions(self, limit: int = 50,
                      status: Optional[SessionStatus] = None,
                      user_id: Optional[int] = None) -> list[Session]:
        sql = "SELECT * FROM sessions"
        args: list[Any] = []
        where: list[str] = []
        if user_id is not None:
            where.append("user_id = ?")
            args.append(user_id)
        if status:
            where.append("status = ?")
            args.append(status)
        if where:
            sql += " WHERE " + " AND ".join(where)
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
                        kind: CorrectionKind = "pronunciation",
                        user_id: Optional[int] = None
                        ) -> list[tuple[str, int]]:
        """
        高频错误单词 —— 用来发现"总是读错的音"。

        传 user_id 时只统计该用户的，否则跨用户混合统计
        会把别人的错误算到你头上。
        """
        sql = """SELECT word, COUNT(*) AS n FROM corrections
                 WHERE word IS NOT NULL AND word <> '' AND kind = ?"""
        args: list[Any] = [kind]
        if user_id is not None:
            sql += """ AND session_id IN
                       (SELECT id FROM sessions WHERE user_id = ?)"""
            args.append(user_id)
        sql += " GROUP BY word ORDER BY n DESC, word LIMIT ?"
        args.append(limit)
        rows = self.conn.execute(sql, args).fetchall()
        return [(r["word"], int(r["n"])) for r in rows]

    # ---------- 个人画像 ----------

    def upsert_fact(self, fact: ProfileFact) -> int:
        """
        写入一条画像事实。

        已存在则更新 value/confidence，并把 confidence 略微上调
        （同一事实被多次提到 → 更可信）。
        匹配键是 (user_id, category, key) —— 不同用户的同名事实互不影响。
        """
        row = self.conn.execute(
            """SELECT id, confidence FROM profile_facts
               WHERE user_id=? AND category=? AND key=?""",
            (fact.user_id, fact.category, fact.key)).fetchone()
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
               (user_id, category, key, value, confidence, source_session_id)
               VALUES (?,?,?,?,?,?)""",
            (fact.user_id, fact.category, fact.key, fact.value,
             fact.confidence, fact.source_session_id))
        self.conn.commit()
        return int(cur.lastrowid)

    def list_facts(self, min_confidence: float = 0.0,
                   user_id: Optional[int] = None) -> list[ProfileFact]:
        sql = "SELECT * FROM profile_facts WHERE confidence >= ?"
        args: list[Any] = [min_confidence]
        if user_id is not None:
            sql += " AND user_id = ?"
            args.append(user_id)
        sql += " ORDER BY category, confidence DESC"
        rows = self.conn.execute(sql, args).fetchall()
        return [_row_to_fact(r) for r in rows]

    # ---------- 复习 ----------

    def upsert_review_habit(self, h: ReviewHabitRow) -> int:
        """
        写入/累加一类反复出现的问题。

        已存在时：累加 occurrences、刷新 last_seen，
        并保留用户的 mastered 标记 —— 复习过的成果不能被
        下一次练习覆盖掉（那样"已掌握"永远归零）。
        """
        row = self.conn.execute(
            "SELECT id, occurrences FROM review_habits "
            "WHERE user_id=? AND habit=?",
            (h.user_id, h.habit)).fetchone()
        if row:
            # 用 MAX 兜底：调用方传进来的 occurrences 是"新一次"的
            # 增量，不是总数，两个来源取较大值都能推进计数。
            self.conn.execute(
                """UPDATE review_habits
                   SET title=?, advice=COALESCE(?, advice),
                       occurrences=occurrences + ?,
                       last_seen=COALESCE(?, last_seen),
                       updated_at=?
                   WHERE id=?""",
                (h.title, h.advice, max(1, h.occurrences),
                 h.last_seen, _now(), row["id"]))
            self.conn.commit()
            return int(row["id"])

        cur = self.conn.execute(
            """INSERT INTO review_habits
               (user_id, habit, title, advice, occurrences,
                first_seen, last_seen)
               VALUES (?,?,?,?,?,?,?)""",
            (h.user_id, h.habit, h.title, h.advice,
             max(1, h.occurrences), h.first_seen, h.last_seen))
        self.conn.commit()
        return int(cur.lastrowid)

    def upsert_review_item(self, it: ReviewItem) -> int:
        """
        写入/累加一条具体说法。

        按 (user_id, dedup_key) 去重：同一句 "I went to the."
        重复出现时只累加次数，不再插入新行 ——
        否则复习清单会被同一句话刷屏。
        """
        row = self.conn.execute(
            "SELECT id FROM review_items WHERE user_id=? AND dedup_key=?",
            (it.user_id, it.dedup_key)).fetchone()
        if row:
            self.conn.execute(
                """UPDATE review_items
                   SET occurrences = occurrences + ?,
                       suggestion=?, note=COALESCE(?, note),
                       collocation=COALESCE(?, collocation),
                       last_seen=COALESCE(?, last_seen),
                       habit_id=COALESCE(?, habit_id),
                       updated_at=?
                   WHERE id=?""",
                (max(1, it.occurrences), it.suggestion, it.note,
                 it.collocation, it.last_seen, it.habit_id, _now(),
                 row["id"]))
            self.conn.commit()
            return int(row["id"])

        cur = self.conn.execute(
            """INSERT INTO review_items
               (user_id, habit_id, dedup_key, original, suggestion, note,
                collocation, occurrences, first_seen, last_seen,
                source_session_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (it.user_id, it.habit_id, it.dedup_key, it.original,
             it.suggestion, it.note, it.collocation,
             max(1, it.occurrences), it.first_seen, it.last_seen,
             it.source_session_id))
        self.conn.commit()
        return int(cur.lastrowid)

    def list_review_habits(self, user_id: int,
                           include_mastered: bool = False
                           ) -> list[ReviewHabitRow]:
        """按出现次数从高到低 —— 最该复习的排最前面。"""
        sql = "SELECT * FROM review_habits WHERE user_id = ?"
        args: list[Any] = [user_id]
        if not include_mastered:
            sql += " AND mastered = 0"
        sql += " ORDER BY occurrences DESC, id"
        rows = self.conn.execute(sql, args).fetchall()
        return [_row_to_review_habit(r) for r in rows]

    def list_review_items(self, user_id: int, habit_id: Optional[int] = None,
                          include_mastered: bool = False) -> list[ReviewItem]:
        sql = "SELECT * FROM review_items WHERE user_id = ?"
        args: list[Any] = [user_id]
        if habit_id is not None:
            sql += " AND habit_id = ?"
            args.append(habit_id)
        if not include_mastered:
            sql += " AND mastered = 0"
        sql += " ORDER BY occurrences DESC, id"
        rows = self.conn.execute(sql, args).fetchall()
        return [_row_to_review_item(r) for r in rows]

    def set_review_mastered(self, user_id: int, *, habit_id: Optional[int] = None,
                            item_id: Optional[int] = None,
                            mastered: bool = True) -> int:
        """
        标记"已掌握" / 取消标记。

        带 user_id 条件：否则传入别人的 id 就能改到别人的记录。
        返回实际影响的行数，便于上层判断 id 是否存在。
        """
        stamp = _now() if mastered else None
        if habit_id is not None:
            cur = self.conn.execute(
                """UPDATE review_habits
                   SET mastered=?, mastered_at=?, updated_at=?
                   WHERE id=? AND user_id=?""",
                (int(mastered), stamp, _now(), habit_id, user_id))
        elif item_id is not None:
            cur = self.conn.execute(
                """UPDATE review_items
                   SET mastered=?, mastered_at=?, updated_at=?
                   WHERE id=? AND user_id=?""",
                (int(mastered), stamp, _now(), item_id, user_id))
        else:
            return 0
        self.conn.commit()
        return cur.rowcount

    def clear_review(self, user_id: int) -> None:
        """清空该用户的复习数据（重新归纳前调用）。"""
        self.conn.execute("DELETE FROM review_items WHERE user_id = ?",
                          (user_id,))
        self.conn.execute("DELETE FROM review_habits WHERE user_id = ?",
                          (user_id,))
        self.conn.commit()

    def list_corrections_for_user(self, user_id: int,
                                  session_id: Optional[int] = None
                                  ) -> list[Correction]:
        """
        某用户的全部纠错（跨会话），带 session_id。

        复习归纳要"跨会话看重复"，所以必须能一次取全，
        而不是像 list_corrections 那样一次只取一个会话。
        """
        sql = """SELECT c.* FROM corrections c
                 JOIN sessions s ON s.id = c.session_id
                 WHERE s.user_id = ?"""
        args: list[Any] = [user_id]
        if session_id is not None:
            sql += " AND c.session_id = ?"
            args.append(session_id)
        sql += " ORDER BY c.session_id, c.id"
        rows = self.conn.execute(sql, args).fetchall()
        return [_row_to_correction(r) for r in rows]

    def count_review(self, user_id: int) -> dict:
        """复习页顶部概览用。"""
        def one(sql: str) -> int:
            return int(self.conn.execute(sql, (user_id,)).fetchone()[0])

        return {
            "habits": one("SELECT COUNT(*) FROM review_habits "
                          "WHERE user_id=? AND mastered=0"),
            "items": one("SELECT COUNT(*) FROM review_items "
                         "WHERE user_id=? AND mastered=0"),
            "mastered": one("SELECT COUNT(*) FROM review_habits "
                            "WHERE user_id=? AND mastered=1")
            + one("SELECT COUNT(*) FROM review_items "
                  "WHERE user_id=? AND mastered=1"),
            "occurrences": one("SELECT COALESCE(SUM(occurrences),0) "
                               "FROM review_habits WHERE user_id=?"),
        }


# ============================================================
#  行 → 模型
# ============================================================

def _row_to_user(r: sqlite3.Row) -> User:
    return User(
        id=r["id"], name=r["name"], avatar=r["avatar"],
        has_voiceprint=bool(r["has_voiceprint"]),
        voiceprint_quality=r["voiceprint_quality"],
        voiceprint_samples=r["voiceprint_samples"],
        created_at=r["created_at"], last_used_at=r["last_used_at"])


def _row_to_session(r: sqlite3.Row) -> Session:
    return Session(
        id=r["id"], user_id=r["user_id"], started_at=r["started_at"],
        ended_at=r["ended_at"],
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
        id=r["id"], user_id=r["user_id"], category=r["category"],
        key=r["key"], value=r["value"],
        confidence=r["confidence"],
        source_session_id=r["source_session_id"],
        created_at=r["created_at"], updated_at=r["updated_at"])


def _row_to_review_habit(r: sqlite3.Row) -> ReviewHabitRow:
    return ReviewHabitRow(
        id=r["id"], user_id=r["user_id"], habit=r["habit"],
        title=r["title"], advice=r["advice"],
        occurrences=r["occurrences"],
        first_seen=r["first_seen"], last_seen=r["last_seen"],
        mastered=bool(r["mastered"]), mastered_at=r["mastered_at"],
        created_at=r["created_at"], updated_at=r["updated_at"])


def _row_to_review_item(r: sqlite3.Row) -> ReviewItem:
    return ReviewItem(
        id=r["id"], user_id=r["user_id"], habit_id=r["habit_id"],
        dedup_key=r["dedup_key"], original=r["original"],
        suggestion=r["suggestion"], note=r["note"],
        collocation=r["collocation"], occurrences=r["occurrences"],
        first_seen=r["first_seen"], last_seen=r["last_seen"],
        mastered=bool(r["mastered"]), mastered_at=r["mastered_at"],
        source_session_id=r["source_session_id"],
        created_at=r["created_at"], updated_at=r["updated_at"])
