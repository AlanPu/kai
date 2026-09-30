-- 迁移 001：引入多用户
--
-- 背景：原本是单用户设计，没有 users 表，所有个人数据
-- （会话、纠错、画像、声纹）都只有一份。
-- 现在要支持多人共用一台机器，每人独立的声纹/画像/历史。
--
-- 关于旧数据：用户明确要求"清空，重新录入"。
-- 旧数据里没有 user_id，无法判断归属 —— 强行塞给某个用户
-- 会产生"训练数据其实来自另一个人"的错误画像，比清空更糟。
-- 所以这里直接重建（旧库已备份到 data/app.db.bak-<时间戳>）。
--
-- 迁移方式：SQLite 不能直接给表加 NOT NULL + REFERENCES 列，
-- 也不能改 UNIQUE 约束，因此用「建新表 → 拷数据 → 换名」。

PRAGMA foreign_keys = OFF;

-- ============================================================
--  users
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL UNIQUE,
    avatar          TEXT,
    has_voiceprint  INTEGER NOT NULL DEFAULT 0,
    voiceprint_quality REAL,
    voiceprint_samples INTEGER,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    last_used_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_users_last_used
    ON users(last_used_at DESC);


-- ============================================================
--  sessions：重建并挂到 user_id
-- ============================================================
DROP TABLE IF EXISTS sessions_old;
ALTER TABLE sessions RENAME TO sessions_old;

CREATE TABLE sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    started_at      TEXT    NOT NULL,
    ended_at        TEXT,
    input_kind      TEXT    NOT NULL
                    CHECK (input_kind IN ('topic','passage','article','url')),
    input_raw       TEXT    NOT NULL,
    input_title     TEXT,
    input_content   TEXT,
    plan_json       TEXT,
    duration_sec    INTEGER,
    turn_count      INTEGER NOT NULL DEFAULT 0,
    user_char_count INTEGER NOT NULL DEFAULT 0,
    ai_char_count   INTEGER NOT NULL DEFAULT 0,
    status          TEXT    NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active','finished','aborted')),
    usage_json      TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

DROP TABLE IF EXISTS sessions_old;

CREATE INDEX IF NOT EXISTS idx_sessions_started
    ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_status
    ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_user_started
    ON sessions(user_id, started_at DESC);


-- ============================================================
--  turns：结构不变，但要跟随 sessions 一起重建
--  （sessions 换了表名，外键指向会失效）
-- ============================================================
DROP TABLE IF EXISTS turns_old;
ALTER TABLE turns RENAME TO turns_old;

CREATE TABLE turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    role        TEXT    NOT NULL CHECK (role IN ('user','assistant')),
    text        TEXT    NOT NULL,
    audio_ms    INTEGER,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (session_id, seq)
);

DROP TABLE IF EXISTS turns_old;

CREATE INDEX IF NOT EXISTS idx_turns_session
    ON turns(session_id, seq);


-- ============================================================
--  corrections
-- ============================================================
DROP TABLE IF EXISTS corrections_old;
ALTER TABLE corrections RENAME TO corrections_old;

CREATE TABLE corrections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_id     INTEGER REFERENCES turns(id) ON DELETE CASCADE,
    kind        TEXT    NOT NULL
                CHECK (kind IN ('grammar','vocabulary','pronunciation','fluency')),
    severity    TEXT    NOT NULL DEFAULT 'minor'
                CHECK (severity IN ('critical','minor','ignore')),
    original    TEXT    NOT NULL,
    suggestion  TEXT    NOT NULL,
    explanation TEXT,
    word        TEXT,
    phonetic    TEXT,
    shown       INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

DROP TABLE IF EXISTS corrections_old;

CREATE INDEX IF NOT EXISTS idx_corrections_session
    ON corrections(session_id);
CREATE INDEX IF NOT EXISTS idx_corrections_kind
    ON corrections(kind, severity);
CREATE INDEX IF NOT EXISTS idx_corrections_word
    ON corrections(word) WHERE word IS NOT NULL;


-- ============================================================
--  profile_facts：UNIQUE 从 (category,key) 改成 (user_id,category,key)
-- ============================================================
DROP TABLE IF EXISTS profile_facts_old;
ALTER TABLE profile_facts RENAME TO profile_facts_old;

CREATE TABLE profile_facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category    TEXT    NOT NULL,
    key         TEXT    NOT NULL,
    value       TEXT    NOT NULL,
    confidence  REAL    NOT NULL DEFAULT 0.5,
    source_session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (user_id, category, key)
);

DROP TABLE IF EXISTS profile_facts_old;

CREATE INDEX IF NOT EXISTS idx_profile_category
    ON profile_facts(user_id, category, confidence DESC);

PRAGMA foreign_keys = ON;
