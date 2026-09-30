-- 英语口语陪练 — 数据库 schema
--
-- 设计原则：
--   1. 一切可回溯：会话、每一轮、每条纠错都能查到原始来源
--   2. 分级明确：发音问题分 critical/minor，对应"明显错误才纠"的需求
--   3. 便于统计：字段可直接用于聚合查询（进步曲线、高频错误）
--   4. 多人隔离：所有个人数据都挂在 users 下，切换用户互不可见
--
-- 迁移：每次结构变更追加一个 migrations/NNN_*.sql，不修改本文件既有内容。

PRAGMA foreign_keys = ON;

-- ============================================================
--  用户：多人共用一台机器时各自独立
--
--  为什么要单独一张表而不是复用 profile_facts：
--  声纹（谁在说话）是"身份"，画像（喜欢什么）是"内容"。
--  身份必须先确立，才能决定往哪个画像里写数据。
--  混在一起会导致切换用户时无法判断该用哪个声纹去匹配。
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL UNIQUE,     -- 显示名，同一台机器上唯一
    avatar          TEXT,                        -- 可选：单个 emoji，纯显示用

    -- 声纹档案：每个用户一份，相对 data/voiceprints/<id>.json
    -- 用 user_id 命名而不是用户名，避免中文/特殊字符变成文件名问题
    has_voiceprint  INTEGER NOT NULL DEFAULT 0,

    -- 声纹质量指标，录入时写入，便于提示"这份声纹可能不太好"
    voiceprint_quality REAL,                     -- 类间区分度，越高越好
    voiceprint_samples INTEGER,                  -- 录入时用了几个样本

    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    last_used_at    TEXT                         -- 最近一次练习，用于排序
);

CREATE INDEX IF NOT EXISTS idx_users_last_used
    ON users(last_used_at DESC);


-- ============================================================
--  会话：一次完整的练习
-- ============================================================
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    -- 归属用户。ON DELETE CASCADE：删用户就删掉他的全部记录
    -- （用户明确要求"清空重来"时不必逐表手工清理）
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    started_at      TEXT    NOT NULL,           -- ISO8601
    ended_at        TEXT,
    -- 输入来源：topic / passage / article / url
    input_kind      TEXT    NOT NULL
                    CHECK (input_kind IN ('topic','passage','article','url')),
    input_raw       TEXT    NOT NULL,           -- 原始输入（网址本身/原文）
    input_title     TEXT,                       -- 文章标题（可空）
    input_content   TEXT,                       -- 抓取/清洗后的正文（可空）

    -- 话题计划（planner 产出，JSON）
    plan_json       TEXT,

    -- 会话统计
    duration_sec    INTEGER,
    turn_count      INTEGER NOT NULL DEFAULT 0,
    user_char_count INTEGER NOT NULL DEFAULT 0, -- 用户说了多少字（衡量"多给我说"）
    ai_char_count   INTEGER NOT NULL DEFAULT 0,
    status          TEXT    NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active','finished','aborted')),

    -- 费用（token 用量，阶段 6 用）
    usage_json      TEXT,

    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sessions_started
    ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_status
    ON sessions(status);
-- 列表页总是"某个用户的最近会话"，这个复合索引直接服务该查询
CREATE INDEX IF NOT EXISTS idx_sessions_user_started
    ON sessions(user_id, started_at DESC);


-- ============================================================
--  轮次：会话里的每一句
-- ============================================================
CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,               -- 第几轮，从 1 开始
    role        TEXT    NOT NULL CHECK (role IN ('user','assistant')),
    text        TEXT    NOT NULL,               -- 转写/回复文本
    audio_ms    INTEGER,                        -- 语音时长
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),

    UNIQUE (session_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_turns_session
    ON turns(session_id, seq);


-- ============================================================
--  纠错项：对应需求 3（语法/地道表达）与需求 4（发音分级）
-- ============================================================
CREATE TABLE IF NOT EXISTS corrections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_id     INTEGER REFERENCES turns(id) ON DELETE CASCADE,

    -- 纠错类型
    kind        TEXT    NOT NULL
                CHECK (kind IN ('grammar','vocabulary','pronunciation','fluency')),

    -- 严重度：直接编码需求 4 的规则
    --   critical → 必须纠正（明显的单词发音错误、硬语法错误）
    --   minor    → 看情况（口音、不地道但能懂）
    --   ignore   → 记录但不展示（可接受变体）
    severity    TEXT    NOT NULL DEFAULT 'minor'
                CHECK (severity IN ('critical','minor','ignore')),

    original    TEXT    NOT NULL,               -- 原句/原词
    suggestion  TEXT    NOT NULL,               -- 建议说法
    explanation TEXT,                           -- 中文解释（为什么要改）
    -- 发音专用：出错的单词 + 音标，便于阶段 5 做高频音素统计
    word        TEXT,
    phonetic    TEXT,

    -- 是否已展示给用户（避免重复提示）
    shown       INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_corrections_session
    ON corrections(session_id);
CREATE INDEX IF NOT EXISTS idx_corrections_kind
    ON corrections(kind, severity);
CREATE INDEX IF NOT EXISTS idx_corrections_word
    ON corrections(word) WHERE word IS NOT NULL;


-- ============================================================
--  个人画像：需求 6（越了解你，越会聊你感兴趣的）
--  key-value 结构，便于增量更新，同时导出成 Markdown 供阅读
--
--  注意 UNIQUE 是 (user_id, category, key)：不同用户可能有
--  同名事实（比如都写了"职业"），必须各自独立累积置信度。
-- ============================================================
CREATE TABLE IF NOT EXISTS profile_facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category    TEXT    NOT NULL,   -- interest / background / goal / habit ...
    key         TEXT    NOT NULL,
    value       TEXT    NOT NULL,
    -- 置信度：多次出现才更可信
    confidence  REAL    NOT NULL DEFAULT 0.5,
    -- 来源会话，便于追溯"我是从哪次对话知道这件事的"
    source_session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now')),

    UNIQUE (user_id, category, key)
);

CREATE INDEX IF NOT EXISTS idx_profile_category
    ON profile_facts(user_id, category, confidence DESC);
