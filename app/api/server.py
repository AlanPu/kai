"""
FastAPI 应用：WebSocket 语音通道 + 静态页面。

音频链路（与原型一致，实测可用）：
  浏览器 getUserMedia → AudioWorklet → 16 kHz int16 PCM
  → WebSocket → 声纹过滤 → Qwen Realtime
  Qwen 返回 24 kHz PCM → WebSocket → 浏览器排队播放
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..core.config import Settings, load_settings
from ..core.content import load_content
from ..core.realtime import SessionFatal
from ..core.voiceprint import SpeakerVerifier, load_voiceprint
from ..services.corrector import Corrector
from ..services.planner import Planner, build_tutor_instructions
from ..services.cost import estimate_cost, format_usage, parse_realtime_usage
from ..services.profile import (INJECT_MIN_CONFIDENCE, ProfileExtractor,
                                ProfileStore)
from ..services.session import ConversationSession
from ..storage.db import Database

log = logging.getLogger(__name__)

# 把应用日志接到 uvicorn 的 handler 上。
# 不配的话 log.info 永远不显示 —— 排查轮转、画像这些
# 后台行为时会误以为「没执行」，实际只是没打印。
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
# server.py 在 app/api/ 下，静态页面在 app/web/
WEB_DIR = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(title="英语口语陪练")
_settings: Optional[Settings] = None
_db: Optional[Database] = None


def settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def profile_store() -> ProfileStore:
    return ProfileStore(db(), settings().profiles_dir)


def db() -> Database:
    global _db
    if _db is None:
        _db = Database(settings().db_path)
        _db.init_schema()
    return _db


# ============================================================
#  页面
# ============================================================

@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
async def health():
    s = settings()
    return {
        "ok": True,
        "voice_ready": s.has_voice_credentials(),
        "text_ready": s.has_text_credentials(),
        "model": s.qwen_model,
        "session_minutes": s.session_minutes,
        "voiceprint": s.voiceprint_path.is_file(),
    }


@app.get("/api/stats")
async def stats():
    """累计统计，含总花费估算。"""
    sessions = db().list_sessions(limit=1000)
    total_sec = sum(x.duration_sec or 0 for x in sessions)
    done = [x for x in sessions if x.status == "finished"]

    tokens = 0
    cost = 0.0
    for x in done:
        if not x.usage_json:
            continue
        try:
            import json as _json
            u = parse_realtime_usage(_json.loads(x.usage_json))
            tokens += u.total_tokens
            cost += estimate_cost(u)
        except Exception:
            continue

    total_chars = sum((x.user_char_count or 0) + (x.ai_char_count or 0)
                      for x in done)
    user_chars = sum(x.user_char_count or 0 for x in done)

    return {
        "sessions": len(sessions),
        "finished": len(done),
        "total_minutes": round(total_sec / 60, 1),
        "total_tokens": tokens,
        "cost_yuan": round(cost, 3),
        "user_ratio": round(user_chars / total_chars, 3) if total_chars else 0,
        "facts_learned": len(db().list_facts()),
    }


# ============================================================
#  备课：把输入变成话题计划（HTTP，非实时）
# ============================================================

from pydantic import BaseModel


class PrepareIn(BaseModel):
    content: str
    topics: int = 5


@app.post("/api/prepare")
async def prepare(body: PrepareIn):
    """
    需求 1：输入主题/段落/文章/网址 → 分析内容并预先规划聊什么。
    """
    text = (body.content or "").strip()
    if not text:
        raise HTTPException(400, "内容不能为空")

    content = load_content(text)
    if not content.ok:
        raise HTTPException(400, f"内容无法处理：{content.error}")

    # 需求 6：把已知画像注入，让话题更贴近本人
    store = profile_store()
    summary = store.summary()

    try:
        plan = await asyncio.to_thread(
            Planner().plan, content, profile_summary=summary,
            n_topics=max(3, min(body.topics, 8)))
    except Exception as e:
        raise HTTPException(502, f"话题规划失败：{e}")

    return {
        "kind": content.kind,
        "title": content.title,
        "chars": len(content.text),
        "plan": plan.to_dict(),
    }


# ============================================================
#  历史与报告
# ============================================================

@app.get("/api/sessions")
async def list_sessions(limit: int = 30):
    out = []
    for s in db().list_sessions(limit=limit):
        out.append({
            "id": s.id, "started_at": s.started_at, "ended_at": s.ended_at,
            "input_kind": s.input_kind,
            "title": s.input_title or (s.input_raw or "")[:60],
            "duration_sec": s.duration_sec, "turn_count": s.turn_count,
            "user_char_count": s.user_char_count,
            "ai_char_count": s.ai_char_count, "status": s.status,
        })
    return {"sessions": out}


@app.get("/api/sessions/{session_id}")
async def session_detail(session_id: int):
    """HTTP 路由：返回会话详情。"""
    data = _build_report(session_id)
    if data is None:
        raise HTTPException(404, "会话不存在")
    return data


def _build_report(session_id: int) -> Optional[dict]:
    """构造会话报告（纯函数，HTTP 与 WebSocket 共用）。"""
    s = db().get_session(session_id)
    if not s:
        return None

    turns = [{"role": t.role, "text": t.text, "seq": t.seq}
             for t in db().list_turns(session_id)]
    corrections = [{
        "kind": c.kind, "severity": c.severity, "original": c.original,
        "suggestion": c.suggestion, "explanation": c.explanation,
        "word": c.word, "phonetic": c.phonetic,
    } for c in db().list_corrections(session_id)]

    total = (s.user_char_count or 0) + (s.ai_char_count or 0)

    # 费用（用量在结束时存进 usage_json）
    usage = None
    if s.usage_json:
        try:
            import json as _json
            usage = _json.loads(s.usage_json)
        except Exception:
            usage = None
    u = parse_realtime_usage(usage)

    return {
        "id": s.id, "status": s.status, "duration_sec": s.duration_sec,
        "input_kind": s.input_kind, "title": s.input_title,
        "turn_count": s.turn_count,
        "user_char_count": s.user_char_count,
        "ai_char_count": s.ai_char_count,
        "user_ratio": round(s.user_char_count / total, 3) if total else 0,
        "turns": turns, "corrections": corrections,
        "usage": {
            "total_tokens": u.total_tokens,
            "text": format_usage(u),
            "cost_yuan": estimate_cost(u),
        } if u.total_tokens else None,
    }


@app.get("/api/profile")
async def get_profile():
    """已积累的画像，供界面展示与用户检查。"""
    store = profile_store()
    facts = db().list_facts()
    return {
        "count": len(facts),
        "facts": [{
            "category": f.category, "key": f.key, "value": f.value,
            "confidence": round(f.confidence, 2),
            "source_session_id": f.source_session_id,
        } for f in facts],
        "summary": store.summary(min_confidence=INJECT_MIN_CONFIDENCE),
        "issues": store.language_issues(),
    }


@app.post("/api/profile/export")
async def export_profile():
    """导出为 Markdown，便于人工查看与修正。"""
    try:
        p = profile_store().write_markdown()
    except Exception as e:
        raise HTTPException(500, f"导出失败：{e}")
    return {"path": str(p), "text": p.read_text(encoding="utf-8")}


# ============================================================
#  语音通道
# ============================================================

@app.websocket("/ws/session")
async def ws_session(ws: WebSocket):
    """
    查询参数：
      content  必填，本次练习的输入内容
      voiceprint  1/0，是否启用声纹过滤（默认 1）
      minutes  本次时长（默认取配置）
    """
    await ws.accept()
    s = settings()

    try:
        raw_content = ws.query_params.get("content", "").strip()
        if not raw_content:
            await ws.send_json({"type": "error", "error": "缺少 content 参数"})
            await ws.close()
            return

        use_vp = ws.query_params.get("voiceprint", "1") != "0"
        try:
            minutes = int(ws.query_params.get("minutes", s.session_minutes))
        except ValueError:
            minutes = s.session_minutes

        # ---- 备课 ----
        content = load_content(raw_content)
        if not content.ok:
            await ws.send_json({"type": "error",
                                "error": f"内容无法处理：{content.error}"})
            await ws.close()
            return

        await ws.send_json({"type": "status", "stage": "planning"})
        try:
            plan = await asyncio.to_thread(Planner().plan, content)
        except Exception as e:
            await ws.send_json({"type": "error",
                                "error": f"话题规划失败：{e}"})
            await ws.close()
            return
        await ws.send_json({"type": "plan", "plan": plan.to_dict()})

        # ---- 建会话 ----
        sid = db().create_session(
            content.kind, content.raw,
            input_title=content.title, input_content=content.text[:20000])
        db().save_plan(sid, plan.to_dict())

        # ---- 声纹 ----
        verifier = None
        if use_vp and s.voiceprint_path.is_file():
            proto, meta = load_voiceprint(s.voiceprint_path)
            if proto is not None and s.voiceprint_model.is_file():
                try:
                    verifier = await asyncio.to_thread(
                        SpeakerVerifier, s.voiceprint_model, proto)
                    log.info("声纹已启用 (label=%s)", meta.get("label"))
                except Exception as e:
                    log.warning("声纹加载失败，改为不过滤: %s", e)

        store = profile_store()
        summary = store.summary()
        instructions = build_tutor_instructions(
            plan, profile_summary=summary, minutes=minutes)

        async def on_client(msg: dict) -> None:
            try:
                await ws.send_json(msg)
            except Exception:
                pass

        sess = ConversationSession(
            s, db(), session_id=sid, instructions=instructions,
            on_client=on_client, voiceprint=verifier,
            corrector=Corrector(), profile_store=store,
            profile_extractor=ProfileExtractor(), minutes=minutes)

        await sess.start()
        await ws.send_json({"type": "session", "session_id": sid,
                            "minutes": minutes})

        # ---- 主循环 ----
        recv_task = asyncio.create_task(_recv_loop(ws, sess))
        timer_task = asyncio.create_task(_timer_loop(ws, sess))

        done, pending = await asyncio.wait(
            {recv_task, timer_task}, return_when=asyncio.FIRST_COMPLETED)

        # asyncio.wait 只是"返回"完成的任务，不会抛出其中的异常。
        # 必须逐个取 exception()，否则真实错误会被静默吞掉。
        fatal = None
        for t in done:
            exc = t.exception()
            if exc is None:
                continue
            if isinstance(exc, (WebSocketDisconnect, _StopRequested)):
                log.info("会话正常结束: %s", type(exc).__name__)
            elif isinstance(exc, SessionFatal):
                # 连接不可用了（额度耗尽/被服务端关闭等）。
                # 必须明确告诉用户「为什么结束」，并让他看报告 ——
                # 只当普通异常记一笔的话，用户看到的是
                # 「莫名其妙就结束了」，也不知道该怎么办。
                fatal = str(exc)
                log.warning("会话因致命错误结束: %s", fatal)
            else:
                log.warning("会话任务异常: %r", exc)

        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        await sess.stop()

        # 致命错误先告诉用户发生了什么、能怎么办
        if fatal:
            await ws.send_json({
                "type": "aborted",
                "reason": fatal,
                "message": _fatal_message(fatal),
            })

        # 结束报告（直接读库，不经过 HTTP 路由层）
        await ws.send_json({"type": "finished", "session_id": sid,
                            "report": _build_report(sid)})

    except (WebSocketDisconnect, _StopRequested):
        log.info("客户端断开/主动结束")
    except Exception as e:
        log.exception("会话异常")
        try:
            await ws.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


async def _recv_loop(ws: WebSocket, sess: ConversationSession) -> None:
    while True:
        msg = await ws.receive()
        if msg.get("type") == "websocket.disconnect":
            return
        if "bytes" in msg and msg["bytes"]:
            await sess.push_audio(msg["bytes"])
        elif "text" in msg and msg["text"]:
            try:
                cmd = json.loads(msg["text"])
            except json.JSONDecodeError:
                continue
            await _handle_cmd(cmd, sess)


def _fatal_message(code: str) -> str:
    """把服务端的错误码翻译成用户能懂的话。"""
    c = (code or "").lower()
    if "idle" in c:
        return ("很久没有听到你说话了，服务端已关闭本次会话。"
                "录音还在，可以接着看报告；想继续就重新开始一次。")
    if "quota" in c or "insufficient" in c:
        # 说清楚怎么办，而不是只说"额度不足"。
        # 免费额度用完后阿里云会在错误里提示两种出路，
        # 不告诉用户的话他只能一脸茫然地反复重试。
        return ("语音模型的免费额度用完了，本次会话无法继续。"
                "到阿里云百炼控制台充值，或关掉「仅使用免费额度」"
                "即可恢复。本次已录到的内容都还在，可以看报告。")
    if "api_key" in c or "auth" in c:
        return "密钥无效或已过期，请检查 .env 配置。"
    if "rate" in c:
        return "请求过于频繁，被限流了，稍后再试。"
    if "expired" in c or "closed" in c:
        return "会话已过期或被服务端关闭，请重新开始。"
    return f"会话被服务端中断（{code}），请重新开始。"


class _StopRequested(Exception):
    """用户主动结束会话。用专门的异常，不复用 WebSocketDisconnect
    （后者语义是"连接意外断开"，拿来当控制流会掩盖真实错误）。"""


async def _handle_cmd(cmd: dict, sess: ConversationSession) -> None:
    t = cmd.get("type")
    if t == "cancel":            # 用户抢话
        if sess.rt:
            await sess.rt.cancel_response()
    elif t == "stop":            # 主动结束
        raise _StopRequested()
    elif t == "status":
        await sess.on_client({"type": "stats", **sess.snapshot()})
    elif t == "pause":
        # 用户中途离开。暂停期间不计时、不收音频，
        # 避免触发 Qwen 的 300 秒空闲关闭。
        sess.pause()
        await sess.on_client({"type": "paused", **sess.snapshot()})
    elif t == "resume":
        sess.unpause()
        await sess.on_client({"type": "resumed", **sess.snapshot()})


async def _timer_loop(ws: WebSocket, sess: ConversationSession) -> None:
    """到点自动结束（需求：每次约 30 分钟）+ 上下文轮转看护。"""
    while not sess.ended:
        await asyncio.sleep(5)

        # 上下文轮转必须在这里做，不能在 Qwen 的 recv 回调里做：
        # 那里关连接会掐断正在执行回调的接收循环，
        # 异常冒泡出去整个会话会被判异常结束（实测踩过）。
        if sess.wants_rotate() and not sess.ended:
            log.info("执行上下文轮转")
            ok = await sess.rotate_context()
            if not ok:
                await sess.on_client({
                    "type": "error",
                    "error": {"code": "rotate_failed",
                              "message": "对话记忆整理失败，"
                                         "建议结束本轮重新开始"},
                })

        # 连接死了就得结束会话，不能继续发 tick 假装一切正常。
        # 实测：服务端报错并断开后，程序又空转了 24 分钟，
        # 用户对着一个死连接说话而界面毫无异常。
        if sess.is_connection_lost():
            log.warning("检测到语音连接已断开，结束会话")
            raise SessionFatal(sess.fatal_error or "connection_closed")

        snap = sess.snapshot()
        await sess.on_client({
            "type": "tick",
            "elapsed_sec": snap["elapsed_sec"],
            "remaining_sec": snap["remaining_sec"],
            "user_chars": snap["user_chars"],
            "ai_chars": snap["ai_chars"],
            "blocked_sec": snap["blocked_sec"],
        })
        if sess.is_expired():
            await sess.on_client({"type": "time_up"})
            return


# ============================================================
#  静态资源
# ============================================================

if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
