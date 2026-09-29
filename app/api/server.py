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
from ..core.voiceprint import SpeakerVerifier, load_voiceprint
from ..services.corrector import Corrector
from ..services.planner import Planner, build_tutor_instructions
from ..services.session import ConversationSession
from ..storage.db import Database

log = logging.getLogger(__name__)
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

    try:
        plan = await asyncio.to_thread(
            Planner().plan, content, n_topics=max(3, min(body.topics, 8)))
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
    return {
        "id": s.id, "status": s.status, "duration_sec": s.duration_sec,
        "input_kind": s.input_kind, "title": s.input_title,
        "turn_count": s.turn_count,
        "user_char_count": s.user_char_count,
        "ai_char_count": s.ai_char_count,
        "user_ratio": round(s.user_char_count / total, 3) if total else 0,
        "turns": turns, "corrections": corrections,
    }


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

        instructions = build_tutor_instructions(plan, minutes=minutes)

        async def on_client(msg: dict) -> None:
            try:
                await ws.send_json(msg)
            except Exception:
                pass

        sess = ConversationSession(
            s, db(), session_id=sid, instructions=instructions,
            on_client=on_client, voiceprint=verifier,
            corrector=Corrector(), minutes=minutes)

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
        for t in done:
            exc = t.exception()
            if exc is None:
                continue
            if isinstance(exc, (WebSocketDisconnect, _StopRequested)):
                log.info("会话正常结束: %s", type(exc).__name__)
            else:
                log.warning("会话任务异常: %r", exc)

        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        await sess.stop()

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


async def _timer_loop(ws: WebSocket, sess: ConversationSession) -> None:
    """到点自动结束（需求：每次约 30 分钟）。"""
    while not sess.ended:
        await asyncio.sleep(5)
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
