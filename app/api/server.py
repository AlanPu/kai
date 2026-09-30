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

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core.config import Settings, load_settings
from ..core.content import load_content
from ..core.enroll import (DEFAULT_PROMPTS_LIST, MIN_SAMPLES,
                           EnrollmentSession)
from ..core.realtime import SessionFatal
from ..core.voiceprint import SpeakerVerifier
from ..services.corrector import Corrector
from ..services.planner import Planner, build_tutor_instructions
from ..services.cost import estimate_cost, format_usage, parse_realtime_usage
from ..services.profile import (INJECT_MIN_CONFIDENCE, ProfileExtractor,
                                ProfileStore)
from ..services.session import ConversationSession
from ..services.users import (DuplicateUser, InvalidUser, UserNotFound,
                              UserStore, quality_label)
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

# 冷场多久之后 AI 主动找话题（秒）。
#
# 20 秒是权衡的结果：真人对话里这个长度刚好是"对方在等你开口"，
# 再长就像冷场，再短就变成催促 —— 而用户需要时间组织英文句子，
# 催太紧反而说不出来。
IDLE_NUDGE_SEC = 20.0

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


def user_store() -> UserStore:
    return UserStore(db(), settings().voiceprints_dir)


def db() -> Database:
    global _db
    if _db is None:
        _db = Database(settings().db_path)
        _db.init_schema()
    return _db


# ============================================================
#  用户：多人共用一台机器
# ============================================================

class UserIn(BaseModel):
    name: str
    avatar: Optional[str] = None


def _user_json(u) -> dict:
    """用户对外表示（含声纹状态），界面直接用。"""
    return {
        "id": u.id,
        "name": u.name,
        "avatar": u.avatar,
        "has_voiceprint": u.has_voiceprint,
        "voiceprint_quality": u.voiceprint_quality,
        "voiceprint_quality_label": quality_label(u.voiceprint_quality),
        "voiceprint_samples": u.voiceprint_samples,
        "created_at": u.created_at,
        "last_used_at": u.last_used_at,
    }


def _resolve_user(user_id: Optional[int]):
    """
    把请求里的 user 参数解析成用户对象。

    没给 user 时回退到「最近用过的那个」：刷新页面、换设备打开时
    不该突然变成空白，而上次在用的人通常就是现在要用的人。

    取不到就返回 None，由调用方决定怎么办 —— 各接口的处理并不相同
    （读接口给空结果、写接口报错）。这里**不能抛异常**：
    用户被删掉后，停在旧页面上的标签页仍会带着失效的 id 发请求，
    那是正常情况，不该变成 500 把页面整个卡住。
    """
    store = user_store()
    try:
        if user_id is None:
            return store.default_user()
        return store.require(user_id)
    except UserNotFound:
        log.info("请求指定的用户 %s 不存在，回退到默认用户", user_id)
        return store.default_user()


@app.get("/api/users")
async def list_users():
    """所有用户 + 默认用户（最近使用）。前端据此渲染切换列表。"""
    store = user_store()
    users = store.list_users()
    current = store.default_user()
    return {
        "users": [_user_json(u) for u in users],
        "current": _user_json(current) if current else None,
    }


@app.post("/api/users")
async def create_user(body: UserIn):
    try:
        u = user_store().create(body.name, body.avatar)
    except DuplicateUser as e:
        raise HTTPException(409, str(e))
    except InvalidUser as e:
        raise HTTPException(400, str(e))
    return _user_json(u)


@app.patch("/api/users/{user_id}")
async def update_user(user_id: int, body: UserIn):
    try:
        u = user_store().rename(user_id, body.name, body.avatar)
    except UserNotFound as e:
        raise HTTPException(404, str(e))
    except DuplicateUser as e:
        raise HTTPException(409, str(e))
    except InvalidUser as e:
        raise HTTPException(400, str(e))
    return _user_json(u)


@app.delete("/api/users/{user_id}")
async def delete_user(user_id: int):
    """
    删除用户及其全部数据。

    返回被删掉的统计，让界面能明确说"删了多少东西"——
    删除不可逆，含糊其辞会让人不敢用或者误删。
    """
    try:
        stats = user_store().delete(user_id)
    except UserNotFound as e:
        raise HTTPException(404, str(e))
    return {"deleted": True, "stats": stats}


@app.get("/api/users/{user_id}/voiceprint")
async def get_voiceprint(user_id: int):
    try:
        return user_store().voiceprint_info(user_id)
    except UserNotFound as e:
        raise HTTPException(404, str(e))


@app.delete("/api/users/{user_id}/voiceprint")
async def delete_voiceprint(user_id: int):
    """只清声纹，保留画像和历史 —— 换麦克风想重录的常见场景。"""
    try:
        user_store().clear_voiceprint(user_id)
    except UserNotFound as e:
        raise HTTPException(404, str(e))
    return {"cleared": True}


@app.get("/api/enroll/prompts")
async def enroll_prompts():
    """录入用的跟读句子。放服务端是为了以后能按用户水平调整。"""
    return {
        "prompts": [{"text": p.text, "hint": p.hint}
                    for p in DEFAULT_PROMPTS_LIST],
        "min_samples": MIN_SAMPLES,
    }


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
        # 至少有一个用户录过声纹就算可用（多用户下不再是单个文件）
        "voiceprint": any(u.has_voiceprint
                          for u in user_store().list_users()),
    }


@app.get("/api/stats")
async def stats(user: Optional[int] = None):
    """某个用户的累计统计，含总花费估算。"""
    u = _resolve_user(user)
    if u is None:
        # 一个用户都没有：返回空统计而不是报错，让首页能正常渲染
        return {"sessions": 0, "finished": 0, "total_minutes": 0,
                "total_tokens": 0, "cost_yuan": 0, "user_ratio": 0,
                "facts_learned": 0, "no_user": True}
    sessions = db().list_sessions(limit=1000, user_id=u.id)
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
        "facts_learned": len(db().list_facts(user_id=u.id)),
        "user_id": u.id,
    }


# ============================================================
#  备课：把输入变成话题计划（HTTP，非实时）
# ============================================================

class PrepareIn(BaseModel):
    content: str
    topics: int = 5


@app.post("/api/prepare")
async def prepare(body: PrepareIn, user: Optional[int] = None):
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
    u = _resolve_user(user)
    store = profile_store()
    summary = store.summary(u.id) if u else ""

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
async def list_sessions(limit: int = 30, user: Optional[int] = None):
    u = _resolve_user(user)
    if u is None:
        return {"sessions": []}
    out = []
    for s in db().list_sessions(limit=limit, user_id=u.id):
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
async def get_profile(user: Optional[int] = None):
    """已积累的画像，供界面展示与用户检查。"""
    u = _resolve_user(user)
    if u is None:
        return {"count": 0, "facts": [], "summary": "", "issues": [],
                "no_user": True}
    store = profile_store()
    facts = db().list_facts(user_id=u.id)
    return {
        "user_id": u.id,
        "count": len(facts),
        "facts": [{
            "category": f.category, "key": f.key, "value": f.value,
            "confidence": round(f.confidence, 2),
            "source_session_id": f.source_session_id,
        } for f in facts],
        "summary": store.summary(u.id, min_confidence=INJECT_MIN_CONFIDENCE),
        "issues": store.language_issues(u.id),
    }


@app.post("/api/profile/export")
async def export_profile(user: Optional[int] = None):
    """导出为 Markdown，便于人工查看与修正。"""
    u = _resolve_user(user)
    if u is None:
        raise HTTPException(400, "还没有用户，无法导出")
    try:
        p = profile_store().write_markdown(u.id)
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
      user  必填，是谁在练（决定用哪份声纹和画像）
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

        # 用户必须能明确认出来。这里**刻意不做默认回退**：
        # 认错人的后果是把甲说的话记进乙的历史和画像里，
        # 而且悄无声息、事后极难发现。宁可报错让前端重新选人。
        raw_user = ws.query_params.get("user", "").strip()
        user = None
        if raw_user:
            try:
                user = user_store().get(int(raw_user))
            except ValueError:
                user = None
        if user is None:
            await ws.send_json({"type": "error",
                                "error": "用户不存在，请重新选择用户"})
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
            user.id, content.kind, content.raw,
            input_title=content.title, input_content=content.text[:20000])
        db().save_plan(sid, plan.to_dict())

        # ---- 声纹（按用户各自的档案）----
        verifier = None
        if use_vp:
            try:
                vp = user_store().load_voiceprint(user.id)
            except Exception as e:
                vp = None
                log.warning("读取声纹失败: %s", e)
            proto = vp[0] if vp else None
            meta = vp[1] if vp else {}
            if proto is not None and s.voiceprint_model.is_file():
                try:
                    verifier = await asyncio.to_thread(
                        SpeakerVerifier, s.voiceprint_model, proto)
                    log.info("声纹已启用 (user=%s %s)", user.name,
                             meta.get("quality", ""))
                except Exception as e:
                    log.warning("声纹加载失败，改为不过滤: %s", e)
            elif use_vp:
                # 明确告知"这次没过滤"，而不是让人困惑外人说话也有回应
                await ws.send_json({
                    "type": "voiceprint_missing",
                    "message": "这个用户还没有录入声纹，"
                               "本次不做声纹过滤（旁人说话也会被回应）",
                })

        store = profile_store()
        summary = store.summary(user.id)
        instructions = build_tutor_instructions(
            plan, profile_summary=summary, minutes=minutes)

        async def on_client(msg: dict) -> None:
            try:
                await ws.send_json(msg)
            except Exception:
                pass

        sess = ConversationSession(
            s, db(), session_id=sid, user_id=user.id,
            instructions=instructions,
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
    elif t == "turn_end":
        # 「按住空格说话」松开时发来：用户明确表示这一轮说完了。
        # 有了这个信号就不必再依赖服务端 VAD 猜断句，
        # 从根上避免"半句话被当成说完 → AI 抢答 → 用户接着说 →
        # AI 被打断"这一连串问题。
        await sess.end_turn()
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

        # 冷场救场：改成「按住空格说话」之后回合完全由用户发起，
        # 用户不开口 AI 就永远不说话。卡住想不出句子时会一直干等，
        # 练习就停在那儿了 —— 所以冷场够久就让 AI 主动找话题。
        # 放在这里（而不是 Qwen 回调里）是因为要发 response.create，
        # 在回调里做容易和别的事件打架。
        await sess.nudge(IDLE_NUDGE_SEC)

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
#  声纹录入：引导式跟读
#
#  和练习会话分开成两条 WebSocket，因为两件事的节奏完全不同：
#  录入是"你说一句 → 我当场判断 → 不合格立刻重来"，
#  而练习是长时间的连续对话。混在一起状态机会乱。
# ============================================================

def _dummy_prototype() -> np.ndarray:
    """
    构造一个占位原型，只为满足 SpeakerVerifier 的构造要求。

    录入阶段只用得到 verifier.ex（特征提取器），不涉及比对，
    所以原型是什么无所谓 —— 给个合法的单位向量即可。
    """
    v = np.zeros(192, dtype=np.float32)
    v[0] = 1.0
    return v


@app.websocket("/ws/enroll")
async def ws_enroll(ws: WebSocket):
    """
    引导式声纹录入。

    查询参数：
      user  必填，给谁录

    协议：
      下行  enroll_start / enroll_progress / enroll_retry
            / enroll_done / enroll_cancelled / error
      上行  二进制帧 = 16 kHz int16 PCM
            {"cmd":"end"|"discard"|"cancel"}
    """
    await ws.accept()
    s = settings()

    # 用户必须明确：录到别人名下会让那个人此后一直认错
    raw_user = ws.query_params.get("user", "").strip()
    user = None
    if raw_user:
        try:
            user = user_store().get(int(raw_user))
        except ValueError:
            user = None
    if user is None:
        await ws.send_json({"type": "error", "error": "用户不存在"})
        await ws.close()
        return

    if not s.voiceprint_model.is_file():
        await ws.send_json({
            "type": "error",
            "error": f"声纹模型不存在：{s.voiceprint_model}。"
                     "请先按 SETUP.md 下载模型。",
        })
        await ws.close()
        return

    # 加载模型可能要几秒，别卡住事件循环
    try:
        verifier = await asyncio.to_thread(
            SpeakerVerifier, s.voiceprint_model, _dummy_prototype(),
            threshold=0.5)
    except Exception as e:
        log.exception("声纹模型加载失败")
        await ws.send_json({"type": "error", "error": f"声纹模型加载失败：{e}"})
        await ws.close()
        return

    sess = EnrollmentSession(user_id=user.id, verifier=verifier)
    store = user_store()
    await ws.send_json({"type": "enroll_start", "user": _user_json(user),
                        **sess.progress()})

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                return

            # 音频帧
            if msg.get("bytes"):
                # 必须显式按 int16 解释：直接喂 bytes 会被 numpy
                # 当成 uint8 数组，特征提取器会把它当字符串解析而报错
                sess.feed(np.frombuffer(msg["bytes"], dtype=np.int16))
                continue

            raw = msg.get("text")
            if not raw:
                continue
            try:
                cmd = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # 前端发的是 {"type": "end"}，早期脚本发的是 {"cmd": "end"}。
            # 两个都认 —— 曾经只认 "cmd"，导致网页上点「读完了」
            # 被静默忽略（不报错、不回应，看起来就是"点了没反应"），
            # 而用脚本测试却一切正常，因为脚本发的是 "cmd"。
            action = (cmd.get("cmd") or cmd.get("type") or "").strip()

            if action == "cancel":
                await ws.send_json({"type": "enroll_cancelled"})
                return

            if action == "discard":
                sess.clear_buffer()
                await ws.send_json({"type": "enroll_progress",
                                    **sess.progress()})
                continue

            if action != "end":
                continue

            # 判定这一句
            res = await asyncio.to_thread(sess.commit)
            if not res.ok:
                await ws.send_json({"type": "enroll_retry",
                                    "reason": res.reason,
                                    **sess.progress()})
                continue

            if not sess.finished:
                await ws.send_json({"type": "enroll_progress",
                                    "speech_sec": round(res.speech_sec, 1),
                                    **sess.progress()})
                continue

            # 句子录满了，检查够不够用
            if not sess.can_save():
                await ws.send_json({
                    "type": "enroll_retry",
                    "reason": f"有效样本不足（至少需要 {MIN_SAMPLES} 句），"
                              "请再读一遍",
                    **sess.progress(),
                })
                continue

            proto = sess.prototype()
            if proto is None:
                await ws.send_json({"type": "enroll_retry",
                                    "reason": "没有拿到有效声纹，请重新录入",
                                    **sess.progress()})
                continue

            quality = float(sess.quality or 0.0)
            await asyncio.to_thread(
                store.save_voiceprint, user.id, proto,
                quality=quality, samples=len(sess.embeddings),
                raw_embeddings=sess.embeddings)

            fresh = user_store().get(user.id)
            await ws.send_json({
                "type": "enroll_done",
                "user": _user_json(fresh) if fresh else None,
                "quality": quality,
                "quality_label": quality_label(quality),
                "samples": len(sess.embeddings),
            })
            return

    except WebSocketDisconnect:
        return
    except Exception as e:
        log.exception("声纹录入异常")
        try:
            await ws.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass


# ============================================================
#  静态资源
# ============================================================

if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
