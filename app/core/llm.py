"""
文本 LLM 客户端（OpenAI 兼容接口）。

话题规划、纠错、画像抽取都走这里 ——
这些任务不需要低延迟，用文本模型比语音通道便宜、可控、可重试。
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Optional

from .config import Settings, load_settings, ssl_context

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """调用失败（网络、鉴权、配额等）。"""


class LLMClient:
    """极简 OpenAI 兼容客户端，只用标准库，不引入额外依赖。"""

    def __init__(self, settings: Optional[Settings] = None, timeout: int = 60):
        self.s = settings or load_settings()
        self.timeout = timeout

    # ---------- 底层 ----------

    def chat(self, messages: list[dict], *, model: Optional[str] = None,
             temperature: float = 0.7, max_tokens: int = 2048,
             json_mode: bool = False) -> str:
        """发一次对话请求，返回文本内容。"""
        if not self.s.has_text_credentials():
            raise LLMError("未配置 DASHSCOPE_API_KEY")

        payload: dict[str, Any] = {
            "model": model or self.s.text_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        req = urllib.request.Request(
            f"{self.s.text_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.s.dashscope_api_key}",
                "Content-Type": "application/json",
            })

        # 国内服务：绕过代理，否则可能失败
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=ssl_context()))

        try:
            with opener.open(req, timeout=self.timeout) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            raise LLMError(f"HTTP {e.code}: {detail}") from e
        except Exception as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e

        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"响应格式异常: {str(data)[:300]}") from e

    def chat_json(self, messages: list[dict], **kw) -> Any:
        """
        要求模型返回 JSON 并解析。

        模型有时会包一层 ```json 代码块，这里做容错处理。
        """
        kw.setdefault("json_mode", True)
        raw = self.chat(messages, **kw)
        return parse_json_loose(raw)


# ============================================================
#  JSON 容错解析
# ============================================================

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json_loose(raw: str) -> Any:
    """
    尽量从模型输出里解析出 JSON。

    依次尝试：直接解析 → 去代码围栏 → 截取首个 {...} 或 [...]
    """
    text = (raw or "").strip()
    if not text:
        raise LLMError("模型返回为空")

    for cand in _candidates(text):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    raise LLMError(f"无法解析为 JSON: {text[:300]}")


def _candidates(text: str) -> list[str]:
    out = [text]

    m = _FENCE_RE.search(text)
    if m:
        out.append(m.group(1).strip())

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        i, j = text.find(open_ch), text.rfind(close_ch)
        if 0 <= i < j:
            out.append(text[i:j + 1])
    return out
