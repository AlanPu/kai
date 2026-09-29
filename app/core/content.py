"""
内容获取与清洗。

需求 1：输入可能是主题、段落、文章或网址。
本模块负责把任意输入归一化成「值得分析的内容」。
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from .config import ssl_context

log = logging.getLogger(__name__)

# 正文提取时的限制，避免把整站导航都灌给模型
MAX_CONTENT_CHARS = 20000
FETCH_TIMEOUT = 25
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120.0 Safari/537.36")


@dataclass
class Content:
    """归一化后的输入内容。"""

    kind: str                  # topic / passage / article / url
    raw: str                   # 原始输入
    title: Optional[str] = None
    text: str = ""             # 用于分析的正文
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


# ============================================================
#  输入类型判定
# ============================================================

def is_url(s: str) -> bool:
    s = (s or "").strip()
    if re.match(r"^https?://", s, re.I):
        return True
    # 形如 example.com/path 也算网址，但要求有域名点且无空格
    return bool(re.match(r"^[\w-]+(\.[\w-]+)+([/?#].*)?$", s)) and " " not in s


def normalize_url(s: str) -> str:
    s = s.strip()
    if not re.match(r"^https?://", s, re.I):
        s = "https://" + s
    return s


def classify(raw: str) -> str:
    """
    判定输入类型。

    规则（宽松优先，宁可判成文章也不要误判成主题）：
      · 网址            → url
      · 多句且较长      → article
      · 单段但较长      → passage
      · 其余（短句）    → topic
    """
    s = (raw or "").strip()
    if not s:
        return "topic"
    if is_url(s):
        return "url"

    # 按句末标点粗略数句子
    sentences = [x for x in re.split(r"[.!?。！？]+", s) if x.strip()]
    length = len(s)

    if length >= 800 or len(sentences) >= 5:
        return "article"
    if length >= 120 or len(sentences) >= 2:
        return "passage"
    return "topic"


# ============================================================
#  网址抓取
# ============================================================

def _assert_public_host(url: str) -> None:
    """
    拒绝内网地址（SSRF 防护）。

    用户可能粘贴 localhost 或 192.168.x.x，
    服务器替他去请求这些地址是不合适的。
    """
    host = urlparse(url).hostname
    if not host:
        raise ValueError("网址缺少主机名")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ValueError(f"域名无法解析: {host}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved):
            raise ValueError(f"拒绝访问内网地址: {host}")


def fetch_url(url: str, timeout: int = FETCH_TIMEOUT) -> Content:
    """抓取网页并提取正文。"""
    url = normalize_url(url)
    try:
        _assert_public_host(url)
    except ValueError as e:
        return Content(kind="url", raw=url, error=str(e))

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en,zh;q=0.9",
        })
        # 外网站点走代理；国内直连。交给系统代理设置处理，
        # 但明确使用 certifi 证书链。
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl_context()))
        with opener.open(req, timeout=timeout) as r:
            charset = r.headers.get_content_charset() or "utf-8"
            html = r.read().decode(charset, "replace")
    except urllib.error.HTTPError as e:
        return Content(kind="url", raw=url, error=f"HTTP {e.code}")
    except Exception as e:
        return Content(kind="url", raw=url,
                       error=f"{type(e).__name__}: {e}")

    title, text = extract_article(html)
    if not text.strip():
        return Content(kind="url", raw=url, title=title,
                       error="未能提取到正文（可能是动态渲染页面）")
    return Content(kind="url", raw=url, title=title,
                   text=text[:MAX_CONTENT_CHARS])


# ============================================================
#  正文提取
# ============================================================

_SCRIPT_RE = re.compile(
    r"<(script|style|noscript|svg|nav|footer|header|aside)\b.*?</\1>",
    re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.S | re.I)
_BLOCK_RE = re.compile(r"</(p|div|h[1-6]|li|br|tr)>", re.I)
_WS_RE = re.compile(r"[ \t\u00a0]+")
_MULTINL_RE = re.compile(r"\n{3,}")


def extract_article(html: str) -> tuple[Optional[str], str]:
    """
    从 HTML 提取标题与正文。

    不引入 BeautifulSoup —— 标准库足够应付大多数文章页，
    且少一个依赖。提取质量不足时由模型兜底。
    """
    if not html:
        return None, ""

    title = None
    m = _TITLE_RE.search(html)
    if m:
        title = _unescape(_TAG_RE.sub("", m.group(1))).strip()
    if not title:
        m = _H1_RE.search(html)
        if m:
            title = _unescape(_TAG_RE.sub("", m.group(1))).strip()

    body = _SCRIPT_RE.sub(" ", html)
    body = _BLOCK_RE.sub("\n", body)
    body = _TAG_RE.sub(" ", body)
    body = _unescape(body)
    body = _WS_RE.sub(" ", body)
    lines = [ln.strip() for ln in body.split("\n")]
    text = "\n".join(ln for ln in lines if ln)
    text = _MULTINL_RE.sub("\n\n", text).strip()
    return title, text


def _unescape(s: str) -> str:
    import html as _h
    return _h.unescape(s)


# ============================================================
#  统一入口
# ============================================================

def load_content(raw: str) -> Content:
    """把任意输入归一化成 Content。"""
    kind = classify(raw)
    if kind == "url":
        return fetch_url(raw)
    return Content(kind=kind, raw=raw, text=raw.strip())
