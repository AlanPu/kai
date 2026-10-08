"""
已备好的话题计划的短期缓存。

解决的问题：「生成话题」和点「开始」各自调一次 Planner，
于是**两次备课、两个结果**。即使两边注入的是同一份画像，
模型重新构思仍会给出不同的话题 —— 界面上预览的是 A，
真正开始练习时聊的是 B，用户只会觉得"AI 没听我刚说的"。

做法是把第一次备课的结果按 (用户, 素材) 记下来，第二次直接取用。

## 为什么按 (user_id, 内容指纹) 而不是 token

直觉上会想让 /api/prepare 返回一个 token，前端再传回来。
但 WS 本来就收到了 `content`（`server.py:584`），服务端能自己
算出同一个键。这样做的好处是**前端一行都不用改**，也就不会
出现"前端忘了传 token"这种新的失效方式。

键里必须带 user_id：两个人输入同一段素材是常事
（同一篇文章一起练），共用一份计划就又串味了。
指纹必须带内容哈希：用户备完课可能又改了输入框里的文字，
拿旧计划去开练，聊的就不是他此刻想聊的东西。

## 这是缓存，不是持久化

不写数据库、不跨重启。重启后 WebSocket 走不到缓存，
会像以前一样现备课 —— 行为正确，只是多花一次调用。
之所以不落库：计划几分钟内就会被用掉，为它建表、
写迁移、处理失效，比省一次调用麻烦得多。
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .planner import Plan

log = logging.getLogger(__name__)

# 备课到开练之间通常只隔几秒。给半小时足够宽松，
# 超过就重新想 —— 久放的素材配上更新过的画像本来就该重备。
TTL_SEC = 1800

# 多人轮流用的机器上，几十条足够；再多也没有一条会被用到。
MAX_ENTRIES = 64


def content_fingerprint(content: str) -> str:
    """
    素材内容的指纹。

    归一化只做 strip：大小写、标点、空格的变化**刻意不归一**。
    "Remote work" 和 "remote work" 可能引出完全不同的话题，
    把它们当成同一份素材，复用旧计划就是在答非所问。
    """
    return hashlib.sha256(
        (content or "").strip().encode("utf-8")).hexdigest()[:32]


@dataclass
class _Entry:
    user_id: int
    fingerprint: str
    plan: Plan
    created_at: float = field(default_factory=time.monotonic)


class PlanCache:
    """
    进程内的计划缓存。

    单进程使用：服务由 `uvicorn.run` 起一个 worker
    （`app/__main__.py:65-73`），所以内存缓存就够了。
    真要多进程部署，这个类得换成共享存储 —— 那时再说，
    现在引入 Redis 只为一个"省一次模型调用"的场景并不划算。
    """

    def __init__(self, ttl_sec: float = TTL_SEC,
                 max_entries: int = MAX_ENTRIES):
        self.ttl = ttl_sec
        self.max_entries = max_entries
        self._data: dict[tuple[int, str], _Entry] = {}

    # ---------- 存取 ----------

    def put(self, user_id: int, content: str, plan: Plan) -> None:
        self._purge()
        key = (user_id, content_fingerprint(content))
        self._data[key] = _Entry(user_id=user_id,
                                 fingerprint=key[1], plan=plan)
        # 满了就丢最旧的一条。字典保序，最旧的在最前。
        while len(self._data) > self.max_entries:
            oldest = next(iter(self._data))
            del self._data[oldest]

    def get(self, user_id: int, content: str) -> Optional[Plan]:
        """命中返回计划，否则 None（调用方改为现备课）。"""
        self._purge()
        e = self._data.get((user_id, content_fingerprint(content)))
        return e.plan if e else None

    def invalidate_user(self, user_id: int) -> None:
        """
        丢掉该用户的全部缓存。

        用户画像被改动后（比如刚导入了新的背景），
        之前按旧画像备好的计划就该作废。
        """
        for k in [k for k in self._data if k[0] == user_id]:
            del self._data[k]

    def __len__(self) -> int:
        return len(self._data)

    # ---------- 内部 ----------

    def _purge(self) -> None:
        """清掉过期的条目。用完即清，不留定时器。"""
        if not self._data:
            return
        cutoff = time.monotonic() - self.ttl
        for k in [k for k, e in self._data.items() if e.created_at < cutoff]:
            del self._data[k]


# 进程内单例。server.py 的依赖注入都走函数（settings/db/user_store），
# 这里沿用同样的写法，方便测试替换。
_plan_cache: Optional[PlanCache] = None


def plan_cache() -> PlanCache:
    global _plan_cache
    if _plan_cache is None:
        _plan_cache = PlanCache()
    return _plan_cache
