"""
用户管理：多人共用一台机器时，各自独立的声纹、画像、历史。

为什么单独一个模块：
    声纹路径、画像路径、当前用户这几件事分散在各处很容易漏改一处，
    集中在这里，路径规则只有一处定义。

关于"当前用户"存在哪：
    存在浏览器 localStorage，不是服务端全局变量。
    服务端全局的话，两个人同时打开页面会互相把对方切走 ——
    这类 bug 很难复现。每个请求带 ?user=<id> 反而简单可靠。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.voiceprint import load_voiceprint
from ..storage.db import Database
from ..storage.models import User

log = logging.getLogger(__name__)

# 声纹质量分级：录入时算出的「本人 vs 其他样本」区分度。
# 太低说明几次录音差异过大（换位置、换音量、有旁人说话），
# 用它做校验会频繁误判，不如重录。
# 声纹质量分档。
#
# 这个分数是"自洽性"：录入时那几句录音彼此的平均余弦相似度。
# 它不是与别人的区分度 —— 录入时现场没有别人可做对比。
#
# 实测（CAM++ / 2 秒窗口，同一说话人 45 对片段）：最低 0.485，均值 0.644；
# 不同说话人 70 对：最高 0.360，均值 0.200。
# 所以本人正常落在 0.5~0.8 之间 —— 阈值必须按这个尺度定。
# 之前写 0.75/0.60 是从别处照搬的，实测下几乎永远只能评"一般"。
QUALITY_GOOD = 0.70
QUALITY_OK = 0.55


def quality_label(q: Optional[float]) -> str:
    """
    把质量分数翻译成给人看的话。

    措辞刻意留了余地：分数低只代表"这次录的几句不太一致"，
    可能是环境噪音、距离变化或没读顺，不等于声纹一定不能用。
    """
    if q is None:
        return "未录入"
    if q >= QUALITY_GOOD:
        return "好"
    if q >= QUALITY_OK:
        return "一般"
    return "偏差"


class UserStore:
    """用户的增删改查 + 每用户声纹文件的管理。"""

    def __init__(self, db: Database, voiceprints_dir: Path):
        self.db = db
        self.dir = Path(voiceprints_dir)

    # ---------- 查询 ----------

    def list_users(self) -> list[User]:
        return self.db.list_users()

    def get(self, user_id: int) -> Optional[User]:
        return self.db.get_user(user_id)

    def get_by_name(self, name: str) -> Optional[User]:
        return self.db.get_user_by_name(name)

    def require(self, user_id: int) -> User:
        """取用户，不存在就抛 —— 调用方据此返回 404。"""
        u = self.db.get_user(user_id)
        if u is None:
            raise UserNotFound(f"用户 {user_id} 不存在")
        return u

    def resolve(self, user_id: Optional[int]) -> Optional[User]:
        """
        把请求里的 user 参数解析成用户。

        user_id 为空时返回 None（调用方决定是报错还是用默认用户）。
        """
        if user_id is None:
            return None
        return self.db.get_user(user_id)

    def default_user(self) -> Optional[User]:
        """
        没指定用户时用谁：最近用过的那个。

        这样刷新页面、换设备打开时不会突然变成空白，
        而"上次在用的人"通常就是现在要用的人。
        """
        users = self.db.list_users()
        return users[0] if users else None

    # ---------- 增删改 ----------

    def create(self, name: str, avatar: Optional[str] = None) -> User:
        name = (name or "").strip()
        if not name:
            raise InvalidUser("名字不能为空")
        if len(name) > 24:
            raise InvalidUser("名字太长了（最多 24 个字）")
        if self.db.get_user_by_name(name):
            raise DuplicateUser(f"已经有一个叫「{name}」的用户了")
        uid = self.db.create_user(name, avatar)
        return self.require(uid)

    def rename(self, user_id: int, name: str,
               avatar: Optional[str] = None) -> User:
        self.require(user_id)
        name = (name or "").strip()
        if not name:
            raise InvalidUser("名字不能为空")
        other = self.db.get_user_by_name(name)
        if other and other.id != user_id:
            raise DuplicateUser(f"已经有一个叫「{name}」的用户了")
        self.db.rename_user(user_id, name, avatar)
        return self.require(user_id)

    def delete(self, user_id: int) -> dict:
        """
        删除用户，连同声纹文件。

        返回被删掉的统计，便于界面提示"已删除 N 场练习记录"。
        """
        self.require(user_id)
        stats = self.db.count_user_data(user_id)
        vp = self.voiceprint_path(user_id)
        self.db.delete_user(user_id)      # 外键 CASCADE 会带走其余数据
        try:
            vp.unlink(missing_ok=True)
            self.profile_path(user_id).unlink(missing_ok=True)
        except OSError as e:
            # 文件删不掉不该让整个操作失败：数据库里已经没有它了，
            # 留个孤立文件不影响使用，只是浪费一点空间。
            log.warning("用户 %s 的声纹/画像文件删除失败: %s", user_id, e)
        return stats

    # ---------- 路径 ----------

    def voiceprint_path(self, user_id: int) -> Path:
        return self.dir / f"{user_id}.json"

    def profile_path(self, user_id: int) -> Path:
        return self.dir.parent / "profiles" / f"{user_id}.md"

    def has_voiceprint(self, user_id: int) -> bool:
        return self.voiceprint_path(user_id).is_file()

    def load_voiceprint(self, user_id: int
                        ) -> tuple[Optional[np.ndarray], dict]:
        """读取该用户的声纹原型。没有则 (None, {})。"""
        return load_voiceprint(self.voiceprint_path(user_id))

    # ---------- 声纹录入 ----------

    def save_voiceprint(self, user_id: int, prototype: np.ndarray,
                        quality: float, samples: int,
                        raw_embeddings: Optional[list] = None) -> None:
        """
        落盘一份新声纹，并更新用户记录。

        存原始样本（raw_embeddings）是有意为之：
        以后想调阈值或改用"多原型匹配"时，不必让人重新录一遍。
        声音是生物特征，多存几 KB 换取重录成本，值得。
        """
        self.require(user_id)
        self.dir.mkdir(parents=True, exist_ok=True)

        proto = np.asarray(prototype, dtype=np.float32)
        n = float(np.linalg.norm(proto))
        if n <= 0:
            raise InvalidUser("声纹样本无效（全零向量）")
        proto = proto / n

        payload = {
            "prototype": [float(x) for x in proto],
            "quality": float(quality),
            "samples": int(samples),
            # 每个样本一个向量，供以后重新计算原型
            "embeddings": [[float(x) for x in np.asarray(e, dtype=np.float32)]
                           for e in (raw_embeddings or [])],
        }
        p = self.voiceprint_path(user_id)
        # 先写临时文件再改名：中途崩溃时不会留下半个残缺的声纹
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(p)

        self.db.set_voiceprint_info(user_id, quality=quality,
                                    samples=samples, has=True)
        log.info("用户 %s 的声纹已保存（质量 %.3f，%d 个样本）",
                 user_id, quality, samples)

    def clear_voiceprint(self, user_id: int) -> None:
        """只清声纹，保留画像和历史（换麦克风时想重录但不想丢数据）。"""
        self.require(user_id)
        try:
            self.voiceprint_path(user_id).unlink(missing_ok=True)
        except OSError as e:
            log.warning("声纹文件删除失败: %s", e)
        self.db.set_voiceprint_info(user_id, quality=None, samples=None,
                                    has=False)

    def voiceprint_info(self, user_id: int) -> dict:
        """给界面用的声纹状态。"""
        u = self.require(user_id)
        return {
            "has": self.has_voiceprint(user_id),
            "quality": u.voiceprint_quality,
            "quality_label": quality_label(u.voiceprint_quality),
            "samples": u.voiceprint_samples,
        }


# ============================================================
#  异常：服务层抛出，API 层翻译成 HTTP 状态码
# ============================================================

class UserError(Exception):
    """用户相关错误的基类。"""


class UserNotFound(UserError):
    """指定的用户不存在 → 404。"""


class DuplicateUser(UserError):
    """用户名重复 → 409。"""


class InvalidUser(UserError):
    """名字非法等 → 400。"""
