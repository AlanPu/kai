"""
配置加载。

集中处理 .env、代理绕行、SSL 证书三个环境坑，
其他模块只从这里取配置，不各自处理。
"""

from __future__ import annotations

import os
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """显式按项目根目录加载，避免 find_dotenv 在非文件入口下失败。"""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass


_load_dotenv()


def ssl_context() -> ssl.SSLContext:
    """
    显式使用 certifi 证书链。

    macOS 上 Python 常找不到系统根证书，
    报 CERTIFICATE_VERIFY_FAILED —— 不是网络问题，别去查代理。
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


@dataclass(frozen=True)
class Settings:
    # ---- 语音：Qwen Realtime ----
    dashscope_api_key: str = ""
    qwen_workspace_id: str = ""
    qwen_region: str = "cn-beijing"
    qwen_model: str = "qwen3.8-omni-flash-realtime"
    qwen_voice: str = "Tina"

    # ---- 文本：话题分析 / 纠错 / 画像 ----
    text_model: str = "qwen-plus"
    text_base_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1")

    # ---- 会话 ----
    session_minutes: int = 30

    # ---- 声纹 ----
    voiceprint_path: Path = PROJECT_ROOT / "data" / "voiceprint.json"
    voiceprint_model: Path = PROJECT_ROOT / "models" / "campplus.onnx"

    # ---- 数据 ----
    db_path: Path = PROJECT_ROOT / "data" / "app.db"
    profiles_dir: Path = PROJECT_ROOT / "data" / "profiles"

    def voice_ws_url(self) -> str:
        """Qwen Realtime WebSocket 地址。"""
        return (f"wss://{self.qwen_workspace_id}.{self.qwen_region}"
                f".maas.aliyuncs.com/api-ws/v1/realtime"
                f"?model={self.qwen_model}")

    def has_voice_credentials(self) -> bool:
        return bool(self.dashscope_api_key and self.qwen_workspace_id)

    def has_text_credentials(self) -> bool:
        return bool(self.dashscope_api_key)


def load_settings(env: Optional[dict] = None) -> Settings:
    """从环境变量读取配置。env 可注入，便于测试。"""
    e = env if env is not None else os.environ

    def get(key: str, default: str) -> str:
        v = e.get(key)
        return v if v else default

    def get_path(key: str, default: Path) -> Path:
        v = e.get(key)
        return Path(v) if v else default

    try:
        minutes = int(get("SESSION_MINUTES", "30"))
    except ValueError:
        minutes = 30

    return Settings(
        dashscope_api_key=get("DASHSCOPE_API_KEY", ""),
        qwen_workspace_id=get("QWEN_WORKSPACE_ID", ""),
        qwen_region=get("QWEN_REGION", "cn-beijing"),
        qwen_model=get("QWEN_MODEL", "qwen3.8-omni-flash-realtime"),
        qwen_voice=get("QWEN_VOICE", "Tina"),
        text_model=get("TEXT_MODEL", "qwen-plus"),
        text_base_url=get("TEXT_BASE_URL",
                          "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        session_minutes=minutes,
        voiceprint_path=get_path("VOICEPRINT_PATH",
                                 PROJECT_ROOT / "data" / "voiceprint.json"),
        voiceprint_model=get_path("VOICEPRINT_MODEL",
                                  PROJECT_ROOT / "models" / "campplus.onnx"),
        db_path=get_path("DB_PATH", PROJECT_ROOT / "data" / "app.db"),
        profiles_dir=get_path("PROFILES_DIR",
                              PROJECT_ROOT / "data" / "profiles"),
    )
