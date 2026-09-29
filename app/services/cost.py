"""
费用统计。

代码本来就在收集 usage，但从未展示出来 —— 用户看不到花了多少钱。
本模块把 token 用量换算成金额，让花费可见。

计价说明：
  Qwen Realtime 按音频时长计费（不是按 token），
  官方单价会调整，因此单价写在配置里而不是硬编码，
  并明确标注"估算值"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# 百炼 qwen3-omni / realtime 系列参考单价（元 / 千 token）
# ⚠️ 这些是估算用的参考值，实际以阿里云账单为准。
DEFAULT_PRICE = {
    "input_per_1k": 0.0008,
    "output_per_1k": 0.002,
}
# 文本模型（qwen-plus 参考价）
TEXT_PRICE = {
    "input_per_1k": 0.0008,
    "output_per_1k": 0.002,
}


@dataclass
class Usage:
    """一次会话的用量。"""

    audio_in_tokens: int = 0
    audio_out_tokens: int = 0
    text_in_tokens: int = 0
    text_out_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (self.audio_in_tokens + self.audio_out_tokens
                + self.text_in_tokens + self.text_out_tokens)


def parse_realtime_usage(usage: Optional[dict]) -> Usage:
    """
    解析 Qwen Realtime 返回的 usage。

    实测的真实结构（字段名与 OpenAI 不同，别照抄文档）：
        {
          "total_tokens": 712,
          "input_tokens": 695,
          "output_tokens": 17,
          "input_tokens_details":  {"text_tokens": 695},
          "output_tokens_details": {"text_tokens": 5, "audio_tokens": 12}
        }
    所以文本/音频 token 藏在 *_tokens_details 里，
    顶层只有总数。兼容多种命名，避免版本变更后静默归零。
    """
    if not usage:
        return Usage()

    def dig(container, *names) -> int:
        """在若干容器里按多个候选名查找一个数值。"""
        if not isinstance(container, dict):
            return 0
        for n in names:
            v = container.get(n)
            if isinstance(v, (int, float)) and v > 0:
                return int(v)
        return 0

    in_det = usage.get("input_tokens_details") or {}
    out_det = usage.get("output_tokens_details") or {}
    if not isinstance(in_det, dict):
        in_det = {}
    if not isinstance(out_det, dict):
        out_det = {}

    audio_in = (dig(in_det, "audio_tokens", "input_audio_tokens")
                or dig(usage, "input_audio_tokens", "audio_input_tokens"))
    audio_out = (dig(out_det, "audio_tokens", "output_audio_tokens")
                 or dig(usage, "output_audio_tokens", "audio_output_tokens"))
    text_in = (dig(in_det, "text_tokens", "input_text_tokens")
               or dig(usage, "input_text_tokens", "text_input_tokens"))
    text_out = (dig(out_det, "text_tokens", "output_text_tokens")
                or dig(usage, "output_text_tokens", "text_output_tokens"))

    # 顶层总数兜底：细节字段缺失时，至少不要把用量记成 0
    if not audio_in and not text_in:
        text_in = dig(usage, "input_tokens")
    if not audio_out and not text_out:
        text_out = dig(usage, "output_tokens")

    cached = (dig(in_det, "cached_tokens")
              or dig(usage, "cached_tokens",
                     "input_cached_tokens"))

    return Usage(
        audio_in_tokens=audio_in, audio_out_tokens=audio_out,
        text_in_tokens=text_in, text_out_tokens=text_out,
        cached_tokens=cached)


def estimate_cost(u: Usage, price: Optional[dict] = None,
                  text_price: Optional[dict] = None) -> float:
    """
    估算费用（元）。

    参数名要说清楚作用范围 —— 早期版本只有一个 price，
    名字看像"全部单价"，实际只管音频，文本偷偷用模块常量，
    调用方改了却没生效。现在音频和文本各自可覆盖。

    注意：Realtime 音频也可能按时长计费，
    这里按 token 估算，仅为量级参考。
    """
    pa = price or DEFAULT_PRICE
    pt = text_price or TEXT_PRICE
    audio_in = u.audio_in_tokens / 1000 * pa["input_per_1k"]
    audio_out = u.audio_out_tokens / 1000 * pa["output_per_1k"]
    text_in = u.text_in_tokens / 1000 * pt["input_per_1k"]
    text_out = u.text_out_tokens / 1000 * pt["output_per_1k"]
    return round(audio_in + audio_out + text_in + text_out, 4)


def format_usage(u: Usage) -> str:
    """给界面看的一行摘要。"""
    parts = []
    if u.audio_in_tokens:
        parts.append(f"输入音频 {u.audio_in_tokens:,}")
    if u.audio_out_tokens:
        parts.append(f"输出音频 {u.audio_out_tokens:,}")
    if u.text_in_tokens:
        parts.append(f"输入文本 {u.text_in_tokens:,}")
    if u.text_out_tokens:
        parts.append(f"输出文本 {u.text_out_tokens:,}")
    if not parts:
        return "无用量数据"
    return " · ".join(parts) + " tokens"
