#!/usr/bin/env python3
"""
提交前检查：这个仓库是公开的，个人数据绝不能进去。

用法：
    python scripts/check_public_safe.py          # 检查待提交的内容
    python scripts/check_public_safe.py --all    # 检查全部已跟踪文件

为什么需要这个脚本：
    .gitignore 只能挡住「未来新增」的文件。一旦某个文件曾经被
    提交过，之后再加进 .gitignore 也无效 —— 它仍然留在历史里，
    会跟着 push 一起上去。而且 git 的历史改写很麻烦，
    所以在提交前拦住，成本最低。

检查内容：
    · 不该入库的路径（.env / data/ / 模型 / 证书）
    · API key 形态的字符串
    · 绝对路径里的用户名
    · 邮箱、内网 IP
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------- 不该被跟踪的路径 ----------
FORBIDDEN_PATHS = [
    (re.compile(r"(^|/)\.env$"), "环境变量文件（含真实 key）"),
    (re.compile(r"(^|/)\.env\.(local|prod|production)$"), "环境变量文件"),
    (re.compile(r"^data/"), "个人数据目录（数据库、画像）"),
    (re.compile(r"\.(db|sqlite|sqlite3)$"), "数据库文件"),
    (re.compile(r"(^|/)voiceprint\.json$"), "声纹样本（生物特征）"),
    (re.compile(r"^models/"), "模型文件（体积大）"),
    (re.compile(r"\.(onnx|pem|key)$"), "模型或证书私钥"),
    (re.compile(r"^prototype/"), "冻结的原型"),
    (re.compile(r"\.(wav|pcm|mp3|m4a)$"), "音频（可能是个人录音）"),
    (re.compile(r"(^|/)certs?/"), "证书目录"),
]

# ---------- 不该出现的内容 ----------
# 注意 sk- 后面要求足够长，避免误伤 .env.example 里的占位符
#
# 关于 IP：不能笼统地禁掉所有 IP。127.0.0.1 / 0.0.0.0 是通用常量，
# content.py 里整段私网段是用来判断「是不是内网地址」的，
# 测试里也需要假的私网地址 —— 一律拦截只会淹没真正的问题。
# 会泄露网络结构的是「本机实际在用的那个私网地址」，
# 而它一定不是 .1、.0 这类典型样例值。
IP_RULE = re.compile(
    r"\b192\.168\.(\d{1,3})\.(\d{1,3})\b"
    r"|\b10\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b"
    r"|\b172\.(1[6-9]|2\d|3[01])\.(\d{1,3})\.(\d{1,3})\b"
)

# 常见到可以忽略的地址：回环、通配、DNS、以及明显是样例的末端值
IP_IGNORE = {"127.0.0.1", "0.0.0.0", "8.8.8.8", "255.255.255.0"}

CONTENT_RULES = [
    (re.compile(r"sk-[A-Za-z0-9_-]{24,}"), "疑似 API key"),
    (re.compile(r"/Users/[A-Za-z0-9._-]+/"), "绝对路径里的用户名"),
    (re.compile(r"/home/[A-Za-z0-9._-]+/"), "绝对路径里的用户名"),
    (IP_RULE, "私网 IP（可能暴露本机网络地址）"),
    (re.compile(r"[A-Za-z0-9._%+-]+@(gmail|qq|163|outlook|hotmail|"
                r"yahoo|icloud|foxmail)\.[A-Za-z]{2,}"),
     "个人邮箱"),
]

# 这些文件允许出现「像 IP / 邮箱」的内容
ALLOWLIST_FILES = {
    ".env.example",           # 占位符
    "scripts/check_public_safe.py",   # 本文件自己就写着这些正则
    # SSRF 防护测试：拿私网地址当被拦截的样例，是测试数据不是本机地址
    "tests/test_stage3_voice.py",
}


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return [f for f in out.stdout.splitlines() if f]


def staged_files() -> list[str]:
    out = subprocess.run(["git", "diff", "--cached", "--name-only",
                          "--diff-filter=ACMR"],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    return [f for f in out.stdout.splitlines() if f]


def check(files: list[str]) -> list[str]:
    problems: list[str] = []

    for rel in files:
        for pat, why in FORBIDDEN_PATHS:
            if pat.search(rel):
                problems.append(f"❌ 不该提交的路径: {rel} —— {why}")
                break

        if rel in ALLOWLIST_FILES:
            continue

        path = ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for pat, why in CONTENT_RULES:
            for m in pat.finditer(text):
                snippet = m.group(0)
                if snippet in IP_IGNORE:
                    continue
                # 末端是 .0 / .1 的多半是网段说明或测试样例，
                # 本机实际地址极少长这样（实测用的是 <your-ip>）
                if why.startswith("私网 IP") and snippet.rsplit(".", 1)[-1] in {"0", "1"}:
                    continue
                line = text[:m.start()].count("\n") + 1
                if len(snippet) > 40:
                    snippet = snippet[:40] + "…"
                problems.append(f"❌ {rel}:{line} {why}: {snippet}")

    return problems


def main() -> int:
    check_all = "--all" in sys.argv
    files = tracked_files() if check_all else staged_files()

    if not files:
        print("✅ 没有待提交的文件")
        return 0

    problems = check(files)
    if problems:
        print(f"\n🚨 发现 {len(problems)} 处可能泄露个人数据的地方：\n")
        for p in problems:
            print("  " + p)
        print("\n这个仓库是公开的。请先处理，再提交。")
        print("如果确认是误报，把文件加进 ALLOWLIST_FILES。")
        return 1

    print(f"✅ 已检查 {len(files)} 个文件，未发现个人数据")
    return 0


if __name__ == "__main__":
    sys.exit(main())
