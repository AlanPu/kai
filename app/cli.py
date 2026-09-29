"""
命令行入口：在不涉及语音的情况下验证文本能力。

用法：
    python -m app.cli plan  "<主题 | 段落 | 文章 | 网址>"
    python -m app.cli check "<一句英文>"
    python -m app.cli try   "<内容>"        # 计划 + 逐步纠错演示
"""

from __future__ import annotations

import argparse
import sys

from .core.content import load_content
from .services.corrector import Corrector
from .services.planner import Planner, validate_plan

C = {"dim": "\033[2m", "b": "\033[1m", "cy": "\033[36m", "gr": "\033[32m",
     "yl": "\033[33m", "rd": "\033[31m", "r": "\033[0m"}

SEV_COLOR = {"critical": C["rd"], "minor": C["yl"], "ignore": C["dim"]}


def cmd_plan(args) -> int:
    content = load_content(args.input)
    print(f"\n{C['b']}输入类型{C['r']}: {content.kind}")
    if content.title:
        print(f"{C['b']}标题{C['r']}: {content.title}")
    if not content.ok:
        print(f"{C['rd']}✗ 内容不可用: {content.error}{C['r']}")
        return 1

    print(f"{C['dim']}正文 {len(content.text)} 字符{C['r']}\n")
    print("生成话题计划中…\n")

    plan = Planner().plan(content, n_topics=args.topics)

    print(f"{C['b']}── 素材摘要 ──{C['r']}")
    print(f"  {plan.background}\n")
    print(f"{C['b']}── 开场白 ──{C['r']}")
    print(f"  {C['cy']}{plan.opening}{C['r']}\n")
    print(f"{C['b']}── 话题 {len(plan.topics)} 个 ──{C['r']}")
    for i, t in enumerate(plan.topics, 1):
        print(f"\n  {C['b']}{i}. {t.title}{C['r']}")
        print(f"     {C['cy']}{t.prompt}{C['r']}")
        if t.why:
            print(f"     {C['dim']}{t.why}{C['r']}")
    if plan.vocabulary:
        print(f"\n{C['b']}── 可能用到的词 ──{C['r']}")
        print(f"  {', '.join(plan.vocabulary)}")

    issues = validate_plan(plan)
    if issues:
        print(f"\n{C['yl']}质量检查:{C['r']}")
        for i in issues:
            print(f"  ⚠️  {i}")
    else:
        print(f"\n{C['gr']}✓ 质量检查通过（话题均为开放式，能让你多说）{C['r']}")
    return 0


def cmd_check(args) -> int:
    text = args.input
    print(f"\n{C['b']}原句{C['r']}: {text}\n")
    print("分析中…\n")
    items = Corrector().check(text)
    if not items:
        print(f"{C['gr']}✓ 没有需要纠正的地方{C['r']}")
        return 0
    for it in items:
        col = SEV_COLOR.get(it.severity, "")
        print(f"  {col}[{it.severity}]{C['r']} {C['dim']}{it.kind}{C['r']}")
        print(f"    {C['rd']}{it.original}{C['r']} → {C['gr']}{it.suggestion}{C['r']}")
        if it.explanation:
            print(f"    {C['dim']}{it.explanation}{C['r']}")
        if it.word:
            print(f"    单词: {it.word} {it.phonetic or ''}")
        print()

    pron_minor = [i for i in items
                  if i.kind == "pronunciation" and i.severity == "minor"]
    if pron_minor:
        print(f"{C['dim']}（其中 {len(pron_minor)} 条为轻微发音问题，"
              f"实际陪练时不会打扰你）{C['r']}")
    return 0


def cmd_try(args) -> int:
    """完整演示：生成计划，再用几句典型发言验证纠错。"""
    rc = cmd_plan(args)
    if rc:
        return rc
    print(f"\n\n{C['b']}{'='*60}{C['r']}")
    print(f"{C['b']}用几句典型发言试试纠错{C['r']}")
    print(f"{C['b']}{'='*60}{C['r']}")
    for s in ["I go to park yesterday with my friend.",
              "I very like playing basketball.",
              "I think this is a good idea."]:
        print()
        cmd_check(argparse.Namespace(input=s))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="app.cli", description="英语口语陪练 — 文本能力验证工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("plan", help="生成话题计划")
    p1.add_argument("input", help="主题 / 段落 / 文章 / 网址")
    p1.add_argument("--topics", type=int, default=5)
    p1.set_defaults(func=cmd_plan)

    p2 = sub.add_parser("check", help="检查一句英文")
    p2.add_argument("input")
    p2.set_defaults(func=cmd_check)

    p3 = sub.add_parser("try", help="完整演示")
    p3.add_argument("input")
    p3.add_argument("--topics", type=int, default=5)
    p3.set_defaults(func=cmd_try)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n已中断")
        return 130


if __name__ == "__main__":
    sys.exit(main())
