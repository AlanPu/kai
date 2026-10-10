#!/usr/bin/env python3
"""
把历史纠错回填成复习清单。

背景
----
复习功能上线前，库里已经积累了几百条纠错（真实数据：76 次会话、
470 条）。这些是现成的复习材料，不回填的话复习页一开始是空的，
要等下一次练习才有内容 —— 那等于把已经付过学费的错误丢掉了。

本脚本按用户归纳历史纠错，写入 review_habits / review_items。

用法
----
    # 先看看会归纳出什么（不写库）
    .venv/bin/python scripts/backfill_review.py --dry-run

    # 真正写入（默认跳过"已掌握"的记录，不覆盖用户的复习进度）
    .venv/bin/python scripts/backfill_review.py

    # 指定用户 / 调高门槛（只看重复 ≥3 次的老毛病）
    .venv/bin/python scripts/backfill_review.py --user 15 --min 3

    # 清空该用户复习数据后重建
    .venv/bin/python scripts/backfill_review.py --reset

注意：默认 **不加 --reset** 时是增量累加，重复执行会把同一批
历史错误重复计数，次数会虚高。想得到准确数字请带 --reset。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import load_settings          # noqa: E402
from app.services.review import ReviewBuilder, summarize, top_items  # noqa: E402
from app.storage.db import Database                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="回填复习清单")
    ap.add_argument("--db", default=None, help="数据库路径（默认取配置）")
    ap.add_argument("--user", type=int, default=None,
                    help="只处理某个用户 id（默认为全部用户）")
    ap.add_argument("--min", type=int, default=2,
                    help="归为「反复犯的问题」所需的最少出现次数（默认 2）")
    ap.add_argument("--reset", action="store_true",
                    help="先清空该用户的复习数据再重建")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印归纳结果，不写数据库")
    args = ap.parse_args()

    settings = load_settings()
    db = Database(args.db or settings.db_path)
    db.init_schema()

    users = db.list_users()
    if args.user is not None:
        users = [u for u in users if u.id == args.user]
        if not users:
            print(f"没有 id={args.user} 的用户", file=sys.stderr)
            return 1

    if not users:
        print("还没有任何用户")
        return 0

    builder = ReviewBuilder(db)
    total_habits = total_items = 0

    for u in users:
        corrections = db.list_corrections_for_user(u.id)
        if not corrections:
            print(f"\n【{u.name}】没有纠错记录，跳过")
            continue

        habits = [h for h in summarize(corrections)
                  if h.occurrences >= args.min]
        items = top_items(corrections)
        kept = [it for h, it in items if h in {x.habit for x in habits}]

        print(f"\n【{u.name}】共 {len(corrections)} 条纠错 → "
              f"{len(habits)} 类问题、{len(kept)} 条具体说法")

        for h in habits:
            print(f"   {h.occurrences:4d} 次  {h.title}")

        # 固定搭配单独列出来 —— 这是用户最想复习的那一类
        collocs = [it for it in kept if it.collocation]
        if collocs:
            print(f"   —— 固定说法（{len(collocs)} 条，按出现次数）——")
            for it in collocs[:15]:
                print(f"     {it.occurrences:3d}x  {it.original[:40]!r}"
                      f" → {it.collocation[:50]!r}")

        if args.dry_run:
            continue

        result = builder.build(u.id, reset=args.reset,
                               min_occurrences=args.min)
        total_habits += result["habits"]
        total_items += result["items"]
        print(f"   ✓ 已写入：{result['habits']} 类问题、"
              f"{result['items']} 条具体说法")

    if args.dry_run:
        print("\n（--dry-run：未写入数据库）")
    else:
        print(f"\n完成：共 {total_habits} 类问题、{total_items} 条具体说法")

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
