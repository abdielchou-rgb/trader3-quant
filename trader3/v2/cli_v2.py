"""3号交易员 v2.0 — CLI 子命令入口

用法:
    python -m trader3.v2.cli_v2 watchlist --action list
    python -m trader3.v2.cli_v2 watchlist --action add --code 600519 --name 贵州茅台
    python -m trader3.v2.cli_v2 watchlist --action transition --code 600519 --status 关注
    python -m trader3.v2.cli_v2 trigger --codes 600519,000858
    python -m trader3.v2.cli_v2 comps --code 600519 --industry 白酒
    python -m trader3.v2.cli_v2 daily --codes 600519,000858
    python -m trader3.v2.cli_v2 collect --codes 600519
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from trader3 import Trader3


def main():
    p = argparse.ArgumentParser(description="3hao v2.0 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("watchlist", help="自选股管理")
    w.add_argument("--action", default="list", choices=["list", "add", "remove", "transition", "events"])
    w.add_argument("--code", default="")
    w.add_argument("--name", default="")
    w.add_argument("--status", default="")
    w.add_argument("--reason", default="")
    w.add_argument("--note", default="")

    tr = sub.add_parser("trigger", help="三因子触发扫描")
    tr.add_argument("--codes", default="")

    c = sub.add_parser("comps", help="可比公司分析")
    c.add_argument("--code", required=True)
    c.add_argument("--industry", default="")

    d = sub.add_parser("daily", help="每日扫描管线")
    d.add_argument("--codes", default="")
    d.add_argument("--collect-only", action="store_true")

    cl = sub.add_parser("collect", help="仅采集事件")
    cl.add_argument("--codes", default="")

    args = p.parse_args()
    t3 = Trader3()

    if args.cmd == "watchlist":
        _watchlist(t3, args)
    elif args.cmd == "trigger":
        _trigger(t3, args)
    elif args.cmd == "comps":
        _comps(t3, args)
    elif args.cmd == "daily":
        _daily(t3, args)
    elif args.cmd == "collect":
        _collect(args)


def _watchlist(t3, args):
    if args.action == "add":
        ok = t3.watchlist_add(args.code, args.name, args.note)
        print(f"加入{'成功' if ok else '已存在'}: {args.code} {args.name}")
    elif args.action == "remove":
        ok = t3.watchlist_remove(args.code, args.reason)
        print(f"移除{'成功' if ok else '不在跟踪'}: {args.code}")
    elif args.action == "transition":
        ok = t3.watchlist_transition(args.code, args.status, args.reason)
        print(f"状态迁移{'成功' if ok else '非法'}: {args.code} -> {args.status}")
    elif args.action == "events":
        for e in t3.watchlist_events(args.code):
            print(f"  {e['event_time']} {e['code']}: {e['from_status']}->{e['to_status']} {e['reason']}")
    else:
        items = t3.watchlist_list()
        print(f"{'状态':<6} {'代码':<12} {'名称':<12} 信号/锚/现价")
        for i in items:
            print(f"  {i['status']:<6} {i['code']:<12} {i['name']:<12} "
                  f"{i['trigger_score']:.2f}/{i['valuation_anchor']:.1f}/{i['current_price']:.1f}")


def _trigger(t3, args):
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] if args.codes else None
    results = t3.trigger_scan()
    if codes:
        results = [r for r in results if r["code"] in codes]
    print("=== 三因子触发扫描 ===")
    for r in results:
        mark = "TRIGGER" if r["triggered"] else "watch"
        print(f"{mark} {r['code']} 催化{r['catalyst_score']:.2f} 估值{r['valuation_score']:.2f} "
              f"技术{r['tech_score']:.2f} 信号{r['score']:.2f}")
        print(f"   {r['reason']}")


def _comps(t3, args):
    table = t3.comps_analysis(args.code, industry=args.industry)
    print(f"=== 可比分析 {args.code}({table['target_name']}) 行业={table['industry']} ===")
    print("结论:", table["conclusion"])
    for row in table["peers"]:
        tag = "T" if row["is_target"] else "P"
        outl = " *离群*" if row["outlier"] else ""
        print(f"  [{tag}] {row['code']} {row['name']:<10} P/E {row['pe']:.1f} "
              f"EV/EBITDA {row['ev_ebitda']:.1f} EV/Rev {row['ev_revenue']:.1f}{outl}")
    for ca in table["caveats"]:
        print(f"  注: {ca}")


def _daily(t3, args):
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] if args.codes else None
    summary = t3.daily_run(codes=codes, collect_only=args.collect_only)
    print(t3.daily_alert_text(summary))


def _collect(args):
    from trader3.v2.collector import DataCollector
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] if args.codes else ["600519"]
    collector = DataCollector()
    total = 0
    for code in codes:
        n = collector.collect_stock(code, source="news")
        print(f"采集 {code}: 新增 {n} 条新闻/公告")
        total += n
    n2 = collector.collect_flow_events()
    print(f"采集市场事件(涨停池/龙虎榜): 新增 {n2} 条")
    collector.close()


if __name__ == "__main__":
    main()