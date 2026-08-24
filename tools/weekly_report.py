"""
P1-② 每周策略汇总报告
用法：python weekly_report.py [--output D:\Marvis\output\] [--week 2026-08-11]
"""
import argparse, json, os, sys
from datetime import datetime, timedelta
from collections import defaultdict

PROJECT = r"D:\Claude\projects\3号交易员"
STATE_FILE = os.path.join(PROJECT, "evolve", "last_daily_run.json")
BY_UNIVERSE = os.path.join(PROJECT, "evolve", "strategies", "by_universe")

def load_daily_states(output_dir: str, week_start: str) -> list:
    """加载本周所有日报"""
    daily_files = sorted([
        f for f in os.listdir(output_dir)
        if f.startswith("daily_") and f.endswith(".json")
        and f >= f"daily_{week_start.replace('-', '')}"
    ])
    states = []
    for fname in daily_files[:7]:
        path = os.path.join(output_dir, fname)
        states.append(json.load(open(path, "r", encoding="utf-8")))
    return states

def build_report(week_start: str, week_end: str, states: list) -> dict:
    """汇总周报"""
    universes = set()
    for state in states:
        for u in state.get("strategies", {}):
            universes.add(u)

    report = {
        "report_type": "weekly",
        "week": f"{week_start} ~ {week_end}",
        "days_with_data": len(states),
        "universes": defaultdict(list),
    }

    for state in states:
        date = state["date"]
        for u in universes:
            s = state.get("strategies", {}).get(u)
            if s:
                report["universes"][u].append({
                    "date": date,
                    "count": s["count"],
                    "top_score": s["top_score"],
                    "top_ic": s["top_ic"],
                })

    # 周均统计
    report["summary"] = {}
    for u in sorted(universes):
        entries = report["universes"][u]
        if entries:
            report["summary"][u] = {
                "avg_count": sum(e["count"] for e in entries) / len(entries),
                "avg_score": sum(e["top_score"] for e in entries) / len(entries),
                "avg_ic": sum(e["top_ic"] or 0 for e in entries) / len(entries),
                "best_day": max(entries, key=lambda e: e["top_score"]),
                "total_days": len(entries),
            }

    return report

def write_markdown_report(report: dict, output_dir: str) -> str:
    """写 Markdown 周报"""
    week_str = report["week"].split("~")[0].strip().replace("-", "")
    md_path = os.path.join(output_dir, f"weekly_{week_str}.md")

    lines = [
        f"# 策略进化周报 — {report['week']}",
        "",
        f"覆盖天数: {report['days_with_data']}",
        "",
        "## 周均指标",
        "",
        "| Universe | 日均策略数 | 周均 Top Score | 周均 IC | 最优日 |",
        "|----------|-----------|---------------|--------|--------|",
    ]

    for u, s in sorted(report.get("summary", {}).items()):
        lines.append(
            f"| {u} | {s['avg_count']:.1f} | {s['avg_score']:.4f} | "
            f"{s['avg_ic']:.4f} | {s['best_day']['date']} ({s['best_day']['top_score']:.4f}) |"
        )

    # 日明细
    lines += ["", "## 每日明细", ""]
    for u in sorted(report.get("universes", {})):
        lines.append(f"### {u}")
        lines.append("")
        lines.append("| 日期 | 策略数 | Top Score | Top IC |")
        lines.append("|------|--------|-----------|--------|")
        for entry in report["universes"][u]:
            top_ic = entry.get("top_ic")
            lines.append(
                f"| {entry['date']} | {entry['count']} | "
                f"{(entry.get('top_score') or 0):.4f} | "
                f"{('%s' % top_ic) if top_ic is None else format(float(top_ic), '.4f')} |"
            )
        lines.append("")

    lines.append(f"---\n*报告由 weekly_report.py 自动生成*")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return md_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=r"D:\Marvis\output")
    parser.add_argument("--week", default=None,
                        help="周一日期 YYYY-MM-DD，默认本周一")
    args = parser.parse_args()

    if args.week:
        week_start_date = datetime.strptime(args.week, "%Y-%m-%d")
    else:
        today = datetime.now()
        week_start_date = today - timedelta(days=today.weekday())

    week_end_date = week_start_date + timedelta(days=6)
    week_start = week_start_date.strftime("%Y-%m-%d")
    week_end = week_end_date.strftime("%Y-%m-%d")

    states = load_daily_states(args.output, week_start)
    report = build_report(week_start, week_end, states)

    # JSON 数据
    json_path = os.path.join(args.output, f"weekly_{week_start.replace('-', '')}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Markdown 报告
    md_path = write_markdown_report(report, args.output)

    print(f"📊 周报: {json_path}")
    print(f"📄 Markdown: {md_path}")
    print(f"   覆盖 {len(states)} 天数据, {len(report['summary'])} 个 universe")

if __name__ == "__main__":
    main()
