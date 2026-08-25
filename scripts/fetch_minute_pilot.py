#!/usr/bin/env python3
"""
fetch_minute_pilot.py — 5分钟线采集试点 CLI

为事件驱动回测储备 5 分钟 K 线（主源 akshare 东财，备源 baostock，双源降级）。
默认 dry-run 只打印计划与 QC 预览；写入必须 --apply（整文件原子替换）。

用法：
  python scripts/fetch_minute_pilot.py --codes 600519,000858 --days 20          # dry-run
  python scripts/fetch_minute_pilot.py --codes 600519,000858 --apply            # 真正落盘
  python scripts/fetch_minute_pilot.py --codes 600519,000858 --limit 1 --apply  # 控制规模
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from trader3.v2 import minute_data as md  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="5分钟线采集试点（默认 dry-run）")
    ap.add_argument("--codes", required=True, help="逗号分隔代码，如 600519,000858")
    ap.add_argument("--days", type=int, default=20, help="回溯自然日窗口（默认 20）")
    ap.add_argument("--apply", action="store_true", help="真正写入（默认 dry-run）")
    ap.add_argument("--limit", type=int, default=0, help="限制处理股票数（调试）")
    ap.add_argument("--out-dir", default="", help="覆盖输出目录（默认 data/minute_5）")
    args = ap.parse_args(argv)

    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    if args.limit > 0:
        codes = codes[:args.limit]
    if not codes:
        print("✗ 无有效股票代码")
        return 1

    base_dir = args.out_dir or md.default_dir()
    print(f"[plan] codes={len(codes)} days={args.days} "
          f"backend={md.active_backend()} out={base_dir} apply={args.apply}")

    ok = fail = 0
    for i, code in enumerate(codes, 1):
        try:
            rows, source = md.fetch_minute_verbose(code, days=args.days)
        except Exception as e:
            print(f"[{i}/{len(codes)}] {code} ✗ 抓取失败: {e}")
            fail += 1
            continue
        qc = md.qc_minute(md.to_frame(rows))
        tag = (f"rows={qc['rows']} dup={qc['dup']} inverted={qc['inverted']} "
               f"jumps={len(qc['jump_offenders'])}")
        if not rows:
            print(f"[{i}/{len(codes)}] {code} ⚠ 空数据 source={source}")
            fail += 1
            continue
        target = os.path.join(base_dir, md.bare_code(code) + md.backend_ext())
        if not args.apply:
            print(f"[{i}/{len(codes)}] {code} [dry-run] source={source} {tag}"
                  f" → {target}（未写入）")
            continue
        try:
            path = md.save_minute_atomic(code, rows, base_dir=base_dir)
        except Exception as e:
            print(f"[{i}/{len(codes)}] {code} ✗ 写入失败: {e}")
            fail += 1
            continue
        back = md.load_minute(code, base_dir=base_dir)
        good = back is not None and len(back) == len(rows)
        mark = "✓" if good else "⚠ 回读校验不一致"
        print(f"[{i}/{len(codes)}] {code} {mark} source={source} {tag} → {path}")
        ok += 1 if good else 0
        fail += 0 if good else 1

    print(f"[done] 写入 {ok} | 失败/空 {fail}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
