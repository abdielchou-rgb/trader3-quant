#!/usr/bin/env python3
"""
run_stock_append_batch.py — 断点续跑个股尾部追加（不重抓指数/不重建成分段）。

用途：update_market_data --stocks 在网络受限下每批只能处理 ~100 只。
本脚本直接调用 update_market_data.plan_stocks + apply_stock_append，
跳过指数段/成分段固定开销，幂等续跑直到 csi300（或指定 universe）全部最新。

用法：
  python scripts/run_stock_append_batch.py            # 追加剩余 ~120 只后退出
  python scripts/run_stock_append_batch.py --batch 60  # 单批 60 只
  python scripts/run_stock_append_batch.py --codes SH601688,SH601689
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

BIN_DTYPE = "<f4"


def _need_list(data_dir: str, universe: str) -> list[dict]:
    """plan_stocks 的只差列表（尾部未到最新）。"""
    import warnings
    warnings.filterwarnings("ignore")
    from trader3.data_provider import QlibDataProvider
    import scripts.update_market_data as um

    dp = QlibDataProvider(data_dir=data_dir)
    codes = dp.instruments(universe, asof_date=dp.calendar()[-1]) or dp.instruments(universe)
    plans = um.plan_stocks(dp, codes)
    return [p for p in plans if p.get("action") == "append"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=120, help="单批最多处理只数")
    ap.add_argument("--universe", default="csi300")
    ap.add_argument("--codes", default="", help="显式指定 code 列表（跳过 plan 扫描）")
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    from trader3.data_provider import QlibDataProvider
    import scripts.update_market_data as um

    dp = QlibDataProvider()
    data_dir = dp.data_dir
    cal_last = dp.calendar()[-1]
    print(f"数据目录: {data_dir}  日历末: {cal_last}", flush=True)

    if args.codes:
        plans = [{"code": c.upper(), "action": "append",
                  "k": 0, "new_dates": dp.calendar()[-8:]} for c in args.codes.split(",") if c.strip()]
    else:
        plans = _need_list(data_dir, args.universe)
    print(f"待追加: {len(plans)} 只", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.join(data_dir, f"_backup_append_{ts}")
    os.makedirs(backup_root, exist_ok=True)
    print(f"备份: {backup_root}", flush=True)

    fetch_start = dp.calendar()[max(0, len(dp.calendar()) - 40)]
    # 分批：每批 batch 只，备份目录共享
    batch = plans[: args.batch]
    summary = um.apply_stock_append(dp, batch, fetch_start, backup_root)
    print(f"本批: 追加 {len(summary['appended'])} 只", flush=True)
    for s in summary["skipped"][:10]:
        print("  skip:", s.get("code"), s.get("reason", "")[:80], flush=True)
    remaining = len(plans) - args.batch
    print(f"剩余: {max(remaining,0)} 只（再跑一次续批）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
