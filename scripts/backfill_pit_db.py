#!/usr/bin/env python3
"""
PIT 真库重灌脚本：financials.db（560 万行，无公告日列）→ financials_pit.db。

双时间戳策略（保守零穿越）：
  - report_date        = quarter（会计期截止）
  - publish_timestamp  = A股法定最晚披露日（季报次年/同年 4-30、半年报 8-31、
                          三季报 10-31、年报次年 4-30）
  真实披露可能更早，按最晚日回填保证 asof 检索零穿越；
  代价是部分时段取不到本可取到的值（保守偏差，方向安全）。

用法：
    python scripts/backfill_pit_db.py [--source <financials.db>] [--dest <financials_pit.db>]
    python scripts/backfill_pit_db.py --verify   # 重灌后抽查穿越数（必须=0）
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")  # GBK 控制台兼容

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from trader3.data.pit_loader import (  # noqa: E402
    PITFundamentalLoader,
    ensure_pit_schema,
    legal_publish_upper_bound,
)

DEFAULT_SOURCE = Path(r"D:\Claude\projects\2hao-analyst\data\financials.db")
DEFAULT_DEST = _PROJECT / "data" / "financials_pit.db"


def backfill(source: Path, dest: Path, batch_size: int = 200_000) -> int:
    if not source.exists():
        raise SystemExit(f"源库不存在: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()  # 全量重灌（幂等：清空重建）

    src = sqlite3.connect(str(source))
    loader = PITFundamentalLoader(str(dest))
    ensure_pit_schema(loader.conn)

    t0 = time.time()
    total = 0
    cur = src.execute(
        "SELECT code, quarter, field, value FROM financials "
        "WHERE value IS NOT NULL AND value != ''"
    )
    batch: list[tuple[str, str, str, int, float, int]] = []
    for code, quarter, field, value in cur.fetchall():
        try:
            pub_ms = legal_publish_upper_bound(str(quarter))
        except ValueError:
            continue  # 非法报告期（如 2026-09-30 中间期）跳过
        batch.append((str(code), str(field), str(quarter), pub_ms,
                      float(value), 0))
        if len(batch) >= batch_size:
            total += loader.insert_many(batch)
            batch = []
            print(f"  ... {total} rows ({time.time() - t0:.0f}s)", flush=True)
    if batch:
        total += loader.insert_many(batch)

    print(f"回灌完成: {total} rows → {dest} ({time.time() - t0:.0f}s)")
    src.close()
    loader.conn.close()
    return total


def verify(dest: Path, n_probe: int = 200) -> int:
    """抽查穿越数：随机 asof 时刻检索，断言每条可见记录 publish <= asof。返回违规数。"""
    import random

    ldr = PITFundamentalLoader(str(dest))
    row = ldr.conn.execute("SELECT COUNT(*) FROM financial_pit").fetchone()
    total = row[0]
    if total == 0:
        raise SystemExit("目标库为空")

    random.seed(42)
    violations = 0
    codes = [r[0] for r in ldr.conn.execute(
        "SELECT DISTINCT symbol FROM financial_pit LIMIT 200").fetchall()]
    for _ in range(n_probe):
        code = random.choice(codes)
        asof = random.randint(0, int(time.time() * 1000))
        cs = ldr.get_asof_cross_section([code], "epsTTM", asof)
        if code not in cs:
            continue
        # 检索到的值必须来自披露时刻 <= asof 的记录
        r = ldr.conn.execute(
            "SELECT publish_timestamp FROM financial_pit WHERE symbol=? "
            "AND field_name='epsTTM' AND value=? AND publish_timestamp<=?",
            (code, cs[code], asof),
        ).fetchone()
        if r is None:
            violations += 1
    ldr.conn.close()
    return violations


def main() -> None:
    ap = argparse.ArgumentParser(description="PIT 真库重灌")
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--verify", action="store_true", help="只做穿越抽查")
    args = ap.parse_args()

    if args.verify:
        v = verify(Path(args.dest))
        print(f"穿越违规数: {v} {'✓ 零穿越' if v == 0 else '✗ 存在穿越！'}")
        return

    backfill(Path(args.source), Path(args.dest))
    v = verify(Path(args.dest))
    print(f"回灌后穿越抽查: {v} 违规 {'✓' if v == 0 else '✗'}")


if __name__ == "__main__":
    main()
