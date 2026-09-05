#!/usr/bin/env python3
"""
repair_splice_20260903.py — 8-24 append 尾部的前复权基线拼接修复(二次)

背景：update_market_data 的治愈追加把"新基准 qfq 价格"拼到"旧基准历史"上，
8月分红季导致 ~80% 成分股在边界日出现 >25% 假跳变。

修复：对每只已治愈股票，按边界处连续性重缩放尾部：
    factor = 边界前最后有效价 / 尾部首个有效价
    bin[boundary:] *= factor   （全部字段同一处理）
保留尾段内真实收益结构，消除跨基准假跳变；重复执行幂等（factor≈1）。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from scripts.update_market_data import _backup, read_bin, write_bin_atomic  # noqa: E402
from trader3.data_provider import QlibDataProvider  # noqa: E402

BOUNDARY_DATE = "2026-08-21"  # 首次治愈追加的日历首日
MARKER = "splice_repair_20260903.json"


def main() -> int:
    dp = QlibDataProvider()
    cal = dp.calendar()
    data_dir = dp.data_dir

    import bisect
    with open(os.path.join(data_dir, "instruments", "all.txt"), encoding="utf-8") as f:
        listing = {}
        for ln in f:
            parts = ln.strip().split("\t")
            if len(parts) >= 2:
                listing[parts[0].upper()] = parts[1]

    b_cal = cal.index(BOUNDARY_DATE)
    codes = dp.instruments("csi300")
    ts = __import__("time").strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.join(data_dir, f"_backup_splicefix_{ts}")
    stats = {"rescaled": 0, "skipped_shape": 0, "skipped_continuous": 0,
             "no_boundary": 0, "errors": 0}

    for code in codes:
        feat = os.path.join(data_dir, "features", code.lower())
        if not os.path.isdir(feat):
            continue
        marker_path = os.path.join(feat, MARKER)
        if os.path.exists(marker_path):
            continue
        files = [f for f in sorted(os.listdir(feat)) if f.endswith(".day.bin")]
        if not files:
            continue
        lst = listing.get(code.upper())
        if not lst or lst not in cal:
            stats["skipped_shape"] += 1
            continue
        i0 = bisect.bisect_left(cal, lst)
        bpos = b_cal - i0
        arr0 = read_bin(os.path.join(feat, files[0]))
        if bpos <= 0 or bpos >= len(arr0):
            stats["no_boundary"] += 1
            continue

        targets = []
        factors = []
        ok_stock = True
        for fname in files:
            arr = read_bin(os.path.join(feat, fname))
            if len(arr) != len(arr0):
                ok_stock = False
                break
            p0 = 0.0
            for k in range(bpos - 1, -1, -1):
                if arr[k] > 0 and np.isfinite(arr[k]):
                    p0 = float(arr[k])
                    break
            v0 = 0.0
            for k in range(bpos, len(arr)):
                if arr[k] > 0 and np.isfinite(arr[k]):
                    v0 = float(arr[k])
                    break
            if p0 <= 0 or v0 <= 0:
                ok_stock = False
                break
            factors.append((fname, arr, p0 / v0))

        if not ok_stock:
            stats["skipped_shape"] += 1
            continue

        max_factor = max(abs(1 - f) for _, _, f in factors)
        if max_factor < 1e-6:
            stats["skipped_continuous"] += 1
            with open(marker_path, "w", encoding="utf-8") as mf:
                json.dump({"status": "already_continuous"}, mf)
            continue

        try:
            _backup([os.path.join(feat, f) for f in files],
                    os.path.join(backup_root, code.lower()), data_dir)
            for fname, arr, fac in factors:
                out = arr.copy()
                out[bpos:] *= fac
                write_bin_atomic(os.path.join(feat, fname), out)
            with open(marker_path, "w", encoding="utf-8") as mf:
                json.dump({"status": "rescaled",
                           "max_factor_dev": round(max_factor, 4),
                           "boundary": BOUNDARY_DATE}, mf)
            stats["rescaled"] += 1
        except Exception as e:
            stats["errors"] += 1
            print(f"  ✗ {code}: {e}")

    print(json.dumps(stats, ensure_ascii=False))
    print("备份:", backup_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
