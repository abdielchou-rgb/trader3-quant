#!/usr/bin/env python3
"""
repair_splice_tail_20260903.py — 修复 8-24 尾部 append 的前复权基准断裂(正确版)。

背景：update_market_data --stocks 对 csi300 尾部追加 8 个交易日时，新浪 qfq 是
"最新复权基准"，与库内"旧基准"历史价在 8-21→8-24 边界出现 50%~2000% 假跳。
原 repair_splice_20260825.py 用全局日历 index 当 bin index —— 对上市日晚于边界
的次新正确，对老股(上市日 < 边界)全错(错位 i0 天)。

本版按"相对上市锚点"定位边界(rel = 边界全局idx - 上市日idx)，对 bin[rel+1:] *= p0/v1
消除跨基准跳变。幂等：8-21 与 8-24 已连续(比例≈1)则跳过。

用法：
  python scripts/repair_splice_tail_20260903.py            # 干跑
  python scripts/repair_splice_tail_20260903.py --apply    # 修复
"""
from __future__ import annotations
import argparse, bisect, json, os, shutil, sys, time, warnings

import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trader3.data_provider import QlibDataProvider

BOUNDARY = "2026-08-21"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dp = QlibDataProvider()
    data_dir = dp.data_dir
    cal = dp.calendar()
    bpos = bisect.bisect_left(cal, BOUNDARY)
    codes = dp.instruments("csi300", asof_date="2026-09-02")
    anch = {}
    with open(os.path.join(data_dir, "instruments", "all.txt"), encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("\t")
            if parts:
                anch.setdefault(parts[0].upper(), parts[1])
    print(f"边界 {BOUNDARY}(idx {bpos}) 候选 {len(codes)}", flush=True)
    bk = os.path.join(data_dir, f"_backup_splice2_{time.strftime('%Y%m%d_%H%M%S')}")
    rescaled = cont = skip = 0
    for i, code in enumerate(codes):
        if args.limit and i >= args.limit:
            break
        lst = anch.get(code, "")
        if not lst or lst > BOUNDARY:
            skip += 1
            continue
        i0 = bisect.bisect_left(cal, lst)
        rel = bpos - i0
        feat = os.path.join(data_dir, "features", code.lower())
        cpath = os.path.join(feat, "close.day.bin")
        if not os.path.exists(cpath):
            skip += 1
            continue
        arr0 = np.fromfile(cpath, dtype="<f4")
        if rel + 1 >= arr0.size:
            skip += 1
            continue
        p0 = float(arr0[rel])       # 8-21
        v1 = float(arr0[rel + 1])   # 8-24
        if p0 <= 0 or v1 <= 0:
            skip += 1
            continue
        fac = p0 / v1
        if abs(fac - 1.0) < 1e-4:
            cont += 1
            continue
        if args.apply:
            ok = True
            for fname in sorted(os.listdir(feat)):
                if not fname.endswith(".day.bin"):
                    continue
                p = os.path.join(feat, fname)
                arr = np.fromfile(p, dtype="<f4")
                if arr.size != arr0.size:
                    ok = False
                    break
                dstdir = os.path.join(bk, code.lower())
                os.makedirs(dstdir, exist_ok=True)
                if not os.path.exists(os.path.join(dstdir, fname)):
                    shutil.copy2(p, os.path.join(dstdir, fname))
                out = arr.copy()
                out[rel + 1 :] *= fac
                tmp = p + ".tmp"
                with open(tmp, "wb") as fh:
                    fh.write(out.astype("<f4").tobytes())
                os.replace(tmp, p)
            if not ok:
                skip += 1
                continue
        rescaled += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(codes)} rescale={rescaled} cont={cont}", flush=True)
    print(f"done rescale={rescaled} cont={cont} skip={skip} 备份={bk if args.apply else '(dry-run 未写)'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
