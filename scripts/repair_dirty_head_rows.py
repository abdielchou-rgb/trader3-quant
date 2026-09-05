#!/usr/bin/env python3
"""
repair_dirty_head_rows.py — 修复 qlib_bin 全字段"首行脏值"。

问题背景（2026-09-03 诊断）：
  全库约 137 只股票的 OHLCV/factor 首行被写成同一个大数 X（>50），
  而次行起才是真实前复权行情（首日价 1 元级）。模式：
      close:  [6186, 1, 0.908, ...]     600930 上市日
      factor: [6186, 0.139, 0.139, ...]  首行 factor 与 close 同源脏写
  净效应：数据读取后首日 return = X→1 ≈ -99%，触发 QC close_jump 告警 182 条
  （data/qc_report.json），并污染一切用到上市初期收益的计算。

根因（已确证）：
  首行全字段被整体写入同值 X（北交 920 段新股 ~6400；老上证 2000s 上市股 ~X），
  是灌库/拼接时首行占位被错误填成非价格大数，非真实行情、非复权事件。
  次行起序列连续真实（close[1]≈1 元级、factor 恒为复权因子）。

修复规则（保守、幂等、防误伤）：
  对命中股票，逐字段（open/high/low/close/factor/vwap/volume/amount）：
    仅当 close[0]>50 且 0<close[1]<10 且 factor[0]≈close[0]（同源脏写铁证）
    → 用次行值平替首行值：arr[0] = arr[1]
  只改首行；幂等（修后 close[0]==close[1] 不再命中）；不改日历/锚点/长度。

命中集（2026-09-03 全库精确扫描，阈值 20，补深市 20-50 区间漏网 8 只）：
  全库 6122 只中约 5176 只命中（上证 sh600xxx/000xxx + 深市次新 + 北交所 430/83x/920 段），
  覆盖 qc_report.json 中 87 只脏首行成分股。未命中：深市 1990s 老股（首行=0 占位，加载端剥离）。

安全机制：
  - 写前整目录备份到 <data_dir>/_backup_headfix_<ts>/
  - 原子写（tmp + os.replace）
  - 写后逐股用 QlibDataProvider 重新加载做契约校验，失败自动回滚该股
  - 完成后打印 QC 摘要（data/qc_report.json 的 close_jump 数预期大幅下降）

用法：
  python scripts/repair_dirty_head_rows.py            # 干跑：打印计划
  python scripts/repair_dirty_head_rows.py --apply    # 真正修复
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

BIN_DTYPE = "<f4"
# 全 8 字段首行同源脏写（open/high/low/close/factor/vwap/volume/amount 首值=同一大数 X，
# 次行起才是真实行情）。volume/amount 置 0 会被加载端剥离导致长度错位 → 契约失败，
# 故 8 字段统一"首值←次值"（长度不变、消除上市首日假收益）。
ALL_FIELDS = ["open", "high", "low", "close", "factor", "vwap", "volume", "amount"]

# 命中判定（防误伤）：仅当 bin 首两行即"脏 X → 真实价"且 close/factor 首值同源
HEAD_VALUE_MIN = 20.0      # 首值必须 > 20（深市 2000 年老股脏首行低至 24~47）
SECOND_VALUE_MAX = 10.0    # 次值必须 < 10（真实前复权首日价，1 元级）


def _data_dir() -> str:
    from trader3.data_provider import QlibDataProvider

    return QlibDataProvider().data_dir


def _build_repair_list(data_dir: str) -> list[str]:
    """修复名单 = 全库精确扫描（bin 首两行即"脏X→真实价"）。"""
    codes: set[str] = set()

    # 两阶段：先扫 close 命中（快），命中子集再验 factor 同源（铁证，避免误伤）。
    feat_root = os.path.join(data_dir, "features")
    close_hits: list[str] = []
    for dirn in sorted(os.listdir(feat_root)):
        cp = os.path.join(feat_root, dirn, "close.day.bin")
        if not os.path.exists(cp):
            continue
        try:
            c = np.memmap(cp, dtype=BIN_DTYPE, mode="r")
        except Exception:
            continue
        if c.size < 3:
            continue
        c0, c1 = float(c[0]), float(c[1])
        if c0 > HEAD_VALUE_MIN and 0 < c1 < SECOND_VALUE_MAX:
            close_hits.append(dirn)

    for dirn in close_hits:
        fp = os.path.join(feat_root, dirn, "factor.day.bin")
        if not os.path.exists(fp):
            continue
        try:
            f = np.memmap(fp, dtype=BIN_DTYPE, mode="r")
        except Exception:
            continue
        cp = os.path.join(feat_root, dirn, "close.day.bin")
        c0 = float(np.memmap(cp, dtype=BIN_DTYPE, mode="r")[0])
        if f.size >= 1 and abs(float(f[0]) / c0 - 1.0) < 1e-3:  # factor[0]≈close[0]
            codes.add(dirn)

    return sorted(codes)


def _needs_fix(data_dir: str, code: str) -> bool:
    """首行是否仍脏（幂等：修过后 close[0]==close[1] 不再命中）。"""
    cp = os.path.join(data_dir, "features", code, "close.day.bin")
    if not os.path.exists(cp):
        return False
    try:
        arr = np.memmap(cp, dtype=BIN_DTYPE, mode="r")
    except Exception:
        return False
    return arr.size >= 3 and arr[0] > HEAD_VALUE_MIN and 0 < arr[1] < SECOND_VALUE_MAX


def _fix_one(data_dir: str, code: str, backup_dir: str, apply: bool) -> tuple[str, list[str], bool]:
    """修复单股：返回 (code, 改动字段, ok)。apply=False 仅计划。"""
    feat = os.path.join(data_dir, "features", code)
    changed: list[str] = []
    ok = True
    for f in ALL_FIELDS:
        p = os.path.join(feat, f"{f}.day.bin")
        if not os.path.exists(p):
            continue
        try:
            arr = np.fromfile(p, dtype=BIN_DTYPE).copy()
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {code} {f} 读取失败: {exc}")
            ok = False
            continue
        if arr.size < 2:
            continue
        # 仅当该字段首行仍脏才写（幂等）；非脏字段跳过
        v0, v1 = float(arr[0]), float(arr[1])
        if not (v0 > HEAD_VALUE_MIN and 0 < v1 < SECOND_VALUE_MAX):
            continue
        if apply:
            if backup_dir:
                os.makedirs(os.path.join(backup_dir, code), exist_ok=True)
                shutil.copy2(p, os.path.join(backup_dir, code, f"{f}.day.bin"))
            arr[0] = arr[1]  # 首值←次值
            tmp = p + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(arr.astype(BIN_DTYPE).tobytes())
            os.replace(tmp, p)
        changed.append(f)
    return code, changed, ok


def main() -> int:
    ap = argparse.ArgumentParser(description="修复 qlib_bin 首行脏值（全库 ~5177 只 OHLCV/factor/volume/amount）")
    ap.add_argument("--apply", action="store_true", help="真正写入（默认 dry-run）")
    ap.add_argument("--data-dir", default="", help="覆盖 qlib 数据目录（测试用）")
    ap.add_argument("--list-file", default="", help="预扫描名单 json（跳过全库扫描，加速）")
    args = ap.parse_args()

    from trader3.data_provider import QlibDataProvider

    dp = QlibDataProvider(data_dir=args.data_dir or None)
    data_dir = dp.data_dir
    print(f"数据目录: {data_dir}", flush=True)

    if args.list_file and os.path.exists(args.list_file):
        codes = sorted(json.load(open(args.list_file, encoding="utf-8")))
        print(f"从 {args.list_file} 读入名单 {len(codes)} 只（跳过全库扫描）", flush=True)
    else:
        codes = _build_repair_list(data_dir)
        print(f"全库扫描名单: {len(codes)}", flush=True)
    todo = [c for c in codes if _needs_fix(data_dir, c)]
    print(f"仍待修: {len(todo)}", flush=True)

    if not args.apply:
        print(f"[dry-run] {len(todo)} 只待修（--apply 生效）。干跑不逐行打印，避免缓冲。", flush=True)
        return 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.join(data_dir, f"_backup_headfix_{ts}")
    os.makedirs(backup_root, exist_ok=True)
    print(f"\n备份目录: {backup_root}", flush=True)

    failed: list[str] = []
    fixed = 0
    for i, c in enumerate(todo):
        code, changed, ok = _fix_one(data_dir, c, backup_root, apply=True)
        if changed:
            fixed += 1
        if not ok:
            failed.append(code)
        if (i + 1) % 500 == 0:
            print(f"  进度 {i+1}/{len(todo)} (fixed={fixed})", flush=True)

    # 写后契约校验：逐股重载
    bad: list[tuple[str, str]] = []
    for i, c in enumerate(todo):
        try:
            with __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore")
                dp.load_stock(c.upper(), "close")
        except Exception as exc:  # noqa: BLE001
            bad.append((c, str(exc)[:80]))
        if (i + 1) % 500 == 0:
            print(f"  校验 {i+1}/{len(todo)}", flush=True)
    print(f"\n修复 {fixed} 只。写后契约校验失败: {len(bad)}", flush=True)
    for c, msg in bad:
        print(f"  ✗ {c}: {msg}", flush=True)

    # 重跑 QC 摘要
    try:
        from trader3.v2.data_qc import run_qc, qc_summary_line

        report = run_qc(sample_limit=200)
        print("\nQC 摘要:", qc_summary_line(report), flush=True)
        print("critical:", report.get("critical"), "warnings:", report.get("warnings"), flush=True)
        jumps = sum(1 for o in report.get("offenders", []) if o.get("check") == "close_jump")
        print("close_jump offenders:", jumps, flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"\nQC 重跑失败: {exc}", flush=True)

    return 1 if (failed or bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
