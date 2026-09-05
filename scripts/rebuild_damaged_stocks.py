#!/usr/bin/env python3
"""
rebuild_damaged_stocks.py — 整段重建被治愈管道截断/伪写的个股 bin。

背景（2026-09-03 诊断）：
  8-23/24 update_market_data --stocks 首次运行时，对部分"对齐漂移"股票执行了治愈追加，
  却把整段 bin 覆盖成 6~20 行的截断数据（bj920xxx 6 行正常；上证 sh600522/sh601058/sh601059
  20 行 6435 系伪序列），并在 all.txt 追加小写重复锚点（把真实上市日覆盖成 2026-07-27）。
  受影响股票被当"新股"，历史被静默砍掉（更隐蔽的：sh600522 连 QC contract 都通过）。

本次重建：
  1. all.txt 去重：删除小写重复行，恢复真实上市锚点（保留大写首行）
  2. 对受影响个股从新浪源整段重拉日线（qfq 前复权），
     写 close/open/high/low/volume/amount/vwap；factor 置 1.0（引擎不读该列，
     仅保持文件存在与长度对齐）。日历/停牌日对齐按上市锚点剥离首尾占位。
  3. 写前备份；写后 QlibDataProvider 契约校验。

用法：
  python scripts/rebuild_damaged_stocks.py --dry-run     # 计划
  python scripts/rebuild_damaged_stocks.py --apply       # 重拉+写入
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

BIN_DTYPE = "<f4"
FIELDS = ["open", "high", "low", "close", "volume", "amount", "vwap", "factor"]


def _list_all_codes(data_dir: str) -> dict[str, list[str]]:
    """解析 all.txt → {CODE: [lines]}（保留所有行，供去重）。"""
    all_txt = os.path.join(data_dir, "instruments", "all.txt")
    out: dict[str, list[str]] = {}
    if not os.path.exists(all_txt):
        return out
    with open(all_txt, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("\t")
            if parts and parts[0].strip():
                out.setdefault(parts[0].strip().upper(), []).append(ln.rstrip("\n"))
    return out


def _dedup_all_txt(data_dir: str) -> list[str]:
    """删除 all.txt 中大小写重复行（保留首行）。返回被删的 code 列表。"""
    path = os.path.join(data_dir, "instruments", "all.txt")
    if not os.path.exists(path):
        return []
    groups: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("\t")
            if parts and parts[0].strip():
                key = parts[0].strip().upper()
                # 同一 code 若首行已存在（大写先到），后续小写重复行丢弃
                if key in groups:
                    groups[key].append(ln.rstrip("\n"))
                else:
                    groups[key] = [ln.rstrip("\n")]
    removed = [k for k, v in groups.items() if len(v) > 1]
    if not removed:
        return []
    with open(path, "w", encoding="utf-8") as f:
        for k in sorted(groups.keys()):
            f.write(groups[k][0] + "\n")
    return removed


def _fetch_sina(code: str) -> list[dict]:
    """新浪全史日线 qfq。返回 [{date,open,high,low,close,volume,amount}, ...]（升序）。"""
    import akshare as ak

    norm = re.sub(r"^(SH|SZ|BJ)", "", code.upper())
    prefix = "sh" if norm.startswith("6") else ("bj" if norm.startswith(("4", "8", "9")) else "sz")
    df = ak.stock_zh_a_daily(symbol=prefix + norm, adjust="qfq")
    rows = []
    for _, r in df.iterrows():
        d = str(r["date"])[:10]
        close = float(r["close"])
        if close <= 0:
            continue
        rows.append({
            "date": d,
            "open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]), "close": close,
            "volume": float(r.get("volume", 0) or 0),
            "amount": float(r.get("amount", 0) or 0),
        })
    return rows


def _find_start_idx(cal: list[str], listing: str) -> int:
    import bisect
    return bisect.bisect_left(cal, listing)


def rebuild_one(data_dir: str, code: str, cal: list[str],
                backup_root: str, apply: bool) -> tuple[str, str, str]:
    """重拉单只。返回 (code, status, msg)。"""
    listing = ""
    all_txt = os.path.join(data_dir, "instruments", "all.txt")
    with open(all_txt, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("\t")
            if parts and parts[0].strip().upper() == code:
                listing = parts[1]
                break
    if not listing:
        return code, "skip", "all.txt 无锚点"
    feat = os.path.join(data_dir, "features", code.lower())
    if not os.path.isdir(feat):
        return code, "skip", "无 features 目录"
    i0 = _find_start_idx(cal, listing)

    try:
        rows = _fetch_sina(code)
    except Exception as exc:  # noqa: BLE001
        return code, "error", f"抓取失败: {str(exc)[:80]}"
    if len(rows) < 100:
        return code, "error", f"抓取过短 {len(rows)}"

    # 对齐：rows 日期 → 日历位置（过滤不在日历的日期；交易日缺失留 0 占位）
    cal_idx = {d: i for i, d in enumerate(cal)}
    length = len(cal) - i0
    buf: dict[str, np.ndarray] = {
        f: np.zeros(length, dtype=np.float32) for f in FIELDS}
    for r in rows:
        i = cal_idx.get(r["date"])
        if i is None or i < i0 or i >= len(cal):
            continue
        j = i - i0
        buf["open"][j] = r["open"]; buf["high"][j] = r["high"]
        buf["low"][j] = r["low"]; buf["close"][j] = r["close"]
        buf["volume"][j] = r["volume"]; buf["amount"][j] = r["amount"]
        v = r["amount"] / r["volume"] if r["volume"] > 0 else 0.0
        buf["vwap"][j] = v
    buf["factor"][:] = 1.0  # 引擎不读该列；保长度

    # 有真实数据的天数
    nz = int(np.sum(buf["close"] > 0))
    if nz < 100:
        return code, "error", f"对齐后真实数据过少 {nz}"

    if apply:
        if backup_root:
            os.makedirs(os.path.join(backup_root, code.lower()), exist_ok=True)
        for fld in FIELDS:
            p = os.path.join(feat, f"{fld}.day.bin")
            if os.path.exists(p) and backup_root:
                shutil.copy2(p, os.path.join(backup_root, code.lower(), f"{fld}.day.bin"))
            tmp = p + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(buf[fld].tobytes())
            os.replace(tmp, p)
    return code, "ok", f"len={length} 真数据={nz}"


def main() -> int:
    ap = argparse.ArgumentParser(description="重建被截断的个股 bin（21 只受影响集 + all.txt 去重）")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--codes", default="", help="逗号分隔重建名单（默认读内置 21 只）")
    ap.add_argument("--limit", type=int, default=0, help="限制处理只数（调试）")
    args = ap.parse_args()

    from trader3.data_provider import QlibDataProvider
    dp = QlibDataProvider()
    data_dir = dp.data_dir
    cal = list(dp.calendar())
    print(f"数据目录: {data_dir}  日历: {len(cal)} 天", flush=True)

    # 0) all.txt 去重（无条件，幂等）
    removed = _dedup_all_txt(data_dir)
    print(f"all.txt 去重: 删除 {len(removed)} 只重复小写锚点行: {removed}", flush=True)

    # 1) 名单
    default = ["SH600522", "SH600523", "SH600525", "SH600526", "SH600527",
               "SH601033", "SH601038", "SH601058", "SH601059", "SH601061",
               "SH601065", "SH603306", "SH688577", "SH688578", "SH688579",
               "BJ920690", "BJ920717", "BJ920718", "BJ920719", "SZ000407", "SZ000408"]
    codes = [c.upper() for c in args.codes.split(",") if c.strip()] or default
    if args.limit:
        codes = codes[:args.limit]

    print(f"\n名单 {len(codes)} 只: {codes}", flush=True)
    if not args.apply:
        print("[dry-run] 预览每只当前 bin 长度（--apply 重拉整段）:", flush=True)
        for c in codes:
            p = os.path.join(data_dir, "features", c.lower(), "close.day.bin")
            if os.path.exists(p):
                arr = np.fromfile(p, dtype=BIN_DTYPE)
                print(f"  {c}: bin {len(arr)} 值，锚点 {cal[:1] and [x for x in []]}",
                      flush=True)
        return 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.join(data_dir, f"_backup_rebuild_{ts}")
    os.makedirs(backup_root, exist_ok=True)
    print(f"\n备份: {backup_root}", flush=True)

    ok = err = 0
    for i, c in enumerate(codes):
        code, status, msg = rebuild_one(data_dir, c, cal, backup_root, apply=True)
        print(f"  [{i+1}/{len(codes)}] {code}: {status} — {msg}", flush=True)
        if status == "ok":
            ok += 1
        elif status == "error":
            err += 1
        # 契约校验
        if status == "ok":
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    close, dates = dp.load_stock(code, "close")
                print(f"      契约 OK: len={len(close)} {dates[0]}~{dates[-1]}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"      契约失败: {str(exc)[:100]}", flush=True)
                err += 1
    print(f"\n完成: ok={ok} err={err}", flush=True)
    return 1 if err else 0


if __name__ == "__main__":
    raise SystemExit(main())
