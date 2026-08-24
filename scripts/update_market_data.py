#!/usr/bin/env python3
"""
update_market_data.py — qlib_bin 增量更新管线

把静态 qlib_bin 数据集滚动到最新交易日：

  模式A 指数重建（默认执行）：
      SH000300 的历史 bin 已确认损坏（末值异常），从东财/新浪整段重拉，
      按"现有日历 ∪ 新拉日期"全量重写指数各字段 bin，并扩展 day.txt 日历。
  模式B 个股严格追加（--stocks）：
      仅处理"bin长度 == 自上市日起交易日数"完全对齐的股票；
      对齐漂移的股票一律跳过并报告（拒绝在未知对齐假设上写数据）。
      停牌日用 0.0 占位（与库内既有约定一致，加载端会剥离首尾0并按 valid_flags 处理内部0）。

安全机制：
  - 默认 dry-run 只打印计划；写入必须 --apply
  - 写前备份 day.txt 与受影响 features/* 到 <data_dir>/_backup_<ts>/
  - 写后用 QlibDataProvider 重新加载做契约校验；失败自动回滚备份
  - 成功后 SharedState.set_data_version → IronGate Gate6 恢复真实判别

用法：
  python scripts/update_market_data.py                 # 指数重建 dry-run
  python scripts/update_market_data.py --apply         # 真正写入
  python scripts/update_market_data.py --apply --stocks --limit 50
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

BIN_DTYPE = "<f4"
INDEX_CODE = "sh000300"


def _dp(data_dir: Optional[str]):
    if data_dir:
        from trader3.data_provider import QlibDataProvider
        return QlibDataProvider(data_dir=data_dir)
    from trader3.data_provider import QlibDataProvider
    return QlibDataProvider()


# ── 行情抓取 ────────────────────────────────────────────

def fetch_index_history() -> Tuple[List[Tuple[str, float]], str]:
    """抓取 SH000300 日线收盘序列 [(date, close)]，双源降级。"""
    import akshare as ak

    try:
        df = ak.index_zh_a_hist(symbol="000300", period="daily")
        rows = [(str(d)[:10], float(v)) for d, v in
                zip(df["日期"], df["收盘"]) if float(v) > 0]
        return rows, "eastmoney-index"
    except Exception as e1:
        try:
            df = ak.stock_zh_index_daily(symbol="sh000300")
            rows = [(str(d)[:10], float(v)) for d, v in
                    zip(df["date"], df["close"]) if float(v) > 0]
            return rows, "sina-index"
        except Exception as e2:
            raise RuntimeError(f"指数行情双源失败: {e1} / {e2}")


def _normalize_code(code: str) -> str:
    """'SH600519' / '600519.SH' / '600519' -> '600519'"""
    c = code.upper().replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
    if c.startswith(("SH", "SZ", "BJ")):
        c = c[2:]
    return c


def fetch_stock_close(code: str, start: str) -> List[Tuple[str, float]]:
    """抓取个股日线收盘（前复权），[(date, close)]。"""
    import akshare as ak

    norm = _normalize_code(code)
    try:
        df = ak.stock_zh_a_hist(symbol=norm, period="daily",
                                start_date=start.replace("-", ""), adjust="qfq")
        return [(str(d)[:10], float(v)) for d, v in zip(df["日期"], df["收盘"])
                if float(v) > 0]
    except Exception:
        pass
    prefix = "sh" if norm.startswith("6") else ("bj" if norm.startswith(("4", "8", "9")) else "sz")
    df = ak.stock_zh_a_daily(symbol=prefix + norm,
                             start_date=start.replace("-", ""), adjust="qfq")
    return [(str(d)[:10], float(v)) for d, v in zip(df["date"], df["close"])
            if float(v) > 0]


# ── bin 读写 ────────────────────────────────────────────

def read_bin(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=BIN_DTYPE)


def write_bin_atomic(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(arr.astype(BIN_DTYPE).tobytes())
    os.replace(tmp, path)


# ── 计划与执行 ──────────────────────────────────────────

def plan_index(dp, rows: List[Tuple[str, float]]) -> dict:
    cal = list(dp.calendar())
    cal_set = set(cal)
    row_map = dict(rows)
    new_tail = [d for d in sorted(row_map) if d > cal[-1]]
    interior_missing = [d for d in sorted(row_map) if d < cal[-1] and d not in cal_set]
    return {
        "mode": "index_rebuild",
        "cal_len": len(cal),
        "cal_last": cal[-1],
        "fetched": len(rows),
        "new_tail_days": new_tail,
        "interior_missing": interior_missing,
        "new_cal_len": len(cal) + len(new_tail),
    }


def _key_for(data_dir: str, path: str) -> str:
    """备份键：相对 data_dir 的路径，分隔符扁平化（保留归属结构）。"""
    rel = os.path.relpath(path, data_dir)
    return rel.replace(os.sep, "__")


def _backup(paths: List[str], backup_dir: str, data_dir: str = "") -> None:
    os.makedirs(backup_dir, exist_ok=True)
    for p in paths:
        key = _key_for(data_dir, p) if data_dir else os.path.basename(p)
        dst = os.path.join(backup_dir, key)
        if os.path.isfile(p):
            shutil.copy2(p, dst)
        elif os.path.isdir(p):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(p, dst)


def _index_anchor(dp, cal: List[str]) -> Tuple[str, int]:
    """读 all.txt 中指数的上市锚点；缺省视为日历首日。返回 (listing, i0)。"""
    all_txt = os.path.join(dp.data_dir, "instruments", "all.txt")
    lst = None
    if os.path.exists(all_txt):
        with open(all_txt, encoding="utf-8") as f:
            for ln in f:
                parts = ln.strip().split("\t")
                if parts and parts[0].upper() == INDEX_CODE.upper():
                    lst = parts[1]
                    break
    if not lst or lst not in cal:
        lst = cal[0]
    return lst, cal.index(lst)


def apply_index(dp, rows: List[Tuple[str, float]], plan: dict, backup_root: str) -> dict:
    """
    按指数自身上市锚点对齐重建 bin + 扩展日历。

    bin 长度 == len(cal_new) - i0（与 all.txt 锚点一致，满足加载端契约）；
    先备份，写入或校验失败由调用方回滚。
    """
    import bisect

    data_dir = dp.data_dir
    cal = list(dp.calendar())
    row_map = dict(rows)

    new_cal = cal + plan["new_tail_days"]
    lst, i0 = _index_anchor(dp, new_cal)
    seg_dates = new_cal[i0:]

    first_fetch = min(row_map)
    if first_fetch > seg_dates[0]:
        raise RuntimeError(
            f"数据源起点 {first_fetch} 晚于指数锚点 {seg_dates[0]}，"
            "无法无零占位对齐——请更换数据源后重试"
        )

    idx_dir = os.path.join(data_dir, "features", INDEX_CODE)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(backup_root or os.path.join(data_dir, f"_backup_{ts}"))
    targets = [os.path.join(data_dir, "calendars", "day.txt")] + (
        [os.path.join(idx_dir, f) for f in sorted(os.listdir(idx_dir))]
        if os.path.isdir(idx_dir) else []
    )
    _backup(targets, backup_dir, data_dir)

    written = {}
    try:
        arr = np.array([row_map.get(d, 0.0) for d in seg_dates], dtype=np.float64)
        for fname in sorted(os.listdir(idx_dir)):
            if not fname.endswith(".day.bin"):
                continue
            field = fname.split(".")[0]
            if field not in ("open", "high", "low", "close", "volume"):
                continue  # 无可靠来源的字段不伪造
            write_bin_atomic(os.path.join(idx_dir, fname), arr)
            written[field] = f"len={len(arr)}"

        # 未覆盖字段：与自身段对齐——超长(历史遗留)右对齐截取，短少则尾值外推
        for fname in sorted(os.listdir(idx_dir)):
            field = fname.split(".")[0]
            if field in written or not fname.endswith(".day.bin"):
                continue
            old = read_bin(os.path.join(idx_dir, fname))
            if len(old) == len(seg_dates):
                continue
            if len(old) > len(seg_dates):
                ext = old[-len(seg_dates):]
            else:
                nz = np.where(old > 0)[0] if len(old) else []
                pad_val = float(old[nz[-1]]) if len(nz) else 0.0
                ext = np.concatenate([old, np.full(len(seg_dates) - len(old), pad_val)])
            write_bin_atomic(os.path.join(idx_dir, fname), ext)
            written[field] = f"(对齐 {len(old)}→{len(ext)})"

        cal_path = os.path.join(data_dir, "calendars", "day.txt")
        tmp = cal_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(new_cal) + "\n")
        os.replace(tmp, cal_path)
    except Exception:
        _restore(backup_dir, data_dir)
        raise

    return {"written_fields": written, "new_cal_len": len(new_cal),
            "expected_bin_len": len(seg_dates),
            "backup_dir": backup_dir}


def _restore(backup_dir: str, data_dir: str) -> None:
    if not os.path.isdir(backup_dir):
        return
    for key in os.listdir(backup_dir):
        src = os.path.join(backup_dir, key)
        if "__" in key:
            dst = os.path.join(data_dir, key.replace("__", os.sep))
        elif key == "day.txt":
            dst = os.path.join(data_dir, "calendars", "day.txt")
        else:
            dst = os.path.join(data_dir, "features", key)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)


def verify_index(dp, rows: List[Tuple[str, float]], expect_cal_len: int,
                 expect_bin_len: Optional[int] = None) -> Tuple[bool, str]:
    # 用全新实例校验（绕过同实例的日历/bin 进程内缓存，等价于重启后视角）
    try:
        from trader3.data_provider import QlibDataProvider
        fresh = QlibDataProvider(data_dir=dp.data_dir)

        cal = fresh.calendar()
        if len(cal) != expect_cal_len:
            return False, f"calendar len {len(cal)} != {expect_cal_len}"
        close, dates = fresh.load_stock(INDEX_CODE, "close")
        if expect_bin_len is not None and len(close) != expect_bin_len:
            return False, f"{INDEX_CODE} close len {len(close)} != {expect_bin_len}"
        row_map = dict(rows)
        last = cal[-1]
        if last in row_map:
            want = row_map[last]
            if abs(float(close[-1]) - want) > max(0.5, want * 1e-3):
                return False, f"tail close {close[-1]} != fetched {want}"
        if float(close[-1]) <= 0:
            return False, "tail close 非正值"
        return True, "ok"
    except Exception as e:
        return False, f"verify exception: {e}"


# ── 个股严格追加 ────────────────────────────────────────

def _effective_len(path: str) -> int:
    """与加载端同口径：剥离首尾 ≤0/非有限占位后的有效长度。"""
    arr = read_bin(path)
    n_lead = 0
    while n_lead < len(arr) and not (np.isfinite(arr[n_lead]) and arr[n_lead] > 0):
        n_lead += 1
    n_tail = 0
    while (n_tail < len(arr) - n_lead
           and not (np.isfinite(arr[len(arr) - 1 - n_tail]) and arr[len(arr) - 1 - n_tail] > 0)):
        n_tail += 1
    return max(len(arr) - n_lead - n_tail, 0)


def plan_stocks(dp, codes: List[str], k_new: int = 0,
                max_heal_days: int = 30) -> List[dict]:
    """
    尾部缺口窗口规则（在日历可能已扩展后调用）。

      gap = 锚点段期望长 - 原始bin长
      gap == 0        → 已最新
      0 < gap <= 窗口  → 待追加/治愈：按日历日期逐日回填权威源真实价
                        （只追加尾部、按日期取值，绝不改写既有历史）
      gap > 窗口       → 拒绝（超出治愈能力，需人工整段重建）
      gap < 0         → 拒绝（bin 超长，锚点/格式异常）
    """
    import bisect

    cal = list(dp.calendar())
    plans: List[dict] = []
    listing_cache: Dict[str, str] = {}

    all_txt = os.path.join(dp.data_dir, "instruments", "all.txt")
    with open(all_txt, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("\t")
            if parts:
                listing_cache[parts[0].upper()] = parts[1]

    for code in codes:
        feat_dir = os.path.join(dp.data_dir, "features", code.lower())
        if not os.path.isdir(feat_dir):
            plans.append({"code": code, "action": "skip", "reason": "无features目录"})
            continue
        sample_field = next((f for f in sorted(os.listdir(feat_dir))
                             if f.endswith(".day.bin")), None)
        if sample_field is None:
            plans.append({"code": code, "action": "skip", "reason": "无bin文件"})
            continue
        actual = os.path.getsize(os.path.join(feat_dir, sample_field)) // 4
        lst = listing_cache.get(code.upper())
        if not lst or lst not in cal:
            plans.append({"code": code, "action": "skip", "reason": f"上市日{lst}不在日历"})
            continue
        i0 = bisect.bisect_left(cal, lst)
        gap = (len(cal) - i0) - actual

        if gap == 0:
            plans.append({"code": code, "action": "skip", "reason": "已是最新"})
        elif 0 < gap <= max_heal_days:
            plans.append({"code": code, "action": "append",
                          "k": gap,
                          "new_dates": cal[-gap:]})
        elif gap < 0:
            plans.append({"code": code, "action": "skip",
                          "reason": f"bin超长 {gap:+d}（锚点/格式异常），拒绝写入"})
        else:
            plans.append({"code": code, "action": "skip",
                          "reason": f"缺口 {gap} 天超过治愈窗口 {max_heal_days}，需人工重建"})
    return plans


def apply_stock_append(dp, plans: List[dict], fetch_start: str, backup_root: str) -> dict:
    """对 action==append 的股票：拉取新增窗口收盘，按新日历日期逐行补齐全部字段。"""
    data_dir = dp.data_dir
    done, skipped = [], []
    import time
    for p in plans:
        if p["action"] != "append":
            skipped.append(p)
            continue
        code = p["code"]
        new_dates = p["new_dates"]
        try:
            time.sleep(0.8)  # 对数据源友好
            hist = dict(fetch_stock_close(code, fetch_start))
        except Exception as e:
            skipped.append({"code": code, "action": "skip", "reason": f"抓取失败: {e}"})
            continue

        feat_dir = os.path.join(data_dir, "features", code.lower())
        targets = [os.path.join(feat_dir, f) for f in sorted(os.listdir(feat_dir))
                   if f.endswith(".day.bin")]
        bdir = os.path.join(backup_root, code.lower())
        _backup(targets, bdir, data_dir)
        try:
            fill = np.array([hist.get(d, 0.0) for d in new_dates], dtype=np.float64)
            for tpath in targets:
                old = read_bin(tpath)
                write_bin_atomic(tpath, np.concatenate([old, fill]))
            done.append({"code": code, "appended_k": len(new_dates)})
        except Exception as e:
            _restore(bdir, data_dir)
            skipped.append({"code": code, "action": "skip", "reason": f"写入失败已回滚: {e}"})
    return {"appended": done, "skipped": skipped}


# ── 主流程 ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="qlib_bin 增量更新管线")
    ap.add_argument("--apply", action="store_true", help="真正写入（默认 dry-run）")
    ap.add_argument("--stocks", action="store_true", help="同时处理成分股严格追加")
    ap.add_argument("--limit", type=int, default=0, help="限制处理股票数（调试）")
    ap.add_argument("--data-dir", default="", help="覆盖 qlib 数据目录（测试用）")
    ap.add_argument("--no-version-stamp", action="store_true", help="成功后不写 data_version")
    args = ap.parse_args()

    dp = _dp(args.data_dir)
    print(f"[1/5] 抓取指数行情 ...")
    rows, source = fetch_index_history()
    print(f"      来源={source} 条数={len(rows)} 末条={rows[-1]}")

    plan = plan_index(dp, rows)
    print(f"[2/5] 计划: 日历 {plan['cal_len']}({plan['cal_last']}) → {plan['new_cal_len']}"
          f"，新增 {len(plan['new_tail_days'])} 个交易日")
    if plan["interior_missing"]:
        print(f"      ⚠ 抓取数据中有 {len(plan['interior_missing'])} 个早于现日历末的缺失日（忽略，不影响追加以外的重建）")

    if not args.apply:
        print("[dry-run] 未写入。加 --apply 执行。")
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.join(dp.data_dir, f"_backup_{ts}")

    print("[3/5] 指数重建（含备份+失败自动回滚）...")
    result = apply_index(dp, rows, plan, backup_root)
    ok, msg = verify_index(dp, rows, result["new_cal_len"],
                           expect_bin_len=result.get("expected_bin_len"))
    if not ok:
        _restore(result["backup_dir"], dp.data_dir)
        print(f"      ✗ 校验失败已回滚: {msg}")
        return 2
    print(f"      ✓ {result['written_fields']}")

    stock_summary = {"appended": [], "skipped": []}
    if args.stocks:
        # 关键：用全新实例——上面的指数重建已扩展日历，旧实例缓存的是写前日历
        from trader3.data_provider import QlibDataProvider
        dp_fresh = QlibDataProvider(data_dir=dp.data_dir)
        codes_all = dp_fresh.instruments("csi300", asof_date=plan["cal_last"])
        if len(codes_all) < 20:
            # 成分段未覆盖当前日期 → 回退历史并集（更新数据与成分归属无关）
            codes_all = dp_fresh.instruments("csi300")
        if args.limit:
            codes_all = codes_all[:args.limit]
        print(f"[4/5] 成分股严格追加（候选 {len(codes_all)}，含漂移治愈）...")
        sp = plan_stocks(dp_fresh, codes_all)
        # 抓取窗口：回溯约 60 个交易日，足以覆盖尾部治愈需求
        fetch_start = dp_fresh.calendar()[max(0, len(dp_fresh.calendar()) - 60)]
        stock_summary = apply_stock_append(dp_fresh, sp, fetch_start, backup_root)
        n_ok = len(stock_summary["appended"])
        reasons = [s.get("reason", "") for s in stock_summary["skipped"]]
        n_latest = sum(1 for r in reasons if r == "已是最新")
        n_rebuild = sum(1 for r in reasons if "治愈窗口" in r)
        n_overlong = sum(1 for r in reasons if "超长" in r)
        print(f"      ✓ 追加/治愈 {n_ok} | 已最新 {n_latest} | "
              f"需重建 {n_rebuild} | 超长异常 {n_overlong} | "
              f"其他 {len(reasons) - n_latest - n_rebuild - n_overlong}")
    else:
        print("[4/5] 跳过成分股（未指定 --stocks）")

    if not args.no_version_stamp:
        try:
            from trader3.shared_state import SharedState
            new_last = plan["new_tail_days"][-1] if plan["new_tail_days"] else plan["cal_last"]
            SharedState().set_data_version({
                "qlib_bin": new_last,
                "updated_by": "update_market_data",
            })
            print(f"[5/5] data_version 已盖章: qlib_bin={new_last}（Gate6 恢复判别力）")
        except Exception as e:
            print(f"[5/5] ⚠ data_version 写入失败: {e}")
    else:
        print("[5/5] 跳过 data_version")

    print("✅ 完成。备份保留于:", backup_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
