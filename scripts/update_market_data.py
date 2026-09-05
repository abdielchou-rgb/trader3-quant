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
  成分段重建（默认执行，--membership-universe 参数化，失败不阻断）：
      instruments/<universe>.txt 半年度切分的段末常滞后日历末，
      导致近期 asof 过滤返回空；拉当前成分把滞后段末延长到日历末并为新成分补段。

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
import sys
from datetime import datetime

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

BIN_DTYPE = "<f4"
INDEX_CODE = "sh000300"


def _dp(data_dir: str | None):
    if data_dir:
        from trader3.data_provider import QlibDataProvider
        return QlibDataProvider(data_dir=data_dir)
    from trader3.data_provider import QlibDataProvider
    return QlibDataProvider()


# ── 行情抓取 ────────────────────────────────────────────

def fetch_index_history() -> tuple[list[tuple[str, float]], str]:
    """抓取 SH000300 日线收盘序列 [(date, close)]，双源降级。"""
    import akshare as ak

    try:
        df = ak.index_zh_a_hist(symbol="000300", period="daily")
        rows = [(str(d)[:10], float(v)) for d, v in
                zip(df["日期"], df["收盘"], strict=False) if float(v) > 0]
        return rows, "eastmoney-index"
    except Exception as e1:
        try:
            df = ak.stock_zh_index_daily(symbol="sh000300")
            rows = [(str(d)[:10], float(v)) for d, v in
                    zip(df["date"], df["close"], strict=False) if float(v) > 0]
            return rows, "sina-index"
        except Exception as e2:
            raise RuntimeError(f"指数行情双源失败: {e1} / {e2}") from e2


def _normalize_code(code: str) -> str:
    """'SH600519' / '600519.SH' / '600519' -> '600519'"""
    c = code.upper().replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
    if c.startswith(("SH", "SZ", "BJ")):
        c = c[2:]
    return c


def fetch_stock_close(code: str, start: str) -> list[tuple[str, float]]:
    """抓取个股日线收盘（前复权），[(date, close)]。"""
    import akshare as ak

    norm = _normalize_code(code)
    try:
        df = ak.stock_zh_a_hist(symbol=norm, period="daily",
                                start_date=start.replace("-", ""), adjust="qfq")
        return [(str(d)[:10], float(v)) for d, v in
                zip(df["日期"], df["收盘"], strict=False) if float(v) > 0]
    except Exception:
        pass
    prefix = "sh" if norm.startswith("6") else ("bj" if norm.startswith(("4", "8", "9")) else "sz")
    df = ak.stock_zh_a_daily(symbol=prefix + norm,
                             start_date=start.replace("-", ""), adjust="qfq")
    return [(str(d)[:10], float(v)) for d, v in
            zip(df["date"], df["close"], strict=False) if float(v) > 0]


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

def plan_index(dp, rows: list[tuple[str, float]]) -> dict:
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


def _backup(paths: list[str], backup_dir: str, data_dir: str = "") -> None:
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


def _index_anchor(dp, cal: list[str]) -> tuple[str, int]:
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


def apply_index(dp, rows: list[tuple[str, float]], plan: dict, backup_root: str) -> dict:
    """
    按指数自身上市锚点对齐重建 bin + 扩展日历。

    bin 长度 == len(cal_new) - i0（与 all.txt 锚点一致，满足加载端契约）；
    先备份，写入或校验失败由调用方回滚。
    """

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


def verify_index(dp, rows: list[tuple[str, float]], expect_cal_len: int,
                 expect_bin_len: int | None = None) -> tuple[bool, str]:
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


def plan_stocks(dp, codes: list[str], k_new: int = 0,
                max_heal_days: int = 30) -> list[dict]:
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
    plans: list[dict] = []
    listing_cache: dict[str, str] = {}

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


def apply_stock_append(dp, plans: list[dict], fetch_start: str, backup_root: str) -> dict:
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


# ── 成分段重建 ──────────────────────────────────────────

_UNIVERSE_SYMBOL = {
    "csi300": "000300",
    "csi500": "000905",
    "csi800": "000906",
    "csi1000": "000852",
}


def fetch_constituents(universe: str = "csi300") -> set[str]:
    """拉取指数当前成分裸码集合；主源中证官网、备源新浪，双源降级。

    接口实测（akshare 1.18.81）：
      index_stock_cons_csindex(symbol="000300") → 列含 "成分券代码"（裸 6 位码）
      index_stock_cons(symbol="000300")         → 列含 "品种代码"

    universe 校验前置于 akshare 导入：参数错误不依赖可选数据源是否安装。
    """
    symbol = _UNIVERSE_SYMBOL.get(universe)
    if not symbol:
        raise ValueError(f"未支持的 universe: {universe}")
    import akshare as ak

    try:
        df = ak.index_stock_cons_csindex(symbol=symbol)
        codes = {str(c).strip().zfill(6) for c in df["成分券代码"]}
    except Exception as e1:
        try:
            df = ak.index_stock_cons(symbol=symbol)
            codes = {str(c).strip().zfill(6) for c in df["品种代码"]}
        except Exception as e2:
            raise RuntimeError(f"成分接口双源失败({universe}): {e1} / {e2}") from e2
    return {c for c in codes if len(c) == 6 and c.isdigit()}


def _prefixed_code(bare: str) -> str:
    """裸码 → 库内 instrument 命名（与 fetch_stock_close 前缀约定一致）。"""
    if bare.startswith("6"):
        return "SH" + bare
    if bare.startswith(("4", "8", "9")):
        return "BJ" + bare
    return "SZ" + bare


def refresh_membership(dp, universe: str = "csi300", cal_last: str = "") -> dict:
    """
    成分段尾部重建：把 instruments/<universe>.txt 的滞后段末对齐日历末。

    data_qc 实跑发现：半年度切分的段末滞后日历末约一个月，
    导致 instruments(universe, asof_date=近期) 返回空。规则：
      - 段末 == max_end 的当前成分 → 段 end 延长至 cal_last
      - 文件中不存在的新成分      → 追加 [max_end后首个交易日, cal_last]（保守起点）
      - 末段早于 max_end / 非当前成分 → 一律不动（历史成员保留）
    原子写（tmp+os.replace），写前备份到 <data_dir>/_backup_<ts>/。
    """
    import bisect

    inst_path = os.path.join(dp.data_dir, "instruments", f"{universe}.txt")
    if not os.path.exists(inst_path):
        raise FileNotFoundError(f"成分文件不存在: {inst_path}")
    cal = list(dp.calendar())
    if not cal_last:
        cal_last = cal[-1]

    cons_now = fetch_constituents(universe)

    with open(inst_path, encoding="utf-8") as f:
        orig_lines = [ln for ln in f.read().splitlines() if ln.strip()]
    parsed: list[list[str]] = []
    ends: list[str] = []
    seen_bare: set[str] = set()
    for ln in orig_lines:
        parts = ln.split("\t") if "\t" in ln else ln.split()
        code = parts[0].upper()
        start = parts[1] if len(parts) > 1 and parts[1] else "1900-01-01"
        end = parts[2] if len(parts) > 2 and parts[2] else "2999-12-31"
        parsed.append([code, start, end])
        ends.append(end)
        seen_bare.add(_normalize_code(code))

    stats = {"universe": universe, "cal_last": cal_last,
             "max_end": max(ends), "cons_n": len(cons_now),
             "extended": [], "appended": [], "changed": False}
    if stats["max_end"] >= cal_last:
        stats["status"] = "fresh"
        return stats
    max_end = stats["max_end"]

    seg_start_idx = bisect.bisect_right(cal, max_end)
    seg_start = cal[seg_start_idx] if seg_start_idx < len(cal) else cal_last

    new_parsed = [list(p) for p in parsed]
    extended_bares: set[str] = set()
    for p in new_parsed:
        bare = _normalize_code(p[0])
        if bare in cons_now and p[2] == max_end:
            p[2] = cal_last
            if bare not in extended_bares:
                stats["extended"].append(p[0])
                extended_bares.add(bare)
    for bare in sorted(cons_now - seen_bare):
        code = _prefixed_code(bare)
        new_parsed.append([code, seg_start, cal_last])
        stats["appended"].append(code)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(dp.data_dir, f"_backup_{ts}")
    _backup([inst_path], backup_dir, dp.data_dir)

    tmp = inst_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join("\t".join(p) for p in new_parsed) + "\n")
    os.replace(tmp, inst_path)

    stats.update({"changed": True, "status": "rebuilt",
                  "seg_start_new": seg_start, "backup_dir": backup_dir,
                  "lines_after": len(new_parsed)})
    return stats


# ── 主流程 ──────────────────────────────────────────────

def post_qc_guard(data_dir: str, universe: str = "csi300",
                  tolerance_critical: int = 2) -> dict:
    """
    更新收尾 QC 护栏：任何 --apply 写入后跑一次 QC，确认没有把数据写坏。

    事故史（2026-08/09）：update --stocks 曾静默截断 21 只老股、追加段前复权基准
    断裂造成 50%~2000% 假跳。容忍基线：critical ≤ 2（两只已知单只历史/次新孤例），
    stock_contract == 0（任何契约失败都说明本次更新写坏了对齐）。
    返回 report 摘要 dict（供调用方打印/告警）。
    """
    from trader3.v2 import data_qc

    report = data_qc.run_qc(data_dir=data_dir, sample_limit=200, universe=universe)
    data_qc.save_report(report)
    line = data_qc.qc_summary_line(report)
    critical = int(report.get("critical", -1))
    contracts = sum(1 for o in report.get("offenders", [])
                    if o.get("check") == "stock_contract")
    return {
        "line": line,
        "critical": critical,
        "stock_contract": contracts,
        "ok": critical <= tolerance_critical and contracts == 0,
        "warnings": int(report.get("warnings", 0)),
    }


def main():
    ap = argparse.ArgumentParser(description="qlib_bin 增量更新管线")
    ap.add_argument("--apply", action="store_true", help="真正写入（默认 dry-run）")
    ap.add_argument("--stocks", action="store_true", help="同时处理成分股严格追加")
    ap.add_argument("--limit", type=int, default=0, help="限制处理股票数（调试）")
    ap.add_argument("--data-dir", default="", help="覆盖 qlib 数据目录（测试用）")
    ap.add_argument("--membership-universe", default="csi300",
                    help="成分段重建的指数 universe（默认 csi300）")
    ap.add_argument("--no-version-stamp", action="store_true", help="成功后不写 data_version")
    args = ap.parse_args()

    dp = _dp(args.data_dir)
    print("[1/6] 抓取指数行情 ...")
    rows, source = fetch_index_history()
    print(f"      来源={source} 条数={len(rows)} 末条={rows[-1]}")

    plan = plan_index(dp, rows)
    print(f"[2/6] 计划: 日历 {plan['cal_len']}({plan['cal_last']}) → {plan['new_cal_len']}"
          f"，新增 {len(plan['new_tail_days'])} 个交易日")
    if plan["interior_missing"]:
        print(f"      ⚠ 抓取数据中有 {len(plan['interior_missing'])} 个早于现日历末的缺失日（忽略，不影响追加以外的重建）")

    if not args.apply:
        print("[dry-run] 未写入。加 --apply 执行。")
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.join(dp.data_dir, f"_backup_{ts}")

    print("[3/6] 指数重建（含备份+失败自动回滚）...")
    result = apply_index(dp, rows, plan, backup_root)
    ok, msg = verify_index(dp, rows, result["new_cal_len"],
                           expect_bin_len=result.get("expected_bin_len"))
    if not ok:
        _restore(result["backup_dir"], dp.data_dir)
        print(f"      ✗ 校验失败已回滚: {msg}")
        return 2
    print(f"      ✓ {result['written_fields']}")

    # 成分段尾部对齐：成分文件段末常滞后日历末（半年度切分），
    # 否则 instruments(asof_date=近期) 返回空。失败仅告警不阻断。
    print("[4/6] 成分段重建（对齐日历末，防近期 asof 过滤为空）...")
    try:
        from trader3.data_provider import QlibDataProvider
        ms_dp = QlibDataProvider(data_dir=dp.data_dir)
        ms_cal_last = plan["new_tail_days"][-1] if plan["new_tail_days"] else plan["cal_last"]
        ms = refresh_membership(ms_dp, universe=args.membership_universe,
                                cal_last=ms_cal_last)
        if ms.get("changed"):
            print(f"      ✓ 延长 {len(ms['extended'])} 段 | 新增成分 {len(ms['appended'])}"
                  f" | 新段起点 {ms['seg_start_new']} | 备份: {ms['backup_dir']}")
        else:
            print(f"      ✓ 成分段已覆盖至 {ms['max_end']}（≥ {ms['cal_last']}），无需更新")
    except Exception as e:
        print(f"      ⚠ 成分段刷新失败（不阻断主流程）: {e}")

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
        print(f"[5/6] 成分股严格追加（候选 {len(codes_all)}，含漂移治愈）...")
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
        print("[5/6] 跳过成分股（未指定 --stocks）")

    if not args.no_version_stamp:
        try:
            from trader3.shared_state import SharedState
            new_last = plan["new_tail_days"][-1] if plan["new_tail_days"] else plan["cal_last"]
            SharedState().set_data_version({
                "qlib_bin": new_last,
                "updated_by": "update_market_data",
            })
            print(f"[6/6] data_version 已盖章: qlib_bin={new_last}（Gate6 恢复判别力）")
        except Exception as e:
            print(f"[6/6] ⚠ data_version 写入失败: {e}")
    else:
        print("[6/6] 跳过 data_version")

    # ── 收尾 QC 护栏（7/6）────────────────────────────────
    # 数据纪律：任何 --apply 写入后必须跑一次 QC，确认没有把数据写坏。
    try:
        guard = post_qc_guard(dp.data_dir, universe=args.membership_universe)
        print(f"[7/6] QC 护栏: {guard['line']}")
        print(f"      critical={guard['critical']} stock_contract={guard['stock_contract']}")
        if not guard["ok"]:
            print("      ⚠ QC 超容忍基线（critical>2 或 contract>0）——"
                  "本次更新可能引入了数据损坏，建议人工检查备份目录并回滚。")
    except Exception as qc_exc:
        print(f"[7/6] ⚠ QC 护栏执行失败: {qc_exc}")

    print("✅ 完成。备份保留于:", backup_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
