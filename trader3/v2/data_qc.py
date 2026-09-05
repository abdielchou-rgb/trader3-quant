"""
3号交易员 — 数据质检管道 (QC)

对 qlib_bin 数据目录做离线质量抽查，产出结构化报告（原子写 data/qc_report.json）
供日报与推送引用。

历史事故背景：前复权拼接断裂（边界日跳变>25%）、上市锚点错位、退市股 bin 截断。
检查项与严重级：
  calendar_order   日历非严格递增 / 含重复日期          critical
  stock_contract   个股契约失败（截断/锚点错位/空序列） critical
  close_jump       相邻有效收盘 |ret| 超阈值            critical
  calendar_stale   日历末日期距今超过 N 天              warning
  zero_ratio       close bin 零值占比超限               warning
  ohlc_high_low    出现 high < low                      warning

通过口径：passed == (critical == 0)，warning 不影响 passed。
offenders 截断至前 100 条；critical/warnings 计数不截断。
"""

from __future__ import annotations

import json
import os
import warnings
from collections import Counter
from datetime import datetime

import numpy as np

from trader3.data_provider import QlibDataProvider

JUMP_THRESHOLD = 0.25      # 相邻有效收盘 |ret| 阈值（前复权拼接断裂特征）
ZERO_RATIO_LIMIT = 0.30    # close bin 零值占比告警阈值
STALE_DAYS = 10            # 日历末日期距今天数告警阈值
OFFENDER_CAP = 100         # offenders 截断上限
PER_STOCK_EVENT_CAP = 3    # 单股跳变事件记录上限（计数不截断）
DEFAULT_UNIVERSE = "csi300"
MAX_DUP_LISTED = 10        # 报告中重复日期最多列出条数
_VALUE_TRUNC = 80

CRITICAL_CHECKS = frozenset({"calendar_order", "stock_contract", "close_jump"})
WARNING_CHECKS = frozenset({
    "calendar_stale", "zero_ratio", "ohlc_high_low", "sample_fallback", "sample_empty",
})

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_report_path() -> str:
    """默认落盘位置：<project>/data/qc_report.json"""
    return os.path.join(_PROJECT_ROOT, "data", "qc_report.json")


def _read_bin_raw(data_dir: str, code: str, field: str) -> np.ndarray | None:
    path = os.path.join(data_dir, "features", code.lower(), f"{field}.day.bin")
    if not os.path.exists(path):
        return None
    return np.fromfile(path, dtype="<f4").astype(np.float64)


def _strip_invalid(arr: np.ndarray) -> np.ndarray:
    """剥离首尾非有效占位（price<=0 / NaN/Inf），与 data_provider 对齐规则一致。"""
    ok = np.isfinite(arr) & (arr > 0)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return arr[:0]
    return arr[idx[0]:idx[-1] + 1]


def _fmt_value(check: str, value) -> str:
    if isinstance(value, (int, float)):
        if check in ("close_jump", "zero_ratio"):
            return f"{value:+.1%}" if check == "close_jump" else f"{value:.1%}"
        if check == "calendar_stale":
            return f"{value:.0f}d"
        return f"{value:g}"
    return str(value)[:_VALUE_TRUNC]


def _offender(code: str, check: str, value, severity: str) -> dict:
    return {"code": code, "check": check, "value": value, "severity": severity}


# ── 日历检查 ────────────────────────────────────────────


def _check_calendar(cal: list[str], today) -> tuple[dict, list[dict]]:
    counts = Counter(cal)
    duplicates = sorted(d for d, n in counts.items() if n > 1)
    monotonic = all(cal[i] < cal[i + 1] for i in range(len(cal) - 1))
    stale_days = None
    stale = False
    offenders: list[dict] = []

    if not monotonic or duplicates:
        if duplicates:
            detail = f"duplicate:{','.join(duplicates[:MAX_DUP_LISTED])}"
        else:
            bad_at = next(i for i in range(len(cal) - 1) if cal[i] >= cal[i + 1])
            detail = f"non-monotonic@{bad_at}:{cal[bad_at]}>={cal[bad_at + 1]}"
        offenders.append(_offender("CALENDAR", "calendar_order", detail, "critical"))

    last = cal[-1] if cal else ""
    try:
        last_date = datetime.strptime(last, "%Y-%m-%d").date()
        stale_days = (today - last_date).days
        stale = stale_days > STALE_DAYS
        if stale:
            offenders.append(
                _offender("CALENDAR", "calendar_stale", int(stale_days), "warning"))
    except ValueError:
        pass

    checks = {
        "days": len(cal),
        "first": cal[0] if cal else "",
        "last": last,
        "monotonic": monotonic,
        "duplicates": duplicates[:MAX_DUP_LISTED],
        "stale": stale,
        "stale_days": int(stale_days) if stale_days is not None else None,
    }
    return checks, offenders


# ── 每股抽查 ────────────────────────────────────────────


def _check_stock(dp: QlibDataProvider, data_dir: str, code: str,
                 jump_threshold: float) -> tuple[list[dict], bool]:
    """返回 (offenders, contract_ok)。跳变/零值/OHLC 检查依赖契约通过。"""
    offenders: list[dict] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 契约失败已转为 offender，无需重复告警
            close, _dates = dp.load_stock(code.upper(), "close")
    except Exception as exc:  # noqa: BLE001 — 契约失败本身就是质检对象
        offenders.append(
            _offender(code, "stock_contract", str(exc)[:_VALUE_TRUNC], "critical"))
        return offenders, False
    if close.size == 0:
        offenders.append(_offender(code, "stock_contract", "close序列为空", "critical"))
        return offenders, False

    raw_close = _read_bin_raw(data_dir, code, "close")
    if raw_close is not None and raw_close.size:
        zero_ratio = float(np.mean(~(np.isfinite(raw_close) & (raw_close > 0))))
        if zero_ratio > ZERO_RATIO_LIMIT:
            offenders.append(
                _offender(code, "zero_ratio", round(zero_ratio, 4), "warning"))

    valid_idx = np.flatnonzero(np.isfinite(close) & (close > 0))
    if valid_idx.size >= 2:
        a, b = close[valid_idx[:-1]], close[valid_idx[1:]]
        rets = b / a - 1.0
        bad = np.flatnonzero(np.abs(rets) > jump_threshold)
        for pos in bad[:PER_STOCK_EVENT_CAP]:
            # 豁免真实"长期停牌 → 复牌"事件：跳变前若相邻有效日之间在 bin 内
            # 存在 ≥1 个 0 占位（停牌期），则该收益跨越停牌段，属真实事件而非脏数据
            # （如 ST 重整复牌 +300%、股改停牌复牌等）。此类 close_jump 降级为 warning。
            i_prev, i_next = valid_idx[pos], valid_idx[pos + 1]
            halted = int(np.sum(~((close[i_prev + 1:i_next] > 0)
                                  & np.isfinite(close[i_prev + 1:i_next]))))
            # 次新豁免：上市后前 5 个交易日（新股上市前 5 日无涨跌幅限制，真实大波动）。
            # 只用前 5 日（非 20）——避免把测试/老股中段跳变误豁免。
            is_newish = int(valid_idx[pos]) < 5
            severity = "warning" if (halted >= 1 or is_newish) else "critical"
            offenders.append(
                _offender(code, "close_jump", round(float(rets[pos]), 4), severity))

    high = _read_bin_raw(data_dir, code, "high")
    low = _read_bin_raw(data_dir, code, "low")
    if high is not None and low is not None:
        n = min(high.size, low.size)
        h_ok = np.isfinite(high[:n]) & (high[:n] > 0)
        l_ok = np.isfinite(low[:n]) & (low[:n] > 0)
        n_bad = int(np.sum((high[:n] < low[:n]) & h_ok & l_ok))
        if n_bad:
            offenders.append(_offender(code, "ohlc_high_low", n_bad, "warning"))
    return offenders, True


# ── 主入口 ──────────────────────────────────────────────


def run_qc(data_dir: str | None = None, sample_limit: int = 200, *,
           universe: str = DEFAULT_UNIVERSE,
           jump_threshold: float = JUMP_THRESHOLD,
           today: str | None = None) -> dict:
    """
    对 qlib_bin 目录做质检抽查，返回报告 dict（可 JSON 化）。

    - 成分抽样：universe（默认 csi300）在 asof=日历末 仍在册的成分，
      超过 sample_limit 时按等距索引抽样（确定性）。
      成分文件滞后于日历（asof 无在册成分）时告警并回退全历史并集，
      避免零覆盖的空洞通过；universe 本身为空则记 sample_empty 告警。
    - today: YYYY-MM-DD，注入当前日期以便离线测试新鲜度告警；None 取真实今天。
    """
    now = datetime.now()
    today_d = (
        datetime.strptime(today, "%Y-%m-%d").date()
        if today else now.date()
    )
    offenders: list[dict] = []

    try:
        dp = QlibDataProvider(data_dir=data_dir)
    except FileNotFoundError as exc:
        return {
            "passed": False, "critical": 1, "warnings": 0,
            "checks": {"calendar": {}, "stocks": {}},
            "offenders": [_offender("<data_dir>", "data_dir_missing",
                                    str(exc)[:_VALUE_TRUNC], "critical")],
            "offenders_truncated": False,
            "data_dir": str(data_dir or ""), "universe": universe,
            "asof_date": "", "generated_at": now.isoformat(timespec="seconds"),
        }

    data_root = dp.data_dir
    cal = list(dp.calendar())
    cal_checks, cal_offenders = _check_calendar(cal, today_d)
    offenders.extend(cal_offenders)
    asof = cal[-1] if cal else ""

    codes = dp.instruments(universe, asof_date=asof) if asof else []
    if not codes:
        codes = dp.instruments(universe)
        if codes:
            offenders.append(_offender(
                "UNIVERSE", "sample_fallback",
                f"asof={asof} 无在册成分，回退全历史并集({len(codes)})", "warning"))
        else:
            offenders.append(_offender(
                "UNIVERSE", "sample_empty", f"universe={universe} 无成分", "warning"))
    total = len(codes)
    if total > sample_limit > 0:
        picks = np.linspace(0, total - 1, sample_limit).round().astype(int)
        sampled = [codes[i] for i in sorted(set(picks.tolist()))]
    else:
        sampled = list(codes)

    n_contract = n_zero = n_ohlc = 0
    n_jump_events = 0
    jump_codes: set[str] = set()
    checked_ok = 0
    for code in sampled:
        stock_offenders, contract_ok = _check_stock(dp, data_root, code, jump_threshold)
        offenders.extend(stock_offenders)
        checks_seen = {o["check"] for o in stock_offenders}
        n_contract += 1 if not contract_ok else 0
        n_zero += 1 if "zero_ratio" in checks_seen else 0
        n_ohlc += 1 if "ohlc_high_low" in checks_seen else 0
        jumps = sum(1 for o in stock_offenders if o["check"] == "close_jump")
        if jumps:
            jump_codes.add(code)
            n_jump_events += jumps
        if contract_ok:
            checked_ok += 1

    truncated = len(offenders) > OFFENDER_CAP
    report = {
        "passed": all(o["severity"] != "critical" for o in offenders),
        "critical": sum(1 for o in offenders if o["severity"] == "critical"),
        "warnings": sum(1 for o in offenders if o["severity"] == "warning"),
        "checks": {
            "calendar": cal_checks,
            "stocks": {
                "universe_total": total,
                "sampled": len(sampled),
                "checked_ok": checked_ok,
                "jump_events": n_jump_events,
                "jump_stocks": len(jump_codes),
                "contract_failed": n_contract,
                "zero_flagged": n_zero,
                "ohlc_flagged": n_ohlc,
            },
        },
        "offenders": offenders[:OFFENDER_CAP],
        "offenders_truncated": truncated,
        "data_dir": data_root,
        "universe": universe,
        "asof_date": asof,
        "generated_at": now.isoformat(timespec="seconds"),
    }
    return report


# ── 落盘与汇总行 ────────────────────────────────────────


def save_report(report: dict, path: str | None = None) -> str:
    """原子写报告 JSON（tmp + os.replace），默认 <project>/data/qc_report.json。"""
    path = path or _default_report_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


def qc_summary_line(report: dict) -> str:
    """
    日报用一行汇总：
      "QC: PASS(0 critical)"
      或 "QC: FAIL(critical=N, top=<code> <check> <value>)"
    top 取 critical offender 中 |value| 最大者（数值可比时），否则取第一条。
    """
    critical = int(report.get("critical") or 0)
    if critical <= 0:
        return "QC: PASS(0 critical)"

    offenders = report.get("offenders") or []
    crit = [o for o in offenders if o.get("severity", "critical") == "critical"]
    if not crit:
        crit = list(offenders)

    def _magnitude(o: dict) -> float:
        v = o.get("value")
        return abs(float(v)) if isinstance(v, (int, float)) else float("-inf")

    top = max(crit, key=_magnitude) if crit else None
    desc = ("<unknown>" if top is None else
            f"{top.get('code')} {top.get('check')} "
            f"{_fmt_value(top.get('check'), top.get('value'))}")
    return f"QC: FAIL(critical={critical}, top={desc})"
