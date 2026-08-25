"""
3号交易员 — 因子衰减监控

对核心在役因子（当前为 F1 = sub(log(vwap), log(close))）做滚动 IC 追踪：
  - 每日收盘后计算近 lookback 窗口的逐日横截面 IC
  - 追加写 shared_state/factor_watch/<name>_ic.csv
  - 近 decay_window 日均值 < 阈值时判定衰减，返回告警（供日报推送引用）

数据获取与 IC 计算解耦：panel_builder 可注入，便于离线测试。
"""

from __future__ import annotations

import csv
import os

import numpy as np

DEFAULT_EXPR = "sub(log(vwap), log(close))"
DEFAULT_LOOKBACK = 260
DEFAULT_DECAY_WINDOW = 20
DEFAULT_UNIVERSE = "csi300"
DEFAULT_N_STOCKS = 40


def _default_panel_builder(lookback_days: int, universe: str, n_stocks: int):
    """默认面板构建器：qlib 真实数据，近 N 个交易日（含因子预热）。"""
    import numpy as _np

    from evolve.core.gp import FIELDS, normalize
    from evolve.core.parser import parse_expr
    from trader3.tools.backtest import (
        _build_aligned_panel,
        _expr_zscores,
        _load_expression_panels,
        _open_qlib_dp,
    )

    dp = _open_qlib_dp()
    cal = dp.calendar()
    end = cal[-1]
    start_idx = max(0, len(cal) - lookback_days - 25)
    start, end_eff = cal[start_idx], end

    codes = dp.instruments(universe, asof_date=end)
    if len(codes) < n_stocks:
        codes = dp.instruments(universe)

    time_axis, codes_list, close_matrix, _returns, valid_flags, _n = (
        _build_aligned_panel(dp, codes[: max(n_stocks * 2, n_stocks)], start, end_eff)
    )
    node = normalize(parse_expr(DEFAULT_EXPR))
    fields: set[str] = set()
    stack = [node]
    while stack:
        nd = stack.pop()
        if getattr(nd, "op", "") in FIELDS:
            fields.add(nd.op)
        stack.extend(getattr(nd, "children", []))

    close_panel = _np.where(close_matrix > 0, close_matrix, _np.nan)
    panels = {"close": close_panel}
    extra = {f for f in fields if f != "close"}
    if extra:
        panels.update(_load_expression_panels(dp, codes_list, time_axis, extra))
    scores = _expr_zscores(node, panels, valid_flags)
    return time_axis, scores, close_matrix, valid_flags


def compute_ic_series(
    panel_builder=None,
    lookback_days: int = DEFAULT_LOOKBACK,
    universe: str = DEFAULT_UNIVERSE,
    n_stocks: int = DEFAULT_N_STOCKS,
) -> dict:
    """计算逐日横截面 IC。返回 {dates, ic_series, ic_mean, n_obs}。"""
    builder = panel_builder or _default_panel_builder
    time_axis, scores, closes, valid_flags = builder(lookback_days, universe, n_stocks)

    fwd = np.full_like(closes, np.nan)
    pos = np.where(closes > 0, closes, np.nan)
    with np.errstate(all="ignore"):
        fwd[:-1] = pos[1:] / pos[:-1] - 1.0

    dates_out: list[str] = []
    ic_vals: list[float] = []
    T, N = scores.shape
    min_names = max(5, N // 3)
    for t in range(T):
        r = fwd[t] if t < T - 1 else np.full(N, np.nan)
        s = np.nan_to_num(scores[t], nan=0.0)
        mask = np.isfinite(r) & np.isfinite(pos[t]) & (pos[t] > 0)
        if mask.sum() < min_names:
            continue
        rr = r[mask]
        ss = s[mask]
        if np.std(rr) < 1e-10:
            continue
        ra = np.argsort(np.argsort(rr)).astype(np.float64)
        sa = np.argsort(np.argsort(ss)).astype(np.float64)
        if np.std(ra) < 1e-10:
            continue
        ic_vals.append(float(np.corrcoef(sa, ra)[0, 1]))
        dates_out.append(time_axis[t])

    arr = np.asarray(ic_vals) if ic_vals else np.array([])
    return {
        "dates": dates_out,
        "ic_series": [round(v, 4) for v in ic_vals],
        "ic_mean": round(float(arr.mean()), 4) if arr.size else None,
        "n_obs": int(arr.size),
    }


def decay_alert(ic_mean_decay_window: float | None,
                threshold: float = 0.0) -> bool:
    """近窗均值低于阈值即判定衰减；无观测视为未衰减（不误报）。"""
    if ic_mean_decay_window is None:
        return False
    return ic_mean_decay_window < threshold


def append_ic_history(state_dir: str, name: str, record: dict,
                      decay_window: int = DEFAULT_DECAY_WINDOW) -> dict:
    """
    追加当日 IC 到 <state_dir>/factor_watch/<name>_ic.csv，
    并基于最近 decay_window 条返回 {ic_mean_recent, alert}。
    """
    d = os.path.join(state_dir, "factor_watch")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}_ic.csv")
    need_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["date", "ic"])
        w.writerow([record.get("last_date", ""), record.get("ic_last", "")])

    vals: list[float] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                vals.append(float(row["ic"]))
            except (TypeError, ValueError):
                continue
    recent = vals[-decay_window:]
    mean_recent = float(np.mean(recent)) if recent else None
    return {"ic_mean_recent": round(mean_recent, 4) if mean_recent is not None else None,
            "alert": decay_alert(mean_recent)}
