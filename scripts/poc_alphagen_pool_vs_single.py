# -*- coding: utf-8 -*-
"""P2-5 AlphaGen PoC：协同因子挖掘对照实验（池增益贪心 vs 单因子贪心）

差距报告 C1/A1 的核心主张（AlphaGen 池化奖励）在真实数据上的轻量验证：
    同一批候选因子，两种建池策略——
      A. 单因子贪心：按个体 OOS RankIC 降序入库（现状基线），
         只受互相关上限 corr_max 约束；
      B. 池增益贪心（alphagen 思想）：每步选"使组合 OOS RankIC 增益最大"者，
         同受 corr_max 约束。
    在留出段（OOS）比较两池等权合成 RankIC——若 B > A，则池化奖励
    在真实 A 股数据上成立（协同性目标优于单因子排序）。

诚实性设计：
  - 选池只用训练段（前 70%）的扩张窗口 IC；比较只用留出段（后 30%）
    —— 选池与评估时间不相交，无前视。
  - 与 factor_factory 同口径：截面 rank 等权合成、扩张窗口 IC（前 60 日起）。
  - 严格门禁版 evaluate 的失败原因分布一并打印（真实数据诚实观察）。

用法：
  python scripts/poc_alphagen_pool_vs_single.py            # 真实 csi300 2022-2023
  python scripts/poc_alphagen_pool_vs_single.py --quick     # 合成数据冒烟
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evolve"))
sys.path.insert(0, str(ROOT))


# ── 数据装配（真实 qlib → DSL panel） ──────────────────

def build_real_panel(start="2022-01-01", end="2023-12-31", n_stocks=60):
    from core.data_loader import load_qlib_panel
    from trader3.data_provider import QlibDataProvider
    import bisect

    panel_dict, fwd_np = load_qlib_panel(universe="csi300", n_stocks=n_stocks,
                                         start=start, end=end)
    dp = QlibDataProvider()
    cal = dp.calendar()
    codes = dp.instruments("csi300", asof_date=end)
    if len(codes) < n_stocks:
        codes = dp.instruments("csi300")
    cal_str = [str(c)[:10] for c in cal]
    s = bisect.bisect_left(cal_str, start)
    e = bisect.bisect_right(cal_str, end)
    taxis = pd.to_datetime(cal_str[s:e])
    T, N = panel_dict["close"].shape
    assets = list(codes[:N])
    cols = pd.MultiIndex.from_product([assets, list(panel_dict.keys())])
    df = pd.DataFrame(index=taxis[:T], columns=cols, dtype=float)
    for f, arr in panel_dict.items():
        for i, a in enumerate(assets):
            df[(a, f)] = arr[:, i]
    fwd = pd.DataFrame(fwd_np, index=df.index, columns=assets)
    return df, fwd


def build_synthetic_panel(n_dates=300, n_assets=30, seed=7):
    """合成面板：隐藏两个互补信号（动量+反转）+ 大量噪声因子。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    assets = [f"S{i}" for i in range(n_assets)]
    fields = ["open", "high", "low", "close", "volume", "vwap", "amount"]
    cols = pd.MultiIndex.from_product([assets, fields])
    df = pd.DataFrame(rng.normal(size=(n_dates, len(cols))), index=dates, columns=cols)
    close = np.cumsum(rng.normal(0, 0.5, (n_dates, n_assets)), axis=0) + 100
    for a, i in zip(assets, range(n_assets)):
        df[(a, "close")] = close[:, i]
        df[(a, "vwap")] = close[:, i]
        df[(a, "high")] = close[:, i] + 0.3
        df[(a, "low")] = close[:, i] - 0.3
        df[(a, "open")] = np.roll(close[:, i], 1)
        df[(a, "volume")] = abs(rng.normal(1e5, 1e4, n_dates))
        df[(a, "amount")] = df[(a, "volume")] * close[:, i]
    mom = np.zeros_like(close)
    mom[5:] = close[5:] - close[:-5]
    rev = -np.roll(mom, 2)  # 与动量低相关的另一信号
    fwd = 0.5 * mom + 0.3 * rev + rng.normal(0, 0.3, close.shape)
    fwd = pd.DataFrame(fwd, index=dates, columns=assets)
    fwd.iloc[:5] = np.nan
    return df, fwd


# ── IC 口径（与 factor_factory._pool_gain 一致：截面 rank + 扩张窗口） ──

def _row_rank_ic(sig_row: np.ndarray, fwd_row: np.ndarray) -> float:
    m = ~np.isnan(sig_row) & ~np.isnan(fwd_row)
    if m.sum() < 10:
        return float("nan")
    def rk(x):
        order = np.argsort(np.argsort(x))
        return order.astype(float)
    s, f = rk(sig_row[m]), rk(fwd_row[m])
    sd, fd_ = s.std(), f.std()
    if sd < 1e-9 or fd_ < 1e-9:
        return float("nan")
    return float(np.corrcoef(s, f)[0, 1])


def expanding_oos_ic(factor: pd.DataFrame, fwd: pd.DataFrame,
                     start_min: int = 60) -> float:
    """扩张窗口 OOS 均值 RankIC（时序因果：t 日信号只标准化于 [:t] 历史）"""
    sig = factor.values
    fwd_seg = fwd.iloc[:len(factor)] if len(fwd) > len(factor) else fwd
    fr = fwd_seg.reindex(columns=factor.columns).values
    ics = []
    for t in range(start_min, len(sig)):
        hist = sig[:t]
        mu, sd = np.nanmean(hist), np.nanstd(hist) + 1e-9
        z = (sig[t] - mu) / sd
        if t >= len(fr):
            break
        ic = _row_rank_ic(z, fr[t])
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


def oos_segment_ic(factors: list[pd.DataFrame], fwd: pd.DataFrame,
                   lo: int, hi: int) -> float:
    """留出段 [lo,hi) 等权 rank 合成的均值 RankIC（评估用，不做标准化因果也可：
    合成只取当日截面 rank，无跨日信息）"""
    fr = fwd.values
    ics = []
    for t in range(lo, hi):
        combo = np.zeros(factors[0].shape[1])
        cnt = 0
        for f in factors:
            row = f.values[t]
            m = ~np.isnan(row)
            if m.sum() < 10:
                continue
            r = np.full(row.shape, np.nan)
            r[m] = np.argsort(np.argsort(row[m]))
            combo = combo + np.nan_to_num(r / max(1e-9, np.nanmax(r)))
            cnt += 1
        if cnt == 0:
            continue
        ic = _row_rank_ic(combo, fr[t])
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


def cross_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    ra, rb = a.rank(axis=1), b.rank(axis=1)
    c = ra.corrwith(rb, axis=1).mean()
    return abs(float(c)) if np.isfinite(c) else 0.0


# ── 两种建池策略 ────────────────────────────────────

def pool_single_greedy(cands: dict[str, pd.DataFrame], train_ic: dict[str, float],
                       fwd: pd.DataFrame, train_end: int, k: int = 8,
                       corr_max: float = 0.7) -> list[str]:
    """A：单因子贪心（现状基线）——按个体训练段 IC 降序，corr 门内即入。"""
    sel, chosen = [], []
    for expr, ic in sorted(train_ic.items(), key=lambda kv: -kv[1]):
        if np.isnan(ic) or ic <= 0:
            continue
        f = cands[expr]
        if all(cross_corr(f, c) <= corr_max for c in chosen):
            sel.append(expr)
            chosen.append(f)
        if len(sel) >= k:
            break
    return sel


def pool_gain_greedy(cands: dict[str, pd.DataFrame], fwd: pd.DataFrame,
                     train_end: int, k: int = 8, corr_max: float = 0.7,
                     min_train: int = 60) -> list[str]:
    """B：池增益贪心（alphagen 池化奖励）——每步选组合训练段扩张 IC 增益最大者。"""
    def train_ic_of(fs: list[pd.DataFrame]) -> float:
        # 等权 rank 合成后的扩张窗口 IC（训练段内）
        combo = sum(f.rank(axis=1) for f in fs) / len(fs)
        return expanding_oos_ic(combo, fwd.iloc[:train_end], start_min=min_train)

    sel, chosen, cur_ic = [], [], None
    remaining = dict(cands)
    while len(sel) < k and remaining:
        best_expr, best_gain, best_f = None, 0.0, None
        for expr, f in remaining.items():
            if any(cross_corr(f, c) > corr_max for c in chosen):
                continue
            new_ic = train_ic_of(chosen + [f])
            base = cur_ic if cur_ic is not None else 0.0
            gain = (new_ic - base) if np.isfinite(new_ic) else float("-inf")
            if np.isnan(base):
                gain = new_ic
            if gain > best_gain:
                best_expr, best_gain, best_f = expr, gain, f
        if best_expr is None or best_gain <= 0:
            break
        cur_ic = train_ic_of(chosen + [best_f])
        sel.append(best_expr)
        chosen.append(best_f)
        del remaining[best_expr]
    return sel


# ── 主实验 ──────────────────────────────────────────

def run(df, fwd, label, k=8, report_gate_stats=False):
    from trader3.v2.factor_factory import (FactorFactory, FactorFactoryConfig,
                                           FactorRegistry)
    T = len(df)
    train_end = int(T * 0.7)

    fac = FactorFactory(FactorRegistry("/tmp/_poc_reg.json"),
                        FactorFactoryConfig(n_propose=48))
    cands_exprs = [c.expr for c in fac.propose_offline(48)]
    cands: dict[str, pd.DataFrame] = {}
    for expr in cands_exprs:
        try:
            f = fac.dsl.full_series(expr, df)
            cands[expr] = f
        except Exception:
            continue
    print(f"[{label}] 候选 {len(cands)} 个（模板展开）")

    # 训练段个体 IC（选池依据 A）
    train_ic = {e: expanding_oos_ic(f, fwd.iloc[:train_end]) for e, f in cands.items()}

    t0 = time.time()
    pool_a = pool_single_greedy(cands, train_ic, fwd, train_end, k=k)
    t_a = time.time() - t0
    t0 = time.time()
    pool_b = pool_gain_greedy(cands, fwd, train_end, k=k)
    t_b = time.time() - t0

    # 留出段评估（选池与评估时间不相交）
    fa = [cands[e] for e in pool_a]
    fb = [cands[e] for e in pool_b]
    ic_a = oos_segment_ic(fa, fwd, train_end, T)
    ic_b = oos_segment_ic(fb, fwd, train_end, T)
    best_single = max((oos_segment_ic([cands[e]], fwd, train_end, T), e)
                      for e in cands)
    result = {
        "label": label,
        "n_candidates": len(cands),
        "pool_A_single_greedy": {"exprs": pool_a, "oos_ic": round(ic_a, 4)},
        "pool_B_gain_greedy": {"exprs": pool_b, "oos_ic": round(ic_b, 4)},
        "best_single_factor": {"expr": best_single[1],
                               "oos_ic": round(best_single[0], 4)},
        "verdict": ("B优于A（池化奖励成立）" if ic_b > ic_a else
                    "A不劣于B（本面板未体现协同增益）"),
        "timing_s": {"A": round(t_a, 1), "B": round(t_b, 1)},
    }
    print(json.dumps(result, ensure_ascii=False, indent=1))

    if report_gate_stats:
        # 严格门禁版 evaluate 的失败原因分布（真实数据诚实观察）
        from trader3.v2.factor_factory import FactorCandidate
        reasons: dict[str, int] = {}
        for e, f in cands.items():
            m = fac.evaluate(FactorCandidate(expr=e, hypothesis="poc"), df, fwd)
            key = m.reason.split("=")[0].split("：")[0][:16]
            reasons[key] = reasons.get(key, 0) + 1
        print(f"[{label}] 严格门禁失败原因分布: {reasons}")
        result["gate_failure_dist"] = reasons
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="合成数据冒烟")
    args = ap.parse_args()
    if args.quick:
        df, fwd = build_synthetic_panel()
        run(df, fwd, "synthetic-quick", k=5)
        return
    df, fwd = build_real_panel()
    res = run(df, fwd, "csi300-2022-2023", k=8, report_gate_stats=True)
    out = ROOT / "docs" / "poc_alphagen_pool_vs_single_20260905.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已落盘 {out}")


if __name__ == "__main__":
    main()
