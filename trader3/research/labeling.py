"""
三重屏障标签 + Meta-Labeling（R2）—— 自研最小实现（AFML 方法论）。

背景：mlfinlab 开源仓库已闭源成空壳（docstring 桩），不应 import；
此处按 AFML 书 Snippet 语义自研，接口/列名对齐业界约定
（t1/trgt/pt/sl/side/bin）。

核心语义：
  - triple_barrier_labels：每进场点架 上(pt×σ) / 下(sl×σ) / 垂直(t1) 三屏障，
    取实际最先被价格触碰者 → +1 / -1 / 0
  - meta_labels：给定主方向 side，产出 bin∈{0,1} —— 该方向交易(扣费后)是否盈利；
    二段式：先定方向、再定"是否执行"（风控内嵌模型）
  - daily_vol：EWMA 式滚动日波动（σ 缩放使标签跨波动率可比）

A股/日频注意：T+1 用 t1 = t0 之后 ≥1 bar；涨跌停不可成交场景由调用方在
t1 触碰判定前预处理（本模块只做价格触碰逻辑，避免假设流动性）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_vol(close: pd.Series, lookback: int = 20,
              span: int | None = None) -> pd.Series:
    """日波动率（EWMA 或滚动 std）。返回与 close 同索引 Series。"""
    if span is not None:
        ret = close.pct_change().dropna()
        vol = ret.ewm(span=span).std()
        return vol.reindex(close.index)
    ret = close.pct_change()
    return ret.rolling(lookback).std()


def _first_touch(price: pd.Series, t0: pd.Timestamp, t1: pd.Timestamp,
                 upper: float, lower: float) -> int:
    """价格在 [t0, t1] 区间内，先触上/下屏障则返回 ±1，否则 0。"""
    seg = price.loc[t0:t1]
    if seg.empty or len(seg) < 2:
        return 0
    # 事件驱动：逐 bar 推进，谁先触（含开盘即越界）
    for _dt, px in seg.items():
        if px >= upper:
            return 1
        if px <= lower:
            return -1
    return 0


def triple_barrier_labels(
    close: pd.Series,
    events: pd.DataFrame,
    pt: float = 1.0,
    sl: float = 1.0,
    vol: pd.Series | None = None,
) -> pd.DataFrame:
    """三重屏障标签。

    Parameters
    ----------
    close : 价格序列（datetime index）
    events : DataFrame，index=t0(进场时间)，含列 t1(垂直屏障时间)、
             trgt(σ 目标，屏障=pt/sl × trgt)。trgt 缺失用 vol 推算。
    pt / sl : 上/下屏障倍数（乘 trgt）
    vol : 可选日波动率；trgt 缺失时用 vol 当日值补

    Returns
    -------
    events + 'bin' 列：先触上→1，先触下→-1，垂直到期→0
    """
    out = events.copy()
    if "trgt" not in out.columns or out["trgt"].isna().all():
        v = daily_vol(close) if vol is None else vol
        out["trgt"] = out.index.map(
            lambda t: float(v.get(t, np.nan)) if pd.notna(t) else np.nan)

    bins: list[int] = []
    close_idx = close.index
    for (t0, row) in out.iterrows():
        t1 = pd.Timestamp(row["t1"]) if pd.notna(row["t1"]) else None
        target = float(row["trgt"]) if pd.notna(row["trgt"]) else np.nan
        if target != target or target <= 0:
            bins.append(0)
            continue
        entry_px = float(close.loc[t0]) if t0 in close_idx else np.nan
        if entry_px != entry_px or entry_px <= 0:
            bins.append(0)
            continue
        upper = entry_px * (1 + pt * target)
        lower = entry_px * (1 - sl * target)
        # 水平屏障若未给 t1（仅垂直=end），需调用方预置；这里兜底取序列末
        if t1 is None:
            t1 = close_idx[-1]
        bins.append(_first_touch(close, pd.Timestamp(t0),
                                 t1, upper, lower))
    out["bin"] = bins
    return out


def meta_labels(
    close: pd.Series,
    events: pd.DataFrame,
    side: pd.Series,
    cost_bps: float = 0.0,
    vol: pd.Series | None = None,
) -> pd.DataFrame:
    """Meta-Labeling：给定主模型方向 side(-1/+1)，产二分类 bin∈{0,1}。

    bin=1 ⟺ side·(close_t1/close_t0 - 1) 扣双边成本后 > 0。
    cost_bps：单边成本（基点），双边收取。
    """
    out = events.copy()
    out["side"] = side.reindex(out.index).fillna(1.0)
    cost = cost_bps / 10_000.0
    bins: list[int] = []
    for (t0, row) in out.iterrows():
        s = float(row["side"])
        if t0 not in close.index:
            bins.append(0)
            continue
        t1 = pd.Timestamp(row["t1"]) if pd.notna(row["t1"]) else None
        if t1 is None or t1 not in close.index:
            # 用后向最近可得价（保守：视为无盈利）
            bins.append(0)
            continue
        p0 = float(close.loc[t0])
        p1 = float(close.loc[t1])
        ret = p1 / p0 - 1.0
        pnl = s * ret - 2 * cost  # 双边成本
        bins.append(1 if pnl > 0 else 0)
    out["bin"] = bins
    return out
