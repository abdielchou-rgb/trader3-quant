"""
3号交易员 — 进化数据加载器

从 qlib bin 或 akshare 拉取的 CSV 构建进化面板:
    panel = {field: (T, N) ndarray}   # 对齐的行情矩阵
    forward_returns = (T, N)          # 前瞻收益

支持两种数据源:
    1. qlib_bin（本地，无网可用，股票/指数）
    2. akshare CSV（本机有网，拉取 ETF 日线）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_qlib_panel(
    universe: str = "csi300",
    n_stocks: int = 60,
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    data_dir: str = "",
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    从 qlib bin 加载股票面板。

    Returns
    -------
    (panel, forward_returns)
    panel: {field: (T,N) float64}
    forward_returns: (T,N) — 次日收益率
    """
    from trader3.data_provider import QlibDataProvider

    dp = QlibDataProvider(data_dir=data_dir if data_dir else None)
    cal = dp.calendar()
    start_idx = dp._lower_bound(cal, start)
    end_idx = dp._upper_bound(cal, end)
    time_axis = cal[start_idx:end_idx]
    T = len(time_axis)

    codes = dp.instruments(universe)[:n_stocks]
    closes = {}
    for code in codes:
        close, dates = dp.load_stock(code.lower(), "close", start, end)
        if len(close) >= 100:
            closes[code] = (close, dates)

    if len(closes) < 20:
        raise RuntimeError(f"可用股票不足 ({len(closes)} < 20)")

    fields = ["open", "high", "low", "close", "volume", "vwap", "amount"]
    panel = {f: np.full((T, len(closes)), np.nan, dtype=np.float64) for f in fields}

    codes_list = list(closes.keys())
    for j, (code, (vals, dates)) in enumerate(closes.items()):
        for f in fields:
            try:
                fvals, fdates = dp.load_stock(code.lower(), f, start, end)
            except Exception:
                continue
            # 对齐到 time_axis
            dmap = {d: v for d, v in zip(fdates, fvals)}
            for i, d in enumerate(time_axis):
                if d in dmap:
                    panel[f][i, j] = dmap[d]

    # 前向收益（次日）：fwd[t] = t→t+1 的未来收益，供信号[t] 配对
    close_panel = panel["close"]
    fwd = np.full_like(close_panel, np.nan)
    for j in range(close_panel.shape[1]):
        col = close_panel[:, j]
        valid_pos = np.where(col > 0)[0]
        for idx in range(len(valid_pos) - 1):
            i_prev, i_curr = valid_pos[idx], valid_pos[idx + 1]
            if i_curr - i_prev <= 5:
                fwd[i_prev, j] = col[i_curr] / col[i_prev] - 1.0

    return panel, fwd


def load_etf_panel(
    csv_dir: str,
    n_etfs: int = 30,
    start: str = "2020-01-01",
    end: str = "2024-12-31",
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    从 akshare 拉取的 ETF CSV 目录加载面板。

    CSV 格式（每只 ETF 一个文件）:
        date,open,high,low,close,volume
        2020-01-02,3.2,3.25,3.15,3.22,1000000
    """
    import pandas as pd

    csv_dir = Path(csv_dir)
    if not csv_dir.exists():
        raise FileNotFoundError(f"ETF CSV 目录不存在: {csv_dir}")

    csvs = sorted(csv_dir.glob("*.csv"))[:n_etfs]
    if not csvs:
        raise RuntimeError(f"ETF 目录 {csv_dir} 无 CSV 文件")

    # 统一日期轴
    all_dates = set()
    frames = {}
    for csv_path in csvs:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df[(df["date"].dt.strftime("%Y-%m-%d") >= start) &
                (df["date"].dt.strftime("%Y-%m-%d") <= end)]
        df = df.set_index("date")
        frames[csv_path.stem] = df
        all_dates.update(df.index)

    dates = sorted(all_dates)
    T = len(dates)
    N = len(frames)
    if T < 30 or N < 5:
        raise RuntimeError(f"ETF 数据不足: T={T}, N={N}")

    fields = ["open", "high", "low", "close", "volume"]
    panel = {f: np.full((T, N), np.nan, dtype=np.float64) for f in fields}
    names = list(frames.keys())

    for j, (name, df) in enumerate(frames.items()):
        df = df.reindex(dates)
        for f in fields:
            if f in df.columns:
                panel[f][:, j] = df[f].values

    # 前向收益（次日）：fwd[t] = t→t+1 的未来收益，供信号[t] 配对
    close_panel = panel["close"]
    fwd = np.full_like(close_panel, np.nan)
    for j in range(N):
        col = close_panel[:, j]
        valid = np.where(col > 0)[0]
        for idx in range(len(valid) - 1):
            i_prev, i_curr = valid[idx], valid[idx + 1]
            if i_curr - i_prev <= 5:
                fwd[i_prev, j] = col[i_curr] / col[i_prev] - 1.0

    return panel, fwd