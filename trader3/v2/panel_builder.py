"""
面板构建器（Panel Builder）—— 让调度器可自主运行。

统一面板格式：(date × (asset, field)) MultiIndex DataFrame，字段 open/high/low/close/
volume/vwap/amount（与 factor_dsl / quant_pipeline 一致）。

数据源可注入（source(code, lookback_days, end) -> 日线 DataFrame）；缺省为合成数据，
便于离线测试与演示。生产接入真实行情时，注入 qa_accessor / 数据库 适配器即可。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume", "vwap", "amount"]


def _synthetic_source(code: str, lookback_days: int, end: datetime) -> pd.DataFrame:
    """合成日线（带轻微截面差异），仅用于离线/演示。"""
    dates = pd.date_range(end=end, periods=lookback_days, freq="B")
    seed = abs(hash(code)) % (2**31)
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, lookback_days))
    open_ = close * (1 + rng.normal(0, 0.001, lookback_days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, lookback_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, lookback_days)))
    vol = np.abs(rng.normal(1e5, 2e4, lookback_days)) + 1e4
    vwap = (high + low + close) / 3
    amount = vol * vwap
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": vol, "vwap": vwap, "amount": amount},
        index=dates,
    )


def build_panel(universe: list[str], lookback_days: int = 250,
                *, source: Callable | None = None,
                end: datetime | None = None) -> pd.DataFrame:
    """构建截面面板。source 缺省用合成数据。"""
    src = source or _synthetic_source
    end = end or datetime.now()
    parts: dict[str, pd.DataFrame] = {}
    for code in universe:
        try:
            bars = src(code, lookback_days, end)
        except Exception:
            bars = None
        if bars is None or (isinstance(bars, pd.DataFrame) and bars.empty):
            continue
        parts[code] = bars
    if not parts:
        raise ValueError("所有标的数据源均返回空，无法构建面板。")
    panel = pd.concat(parts, axis=1, names=["asset", "field"])
    panel = panel.reindex(columns=pd.MultiIndex.from_product([list(parts.keys()), FIELDS]))
    panel = panel.sort_index(axis=1)
    return panel
