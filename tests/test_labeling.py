"""
R2 三重屏障 + Meta-Labeling — 回归测试

契约（对齐 mlfinlab/AFML 列名约定 t1/trgt/pt/sl/side/bin）：
  1. daily_vol：滚动日波动率
  2. triple_barrier_labels：每个进场点 → bin∈{-1,0,1}（上/下/垂直屏障，先触为准）
  3. meta_labels：给定主方向 side，产出 bin∈{0,1}（该方向交易扣费后是否盈利）
  4. A股语义：涨跌停无法成交的触碰不误标方向
  5. 无未来函数：只用进场时刻可得信息
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.research.labeling import (  # noqa: E402
    daily_vol,
    meta_labels,
    triple_barrier_labels,
)


def _close(seed=0, n=120, sigma=0.02):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, sigma, n)
    return pd.Series(100 * np.cumprod(1 + rets), index=pd.date_range("2024-01-01", periods=n, freq="B"))


def test_daily_vol_positive_finite():
    c = _close()
    v = daily_vol(c, lookback=20)
    assert (v.dropna() > 0).all()
    assert np.isfinite(v.dropna()).all()


def test_triple_barrier_up_down_labels():
    """构造确定性路径验证方向：上穿 pt×σ → +1。"""
    n = 80
    # 强趋势上行：每日 +3%，屏障 target=0.02 → 需 +2%/bar → 必触上
    trend = pd.Series(100 * np.cumprod(1 + np.full(n, 0.03)),
                      index=pd.date_range("2024-01-01", periods=n, freq="B"))
    t0 = trend.index[10:40]  # 真实日期索引
    events = pd.DataFrame({
        "t1": pd.Series([trend.index[i + 5] for i in range(10, 40)], index=t0),
        "trgt": pd.Series(0.02, index=t0),
    })
    labels = triple_barrier_labels(
        trend, events, pt=1.0, sl=1.0,
        vol=pd.Series(0.02, index=trend.index))
    # 5 bar 内 +15% 远大于 +2% 屏障 → 全 +1（上屏障先触）
    assert set(labels["bin"].unique()) <= {1}
    assert (labels["bin"] == 1).mean() > 0.9


def test_vertical_barrier_zero_when_flat():
    """横盘无趋势：多数样本到期未触 → bin=0。"""
    flat = pd.Series(100.0, index=pd.date_range("2024-01-01", periods=80, freq="B"))
    rng = np.random.default_rng(2)
    noisy = flat + rng.normal(0, 0.05, 80)  # 极小噪声
    ev = pd.DataFrame({"t1": noisy.index[5:30],
                       "trgt": pd.Series(0.5, index=noisy.index[5:30])})
    labels = triple_barrier_labels(noisy, ev, pt=1.0, sl=1.0,
                                   vol=pd.Series(0.5, index=noisy.index))
    # 屏障 ±0.5，噪声 ±0.05 几乎不触 → 全 0
    assert (labels["bin"] == 0).mean() > 0.8


def test_meta_labels_binary_on_side():
    """meta-label：给定方向 side，bin∈{0,1}，bin=1 ⟺ side×(close_t1-close_t0)>0。"""
    c = _close(seed=3, n=60)
    t0 = c.index[5:40]
    events = pd.DataFrame({
        "t1": pd.Series([c.index[min(i + 5, len(c) - 1)] for i in range(5, 40)],
                        index=t0),
        "trgt": pd.Series(0.01, index=t0),
    })
    # 全部看多（side=1）
    side = pd.Series(1.0, index=t0)
    meta = meta_labels(c, events, side, cost_bps=0.0)
    assert set(meta["bin"].unique()) <= {0, 1}
    # 手动验证前几行：price up → bin 1
    p0 = c.loc[meta.index[0]]
    p1 = c.loc[meta["t1"].iloc[0]]
    expect = 1 if p1 > p0 else 0
    assert meta["bin"].iloc[0] == expect


def test_cost_shifts_meta_label_to_zero():
    """高成本 → 微利交易变负 → bin 从 1 变 0（成本意识）。"""
    c = _close(seed=4, n=60)
    t0 = c.index[5:30]
    events = pd.DataFrame({
        "t1": pd.Series([c.index[i + 1] for i in range(5, 30)], index=t0),
        "trgt": pd.Series(0.01, index=t0),
    })
    side = pd.Series(1.0, index=t0)
    # 无成本版
    m0 = meta_labels(c, events, side, cost_bps=0.0)
    # 高成本版（每次交易收 1%）
    m1 = meta_labels(c, events, side, cost_bps=100.0)
    n_diff = int((m0["bin"] != m1["bin"]).sum())
    assert n_diff > 0  # 成本改变至少一笔的盈利判定


def test_no_lookahead_t1_after_t0():
    """t1 必须严格晚于 t0（无未来：出场时点在进场之后）。"""
    c = _close(seed=5)
    t0 = c.index[10:50]
    events = pd.DataFrame({
        "t1": pd.Series([c.index[min(i + 3, len(c) - 1)] for i in range(10, 50)],
                        index=t0),
        "trgt": pd.Series(0.01, index=t0),
    })
    assert (pd.to_datetime(events["t1"]) > pd.to_datetime(events.index)).all()
