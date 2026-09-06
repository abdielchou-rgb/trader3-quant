"""
唯一性权重 / 样本去重加权（R4，AFML Ch.4）。

金融标签有区间（t0 进 t1 出）：不重叠样本信息独立；重叠样本共享同一段
价格路径 → 按 IID 训练 = 同一信息重复计数。解法：按"独占度"加权——
同一根 bar 上同时活跃的样本越多，每样本分享的信息越少、权重越低。

  concurrency[t] = Σ_i 1[t ∈ 样本i区间]      # 每 bar 活跃样本数
  uniqueness[i]  = Σ_{t∈[t0,t1]} (1/concurrency[t]) / |区间|
  w[i] = uniqueness[i]（可选 × 时间衰减/收益权重）
"""

from __future__ import annotations

import pandas as pd


def concurrency(t1: pd.Series, close_index) -> pd.Series:
    """每根 bar 上同时"活着"（区间含该 bar）的样本数。

    t1 : 每样本的出场时间（index=t0）；close_index : 时间轴（升序）。
    """
    idx = pd.DatetimeIndex(close_index)
    cc = pd.Series(0, index=idx, dtype=float)
    for t0, t1v in t1.items():
        # 区间 [t0, t1v] 内的 bar 都 +1
        mask = (idx >= pd.Timestamp(t0)) & (idx <= pd.Timestamp(t1v))
        cc[mask] += 1.0
    return cc


def _indicator_row(start, end, idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(
        ((idx >= start) & (idx <= end)).astype(float), index=idx)


def average_uniqueness(t1: pd.Series, close_index) -> pd.Series:
    """每样本的平均唯一性 = 区间内 Σ(1/concurrency)/区间长（独占份额）。"""
    idx = pd.DatetimeIndex(close_index)
    cc = concurrency(t1, idx)
    out = pd.Series(0.0, index=t1.index)
    for t0, t1v in t1.items():
        span = (idx >= pd.Timestamp(t0)) & (idx <= pd.Timestamp(t1v))
        if span.sum() == 0:
            out.loc[t0] = 0.0
            continue
        seg = cc[span]
        out.loc[t0] = float((1.0 / seg).sum() / len(seg))
    return out


def sample_weights(
    t1: pd.Series,
    close_index,
    uniqueness: pd.Series | None = None,
    time_decay: float = 1.0,
) -> pd.Series:
    """样本权重 = 唯一性（可选 × 时间衰减）。

    Parameters
    ----------
    time_decay : 越早样本权重越低；1.0 = 不衰减。
        w[t0] 按与最新样本时间距离指数衰减 (last_w 语义用简单线性近似)。
    """
    u = uniqueness if uniqueness is not None else \
        average_uniqueness(t1, close_index)
    w = u.copy()
    if time_decay != 1.0:
        # 时间衰减：位置越早权重 × decay^(距末位置)
        n = len(w)
        for j, (t0, _v) in enumerate(w.items()):
            w.loc[t0] *= (time_decay ** (n - 1 - j))
    # 归一可选：保留原始比例（调用方可自行归一）
    return w
