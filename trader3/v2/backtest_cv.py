"""严谨 OOS 回测（对标 P0-4）：Purged K-fold + embargo + 组合式回测 + 缩水夏普。

解决单扩张窗口 RankIC 的泄漏/乐观偏差：
  - 时序 walk-forward 测试折，折间 embargo 间隔、训练折对重叠标签做 purge；
  - 逐折截面 RankIC 聚合为 OOS t 统计量（比逐期 t 更保守、更诚实）；
  - 多空组合式回测，报告夏普与 *Deflated Sharpe Ratio*（多重检验校正）。

 基于 numpy/pandas；缩水夏普分位函数优先使用 scipy（否则回退至标准库）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm

    def _norm_ppf(p: float) -> float:
        return float(norm.ppf(p))
except Exception:  # pragma: no cover - scipy 通常可用
    import statistics

    def _norm_ppf(p: float) -> float:
        from math import sqrt

        if p <= 0.0:
            return float("-inf")
        if p >= 1.0:
            return float("inf")
        return sqrt(2.0) * statistics.NormalDist().inv_cdf(p)


def embargo_purged_splits(n: int, test_size: int, embargo: int,
                          min_train: int, horizon: int):
    """时序 walk-forward 切分，带 embargo 间隔与标签 purge。

    对测试折 [i, i+test_size)，从训练集中剔除：
      - 标签窗口与测试折重叠的样本（purge: j ∈ [i-horizon+1, i+test_size-1]）；
      - 测试折之后 embargo 长度的样本（避免信息泄漏）。
    产出 (train_idx, test_idx) 的 numpy 数组（全局行下标）。
    """
    i = min_train
    while i + test_size <= n:
        te = np.arange(i, i + test_size)
        purge_lo = max(0, i - horizon + 1)
        emb_hi = min(n, i + test_size + embargo)
        tr = np.concatenate([np.arange(min_train, purge_lo), np.arange(emb_hi, n)])
        tr = tr[(tr >= min_train) & (tr < n)]
        if len(tr) >= min_train:
            yield tr, te
        i += test_size


def _mean_rankic(scores_df: pd.DataFrame, fwd_df: pd.DataFrame) -> float:
    ics: list[float] = []
    for t in scores_df.index:
        a = scores_df.loc[t]
        b = fwd_df.loc[t]
        m = a.notna() & b.notna()
        if m.sum() < 5:
            continue
        aa = a[m].rank()
        bb = b[m].rank()
        if aa.std() == 0 or bb.std() == 0:
            continue
        ics.append(float(np.corrcoef(aa.values, bb.values)[0, 1]))
    return float(np.nanmean(ics)) if ics else float("nan")


def purged_cv_rankic(scores: pd.DataFrame, fwd: pd.DataFrame, n_splits: int = 5,
                     embargo: int = 5, min_train: int | None = None,
                     horizon: int = 5) -> dict:
    """Purged-CV 截面 RankIC：逐折 IC 聚合为 OOS 统计量。"""
    n = len(scores)
    min_train = min_train or max(20, n // 4)
    test_size = max(5, n // (n_splits + 1))
    fold_ics: list[float] = []
    for _tr, te in embargo_purged_splits(n, test_size, embargo, min_train, horizon):
        if len(te) == 0:
            continue
        ic = _mean_rankic(scores.iloc[te], fwd.iloc[te])
        if np.isfinite(ic):
            fold_ics.append(ic)
    fold_ics = np.array(fold_ics)
    if len(fold_ics) < 1:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "ic_t": float("nan"),
                "pos_ratio": float("nan"), "stability": float("nan"), "fold_ics": fold_ics}
    ic_mean = float(fold_ics.mean())
    ic_std = float(fold_ics.std())
    k = len(fold_ics)
    ic_t = ic_mean / (ic_std / math.sqrt(k)) if ic_std > 0 else 0.0
    pos_ratio = float((fold_ics > 0).mean())
    stability = 1.0 - abs(ic_std) / abs(ic_mean) if abs(ic_mean) > 0 else 0.0
    return {"ic_mean": ic_mean, "ic_std": ic_std, "ic_t": float(ic_t),
            "pos_ratio": pos_ratio, "stability": float(stability), "fold_ics": fold_ics}


def probabilistic_sharpe(sr: float, n_obs: int, skew: float = 0.0,
                         kurt: float = 3.0, sr0: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio（PSR, Bailey & López de Prado）。"""
    denom = math.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr)
    if denom <= 0:
        denom = 1e-9
    z = (sr - sr0) * math.sqrt(max(n_obs, 1)) / denom
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def deflated_sharpe(sr: float, n_obs: int, skew: float = 0.0,
                    kurt: float = 3.0, n_trials: int = 5) -> float:
    """Deflated Sharpe Ratio：以多重检验期望最大夏普为基准的 PSR。

    ``n_trials`` 为搜索过的策略数（因子工厂中可传候选数），用于校正过拟合。
    """
    if n_trials <= 1:
        return probabilistic_sharpe(sr, n_obs, skew, kurt, sr0=0.0)
    sr0 = sr * _norm_ppf(1.0 - 1.0 / n_trials)
    return probabilistic_sharpe(sr, n_obs, skew, kurt, sr0=sr0)


def combinatorial_backtest(scores: pd.DataFrame, fwd: pd.DataFrame, n_splits: int = 5,
                           embargo: int = 5, top_frac: float = 0.2, bottom_frac: float = 0.2,
                           min_train: int | None = None, horizon: int = 5,
                           n_trials: int = 5) -> dict:
    """组合式（多空）回测：测试折上按截面分位多空，聚合为收益序列。"""
    n = len(scores)
    min_train = min_train or max(20, n // 4)
    test_size = max(5, n // (n_splits + 1))
    rets: list[float] = []
    for _tr, te in embargo_purged_splits(n, test_size, embargo, min_train, horizon):
        for t in te:
            row = scores.iloc[t]
            ff = fwd.iloc[t]
            r = row.rank()
            if r.notna().sum() < 5 or ff.notna().sum() == 0:
                continue
            q_top = r.quantile(1.0 - top_frac)
            q_bot = r.quantile(bottom_frac)
            longs = r >= q_top
            shorts = r <= q_bot
            if longs.sum() == 0 or shorts.sum() == 0:
                continue
            ret = float(ff[longs].mean() - ff[shorts].mean())
            if np.isfinite(ret):
                rets.append(ret)
    rets = np.array(rets)
    if len(rets) < 3:
        return {"sharpe": 0.0, "deflated_sharpe": 0.0, "max_dd": 0.0,
                "mean_ret": 0.0, "n_folds": n_splits, "n_obs": 0}
    mean = float(rets.mean())
    std = float(rets.std())
    sharpe = mean / (std / math.sqrt(252)) if std > 0 else 0.0
    cum = np.cumprod(1.0 + rets)
    run_max = np.maximum.accumulate(cum)
    dd = float(((run_max - cum) / run_max).max()) if len(cum) else 0.0
    skew = float(pd.Series(rets).skew() or 0.0)
    kurt = float((pd.Series(rets).kurt() or 0.0) + 3.0)
    dsr = deflated_sharpe(sharpe, len(rets), skew, kurt, n_trials)
    return {"sharpe": float(sharpe), "deflated_sharpe": float(dsr), "max_dd": dd,
            "mean_ret": mean, "n_folds": n_splits, "n_obs": len(rets)}
