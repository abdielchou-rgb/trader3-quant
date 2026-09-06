"""
回测严谨性模块（R3）—— PSR / DSR / MTRL 自研实现。

解决 GP/网格挖掘最深的坑：在同一份历史上反复试错后，"最佳回测"
大概率是运气。若不惩罚试验次数，任何夏普都是虚的。

口径约定：内部一律用「期」夏普（不年化），仅 MTRL 输出按年。
  SR̂ = mean(r)/std(r)  （与样本量 √(T-1) 自然匹配）

公式（Bailey & López de Prado, 2014）：
  PSR(SR*) = Φ( (SR̂ − SR*)·√(T−1) / √(1 − γ3·SR̂ + (γ4/4)·SR̂²) )
  SR* = √V[{SR̂_n}]·[ (1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)) ]  # N=试验次数, γ≈0.5772
  DSR = PSR(SR*)；SR*=0 即退化 PSR
γ3=偏度(excess)、γ4=峰度(excess)。
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

_EULER = 0.5772156649015329


def _moments(rets: np.ndarray) -> tuple[float, float, float, float]:
    """(mean, std, skew_excess, kurt_excess)。"""
    r = np.asarray(rets, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return 0.0, 1.0, 0.0, 0.0
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd < 1e-12:
        return mu, 1e-12, 0.0, 0.0
    z = (r - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4)) - 3.0
    return mu, sd, skew, kurt


def _period_sharpe(rets: np.ndarray) -> float:
    """期夏普（不年化）：mean/std。"""
    mu, sd, _, _ = _moments(rets)
    return float(mu / sd) if sd > 1e-12 else 0.0


def _denom(sr: float, skew: float, kurt: float) -> float:
    """PSR 分母（excess 口径）：√(1 − γ3·SR̂ + (γ4/4)·SR̂²)。"""
    return math.sqrt(max(1.0 - skew * sr + kurt / 4.0 * sr ** 2, 1e-12))


def probabilistic_sharpe_ratio(
    returns: np.ndarray, sr_benchmark: float = 0.0,
) -> float:
    """PSR：样本夏普显著高于基准 SR* 的概率（期口径基准）。"""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 5:
        return 0.5
    mu, sd, skew, kurt = _moments(r)
    sr = _period_sharpe(r)
    d = _denom(sr, skew, kurt)
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / d
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int = 1) -> float:
    """DSR：考虑试验次数 N 后的显著夏普概率（期口径）。"""
    if n_trials < 1:
        n_trials = 1
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 5:
        return 0.5
    mu, sd, skew, kurt = _moments(r)
    sr = _period_sharpe(r)
    # SR̂ 的估计方差 → SR* 期望最大值（LdP 公式，期口径）
    est_var = 1.0 - skew * sr + kurt / 4.0 * sr ** 2
    sr_std = math.sqrt(max(est_var, 1e-12) / max(n - 1, 1))
    inv_n = stats.norm.ppf(1.0 - 1.0 / n_trials)
    inv_ne = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    sr_star = sr_std * ((1.0 - _EULER) * inv_n + _EULER * inv_ne)
    return probabilistic_sharpe_ratio(r, sr_benchmark=max(sr_star, 0.0))


def minimum_track_record_length(
    returns: np.ndarray, sr_benchmark: float = 0.0, prob: float = 0.95,
) -> float:
    """最小可信记录长度（年）：该夏普在 prob 置信下显著所需的样本年数。"""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 5:
        return float("inf")
    mu, sd, skew, kurt = _moments(r)
    sr = _period_sharpe(r)
    if sr <= sr_benchmark:
        return float("inf")
    z_target = stats.norm.ppf(prob)
    d = _denom(sr, skew, kurt)
    spread = max(sr - sr_benchmark, 1e-12)
    t_samples = (z_target * d / spread) ** 2 + 1.0  # 所需期数
    return float(max(t_samples / 252.0, 0.0))
