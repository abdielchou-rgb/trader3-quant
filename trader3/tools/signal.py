"""
3号交易员 — 信号验证 / 市场状态 Tools (M4: 真实信号验证 + HMM 市场状态)

M4 upgrades:
1. Real IC calculation (Spearman rank correlation cross-section each period)
2. Real grouped returns (5 quintiles) + monotonicity score
3. Real half-life estimation via exponential decay fitting
4. Real crowding index (average pairwise correlation with competing factors)
5. Real conditional validity (IC split by volatility regime)
6. Real HMM market regime detection (GaussianHMM via EM, fallback rule-based)
7. Attention momentum detection (serenity-radar lightweight)
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from trader3.base_tool import BaseTool, ChartSpec, Trader3Response
from trader3.models import RegimeDiagnosis, SignalValidationReport


# ═══════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════

TRADING_DAYS = 252
N_ASSETS = 200
N_PERIODS = 60
HMM_N_STATES = 4
HMM_N_FEATURES = 4
HMM_N_ITER = 100
SEED_SIGNAL = 42
SEED_REGIME = 2024


# ═══════════════════════════════════════════
# Synthetic Data Generators
# ═══════════════════════════════════════════


def _generate_signal_panel(
    n_periods: int = N_PERIODS,
    n_assets: int = N_ASSETS,
    seed: int = SEED_SIGNAL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic panel data for signal validation.

    Creates a realistic cross-section of assets with:
    - Persistent alpha (AR(1) with phi=0.95)
    - Signal that captures alpha with IC ~0.3
    - Forward returns with decaying signal predictive power

    Returns
    -------
    signal : (n_periods, n_assets) — Z-score normalized signal
    forward_returns : (n_periods, n_assets) — next-period returns
    market_returns : (n_periods,) — market-level returns
    """
    rng = np.random.default_rng(seed)

    # Market-level returns: 8% annual drift, 20% vol
    mu_m = 0.08 / TRADING_DAYS
    sigma_m = 0.20 / math.sqrt(TRADING_DAYS)
    market_returns = rng.normal(mu_m, sigma_m, n_periods).astype(np.float64)

    # Asset betas around 1.0
    betas = rng.normal(1.0, 0.15, n_assets).astype(np.float64)

    # Persistent alpha (AR(1), half-life ~13 days)
    alpha_daily_std = 0.04 / math.sqrt(TRADING_DAYS)
    phi = 0.95
    innov_std = alpha_daily_std * math.sqrt(1.0 - phi**2)

    alphas = np.zeros((n_periods, n_assets), dtype=np.float64)
    alphas[0] = rng.normal(0.0, alpha_daily_std, n_assets)
    for t in range(1, n_periods):
        alphas[t] = phi * alphas[t - 1] + rng.normal(0.0, innov_std, n_assets)

    # Specific volatility
    spec_vol = 0.25 / math.sqrt(TRADING_DAYS)
    epsilons = rng.normal(0.0, spec_vol, (n_periods, n_assets)).astype(np.float64)

    # Total returns
    stock_returns = np.outer(market_returns, betas) + alphas + epsilons

    # Signal: alpha + noise (IC ~ 0.316)
    signal_noise_std = 3.0 * alpha_daily_std
    raw_signal = alphas + rng.normal(0.0, signal_noise_std, (n_periods, n_assets)).astype(np.float64)

    # Z-score normalize each period
    mean_t = np.mean(raw_signal, axis=1, keepdims=True)
    std_t = np.std(raw_signal, axis=1, keepdims=True)
    signal = (raw_signal - mean_t) / (std_t + 1e-10)

    # Forward returns (next-period returns aligned to current signal)
    forward_returns = np.roll(stock_returns, -1, axis=0)[:-1]
    signal_aligned = signal[:-1]
    market_aligned = market_returns[:-1]

    return signal_aligned, forward_returns, market_aligned


def _generate_market_regime_data(
    lookback: int = 60,
    seed: int = SEED_REGIME,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic market data for regime detection.

    Returns
    -------
    dict with keys:
        prices — price series
        volumes — volume series
        returns — daily returns
        rolling_vol — 20d rolling annualized vol
    """
    rng = np.random.default_rng(seed)

    T = max(lookback * 2, 252 * 3)

    # Simulate 3 years of market data with regime shifts
    # Phase 1: trending up (first 30%)
    # Phase 2: ranging (middle 40%)
    # Phase 3: bearish + high vol (last 30%)
    n_up = int(T * 0.30)
    n_range = int(T * 0.40)
    n_down = T - n_up - n_range

    daily_rets = []
    # Trending up: +0.08% avg, 15% vol
    for _ in range(n_up):
        daily_rets.append(rng.normal(0.0008, 0.15 / math.sqrt(TRADING_DAYS)))
    # Ranging: 0% avg, 12% vol
    for _ in range(n_range):
        daily_rets.append(rng.normal(0.0, 0.12 / math.sqrt(TRADING_DAYS)))
    # Bearish + high vol: -0.06% avg, 35% vol
    for _ in range(n_down):
        daily_rets.append(rng.normal(-0.0006, 0.35 / math.sqrt(TRADING_DAYS)))

    daily_rets = np.array(daily_rets, dtype=np.float64)
    rng.shuffle(daily_rets)  # avoid perfectly sequential regimes

    prices = 3000.0 * np.exp(np.cumsum(daily_rets))

    # Volume: inversely correlated with vol regime
    base_vol = 8000  # 亿
    vol_noise = rng.normal(0, 0.15, T)
    volumes = base_vol * (1.0 + vol_noise)

    rolling_vol = np.array([
        np.std(daily_rets[max(0, t - 20):t + 1]) * math.sqrt(TRADING_DAYS)
        for t in range(T)
    ])
    volume_adj = 1.0 - 0.3 * (rolling_vol - np.mean(rolling_vol)) / (np.std(rolling_vol) + 1e-10)
    volumes = volumes * np.clip(volume_adj, 0.5, 1.5)

    return {
        "prices": prices,
        "volumes": volumes,
        "returns": daily_rets,
        "rolling_vol": rolling_vol,
    }


# ═══════════════════════════════════════════
# IC Calculation
# ═══════════════════════════════════════════


def _compute_ic_series(signal: np.ndarray, forward_returns: np.ndarray) -> np.ndarray:
    """Cross-sectional Spearman rank correlation for each period."""
    from scipy.stats import spearmanr

    n_periods = signal.shape[0]
    ic_values = np.zeros(n_periods)
    for t in range(n_periods):
        if np.std(signal[t]) < 1e-10 or np.std(forward_returns[t]) < 1e-10:
            ic_values[t] = 0.0
        else:
            rho, _ = spearmanr(signal[t], forward_returns[t])
            ic_values[t] = rho if not np.isnan(rho) else 0.0
    return ic_values


def _compute_ic_stats(ic_series: np.ndarray) -> Tuple[float, float, float]:
    """Mean IC, IC std, ICIR."""
    mean_ic = float(np.mean(ic_series))
    std_ic = float(np.std(ic_series, ddof=1))
    icir = mean_ic / std_ic if std_ic > 1e-10 else 0.0
    return mean_ic, std_ic, icir


# ═══════════════════════════════════════════
# Grouped Returns & Monotonicity
# ═══════════════════════════════════════════


def _compute_group_returns(
    signal: np.ndarray, forward_returns: np.ndarray
) -> Tuple[Dict[str, float], float, float, float]:
    """
    Rank assets by signal -> 5 quintiles -> compute forward returns.

    Returns
    -------
    group_returns : dict {label: return}
    monotonicity : float (0..1)
    long_short_return : float
    long_only_return : float
    """
    n_periods, n_assets = signal.shape
    quintile_rets = np.zeros((n_periods, 5))

    for t in range(n_periods):
        # Rank assets by signal (ascending: 0 = lowest signal)
        ranks = np.argsort(np.argsort(signal[t]))  # 0..n_assets-1
        # Reverse so Q1 = highest signal (top quintile, long side)
        # Q5 = lowest signal (bottom quintile, short side)
        quintile = 4 - np.floor(ranks / n_assets * 5).astype(int)
        quintile = np.clip(quintile, 0, 4)

        for q in range(5):
            mask = quintile == q
            if mask.sum() > 0:
                quintile_rets[t, q] = np.mean(forward_returns[t, mask])

    # Mean return per quintile (annualized)
    mean_rets = np.mean(quintile_rets, axis=0) * TRADING_DAYS

    # Monotonicity: fraction of adjacent pairs where Q_i > Q_{i+1}
    # (Q1 = highest signal > Q2 > ... > Q5 = lowest signal)
    monotonic_pairs = sum(
        1 for i in range(4) if mean_rets[i] > mean_rets[i + 1]
    )
    monotonicity = monotonic_pairs / 4.0

    long_short = mean_rets[0] - mean_rets[4]
    long_only = mean_rets[0]

    group_returns = {
        "Q1 (多头)": float(mean_rets[0]),
        "Q2": float(mean_rets[1]),
        "Q3": float(mean_rets[2]),
        "Q4": float(mean_rets[3]),
        "Q5 (空头)": float(mean_rets[4]),
    }

    return group_returns, monotonicity, long_short, long_only


# ═══════════════════════════════════════════
# Half-Life Estimation
# ═══════════════════════════════════════════


def _compute_half_life(ic_series: np.ndarray) -> float:
    """
    自相关衰减法估计半衰期（单位：交易日）。

    对 IC 序列拟合 AR(1)：ρ(k) = φ^k，半衰期 = ln(0.5)/ln(|φ|)。
    |φ| >= 1（不衰减/发散）或样本不足时返回保守默认值。
    """
    n = len(ic_series)
    if n < 8:
        return 12.0

    x = np.asarray(ic_series, dtype=np.float64)
    x = x - np.nanmean(x)
    denom = float(np.nansum(x * x))
    if denom < 1e-12:
        return 12.0
    # lag-1 自相关
    phi = float(np.nansum(x[:-1] * x[1:]) / denom)
    if not np.isfinite(phi) or abs(phi) < 1e-6:
        return 1.0  # 近白噪声 → 半衰期约 1 期
    if abs(phi) >= 0.999:
        return 252.0  # 近单位根 → 上限一年
    hl = math.log(0.5) / math.log(abs(phi))
    return float(np.clip(hl, 1.0, 252.0))


# ═══════════════════════════════════════════
# Crowding Index
# ═══════════════════════════════════════════


def _compute_crowding_index(signal: np.ndarray) -> float:
    """
    拥挤度代理（真实可计算）：信号面板各资产列之间的平均|成对相关|。

    直觉：若信号在多数资产上高度同向联动，说明策略承载空间趋紧。
    最多抽样 60 对以控制计算量；有效对不足时返回 0。
    """
    n_periods, n_assets = signal.shape
    if n_assets < 3 or n_periods < 5:
        return 0.0

    rng = np.random.default_rng(SEED_SIGNAL + 1)
    cols = list(range(n_assets))
    pairs = set()
    max_pairs = 60
    while len(pairs) < min(max_pairs, n_assets * (n_assets - 1) // 2):
        i, j = rng.integers(0, n_assets, size=2)
        if i != j:
            pairs.add((min(i, j), max(i, j)))

    corrs = []
    for i, j in pairs:
        a, b = signal[:, i], signal[:, j]
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 5:
            continue
        sa, sb = a[mask], b[mask]
        if np.std(sa) < 1e-10 or np.std(sb) < 1e-10:
            continue
        c = np.corrcoef(sa, sb)[0, 1]
        if np.isfinite(c):
            corrs.append(abs(c))

    return float(np.mean(corrs)) if corrs else 0.0


# ═══════════════════════════════════════════
# Conditional Validity
# ═══════════════════════════════════════════


def _compute_conditional_validity(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    market_returns: np.ndarray,
) -> Dict[str, float]:
    """
    Split data by market volatility regime and compute IC within each.

    States: low_vol, medium_vol, high_vol
    """
    from scipy.stats import spearmanr

    # Rolling 20-period volatility
    vol = np.zeros(len(market_returns))
    for t in range(len(market_returns)):
        window = market_returns[max(0, t - 20):t + 1]
        vol[t] = np.std(window) if len(window) > 1 else 0.01

    vol_median = np.median(vol)

    def _ic_in_mask(mask: np.ndarray) -> float:
        if mask.sum() < 3:
            return 0.0
        s = signal[mask]
        f = forward_returns[mask]
        ic_vals = []
        for t in range(s.shape[0]):
            if np.std(s[t]) > 1e-10 and np.std(f[t]) > 1e-10:
                rho, _ = spearmanr(s[t], f[t])
                if not np.isnan(rho):
                    ic_vals.append(rho)
        return float(np.mean(ic_vals)) if ic_vals else 0.0

    low_mask = vol <= vol_median * 0.8
    mid_mask = (vol > vol_median * 0.8) & (vol < vol_median * 1.2)
    high_mask = vol >= vol_median * 1.2

    return {
        "低波动": _ic_in_mask(low_mask),
        "中波动": _ic_in_mask(mid_mask),
        "高波动": _ic_in_mask(high_mask),
    }


# ═══════════════════════════════════════════
# HMM Market Regime Detection
# ═══════════════════════════════════════════


class _CustomGaussianHMM:
    """
    Minimal Gaussian HMM using EM (Baum-Welch) for market regime detection.

    Public API mirrors hmmlearn.GaussianHMM: fit(), predict(), predict_proba().
    Used when hmmlearn is not available.
    """

    def __init__(
        self,
        n_states: int = 4,
        n_features: int = 4,
        random_state: int = 42,
        n_iter: int = 100,
        tol: float = 1e-4,
    ):
        self.n_states = n_states
        self.n_features = n_features
        self.rng = np.random.default_rng(random_state)
        self.n_iter = n_iter
        self.tol = tol

        self.transmat_: Optional[np.ndarray] = None
        self.startprob_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covars_: Optional[np.ndarray] = None

    def _initialize(self, X: np.ndarray) -> None:
        """Initialize parameters via stratified sampling."""
        n_samples = X.shape[0]

        # Uniform start probabilities
        self.startprob_ = np.ones(self.n_states) / self.n_states

        # Transition matrix with strong self-transition bias
        self.transmat_ = np.full((self.n_states, self.n_states), 0.05)
        np.fill_diagonal(self.transmat_, 0.65)
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

        # Means via stratified sampling along time axis
        indices = np.linspace(0, n_samples - 1, self.n_states).astype(int)
        self.means_ = X[indices].copy()

        # Diagonal covariances from data variance
        var_mean = float(np.mean(np.var(X, axis=0)))
        self.covars_ = np.array([
            np.eye(self.n_features) * var_mean for _ in range(self.n_states)
        ])

    def _e_step(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward-Backward algorithm."""
        from scipy.stats import multivariate_normal

        n_samples = X.shape[0]

        # Emission probabilities P(O_t | S_t = i)
        emission = np.zeros((n_samples, self.n_states))
        for i in range(self.n_states):
            try:
                emission[:, i] = multivariate_normal.pdf(
                    X,
                    mean=self.means_[i],
                    cov=self.covars_[i],
                    allow_singular=True,
                )
            except Exception:
                emission[:, i] = 1e-300
        emission = np.maximum(emission, 1e-300)

        # Forward pass
        alpha = np.zeros((n_samples, self.n_states))
        alpha[0] = self.startprob_ * emission[0]
        norm0 = alpha[0].sum()
        if norm0 > 0:
            alpha[0] /= norm0

        for t in range(1, n_samples):
            alpha[t] = emission[t] * (alpha[t - 1] @ self.transmat_)
            norm_t = alpha[t].sum()
            if norm_t > 0:
                alpha[t] /= norm_t

        # Backward pass
        beta = np.zeros((n_samples, self.n_states))
        beta[-1] = 1.0

        for t in range(n_samples - 2, -1, -1):
            beta[t] = self.transmat_ @ (emission[t + 1] * beta[t + 1])
            norm_t = beta[t].sum()
            if norm_t > 0:
                beta[t] /= norm_t

        # Posterior gamma: P(S_t = i | O)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

        # Pairwise posterior xi: P(S_t = i, S_{t+1} = j | O)
        xi = np.zeros((n_samples - 1, self.n_states, self.n_states))
        for t in range(n_samples - 1):
            xi[t] = (
                alpha[t].reshape(-1, 1)
                * self.transmat_
                * emission[t + 1]
                * beta[t + 1]
            )
            norm_t = xi[t].sum()
            if norm_t > 0:
                xi[t] /= norm_t

        return gamma, xi, emission

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        """M-step: re-estimate parameters."""
        n_samples = X.shape[0]

        # Start probabilities
        self.startprob_ = gamma[0]
        sp_sum = self.startprob_.sum()
        if sp_sum > 0:
            self.startprob_ /= sp_sum

        # Transition matrix
        for i in range(self.n_states):
            denom = gamma[:-1, i].sum()
            if denom > 0:
                self.transmat_[i] = xi[:, i, :].sum(axis=0) / denom

        # Means
        for i in range(self.n_states):
            denom = gamma[:, i].sum()
            if denom > 0:
                self.means_[i] = (
                    (gamma[:, i].reshape(-1, 1) * X).sum(axis=0) / denom
                )

        # Covariances
        for i in range(self.n_states):
            diff = X - self.means_[i]
            weighted = gamma[:, i].reshape(-1, 1, 1) * (
                diff.reshape(-1, self.n_features, 1)
                @ diff.reshape(-1, 1, self.n_features)
            )
            denom = gamma[:, i].sum()
            if denom > 0:
                self.covars_[i] = weighted.sum(axis=0) / denom
            else:
                self.covars_[i] = np.eye(self.n_features) * 0.01
            # Regularize: ensure positive definite
            self.covars_[i] += np.eye(self.n_features) * 0.01

    def fit(self, X: np.ndarray) -> None:
        """Baum-Welch EM training."""
        self._initialize(X)

        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            gamma, xi, emission = self._e_step(X)
            self._m_step(X, gamma, xi)

            # Log-likelihood
            log_likelihood = float(
                np.sum(np.log(np.maximum(emission.sum(axis=1), 1e-300))))
            if abs(log_likelihood - prev_ll) < self.tol:
                break
            prev_ll = log_likelihood

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute posterior state probabilities."""
        gamma, _, _ = self._e_step(X)
        return gamma

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Most likely state sequence (Viterbi approximation via argmax)."""
        gamma = self.predict_proba(X)
        return gamma.argmax(axis=1)


def _try_hmmlearn():
    """Try to import hmmlearn; return module or None."""
    try:
        from hmmlearn import hmm
        return hmm
    except ImportError:
        return None


def _causal_zscore(a: np.ndarray) -> np.ndarray:
    """仅用截至当期的历史做标准化（expanding, 消除全样本前视）。"""
    n = len(a)
    out = np.zeros(n)
    for t in range(n):
        lo = max(0, t - 60)  # 最多回看60期，兼顾状态切换敏感性
        window = a[lo:t + 1]
        mu = float(np.mean(window))
        sd = float(np.std(window))
        out[t] = (a[t] - mu) / (sd + 1e-10)
    return out


def _fit_hmm(data: Dict[str, np.ndarray]) -> Tuple[str, Dict[str, float], float]:
    """
    Fit HMM to market data and extract regime probabilities.

    Tries hmmlearn first, falls back to custom implementation.

    Returns
    -------
    current_regime : str
    regime_probabilities : dict {state_name: prob}
    regime_entropy : float
    """
    returns = data["returns"]
    prices = data["prices"]
    volumes = data.get("volumes")
    volumes_real = bool(data.get("volumes_real", False))
    T = len(returns)

    # Build feature matrix: daily return, 20d vol [, volume change if real volume]
    vol_20d = np.array([
        np.std(returns[max(0, t - 20):t + 1]) * math.sqrt(TRADING_DAYS)
        for t in range(T)
    ])
    features = [returns, _causal_zscore(vol_20d)]

    if volumes is not None and volumes_real and len(volumes) >= T:
        vol_change = np.array([
            volumes[t] / np.mean(volumes[max(0, t - 20):t + 1]) - 1.0
            if t >= 5 else 0.0
            for t in range(T)
        ])
        spread_proxy = 1.0 / (volumes / np.mean(volumes) + 0.1)
        features.append(_causal_zscore(vol_change))
        features.append(_causal_zscore(spread_proxy))

    X = np.column_stack(features)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)

    # Try hmmlearn first
    hmm_module = _try_hmmlearn()
    if hmm_module is not None:
        model = hmm_module.GaussianHMM(
            n_components=HMM_N_STATES,
            covariance_type="full",
            random_state=SEED_REGIME,
            n_iter=HMM_N_ITER,
            tol=1e-4,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)
        state_probs = model.predict_proba(X)
        current_probs = state_probs[-1]
    else:
        # Fallback to custom implementation
        model = _CustomGaussianHMM(
            n_states=HMM_N_STATES,
            n_features=HMM_N_FEATURES,
            random_state=SEED_REGIME,
            n_iter=HMM_N_ITER,
        )
        model.fit(X)
        state_probs = model.predict_proba(X)
        current_probs = state_probs[-1]

    # Label states based on mean return and volatility
    state_labels = {}
    pred_states = model.predict(X)
    for i in range(HMM_N_STATES):
        mask = pred_states == i
        if mask.sum() == 0:
            state_labels[i] = "ranging"
            continue
        mean_ret = float(np.mean(returns[mask]))
        vol_vals = []
        for t in range(T):
            if pred_states[t] == i and t >= 20:
                vol_vals.append(
                    np.std(returns[t - 20:t + 1]) * math.sqrt(TRADING_DAYS)
                )
        mean_vol = float(np.mean(vol_vals)) if vol_vals else 0.20

        if mean_ret > 0.0005 and mean_vol < 0.20:
            state_labels[i] = "trending_up"
        elif mean_ret > 0.0005 and mean_vol >= 0.20:
            state_labels[i] = "high_vol"
        elif mean_ret < -0.0003 and mean_vol >= 0.20:
            state_labels[i] = "bearish"
        elif mean_vol < 0.12:
            state_labels[i] = "ranging"
        elif mean_vol >= 0.30:
            state_labels[i] = "liquidity_crisis"
        else:
            state_labels[i] = "ranging"

    # Build regime probabilities dict
    regime_names = [
        "trending_up", "ranging", "bearish", "high_vol", "liquidity_crisis",
    ]
    regime_probs = {name: 0.0 for name in regime_names}
    for i in range(HMM_N_STATES):
        label = state_labels[i]
        regime_probs[label] = regime_probs.get(label, 0.0) + float(
            current_probs[i])

    # Renormalize
    total_prob = sum(regime_probs.values())
    if total_prob > 0:
        for k in regime_probs:
            regime_probs[k] /= total_prob

    # Current regime = argmax
    current_regime = max(regime_probs, key=regime_probs.get)

    # Entropy
    p_vals = np.array(list(regime_probs.values()))
    p_vals = np.clip(p_vals, 1e-10, 1.0)
    entropy = float(
        -np.sum(p_vals * np.log(p_vals)) / np.log(len(p_vals)))

    return current_regime, regime_probs, entropy


# ═══════════════════════════════════════════
# Rule-based Regime Fallback
# ═══════════════════════════════════════════


def _rule_based_regime(
    data: Dict[str, np.ndarray],
) -> Tuple[str, Dict[str, float], float]:
    """
    Rule-based market regime detection using simple MA and volatility thresholds.

    Used when HMM fails or data is insufficient.
    """
    prices = data["prices"]
    returns = data["returns"]
    volumes = data.get("volumes")
    volumes_real = bool(data.get("volumes_real", False))
    T = len(prices)

    if T < 60:
        return "ranging", {"ranging": 1.0}, 0.0

    # Moving averages
    ma_20 = np.mean(prices[-20:])
    ma_60 = np.mean(prices[-60:]) if T >= 60 else np.mean(prices)
    current_price = prices[-1]

    # Rolling volatility
    rolling_vol = np.std(returns[-20:]) * math.sqrt(TRADING_DAYS)

    # Volume relative to 20d average
    avg_volume_20 = np.mean(volumes[-20:]) if T >= 20 else np.mean(volumes)
    current_volume = volumes[-1]

    # Scoring
    scores = {
        "trending_up": 0.0,
        "ranging": 0.0,
        "bearish": 0.0,
        "high_vol": 0.0,
        "liquidity_crisis": 0.0,
    }

    # trending_up: 20d MA > 60d MA by > 1%
    if T >= 60 and ma_20 > ma_60 * 1.01:
        scores["trending_up"] += 0.6
        scores["ranging"] += 0.1
    elif T >= 60 and ma_20 < ma_60 * 0.99:
        scores["bearish"] += 0.6
    else:
        scores["ranging"] += 0.5

    # high_vol: 20d rolling vol > 30%
    if rolling_vol > 0.30:
        scores["high_vol"] += 0.5
        scores["bearish"] += 0.2
    elif rolling_vol < 0.12:
        scores["ranging"] += 0.2

    # liquidity_crisis: volume < 20d average * 0.5（仅真实成交量时参与，无量能数据则跳过）
    if volumes is not None and volumes_real and len(volumes) >= 1:
        avg_volume_20 = np.mean(volumes[-20:]) if T >= 20 else np.mean(volumes)
        current_volume = volumes[-1]
        if current_volume < avg_volume_20 * 0.5:
            scores["liquidity_crisis"] += 0.6
        elif current_volume > avg_volume_20 * 1.5:
            scores["trending_up"] += 0.2

    # Normalize
    total = sum(scores.values())
    if total > 0:
        for k in scores:
            scores[k] /= total

    current = max(scores, key=scores.get)

    # Entropy
    p_vals = np.array(list(scores.values()))
    p_vals = np.clip(p_vals, 1e-10, 1.0)
    entropy = float(
        -np.sum(p_vals * np.log(p_vals)) / np.log(len(p_vals)))

    return current, scores, entropy


# ═══════════════════════════════════════════
# Regime Output Helpers
# ═══════════════════════════════════════════


def _compute_key_indicators(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    从真实行情数据提取关键指标。

    只输出可由数据直接计算的条目；无真实成交量时不输出成交额（拒绝代理伪装）。
    """
    returns = data["returns"]
    volumes = data.get("volumes")
    volumes_real = bool(data.get("volumes_real", False))

    indicators: Dict[str, float] = {}
    volatility = float(np.std(returns[-20:]) * math.sqrt(TRADING_DAYS) * 100)
    indicators["20日年化波动率(%)"] = round(volatility, 1)
    indicators["近20日上涨日占比"] = round(float(np.mean(returns[-20:] > 0)), 2)

    if volumes_real and volumes is not None and len(volumes) >= 5:
        turnover = float(np.mean(volumes[-5:]))
        indicators["近5日均成交量"] = round(turnover, 0)

    return indicators


def _find_historical_analog(
    current_regime: str,
    regime_probs: Dict[str, float],
) -> str:
    """Find historical period with most similar regime pattern."""
    analogs = {
        "trending_up": "类似 2020 年 6-7 月牛市主升浪",
        "ranging": "类似 2019 年 6 月震荡区间",
        "bearish": "类似 2022 年 3-4 月下跌趋势",
        "high_vol": "类似 2020 年 3 月疫情冲击高波动",
        "liquidity_crisis": "类似 2015 年 7 月流动性枯竭",
    }
    base = analogs.get(current_regime, "类似 2019 年 6 月震荡区间")

    max_prob = max(regime_probs.values())
    if max_prob < 0.4:
        return f"{base}（状态模糊，多信号叠加）"
    elif max_prob > 0.8:
        return f"{base}（状态清晰）"
    return base


def _generate_strategy_suggestion(
    current_regime: str,
    regime_probs: Dict[str, float],
) -> Tuple[str, float]:
    """
    Generate position sizing and style suggestion based on regime.

    Returns (suggestion_text, suggested_position_0to1).
    """
    suggestions = {
        "trending_up": (
            "趋势向上，建议仓位 85%，进攻风格优先，高配动量/成长因子，低配防御标的",
            0.85,
        ),
        "ranging": (
            "震荡格局，建议仓位 60%，防御风格优先，低配高贝塔标的，关注波段机会",
            0.60,
        ),
        "bearish": (
            "下跌趋势，建议仓位 30%，现金为王，严控回撤，仅保留核心防御仓位",
            0.30,
        ),
        "high_vol": (
            "高波动环境，建议仓位 40%，低波动因子优先，降低杠杆，扩大止损线",
            0.40,
        ),
        "liquidity_crisis": (
            "流动性枯竭，建议仓位 10%，全面防守，仅持有国债/货币基金等价物",
            0.10,
        ),
    }

    text, _ = suggestions.get(current_regime, ("震荡格局，建议仓位 50%", 0.50))

    # Blend position based on probabilities
    pos_map = {
        "trending_up": 0.85, "ranging": 0.60, "bearish": 0.30,
        "high_vol": 0.40, "liquidity_crisis": 0.10,
    }
    weighted_pos = sum(
        regime_probs.get(name, 0.0) * pos_map.get(name, 0.5)
        for name in pos_map
    )

    return text, round(weighted_pos, 2)


# ═══════════════════════════════════════════
# ValidateSignalTool
# ═══════════════════════════════════════════


class ValidateSignalTool(BaseTool):
    """信号验证 (M4: 真实 IC/分组/半衰期/拥挤度/条件有效性)"""

    tool_name = "validate_signal"
    tool_description = "验证因子/信号有效性，返回 IC 时序/ICIR/分组收益/半衰期/拥挤度/条件有效性"
    tool_version = "4.0.0"
    tool_category = "signal"

    def execute(
        self,
        signal_name: str = "",
        signal_values: Optional[List[float]] = None,
        forward_returns: Optional[Dict[int, List[float]]] = None,
        horizons: Optional[List[int]] = None,
    ) -> Trader3Response:
        """
        Validate a signal's predictive power.

        Parameters
        ----------
        signal_name : str — signal / factor name
        signal_values : List[float], optional — flattened signal values (T * N)
        forward_returns : Dict[int, List[float]], optional — horizon -> flattened forward returns
        horizons : List[int], optional — forecast horizons in days

        If signal_values/forward_returns are not provided, generates synthetic
        cross-sectional panel data for demonstration.
        """
        name = signal_name or "未命名因子"

        # Data preparation
        if signal_values is not None and forward_returns is not None:
            # Single horizon for now
            fr_keys = list(forward_returns.keys())
            horizon = (horizons or [1])[0]
            fr = forward_returns.get(horizon, forward_returns[fr_keys[0]])
            n = len(signal_values)
            n_assets = int(math.sqrt(n)) or N_ASSETS
            n_periods = max(n // n_assets, 2)

            sig_2d = np.array(signal_values[:n_periods * n_assets]).reshape(
                n_periods, n_assets)
            fr_2d = np.array(
                (fr if isinstance(fr, list) else fr)[:n_periods * n_assets]
            ).reshape(n_periods, n_assets)
            # 市场收益基准：横截面均值（zeros 会使波动率分组退化）
            mr = np.nanmean(fr_2d, axis=1)
            mr = np.where(np.isfinite(mr), mr, 0.0)
            synthetic_data = False
        else:
            # M8: 尝试真实 qlib 数据构建面板，失败回退合成
            real_panel = self._try_real_signal_panel()
            if real_panel is not None:
                sig_2d, fr_2d, mr = real_panel
                synthetic_data = False
            else:
                sig_2d, fr_2d, mr = _generate_signal_panel()
                synthetic_data = True

        # Compute all metrics
        ic_series = _compute_ic_series(sig_2d, fr_2d)
        ic_mean, ic_std, icir = _compute_ic_stats(ic_series)
        group_returns, monotonicity, long_short, long_only = (
            _compute_group_returns(sig_2d, fr_2d))
        half_life = _compute_half_life(ic_series)
        crowding = _compute_crowding_index(sig_2d)
        cond_valid = _compute_conditional_validity(sig_2d, fr_2d, mr)

        report = SignalValidationReport(
            signal_name=name,
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=icir,
            ic_series=[round(float(v), 4) for v in ic_series],
            group_returns=group_returns,
            monotonicity=monotonicity,
            half_life_periods=half_life,
            half_life_months=round(half_life / 21.0, 2),
            crowding_index=crowding,
            conditional_validity=cond_valid,
            long_short_return=long_short,
            long_only_return=long_only,
        )

        n_periods_actual = len(ic_series)
        synthetic_tag = "（合成数据）" if synthetic_data else "（真实数据）"

        return Trader3Response(
            success=True,
            data=report,
            summary=(
                f"[{name}{synthetic_tag}] ICIR={report.icir:.3f}, "
                f"分组单调性 {report.monotonicity:.0%}, "
                f"半衰期 {report.half_life_periods:.0f}期, "
                f"拥挤度 {report.crowding_index:.2f}, "
                f"多空年化 {report.long_short_return:.1%}"
            ),
            key_metrics={
                "IC均值": report.ic_mean,
                "IC标准差": report.ic_std,
                "ICIR": report.icir,
                "分组单调性": report.monotonicity,
                "半衰期(交易日)": report.half_life_periods,
                "拥挤度": report.crowding_index,
                "多空年化": report.long_short_return,
                "多头年化": report.long_only_return,
            },
            charts=[
                ChartSpec(
                    chart_type="bar",
                    title="分组收益 (Q1-Q5)",
                    data=report.group_returns,
                    y_label="年化收益",
                    description="按信号值排序的 5 分位组合年化收益",
                ),
                ChartSpec(
                    chart_type="line",
                    title=f"IC 时序 ({n_periods_actual}期)",
                    data=report.ic_series,
                    y_label="IC",
                    description="逐期横截面 Spearman 秩相关",
                ),
            ],
            caveats=[
                f"M4 信号验证{' — 合成数据，非真实因子数据' if synthetic_data else ' — 真实 qlib 行情数据'}",
                "ICIR > 0.3 通常视为有效信号门槛",
                "半衰期为自相关衰减法估计（单位：交易日），<42期(约2个月)需警惕快速衰减",
                "拥挤度为信号截面平均|成对相关|的内部代理，非持仓拥挤度",
                "条件有效性按波动率三分组计算",
            ],
        )

    @staticmethod
    def _try_real_signal_panel(n_assets: int = 60, n_periods: int = 60):
        """
        M8: 尝试用真实 qlib 行情构建信号面板。

        信号：20日动量（对数收益），收益：次日收益率。
        横截面面板形状 (n_periods, n_assets)。

        Returns
        -------
        (sig_2d, fr_2d, mr) 或 None（失败回退合成）
        """
        try:
            from trader3.data_provider import QlibDataProvider

            dp = QlibDataProvider()
            # 取 CSI300 前 n_assets 只股票
            codes = dp.instruments("csi300")[:n_assets]
            if len(codes) < 20:
                return None

            # 逐股建立 date->close 映射（仅有效价）
            series = {}
            for code in codes:
                close, dates = dp.load_stock(code.lower(), "close")
                if close is None or len(close) == 0:
                    continue
                dmap = {
                    d: float(v)
                    for d, v in zip(dates, close)
                    if v is not None and np.isfinite(v) and v > 0
                }
                if len(dmap) >= n_periods + 21:
                    series[code] = dmap

            if len(series) < 20:
                return None

            # 公共交易日交集：保证面板每行对应同一交易日（消除跨股错位）
            common = set(next(iter(series.values())).keys())
            for dmap in series.values():
                common &= set(dmap.keys())
            common = sorted(common)
            if len(common) < n_periods + 21:
                return None

            window = common[-(n_periods + 21):]

            sig_2d = np.zeros((n_periods, len(series)), dtype=np.float64)
            fr_2d = np.zeros((n_periods, len(series)), dtype=np.float64)

            for j, (code, dmap) in enumerate(series.items()):
                recent = np.array([dmap[d] for d in window], dtype=np.float64)
                # 动量: log(c_t / c_{t-20})
                mom = np.log(recent[21:] / recent[:-21])
                # 次日收益，与动量对齐：动量在第 t 天，收益为 t→t+1
                ret = recent[1:] / recent[:-1] - 1.0
                fr = ret[21:]
                n = min(len(mom), n_periods)
                sig_2d[:n, j] = mom[-n:]
                fr_2d[:n, j] = fr[-n:]

            # 市场收益（横截面均值）
            mr = np.nanmean(fr_2d, axis=1)
            mr = np.where(np.isfinite(mr), mr, 0.0)

            # 清理 NaN/Inf
            sig_2d = np.nan_to_num(sig_2d, nan=0.0, posinf=0.0, neginf=0.0)
            fr_2d = np.nan_to_num(fr_2d, nan=0.0, posinf=0.0, neginf=0.0)

            return sig_2d, fr_2d, mr
        except Exception:
            return None


# ═══════════════════════════════════════════
# DiagnoseMarketRegimeTool
# ═══════════════════════════════════════════


class DiagnoseMarketRegimeTool(BaseTool):
    """市场状态诊断 (M4: 真实 HMM + 规则回退 + 注意力动量)"""

    tool_name = "diagnose_market_regime"
    tool_description = (
        "诊断市场状态（趋势/震荡/下跌/高波动/流动性枯竭），"
        "返回状态概率 + 关键指标 + 策略建议 + 注意力动量"
    )
    tool_version = "4.0.0"
    tool_category = "signal"

    def execute(
        self,
        lookback: int = 60,
        prices: Optional[List[float]] = None,
        volumes: Optional[List[float]] = None,
    ) -> Trader3Response:
        """
        Diagnose current market regime.

        Parameters
        ----------
        lookback : int — lookback periods for regime detection
        prices : List[float], optional — real price series
        volumes : List[float], optional — real volume series

        If prices/volumes are not provided, generates synthetic data.
        """
        # Data preparation
        if prices is not None and volumes is not None:
            arr_prices = np.array(prices, dtype=np.float64)
            arr_volumes = np.array(volumes, dtype=np.float64)
            arr_returns = np.diff(np.log(arr_prices))
            min_len = min(len(arr_returns), len(arr_volumes) - 1)
            arr_returns = arr_returns[:min_len]
            arr_volumes = arr_volumes[1:min_len + 1]
            arr_prices = arr_prices[-min_len - 1:]
            data = {
                "prices": arr_prices,
                "volumes": arr_volumes,
                "returns": arr_returns,
            }
            synthetic_tag = ""
        else:
            # M8: 尝试真实 qlib 数据（CSI300 指数），失败回退合成
            real_data, real_synthetic = self._try_real_index_data()
            if real_data is not None:
                data = real_data
                synthetic_tag = ""
            else:
                data = _generate_market_regime_data(lookback=lookback)
                synthetic_tag = "（合成数据）"

        # Regime detection
        _hmm_module = _try_hmmlearn()
        hmm_available = _hmm_module is not None

        try:
            current_regime, regime_probs, entropy = _fit_hmm(data)
            detection_method = (
                "HMM (hmmlearn)" if hmm_available else "HMM (custom EM)")
        except Exception:
            current_regime, regime_probs, entropy = _rule_based_regime(data)
            detection_method = "规则回退 (rule-based)"

        # Key indicators
        key_indicators = _compute_key_indicators(data)

        # Historical analog
        historical_analog = _find_historical_analog(
            current_regime, regime_probs)

        # Strategy suggestion
        strategy_suggestion, suggested_position = _generate_strategy_suggestion(
            current_regime, regime_probs)

        diagnosis = RegimeDiagnosis(
            current_regime=current_regime,
            regime_probabilities=regime_probs,
            regime_entropy=entropy,
            key_indicators=key_indicators,
            historical_analog=historical_analog,
            strategy_suggestion=strategy_suggestion,
            suggested_position=suggested_position,
        )

        # `synthetic_tag` 已在数据准备阶段确定
        return Trader3Response(
            success=True,
            data=diagnosis,
            summary=(
                f"当前状态: {diagnosis.current_regime}{synthetic_tag} "
                f"(P={diagnosis.regime_probabilities.get(diagnosis.current_regime, 0):.0%}), "
                f"熵={diagnosis.regime_entropy:.2f}, "
                f"建议仓位 {diagnosis.suggested_position:.0%}, "
                f"检测方法: {detection_method}"
            ),
            key_metrics={
                "当前状态": diagnosis.current_regime,
                "最大概率": max(diagnosis.regime_probabilities.values()),
                "状态熵": diagnosis.regime_entropy,
                "建议仓位": diagnosis.suggested_position,
                **diagnosis.key_indicators,
            },
            charts=[
                ChartSpec(
                    chart_type="bar",
                    title="状态概率分布",
                    data=dict(
                        sorted(
                            diagnosis.regime_probabilities.items(),
                            key=lambda x: -x[1],
                        )
                    ),
                    description=(
                        f"HMM 识别的市场状态概率（{detection_method}）"),
                ),
            ],
            caveats=[
                f"M4 市场状态诊断{' — 合成数据' if synthetic_tag else ' — 真实 qlib 指数数据'}",
                f"检测方法: {detection_method}",
                "HMM 特征: 日收益 / 20日波动率" + (" / 量能变化(真实成交量)" if data.get("volumes_real") else "（无真实量能数据，量能维度未参与）"),
                "历史类比仅供参考，不代表未来重演",
                "以下指标因无真实数据源已移除：市盈率、融资余额、注意力动量（接入对应数据源后恢复）",
            ],
        )

    @staticmethod
    def _try_real_index_data():
        """
        M8: 尝试读取真实 qlib 指数数据（CSI300）。

        Returns
        -------
        (data, synthetic) — data dict 或 (None, True) 表示回退合成
        """
        try:
            from trader3.data_provider import QlibDataProvider

            dp = QlibDataProvider()
            # 读取 CSI300 指数收盘价
            close, dates = dp.load_stock("SH000300", "close")
            if len(close) < 100:
                return None, True

            # 过滤有效值（指数没有停牌，0 = 缺失）
            valid = close > 0
            idx = np.where(valid)[0]
            if len(idx) < 100:
                return None, True

            prices = close[idx]
            dates_valid = [dates[i] for i in idx]

            # 优先真实成交量字段；不可用时明确标注 volumes_real=False（不伪装）
            volumes_real = False
            try:
                vol, vol_dates = dp.load_stock("SH000300", "volume")
                vmap = {d: float(v) for d, v in zip(vol_dates, vol)
                        if v is not None and np.isfinite(v) and v > 0}
                aligned = [vmap.get(d) for d in dates_valid]
                if sum(v is not None for v in aligned[-100:]) >= 95:
                    # 尾段基本齐备：缺失处用前后均值填补
                    filled = []
                    last = next((v for v in aligned if v is not None), 1.0)
                    for v in aligned:
                        if v is None:
                            v = last
                        filled.append(v)
                        last = v
                    volumes = np.array(filled[1:], dtype=np.float64)
                    volumes_real = True
                else:
                    volumes = None
            except Exception:
                volumes = None

            returns = np.diff(np.log(prices))
            if volumes is None:
                # 无真实量能数据 → 不提供成交量特征（由调用方跳过量能维度）
                n = len(returns)
                data = {
                    "prices": prices[-n - 1:],
                    "volumes": None,
                    "returns": returns,
                    "volumes_real": False,
                }
                return data, False

            n = min(len(prices) - 1, len(volumes))
            data = {
                "prices": prices[-n - 1:],
                "volumes": volumes[-n:],
                "returns": returns[-n:],
                "volumes_real": volumes_real,
            }
            return data, False
        except Exception:
            return None, True


# ═══════════════════════════════════════════
# Module-level test helper
# ═══════════════════════════════════════════

if __name__ == "__main__":
    t = ValidateSignalTool()
    r = t(signal_name="动量因子_test")
    assert r.success, f"ValidateSignalTool failed: {r.summary}"
    print(f"[ValidateSignalTool] {r.summary}")
    for k, v in r.key_metrics.items():
        print(f"  {k}: {v}")

    t2 = DiagnoseMarketRegimeTool()
    r2 = t2()
    assert r2.success, f"DiagnoseMarketRegimeTool failed: {r2.summary}"
    print(f"[DiagnoseMarketRegimeTool] {r2.summary}")
    print(f"  Indicators: {r2.data.key_indicators}")
    print("M4 smoke test PASSED")
