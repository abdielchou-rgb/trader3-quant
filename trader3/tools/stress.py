"""
压力测试与尾部风险指标。

包含：
- VaR / CVaR：历史模拟、参数法（正态/Cornish-Fisher）、蒙特卡洛
- 历史情景回放：2008、2015股灾、2018贸易战、2020疫情等（相对收益冲击）
- 假想情景：利率+200bp、波动率×2、流动性骤降等
- 尾部指标：最大回撤、尾部依赖、偏度/峰度、Calmar、Sortino
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TailMetrics:
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    skewness: float
    kurtosis: float
    sortino: float
    calmar: float
    worst_day: float
    best_day: float


@dataclass
class ScenarioResult:
    name: str
    portfolio_pnl_pct: float        # 情景下组合损益（%）
    stressed_var: float             # 冲击后 99% VaR（日）
    factor_shocks: dict[str, float] = field(default_factory=dict)


# ---------- 基础尾部指标 ----------

def compute_tail_metrics(returns: pd.Series, risk_free: float = 0.0) -> TailMetrics:
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 20:
        raise ValueError(f"收益样本过少: {len(r)} < 20")

    equity = (1 + r).cumprod()
    peak = equity.cummax()
    max_dd = float((equity / peak - 1).min())

    downside = r[r < 0]
    downside_std = downside.std(ddof=0)
    ann_ret = (equity.iloc[-1]) ** (252 / len(r)) - 1
    ann_dd_std = downside_std * np.sqrt(252) if downside_std > 0 else np.nan

    from scipy import stats
    return TailMetrics(
        var_95=float(-np.quantile(r, 0.05)),
        var_99=float(-np.quantile(r, 0.01)),
        cvar_95=float(-r[r <= np.quantile(r, 0.05)].mean()),
        cvar_99=float(-r[r <= np.quantile(r, 0.01)].mean()) if (r <= np.quantile(r, 0.01)).any() else float(-r.min()),
        max_drawdown=max_dd,
        skewness=float(stats.skew(r)),
        kurtosis=float(stats.kurtosis(r)),  # 超额峰度
        sortino=float((ann_ret - risk_free) / ann_dd_std) if ann_dd_std and not np.isnan(ann_dd_std) and ann_dd_std > 0 else 0.0,
        calmar=float(ann_ret / abs(max_dd)) if max_dd < 0 else 0.0,
        worst_day=float(r.min()),
        best_day=float(r.max()),
    )


# ---------- VaR 家族 ----------

def historical_var(returns: pd.Series, confidence: float = 0.99) -> float:
    r = pd.Series(returns).dropna()
    return float(-np.quantile(r, 1 - confidence))


def parametric_var(returns: pd.Series, confidence: float = 0.99,
                   method: str = "normal") -> tuple[float, float]:
    """返回正的损失量级 (VaR, CVaR)。method: normal | cornish_fisher。"""
    from scipy import stats
    r = pd.Series(returns).dropna()
    mu, sigma = float(r.mean()), float(r.std(ddof=0))
    z = stats.norm.ppf(confidence)

    if method == "cornish_fisher":
        s, k = float(stats.skew(r)), float(stats.kurtosis(r))
        # Cornish-Fisher 修正分位数
        z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 \
            - (2 * z**3 - 5 * z) * s**2 / 36
        z_cf = np.clip(z_cf, -6.0, 6.0)  # 防 CF 展开爆炸
        var = z_cf * sigma - mu
        pdf = stats.norm.pdf(z_cf)
        adj = max(1 + (z_cf**2 - 1) * k / 24, 0.05)
        cvar = sigma * pdf / (1 - confidence) * adj - mu
        return float(var), float(cvar)

    pdf = stats.norm.pdf(z)
    var = z * sigma - mu
    cvar = sigma * pdf / (1 - confidence) - mu
    return float(var), float(cvar)


def monte_carlo_var(returns: pd.Series, confidence: float = 0.99,
                    n_sims: int = 100_000, horizon_days: int = 1,
                    seed: int = 42) -> tuple[float, float]:
    """Bootstrap 蒙特卡洛 VaR/CVaR（保留经验分布的胖尾特征）。"""
    rng = np.random.default_rng(seed)
    r = pd.Series(returns).dropna().values
    sims = rng.choice(r, size=(n_sims, horizon_days), replace=True).sum(axis=1)
    var = float(-np.quantile(sims, 1 - confidence))
    tail = sims[sims <= np.quantile(sims, 1 - confidence)]
    cvar = float(-tail.mean()) if len(tail) else var
    return var, cvar


# ---------- 情景压力测试 ----------

HISTORICAL_SCENARIOS: dict[str, dict[str, float]] = {
    "2008金融危机": {"shock": -0.35, "vol_multiplier": 2.5},
    "2015A股股灾": {"shock": -0.45, "vol_multiplier": 2.8},
    "2018贸易战":   {"shock": -0.25, "vol_multiplier": 1.8},
    "2020疫情":     {"shock": -0.30, "vol_multiplier": 2.2},
    "2022美联储加息": {"shock": -0.22, "vol_multiplier": 1.5},
}

HYPOTHETICAL_SCENARIOS: dict[str, dict[str, float]] = {
    "利率+200bp":      {"shock": -0.12, "vol_multiplier": 1.4},
    "波动率翻倍":       {"shock": -0.08, "vol_multiplier": 2.0},
    "流动性骤降":       {"shock": -0.15, "vol_multiplier": 1.8},
    "极端黑天鹅-25%":  {"shock": -0.25, "vol_multiplier": 2.5},
}


def apply_scenario(portfolio_returns: pd.Series, shock: float,
                   vol_multiplier: float, horizon_days: int = 20,
                   seed: int | None = None) -> ScenarioResult:
    """
    将冲击按 horizon 分摊到组合收益上，并放大波动率，
    返回情景 PnL 与冲击后 VaR。仅使用 horizon_days 窗口（不足则全部）。
    """
    r = pd.Series(portfolio_returns).dropna()
    if len(r) > horizon_days:
        if seed is not None:
            rng = np.random.default_rng(seed)
            start = int(rng.integers(0, len(r) - horizon_days))
        else:
            start = 0
        window = r.values[start:start + horizon_days]
    else:
        window = r.values
    daily_shock = (1 + shock) ** (1 / horizon_days) - 1
    scaled = daily_shock + window * vol_multiplier
    pnl = float(np.prod(1 + scaled) - 1)
    stressed_var = float(-np.quantile(scaled, 0.01))
    return ScenarioResult(
        name=f"shock={shock:+.0%}, vol×{vol_multiplier}",
        portfolio_pnl_pct=pnl,
        stressed_var=stressed_var,
    )


def run_stress_suite(portfolio_returns: pd.Series,
                     include_historical: bool = True,
                     include_hypothetical: bool = True) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    scenarios: dict[str, dict[str, float]] = {}
    if include_historical:
        scenarios.update(HISTORICAL_SCENARIOS)
    if include_hypothetical:
        scenarios.update(HYPOTHETICAL_SCENARIOS)
    for name, spec in scenarios.items():
        sr = apply_scenario(portfolio_returns, spec["shock"], spec["vol_multiplier"])
        sr.name = name
        results.append(sr)
    results.sort(key=lambda x: x.portfolio_pnl_pct)
    return results


# ---------- 因子层面压力（暴露 × 因子冲击）----------

def factor_stress(exposures: pd.Series, factor_cov: pd.DataFrame,
                  factor_shocks_z: dict[str, float] = None,
                  n_mc: int = 50_000, seed: int = 7) -> dict[str, float]:
    """
    给定组合因子暴露与因子协方差，蒙特卡洛联合冲击下的组合损失分布。
    factor_shocks_z: {factor: z-shock}，未指定因子的用随机抽样。
    返回 loss 分布的分位数摘要。
    """
    rng = np.random.default_rng(seed)
    f = list(factor_cov.index)
    mu = np.array([factor_shocks_z.get(k, 0.0) for k in f])
    cov = factor_cov.loc[f, f].values
    try:
        L = np.linalg.cholesky(cov + 1e-10 * np.eye(len(f)))
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-10, None)
        L = vecs @ np.diag(np.sqrt(vals))

    x = rng.standard_normal((n_mc, len(f))) @ L.T + mu
    port = x @ exposures.reindex(f).fillna(0).values
    losses = -port  # 正值=亏损
    return {
        "mean_loss": float(losses.mean()),
        "p95_loss": float(np.quantile(losses, 0.95)),
        "p99_loss": float(np.quantile(losses, 0.99)),
        "max_loss": float(losses.max()),
    }
