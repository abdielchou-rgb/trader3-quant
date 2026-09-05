"""
Barra风格多因子风险模型。

包含：
- 风格因子暴露：Size / Value / Momentum / Volatility / Beta / Liquidity
- 行业因子暴露（one-hot）
- 因子协方差矩阵（Newey-West 调整可选）
- 特异性收益与特异性风险
- 组合风险分解：系统性 vs 特异性，因子贡献
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STYLE_FACTORS = ["size", "value", "momentum", "volatility", "beta", "liquidity"]


@dataclass
class RiskModelConfig:
    """风险模型配置。"""
    half_life: int = 66              # EWMA 半衰期（交易日）
    newey_west_lags: int = 4         # Newey-West 调整滞后阶数
    winsorize_z: float = 5.0         # 暴露去极化 z 分数阈值
    standardize: bool = True         # 暴露标准化
    regress_lw_shrink: float = 0.1   # Ledoit-Wolf 收缩强度（对特定协方差）


@dataclass
class RiskDecomposition:
    """组合风险分解结果。"""
    total_risk: float                      # 年化总风险（波动率）
    systematic_risk: float                 # 系统性风险
    specific_risk: float                   # 特异性风险
    factor_contributions: dict[str, float] # 各因子风险贡献（方差占比）
    factor_exposure: pd.Series             # 组合加权因子暴露
    var_decomposition: pd.Series           # 各因子方差贡献绝对值


class FactorRiskModel:
    """
    Barra风格横截面回归风险模型：

        r_i = sum_k X_ik * f_k + u_i

    其中 X 为暴露矩阵，f 为因子收益，u 为特异性收益。
    因子收益通过 WLS（权重 = 1/特异性方差 的倒数近似）逐期估计，
    因子协方差用 EWMA + Newey-West 调整。
    """

    def __init__(self, config: RiskModelConfig | None = None):
        self.config = config or RiskModelConfig()
        self.exposures_: pd.DataFrame | None = None       # T x N x K -> 存最近一期 (N x K)
        self.factor_returns_: pd.DataFrame | None = None  # T x K
        self.specific_returns_: pd.DataFrame | None = None
        self.factor_cov_: pd.DataFrame | None = None      # K x K
        self.specific_vol_: pd.Series | None = None       # N
        self._fitted = False

    # ---------- 暴露构建 ----------

    @staticmethod
    def build_style_exposures(panel: pd.DataFrame) -> pd.DataFrame:
        """
        从股票面板数据构建风格暴露。

        panel 为宽表：index=date, columns=MultiIndex(asset 或 field)
        自动探测字段所在层级（"close"/"volume"）。
        返回: index=asset, columns=STYLE_FACTORS
        """
        if not isinstance(panel.columns, pd.MultiIndex):
            raise ValueError("panel 必须为 MultiIndex 列宽表")
        levels = [panel.columns.get_level_values(i) for i in range(panel.columns.nlevels)]
        field_lvl = next((i for i, lv in enumerate(levels) if "close" in set(lv)), None)
        if field_lvl is None:
            raise KeyError("panel 列中找不到 'close' 字段")

        close = panel.xs("close", axis=1, level=field_lvl)
        vol_lvl = next((i for i, lv in enumerate(levels) if "volume" in set(lv)), field_lvl)
        volume = panel.xs("volume", axis=1, level=vol_lvl)

        ret = close.pct_change()
        n_days = min(252, len(close))

        exp = pd.DataFrame(index=close.columns)

        # Size: log 市值代理（价格 × 均量）
        mcap_proxy = (close * volume.rolling(20).mean()).iloc[-1]
        exp["size"] = np.log(mcap_proxy.clip(lower=1e-8))

        # Value: 反转代理（长期跌幅大 → 便宜）。真实模型用 BP/EP。
        exp["value"] = -(close.iloc[-1] / close.iloc[-n_days] - 1)

        # Momentum: 过去12-1月收益
        if len(close) > 252:
            exp["momentum"] = close.iloc[-22] / close.iloc[-252] - 1
        else:
            exp["momentum"] = close.iloc[-1] / close.iloc[0] - 1

        # Volatility: 近60日已实现波动
        exp["volatility"] = ret.tail(60).std()

        # Beta: 对等权市场收益的回归 beta
        mkt = ret.mean(axis=1)
        tail_ret, tail_mkt = ret.tail(120), mkt.tail(120)
        cov = tail_ret.apply(lambda s: s.cov(tail_mkt))
        var_m = tail_mkt.var()
        exp["beta"] = cov / max(var_m, 1e-12)

        # Liquidity: log 成交额均值
        turnover = (close * volume)
        exp["liquidity"] = np.log(turnover.rolling(20).mean().iloc[-1].clip(lower=1e-8))

        return exp.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def build_industry_exposures(industries: pd.Series) -> pd.DataFrame:
        """行业 one-hot 暴露。industries: asset -> industry name。"""
        return pd.get_dummies(industries, dtype=float)

    def _winsorize_standardize(self, exposures: pd.DataFrame) -> pd.DataFrame:
        z = self.config.winsorize_z
        out = exposures.copy()
        med = out.median()
        mad = (out - med).abs().median()
        # MAD-based robust winsorize（numpy 广播，避免 pandas clip 的 axis 歧义）
        scale = (mad * 1.4826).replace(0, np.nan)
        lower = (med - z * scale).values[None, :]
        upper = (med + z * scale).values[None, :]
        arr = np.clip(out.values, lower, upper)
        out = pd.DataFrame(arr, index=out.index, columns=out.columns)
        if self.config.standardize:
            std = out.std(ddof=0).replace(0, np.nan)
            out = (out - out.mean()) / std
        return out.fillna(0.0)

    # ---------- 因子收益估计 ----------

    def fit_factor_returns(self, returns: pd.DataFrame, exposures: pd.DataFrame,
                           weights: pd.Series | None = None) -> pd.DataFrame:
        """
        逐期横截面回归估计因子收益。

        returns:   index=date, columns=asset
        exposures: index=asset, columns=factor（单期暴露；也可传 dict[date->exp]）
        weights:   回归权重（asset），默认等权
        """
        common_assets = returns.columns.intersection(exposures.index)
        X = self._winsorize_standardize(exposures.loc[common_assets])
        if weights is not None:
            w = weights.reindex(common_assets).fillna(1.0)
            sw = np.sqrt(w.values)
            Xw = X.mul(sw, axis=0)
        else:
            Xw = X

        fr_rows, spec_rows = {}, {}
        w_sqrt = sw if weights is not None else None
        for date, r in returns.iterrows():
            y = r.reindex(common_assets).values.astype(float)
            mask = ~np.isnan(y)
            X_masked = Xw.values[mask]
            y_masked = y[mask]
            if len(y_masked) < X.shape[1] + 5:
                continue
            if weights is not None:
                w_masked = w_sqrt[mask]
                XtX = (X_masked * w_masked[:, None]).T @ X_masked
                Xty = X_masked.T @ (w_masked * y_masked)
            else:
                XtX = X_masked.T @ X_masked
                Xty = X_masked.T @ y_masked
            try:
                beta = np.linalg.solve(XtX + 1e-8 * np.eye(X.shape[1]), Xty)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(XtX) @ Xty
            resid = y_masked - X_masked @ beta
            fr_rows[date] = dict(zip(X.columns, beta, strict=False))
            spec_rows[date] = dict(zip(common_assets[mask], resid, strict=False))

        self.factor_returns_ = pd.DataFrame(fr_rows).T.sort_index()
        self.specific_returns_ = pd.DataFrame(spec_rows).T.sort_index()
        self.exposures_ = X
        self._fitted = True
        return self.factor_returns_

    # ---------- 协方差估计 ----------

    def fit_factor_covariance(self) -> pd.DataFrame:
        """EWMA + Newey-West 调整的因子协方差矩阵。"""
        if not self._fitted or self.factor_returns_ is None or len(self.factor_returns_) < 10:
            raise RuntimeError("先调用 fit_factor_returns")
        fr = self.factor_returns_.dropna(how="all")
        lam = 0.5 ** (1.0 / self.config.half_life)
        t = len(fr)
        w = np.array([lam ** (t - 1 - i) for i in range(t)])
        w /= w.sum()

        mu = (fr.values * w[:, None]).sum(axis=0)
        d = fr.values - mu
        cov = (d * w[:, None]).T @ d

        # Newey-West：加入滞后自协方差
        for lag in range(1, self.config.newey_west_lags + 1):
            gamma = np.zeros_like(cov)
            for i in range(t - lag):
                gamma += w[i + lag] * np.outer(d[i], d[i + lag])
            gamma /= w.sum()
            cov += (1 - lag / (self.config.newey_west_lags + 1)) * (gamma + gamma.T)

        # PSD 投影
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-12, None)
        cov_psd = vecs @ np.diag(vals) @ vecs.T

        k = list(fr.columns)
        self.factor_cov_ = pd.DataFrame((cov_psd + cov_psd.T) / 2, index=k, columns=k)

        # 特异性波动
        if self.specific_returns_ is not None and len(self.specific_returns_) > 5:
            sr = self.specific_returns_.dropna(how="all").fillna(0)
            self.specific_vol_ = sr.std(ddof=0) * np.sqrt(252)
        return self.factor_cov_

    # ---------- 组合风险分解 ----------

    def decompose_portfolio(self, holdings: pd.Series) -> RiskDecomposition:
        """
        holdings: asset -> 权重（和为1的多头组合）

        sigma_p^2 = h' X F X' h + D
        """
        if not self._fitted or self.factor_cov_ is None:
            raise RuntimeError("模型未拟合完成")
        h = holdings.reindex(self.exposures_.index).fillna(0.0)
        h = h / h.sum()
        X = self.exposures_.loc[h.index]
        port_exp = X.T @ h

        F = self.factor_cov_.reindex(index=X.columns, columns=X.columns).fillna(0)
        sys_var = float(port_exp.values @ F.values @ port_exp.values)

        sv = self.specific_vol_.reindex(h.index).fillna(0) if self.specific_vol_ is not None else pd.Series(0.0, index=h.index)
        spec_var = float(((h.values ** 2) * (sv.values / np.sqrt(252)) ** 2).sum())

        total_var = sys_var + spec_var
        total_risk = np.sqrt(max(total_var, 0)) * np.sqrt(252)

        # 各因子方差贡献
        marginal = F.values @ port_exp.values          # ∂σ²/∂f
        contrib_var = port_exp.values * marginal       # f_k * (F f)_k
        contrib = pd.Series(contrib_var, index=F.index)
        contrib[contrib > 0].sum()
        factor_pct = (contrib / total_var).fillna(0) if total_var > 0 else contrib * 0

        return RiskDecomposition(
            total_risk=total_risk,
            systematic_risk=np.sqrt(max(sys_var, 0)) * np.sqrt(252),
            specific_risk=np.sqrt(max(spec_var, 0)) * np.sqrt(252),
            factor_contributions=factor_pct.to_dict(),
            factor_exposure=port_exp,
            var_decomposition=contrib,
        )

    # ---------- 宏观暴露 ----------
    @staticmethod
    def macro_exposures(macro: pd.DataFrame) -> pd.DataFrame:
        """宏观因子暴露：macro 为 (date×macro) 或 (asset×macro) 宽表。

        返回 index=asset 的暴露（单期取末行 / 直接透传）。
        """
        if isinstance(macro, pd.Series):
            macro = macro.to_frame().T
        if not isinstance(macro.columns, pd.MultiIndex) and len(macro) > 1 and "date" not in str(macro.index.dtype):
            # 时序表 → 取末行截面暴露
            return macro.iloc[-1].to_frame().T
        return macro

    # ---------- 尾部风险 ----------
    def var_cvar(self, holdings: pd.Series, returns: pd.DataFrame,
                 alpha: float = 0.95, method: str = "historical") -> tuple[float, float]:
        """组合 VaR / CVaR（单期收益口径，返回正数表示损失幅度）。

        returns: index=date, columns=asset。
        """
        h = holdings.reindex(returns.columns).fillna(0.0)
        if h.sum() == 0:
            return 0.0, 0.0
        port_ret = returns.reindex(columns=h.index).fillna(0.0).values @ h.values
        if method == "parametric":
            mu = float(port_ret.mean())
            sd = float(port_ret.std(ddof=0)) + 1e-12
            z = _norm_ppf_safe(1.0 - alpha)
            var = max(0.0, -(mu - z * sd))
            # CVaR 近似（正态尾部期望）
            cvar = max(0.0, -(mu - sd * _norm_pdf_safe(z)))
            return var, cvar
        # 历史法
        q = np.percentile(port_ret, (1.0 - alpha) * 100.0)
        tail = port_ret[port_ret <= q]
        cvar = float(tail.mean()) if len(tail) else q
        return max(0.0, -q), max(0.0, -cvar)

    def tracking_error(self, holdings: pd.Series, benchmark: pd.Series,
                        cov: pd.DataFrame | None = None,
                        returns: pd.DataFrame | None = None) -> float:
        """主动收益年化跟踪误差。"""
        h = holdings.reindex(benchmark.index).fillna(0.0)
        act = h - benchmark.reindex(h.index).fillna(0.0)
        if cov is not None:
            c = cov.reindex(index=h.index, columns=h.index).fillna(0.0).values
            return float(np.sqrt(max(act.values @ c @ act.values, 0.0)) * np.sqrt(252))
        if returns is not None:
            pr = returns.reindex(columns=h.index).fillna(0.0).values @ h.values
            br = returns.reindex(columns=benchmark.index).fillna(0.0).values @ benchmark.reindex(returns.columns).fillna(0.0).values
            return float(np.std(pr - br, ddof=0) * np.sqrt(252))
        return 0.0


def _norm_ppf_safe(p: float) -> float:
    try:
        from scipy.stats import norm

        return float(norm.ppf(p))
    except Exception:  # pragma: no cover
        import math

        if p <= 0.0:
            return float("-inf")
        if p >= 1.0:
            return float("inf")
        return math.sqrt(2.0) * __import__("statistics").NormalDist().inv_cdf(p)


def _norm_pdf_safe(x: float) -> float:
    import math

    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def neutralize_beta(weights: pd.Series, exposures: pd.DataFrame,
                    target: float = 0.0) -> pd.Series:
    """β 中性化：投影权重使组合 beta = target（最小改动）。"""
    if "beta" not in exposures.columns:
        return weights
    b = exposures["beta"].reindex(weights.index).fillna(0.0).values
    w = weights.reindex(exposures.index).fillna(0.0).values
    bb = float(b @ b)
    if bb <= 1e-12:
        return weights
    cur = float(b @ w)
    w = w - (cur - target) / bb * b
    s = w.sum()
    if s != 0:
        w = w / s
    return pd.Series(w, index=exposures.index).reindex(weights.index).fillna(0.0)


def cap_industry(weights: pd.Series, industries: dict[str, str] | pd.Series,
                 max_w: float) -> pd.Series:
    """行业权重上限：单行业合计权重不超过 max_w（超限按比例缩放）。"""
    if not industries or max_w >= 1.0:
        return weights
    w = weights.copy()
    groups: dict[str, list[str]] = {}
    if isinstance(industries, pd.Series):
        for ind, mem in industries.groupby(level=0).groups.items():
            groups[ind] = list(mem)
    else:
        for a, ind in industries.items():
            groups.setdefault(ind, []).append(a)
    for mem in groups.values():
        mem = [m for m in mem if m in w.index]
        if not mem:
            continue
        s = float(w[mem].sum())
        if s > max_w:
            w[mem] = w[mem] * (max_w / s)
    return w


def apply_risk_constraints(
    weights: pd.Series,
    exposures: pd.DataFrame | None = None,
    cov: pd.DataFrame | None = None,
    industries: dict[str, str] | None = None,
    beta_neutral: bool = False,
    industry_max_weight: float = 1.0,
    tracking_error_max: float = 0.0,
    benchmark: pd.Series | None = None,
) -> pd.Series:
    """组合风险约束：β 中性 + 行业上限 + 跟踪误差上限（迭代投影）。"""
    w = weights.copy()
    gross = float(w.sum())
    if gross <= 0:
        return w
    if beta_neutral and exposures is not None:
        w = neutralize_beta(w, exposures, target=0.0)
    if industries is not None and industry_max_weight < 1.0:
        w = cap_industry(w, industries, industry_max_weight)
    if tracking_error_max > 0 and benchmark is not None and (cov is not None or True):
        te = 0.0
        if cov is not None:
            te = _te(w, benchmark, cov)
        if te > tracking_error_max > 0:
            scale = tracking_error_max / te
            b = benchmark.reindex(w.index).fillna(0.0)
            w = b + (w - b) * min(scale, 1.0)
    s = w.sum()
    if s != 0:
        w = w / s * gross
    return w


def _te(w: pd.Series, benchmark: pd.Series, cov: pd.DataFrame) -> float:
    h = w.reindex(benchmark.index).fillna(0.0)
    act = h - benchmark.reindex(h.index).fillna(0.0)
    c = cov.reindex(index=h.index, columns=h.index).fillna(0.0).values
    return float(np.sqrt(max(act.values @ c @ act.values, 0.0)) * np.sqrt(252))
