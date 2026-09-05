"""
组合构建（Portfolio Construction）。

把因子得分 / ensemble 输出 → 目标权重（pd.Series，截面）。

方法（method）：
- equal_weight : 等权（含正得分资产）
- ic_weighted  : 权重 ∝ 得分（多头，clip 负值后归一）—— 默认
- hrp          : 层次风险平价（需 returns 矩阵；取自 trader3.portfolio.hrp）
- risk_budget  : 风险平价（scipy-SLSQP，取 OptimizePortfolioTool）
- mean_variance: 均值-方差（取 OptimizePortfolioTool）
- regime_aware : 状态路由分配（取 RegimeAwareAllocationTool）

风控覆盖层（risk_overlay.OverlayResult）：
- halt_new_buys=True  → 返回空权重（不开新仓）
- 否则 gross 暴露 = size_multiplier（留现金）

外部协方差（risk_model.cov / returns）优先于工具内部合成估计。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("trader3.v2.portfolio")

VALID_METHODS = (
    "equal_weight", "ic_weighted", "hrp",
    "risk_budget", "mean_variance", "regime_aware",
)


@dataclass
class PortfolioConfig:
    method: str = "ic_weighted"
    top_n: int = 0                     # >0 时仅取绝对得分最高 n 只（ic_weighted/equal_weight 生效）
    long_only: bool = True
    max_single: float = 0.25          # 单票上限（mean_variance/risk_budget 生效）
    industry_neutral: bool = False
    gross_cap: float = 1.0            # 目标总敞口上限（与 overlay.size_multiplier 取小）
    beta_neutral: bool = False         # β 中性（需 exposures 含 beta 列）
    industry_max_weight: float = 1.0   # 单行业合计权重上限（<1 时启用，需 industries）
    tracking_error_max: float = 0.0    # 年化跟踪误差上限（>0 时启用，需 cov + benchmark）
    lot_sizes: dict[str, int] | None = None  # 仅用于下单阶段，这里仅透传


def _latest_cross_section(scores: pd.Series | pd.DataFrame) -> pd.Series:
    """从 (date×asset) 宽表或 (date,asset) 序列取最新一期截面得分。"""
    if isinstance(scores, pd.DataFrame):
        last = scores.iloc[-1]
        return last.dropna()
    if isinstance(scores.index, pd.MultiIndex):
        # 取最大日期那一期
        dates = scores.index.get_level_values(0)
        last_date = dates.max()
        sub = scores.xs(last_date, level=0)
        return sub.dropna()
    return scores.dropna()


class PortfolioConstruction:
    def __init__(self, config: PortfolioConfig | None = None):
        self.config = config or PortfolioConfig()

    def construct(
        self,
        scores: pd.Series | pd.DataFrame,
        *,
        cov: pd.DataFrame | np.ndarray | None = None,
        returns: pd.DataFrame | np.ndarray | None = None,
        overlay=None,
        industries: dict[str, str] | None = None,
        regime_probs: dict[str, float] | None = None,
        regime_weights: dict[str, dict] | None = None,
        exposures: pd.DataFrame | None = None,
        benchmark: pd.Series | None = None,
    ) -> pd.Series:
        """
        返回目标权重 pd.Series（index=asset，和为 gross_cap×size_multiplier ≤ 1）。
        """
        cfg = self.config
        method = cfg.method
        if method not in VALID_METHODS:
            raise ValueError(f"未知方法 {method}，可选 {VALID_METHODS}")

        # 风控熔断：停止开新仓
        if overlay is not None and getattr(overlay, "halt_new_buys", False):
            logger.warning("[portfolio] 覆盖层熔断：返回空权重")
            return pd.Series(dtype=float)

        cs = _latest_cross_section(scores)
        if cs.empty:
            return pd.Series(dtype=float)
        assets = list(cs.index)

        if method == "equal_weight":
            w = self._equal_weight(cs)
        elif method == "ic_weighted":
            w = self._ic_weighted(cs)
        elif method == "hrp":
            w = self._hrp(returns, assets)
        elif method == "risk_budget":
            w = self._optimize_tool(cs, "risk_budget", cov, industries)
        elif method == "mean_variance":
            w = self._optimize_tool(cs, "mean_variance", cov, industries)
        elif method == "regime_aware":
            w = self._regime_aware(cs, regime_probs, regime_weights, industries)
        else:  # pragma: no cover
            w = self._ic_weighted(cs)

        if w is None or len(w) == 0:
            return pd.Series(dtype=float)

        # 单票上限
        if cfg.long_only:
            w = w.clip(lower=0.0)
        w = w.clip(upper=cfg.max_single)
        total = float(w.sum())
        if total <= 0:
            return pd.Series(dtype=float)
        w = w / total

        # 风险约束：β 中性 / 行业上限 / 跟踪误差上限
        if cfg.beta_neutral or (cfg.industry_max_weight < 1.0) or cfg.tracking_error_max > 0:
            from trader3.v2.risk_model import apply_risk_constraints

            cov_df = cov if isinstance(cov, pd.DataFrame) else None
            w = apply_risk_constraints(
                w, exposures=exposures, cov=cov_df, industries=industries,
                beta_neutral=cfg.beta_neutral,
                industry_max_weight=cfg.industry_max_weight,
                tracking_error_max=cfg.tracking_error_max, benchmark=benchmark,
            )

        # 总敞口上限：gross_cap 与 overlay.size_multiplier 取小
        gross = cfg.gross_cap
        if overlay is not None:
            gross = min(gross, getattr(overlay, "size_multiplier", 1.0))
        gross = min(max(gross, 0.0), 1.0)
        w = w * gross
        return w

    # ── 各方法实现 ────────────────────────────────

    def _equal_weight(self, cs: pd.Series) -> pd.Series:
        sel = cs if self.config.top_n <= 0 else cs.reindex(
            cs.abs().sort_values(ascending=False).index[: self.config.top_n])
        n = len(sel)
        if n == 0:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / n, index=sel.index)

    def _ic_weighted(self, cs: pd.Series) -> pd.Series:
        sel = cs if self.config.top_n <= 0 else cs.reindex(
            cs.abs().sort_values(ascending=False).index[: self.config.top_n])
        x = sel.clip(lower=0.0) if self.config.long_only else sel.copy()
        tot = float(x.abs().sum())
        if tot <= 0:
            return self._equal_weight(cs)
        return x / tot

    def _hrp(self, returns, assets: list[str]) -> pd.Series | None:
        from trader3.portfolio.hrp import hrp_weights
        if returns is None:
            logger.warning("[portfolio] hrp 需要 returns，缺省回退 equal_weight")
            return self._equal_weight(pd.Series(1.0, index=assets))
        if isinstance(returns, pd.DataFrame):
            returns = returns.reindex(columns=assets).values
        arr = np.asarray(returns, dtype=float)
        try:
            w = hrp_weights(arr, names=assets)
        except Exception as e:  # noqa: BLE001
            logger.warning("[portfolio] hrp 失败: %s，回退 equal_weight", e)
            return self._equal_weight(pd.Series(1.0, index=assets))
        return pd.Series(w)

    def _optimize_tool(self, cs: pd.Series, method: str,
                       cov, industries) -> pd.Series | None:
        from trader3.models import PortfolioConstraints
        from trader3.tools.optimize import OptimizePortfolioTool

        signals = {a: float(v) for a, v in cs.items()}
        cons = PortfolioConstraints(
            max_single_weight=self.config.max_single,
            long_only=self.config.long_only,
        )
        risk_model = None
        if cov is not None:
            if isinstance(cov, pd.DataFrame):
                cov_arr = cov.reindex(index=cs.index, columns=cs.index).values
            else:
                cov_arr = np.asarray(cov, dtype=float)
            if cov_arr.shape == (len(cs), len(cs)):
                risk_model = {"cov": cov_arr}
        try:
            resp = OptimizePortfolioTool().execute(
                signals=signals, method=method, constraints=cons,
                risk_model=risk_model,
                industry_neutral=self.config.industry_neutral,
                industries=industries,
            )
            if not resp or not getattr(resp, "success", False):
                logger.warning("[portfolio] %s 优化未成功，回退 ic_weighted", method)
                return self._ic_weighted(cs)
            tw = getattr(resp.data, "target_weights", None) or {}
            return pd.Series({k: float(v) for k, v in tw.items()})
        except Exception as e:  # noqa: BLE001
            logger.warning("[portfolio] %s 异常: %s，回退 ic_weighted", method, e)
            return self._ic_weighted(cs)

    def _regime_aware(self, cs: pd.Series, regime_probs, regime_weights,
                     industries) -> pd.Series | None:
        from trader3.tools.optimize import RegimeAwareAllocationTool
        if not regime_probs or not regime_weights:
            logger.warning("[portfolio] regime_aware 缺 regime_probs/weights，回退 ic_weighted")
            return self._ic_weighted(cs)
        signals = {a: float(v) for a, v in cs.items()}
        try:
            resp = RegimeAwareAllocationTool().execute(
                signals=signals, regime_probs=regime_probs,
                regime_weights=regime_weights,
                industry_neutral=self.config.industry_neutral,
                industries=industries,
            )
            if not resp or not getattr(resp, "success", False):
                return self._ic_weighted(cs)
            tw = getattr(resp.data, "target_weights", None) or {}
            return pd.Series({k: float(v) for k, v in tw.items()})
        except Exception as e:  # noqa: BLE001
            logger.warning("[portfolio] regime_aware 异常: %s，回退 ic_weighted", e)
            return self._ic_weighted(cs)


def construct_portfolio(
    scores: pd.Series | pd.DataFrame,
    method: str = "ic_weighted",
    *,
    cov=None, returns=None, overlay=None,
    industries=None, regime_probs=None, regime_weights=None,
    exposures=None, benchmark=None,
    config: PortfolioConfig | None = None,
) -> pd.Series:
    """便捷函数：construct_portfolio(scores, method=...)(panel) -> weights"""
    cfg = config or PortfolioConfig(method=method)
    cfg.method = method
    return PortfolioConstruction(cfg).construct(
        scores, cov=cov, returns=returns, overlay=overlay,
        industries=industries, regime_probs=regime_probs,
        regime_weights=regime_weights, exposures=exposures, benchmark=benchmark,
    )
