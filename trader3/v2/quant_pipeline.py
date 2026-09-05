"""
量化交易端到端管线（Quant Pipeline）。

把分散的模块串成可执行的闭环：
    面板数据 → 因子(DSL) → ensemble 合成得分 → 组合构建(组合优化)
             → 订单管理 → 经纪商下单(paper / CTP)

设计原则：
- 数据流全部可注入（factors / scores / broker / overlay），便于离线测试
- ensemble 与 portfolio 的失败都有回退，不抛到主链路
- 与 trigger 系 daily_pipeline 解耦：这里是「因子→权重→下单」的纯量化链路
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trader3.v2.execution import KillSwitch, LiquidityEngine, reconcile_positions
from trader3.v2.order_manager import OrderManager, OrderManagerConfig
from trader3.v2.portfolio import PortfolioConfig, PortfolioConstruction

logger = logging.getLogger("trader3.v2.quant_pipeline")


@dataclass
class QuantPipelineConfig:
    factor_exprs: dict[str, str] = field(default_factory=dict)  # name -> DSL expr
    method: str = "ic_weighted"
    top_n: int = 0
    max_single: float = 0.25
    use_ensemble: bool = True
    ensemble_model: str = "auto"
    min_train: int = 250
    order_config: OrderManagerConfig | None = None
    # ── 状态 / 风险模型接入 ──
    use_regime: bool = False               # 用 HMM 检测状态并路由方法 + 缩放敞口
    risk_model_cov: bool = False           # 用面板收益估计资产协方差喂给 MV/RB
    risk_attribution: bool = False         # 用 Barra 风格风险模型对最终权重做分解
    use_moe: bool = False                  # 多专家集成（门控融合）替代单一 ensemble
    moe_experts: list[str] | None = None   # 专家模型列表（None 用默认）
    ewma_cov_lambda: float = 0.94          # 协方差 EWMA 衰减
    factor_registry_path: str | None = None  # 因子工厂仓库：通过闸门的因子并入 factor_exprs
    # ── 执行层加固 ──
    use_cost_gate: bool = False            # 流动性/冲击成本门禁（预期收益需覆盖成本）
    use_reconcile: bool = False            # 下单后与券商持仓对账
    kill_switch_dd: float = 0.05           # 回撤熔断阈值
    edge_per_score_bps: float = 50.0       # 得分每单位→预期收益(bp)，用于成本门禁
    max_participation: float = 0.1         # 单笔参与率上限（流动性引擎）
    # ── 嵌套执行 ──
    use_nested_execution: bool = False     # 组合层+执行层联合优化（冲击回灌+交易轨迹）
    nested_horizon: int = 5                # 执行跨度（期）
    # ── Barra 风险约束（P1-2）──
    beta_neutral: bool = False             # β 中性（需 panel 构建风格暴露）
    industry_max_weight: float = 1.0        # 单行业合计权重上限（<1 启用，需 industries 映射）
    tracking_error_max: float = 0.0         # 年化跟踪误差上限（>0 启用，需 cov）
    # ── 子单调度（P1-6）──
    use_child_scheduling: bool = False      # 把父单拆 TWAP/VWAP 子单按节奏提交
    child_method: str = "twap"              # twap | vwap
    child_slices: int = 10
    child_horizon: float = 60.0             # 执行跨度（秒，回测可缩放为 0）


# regime 标签 -> (方法, gross 敞口系数)
REGIME_PLAN: dict[str, tuple[str, float]] = {
    "low-vol": ("ic_weighted", 1.0),
    "mid-vol": ("ic_weighted", 0.8),
    "high-vol": ("risk_budget", 0.5),
    "calm": ("ic_weighted", 1.0),
    "turbulent": ("risk_budget", 0.5),
    "bear": ("risk_budget", 0.5),
    "flat": ("ic_weighted", 0.8),
    "bull": ("mean_variance", 1.0),
}


def _panel_last_prices(panel: pd.DataFrame) -> dict[str, float]:
    """从面板取各资产最新收盘价。"""
    try:
        close = panel.xs("close", axis=1, level=1)
    except Exception:
        return {}
    last = close.iloc[-1]
    return {a: float(v) for a, v in last.items() if np.isfinite(v) and v > 0}


def _adv_from_panel(panel: pd.DataFrame) -> dict[str, float]:
    """由面板 volume 字段计算各标的日均成交量（ADV），供流动性引擎使用。"""
    try:
        vol = panel.xs("volume", axis=1, level=1)
    except Exception:
        return {}
    adv = vol.mean(axis=0)
    return {a: float(v) for a, v in adv.items() if np.isfinite(v) and v > 0}


def _default_forward(panel: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    close = panel.xs("close", axis=1, level=1)
    return close.pct_change(horizon).shift(-horizon)


def _asset_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """面板 → 资产收益宽表 (date × asset)，含截面交集。"""
    close = panel.xs("close", axis=1, level=1)
    return close.pct_change().dropna(how="all")


def _asset_covariance(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """EWMA 资产协方差矩阵（年化前的日频估计）。"""
    r = returns.dropna(how="all")
    if r.shape[0] < 5 or r.shape[1] < 2:
        return pd.DataFrame(np.eye(returns.shape[1]),
                           index=returns.columns, columns=returns.columns)
    arr = r.values.astype(float)
    T = arr.shape[0]
    w = np.array([lam ** (T - 1 - i) for i in range(T)])
    w /= w.sum()
    mu = (arr * w[:, None]).sum(axis=0)
    d = arr - mu
    cov = (d * w[:, None]).T @ d
    # PSD 投影
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    cov_psd = vecs @ np.diag(vals) @ vecs.T
    return pd.DataFrame((cov_psd + cov_psd.T) / 2, index=r.columns, columns=r.columns)


def _regime_state(panel: pd.DataFrame) -> dict[str, Any] | None:
    """对等权组合收益做 HMM 状态检测，返回 {label, confidence, probs}。"""
    try:
        from trader3.v2.regime import current_regime, detect_regimes
        rets = _asset_returns(panel)
        if rets.shape[0] < 20 or rets.shape[1] < 2:
            return None
        eq_ret = rets.mean(axis=1).dropna()
        if len(eq_ret) < 20:
            return None
        res = detect_regimes(eq_ret.values, n_states=3, label_scheme="vol")
        label, conf = current_regime(res, lookback=5)
        return {
            "label": label,
            "confidence": round(float(conf), 4),
            "state_probs": {k: round(float(v), 4)
                            for k, v in zip(
                                [res.labels_by_vol.get(i, f"s{i}") for i in range(len(res.means))],
                                res.state_probs[-1], strict=False)},
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("[quant_pipeline] 状态检测失败: %s", e)
        return None


def _risk_decomposition(weights: pd.Series, panel: pd.DataFrame) -> dict[str, Any]:
    """Barra 风格风险分解：系统/特异性/总风险 + 因子暴露。"""
    from trader3.v2.risk_model import FactorRiskModel

    rets = _asset_returns(panel).reindex(columns=weights.index)
    if rets.shape[0] < 30:
        return {}
    model = FactorRiskModel()
    exposures = FactorRiskModel.build_style_exposures(panel)
    exposures = exposures.reindex(weights.index)
    if exposures.isna().all().all():
        return {}
    model.fit_factor_returns(rets, exposures)
    model.fit_factor_covariance()
    dec = model.decompose_portfolio(weights)
    fe = getattr(dec, "factor_exposure", None)
    if fe is not None and hasattr(fe, "to_dict"):
        fe = {k: round(float(v), 4) for k, v in fe.to_dict().items()}
    else:
        fe = {}
    return {
        "total_risk": round(float(getattr(dec, "total_risk", 0.0)), 4),
        "systematic_risk": round(float(getattr(dec, "systematic_risk", 0.0)), 4),
        "specific_risk": round(float(getattr(dec, "specific_risk", 0.0)), 4),
        "factor_exposure": fe,
        "factor_contributions": {k: round(float(v), 4)
                                 for k, v in (getattr(dec, "factor_contributions", {}) or {}).items()},
    }


async def run_quant_pipeline(
    panel: pd.DataFrame | None = None,
    *,
    factor_exprs: dict[str, str] | None = None,
    scores: pd.Series | pd.DataFrame | None = None,
    forward_returns: pd.DataFrame | None = None,
    broker=None,
    overlay=None,
    config: QuantPipelineConfig | None = None,
    build_scores: Callable | None = None,
    kill_switch: KillSwitch | None = None,
) -> dict[str, Any]:
    """
    执行端到端量化管线。返回 {scores, weights, orders, equity, meta}。

    - 若 scores 给定，跳过 ensemble；否则由 factor_exprs 经 DSL 计算 + ensemble 合成
    - broker=None 时只算到权重，不下单（便于回测/审计）
    - kill_switch 可注入（跨调用持久），回撤超阈即不下单
    """
    cfg = config or QuantPipelineConfig()
    factor_exprs = dict(factor_exprs or cfg.factor_exprs or {})
    # 并入因子工厂仓库中已通过闸门的因子
    if cfg.factor_registry_path:
        try:
            from trader3.v2.factor_factory import FactorRegistry
            reg = FactorRegistry(cfg.factor_registry_path)
            mined = reg.as_factor_exprs()
            if mined:
                factor_exprs = {**mined, **factor_exprs}
                logger.info("[quant_pipeline] 并入因子工厂仓库 %d 个因子", len(mined))
        except Exception as e:  # noqa: BLE001
            logger.warning("[quant_pipeline] 读取因子仓库失败: %s", e)

    # 1. 得分（因子 → ensemble）
    if scores is None and build_scores is not None:
        scores = build_scores(panel)
    if scores is None and panel is not None and factor_exprs:
        scores = _run_ensemble(panel, factor_exprs, forward_returns, cfg)

    if scores is None:
        return {"scores": None, "weights": pd.Series(dtype=float),
                "orders": [], "equity": 0.0, "meta": {"error": "no scores"}}

    result: dict[str, Any] = {
        "scores": scores, "weights": pd.Series(dtype=float),
        "orders": [], "equity": 0.0, "meta": {},
    }

    # 2. 组合构建（含状态路由 + 风险模型协方差）
    method = cfg.method
    gross_cap = 1.0
    regime_info: dict[str, Any] | None = None
    if cfg.use_regime and panel is not None:
        regime_info = _regime_state(panel)
        if regime_info:
            rm, rg = REGIME_PLAN.get(
                regime_info["label"], (cfg.method, 0.8))
            method = rm
            gross_cap = min(gross_cap, rg)
            logger.info("[quant_pipeline] regime=%s conf=%.2f -> method=%s gross=%.2f",
                        regime_info["label"], regime_info["confidence"], method, gross_cap)

    cov = None
    rets = None
    if cfg.risk_model_cov and panel is not None and method in ("mean_variance", "risk_budget"):
        cov = _asset_covariance(_asset_returns(panel), cfg.ewma_cov_lambda)
    if method == "hrp" and panel is not None:
        rets = _asset_returns(panel)

    pc = PortfolioConstruction(PortfolioConfig(
        method=method, top_n=cfg.top_n, max_single=cfg.max_single, gross_cap=gross_cap,
        beta_neutral=cfg.beta_neutral, industry_max_weight=cfg.industry_max_weight,
        tracking_error_max=cfg.tracking_error_max))
    exposures = None
    if (cfg.beta_neutral or cfg.industry_max_weight < 1.0 or cfg.tracking_error_max > 0.0) and panel is not None:
        from trader3.v2.risk_model import FactorRiskModel

        exposures = FactorRiskModel.build_style_exposures(panel)
    weights = pc.construct(scores, cov=cov, returns=rets, overlay=overlay, exposures=exposures)
    if weights is None or len(weights) == 0:
        return {"scores": scores, "weights": pd.Series(dtype=float),
                "orders": [], "equity": 0.0, "meta": {"halted": True}}
    benchmark = None
    if cfg.tracking_error_max > 0.0 and exposures is not None:
        benchmark = pd.Series(1.0 / len(weights), index=weights.index)
        weights = pc.construct(scores, cov=cov, returns=rets, overlay=overlay,
                               exposures=exposures, benchmark=benchmark)
    if weights is None or len(weights) == 0:
        return {"scores": scores, "weights": pd.Series(dtype=float),
                "orders": [], "equity": 0.0, "meta": {"halted": True}}

    result["weights"] = weights
    if regime_info:
        result["meta"]["regime"] = regime_info
    if cov is not None:
        result["meta"]["cov_source"] = "panel_ewma"

    # 2b. Barra 风格风险分解（可选，仅报告用）
    if cfg.risk_attribution and panel is not None and len(weights) > 1:
        try:
            result["meta"]["risk_decomp"] = _risk_decomposition(weights, panel) or {}
        except Exception as e:  # noqa: BLE001
            logger.warning("[quant_pipeline] 风险分解失败: %s", e)
            result["meta"]["risk_decomp"] = {}

    # 3. 下单（可选）
    if broker is None:
        return result

    # kill switch 预检（跨调用持久对象可能已熔断）
    if kill_switch is not None and kill_switch.tripped:
        result["meta"]["halted_by_kill_switch"] = True
        return result

    try:
        acct = await broker.get_account()
        equity = float(getattr(acct, "equity", 0.0) or 0.0)
        if equity <= 0:
            equity = float(getattr(acct, "cash", 0.0) or 0.0)
        result["equity"] = equity

        # kill switch 更新（单调用内建实例）
        if kill_switch is None and cfg.kill_switch_dd > 0:
            kill_switch = KillSwitch(max_drawdown=cfg.kill_switch_dd)
        if kill_switch is not None:
            killed = kill_switch.update(equity)
            result["meta"]["kill_switch_tripped"] = kill_switch.tripped
            result["meta"]["drawdown"] = round(kill_switch.current_drawdown, 4)
            if killed:
                result["meta"]["halted_by_kill_switch"] = True
                return result
            kill_switch.set_last_good(weights)  # 记录最后已知良好权重（自动回滚用）

        prices = _panel_last_prices(panel) if panel is not None else {}
        # 优先用 broker 行情补全
        try:
            md = await broker.get_market_data(list(weights.index))
            for s, m in md.items():
                if m and getattr(m, "last_price", 0):
                    prices[s] = float(m.last_price)
        except Exception:
            pass

        positions = await broker.get_positions()
        om = OrderManager(cfg.order_config or OrderManagerConfig())
        from trader3.v2.live.broker_base import OrderSide

        # 嵌套执行：用冲击成本回灌重新求解可执行权重 + 交易轨迹
        if cfg.use_nested_execution and panel is not None:
            try:
                from trader3.v2.nested_execution import NestedExecutor, NestedExecutorConfig
                # 规整 alpha 为 1D asset Series（solve 契约）：
                # MoE/ensemble 产出通常为 MultiIndex (date, asset) Series；取最新 date 截面，
                # 保证 asset 唯一且与 cov/weights 对齐（否则 pandas 抛
                # "cannot join with no overlapping index names" / duplicate labels）。
                alpha = scores
                if isinstance(alpha, pd.DataFrame):
                    # (date×asset) 宽表：取末行
                    if isinstance(alpha.index, pd.MultiIndex):
                        alpha = alpha.iloc[-1]
                    else:
                        alpha = alpha.iloc[-1]
                if isinstance(alpha.index, pd.MultiIndex):
                    # (date, asset) 长表：取最新 date 的截面
                    date_lv = alpha.index.get_level_values(0)
                    last_date = date_lv[-1]
                    alpha = alpha.xs(last_date, level=0)
                alpha = alpha.reindex(weights.index).fillna(0.0)
                alpha.index = weights.index
                ne_cov = _asset_covariance(_asset_returns(panel), cfg.ewma_cov_lambda)
                ne_cov = ne_cov.reindex(index=weights.index, columns=weights.index).fillna(0.0)
                ne_cov.index = weights.index
                ne_cov.columns = weights.index
                ne = NestedExecutor(NestedExecutorConfig(horizon=cfg.nested_horizon))
                plan = ne.solve(alpha, ne_cov, weights, equity,
                                prices, _adv_from_panel(panel))
                weights = plan.target_weights
                result["weights"] = weights
                result["meta"]["execution_plan"] = {
                    "expected_cost_bps": plan.expected_cost_bps,
                    "expected_shortfall": plan.expected_shortfall,
                    "n_slices": len(plan.schedule),
                    "horizon": cfg.nested_horizon,
                    "n_target_assets": int((plan.target_weights.abs() > 1e-6).sum()),
                }
            except Exception as e:  # noqa: BLE001
                logger.warning("[quant_pipeline] 嵌套执行失败，退回原权重: %s", e)

        orders = om.generate_orders(weights, equity, prices, current_positions=positions)

        # 成本门禁：预期收益需覆盖流动性+佣金成本
        if cfg.use_cost_gate:
            adv_map = _adv_from_panel(panel) if panel is not None else {}
            liq = LiquidityEngine(adv=adv_map, max_participation=cfg.max_participation)
            kept, gated = [], []
            for o in orders:
                est = liq.estimate(o.symbol, o.quantity, prices.get(o.symbol, 0.0),
                                   adv_map.get(o.symbol))
                edge = abs(float(weights.get(o.symbol, 0.0))) * cfg.edge_per_score_bps
                if liq.cost_gate(edge, est):
                    kept.append(o)
                else:
                    gated.append(o.symbol)
            orders = kept
            result["meta"]["cost_gated_out"] = gated

        # 子单调度：把父单拆 TWAP/VWAP 子单按节奏提交（否则直接提交父单）
        if cfg.use_child_scheduling and orders:
            from trader3.v2.child_orders import ChildOrderSchedulerConfig, execute_child_orders

            cos_cfg = ChildOrderSchedulerConfig(
                method=cfg.child_method, n_slices=cfg.child_slices, horizon=cfg.child_horizon)
            try:
                placed = await execute_child_orders(orders, broker, cos_cfg)
                result["meta"]["child_scheduling"] = {
                    "method": cfg.child_method, "n_slices": cfg.child_slices, "n_parent": len(orders)}
            except Exception as e:  # noqa: BLE001
                logger.warning("[quant_pipeline] 子单调度失败，退回原父单提交: %s", e)
                placed = await om.submit(orders, broker)
        else:
            placed = await om.submit(orders, broker)

        result["orders"] = placed
        result["meta"]["n_orders"] = len(placed)

        # 持仓对账：内部预期（当前+成交）vs 券商实际
        if cfg.use_reconcile:
            broker_pos = {s: float(p.quantity)
                          for s, p in (await broker.get_positions()).items()}
            internal_after: dict[str, float] = {
                s: float(p.quantity) for s, p in positions.items()}
            for o in placed:
                st = str(getattr(o, "status", "")).lower()
                if "fill" in st:
                    q = float(getattr(o, "filled_qty", 0.0) or getattr(o, "quantity", 0.0))
                    internal_after[o.symbol] = internal_after.get(o.symbol, 0.0) + (
                        q if o.side == OrderSide.BUY else -q)
            disc = reconcile_positions(internal_after, broker_pos)
            result["meta"]["reconcile"] = [d.__dict__ for d in disc]
            result["meta"]["reconcile_breaches"] = sum(
                1 for d in disc if d.severity == "breach")
    except Exception as e:  # noqa: BLE001
        logger.warning("[quant_pipeline] 下单阶段异常: %s", e)
        result["meta"]["order_error"] = str(e)

    return result


def _run_ensemble(panel, factor_exprs, forward_returns, cfg) -> pd.Series | pd.DataFrame:
    from trader3.v2.ensemble import EnsembleConfig, build_ensemble
    from trader3.v2.factor_dsl import get_dsl

    dsl = get_dsl()
    features_wide: dict[str, pd.DataFrame] = {}
    for name, expr in factor_exprs.items():
        try:
            features_wide[name] = dsl.full_series(expr, panel)  # (date × asset)
        except Exception as e:  # noqa: BLE001
            logger.warning("[quant_pipeline] 因子 %s 计算失败: %s", name, e)
    if not features_wide:
        return pd.Series(dtype=float)

    fwd = forward_returns if forward_returns is not None else _default_forward(panel)
    try:
        if cfg.use_moe:
            from trader3.v2.moe_ensemble import MoEConfig, train_moe
            moe_cfg = MoEConfig(
                experts=cfg.moe_experts or ["lgbm", "et", "ridge"],
                min_train=cfg.min_train,
            )
            moe_res = train_moe(features_wide, fwd, moe_cfg)
            logger.info("[quant_pipeline] MoE 融合完成，专家=%s OOS_IC=%.4f",
                        moe_res.used_experts, moe_res.oos_rank_ic)
            return moe_res.scores
        res = build_ensemble(
            features_wide, fwd,
            config=EnsembleConfig(model=cfg.ensemble_model, min_train=cfg.min_train),
        )
        return res.scores
    except Exception as e:  # noqa: BLE001
        logger.warning("[quant_pipeline] ensemble 失败，回退等权: %s", e)
        # 回退：各因子最新截面等权
        last = {n: df.iloc[-1] for n, df in features_wide.items()}
        stacked = pd.concat(last.values(), axis=1)
        return stacked.mean(axis=1)
