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


def _panel_last_prices(panel: pd.DataFrame) -> dict[str, float]:
    """从面板取各资产最新收盘价。"""
    try:
        close = panel.xs("close", axis=1, level=1)
    except Exception:
        return {}
    last = close.iloc[-1]
    return {a: float(v) for a, v in last.items() if np.isfinite(v) and v > 0}


def _default_forward(panel: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    close = panel.xs("close", axis=1, level=1)
    return close.pct_change(horizon).shift(-horizon)


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
) -> dict[str, Any]:
    """
    执行端到端量化管线。返回 {scores, weights, orders, equity, meta}。

    - 若 scores 给定，跳过 ensemble；否则由 factor_exprs 经 DSL 计算 + ensemble 合成
    - broker=None 时只算到权重，不下单（便于回测/审计）
    """
    cfg = config or QuantPipelineConfig()
    factor_exprs = factor_exprs or cfg.factor_exprs

    # 1. 得分（因子 → ensemble）
    if scores is None and build_scores is not None:
        scores = build_scores(panel)
    if scores is None and panel is not None and factor_exprs:
        scores = _run_ensemble(panel, factor_exprs, forward_returns, cfg)

    if scores is None:
        return {"scores": None, "weights": pd.Series(dtype=float),
                "orders": [], "equity": 0.0, "meta": {"error": "no scores"}}

    # 2. 组合构建
    pc = PortfolioConstruction(PortfolioConfig(
        method=cfg.method, top_n=cfg.top_n, max_single=cfg.max_single))
    weights = pc.construct(scores, overlay=overlay)
    if weights is None or len(weights) == 0:
        return {"scores": scores, "weights": pd.Series(dtype=float),
                "orders": [], "equity": 0.0, "meta": {"halted": True}}

    result: dict[str, Any] = {
        "scores": scores, "weights": weights,
        "orders": [], "equity": 0.0, "meta": {},
    }

    # 3. 下单（可选）
    if broker is None:
        return result

    try:
        acct = await broker.get_account()
        equity = float(getattr(acct, "equity", 0.0) or 0.0)
        if equity <= 0:
            equity = float(getattr(acct, "cash", 0.0) or 0.0)
        result["equity"] = equity

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
        orders = om.generate_orders(weights, equity, prices, current_positions=positions)
        placed = await om.submit(orders, broker)
        result["orders"] = placed
        result["meta"]["n_orders"] = len(placed)
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
