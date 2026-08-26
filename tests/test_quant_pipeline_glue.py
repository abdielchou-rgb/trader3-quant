"""组合构建 / 订单管理 / 量化管线 集成测试。"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from trader3.v2.live.broker_base import OrderSide, Position
from trader3.v2.order_manager import OrderManager, OrderManagerConfig
from trader3.v2.portfolio import PortfolioConfig, PortfolioConstruction, construct_portfolio
from trader3.v2.quant_pipeline import QuantPipelineConfig, run_quant_pipeline


def _make_panel(n_dates=80, n_assets=6, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    assets = [f"A{i}" for i in range(n_assets)]
    fields = ["open", "high", "low", "close", "volume", "vwap", "amount"]
    cols = pd.MultiIndex.from_product([assets, fields])
    df = pd.DataFrame(rng.normal(size=(n_dates, len(cols))), index=dates, columns=cols)
    for a in assets:
        walk = np.cumsum(rng.normal(0, 1, n_dates))
        df[(a, "close")] = walk + 100
        df[(a, "vwap")] = df[(a, "close")]
    return df


def test_portfolio_ic_weighted():
    scores = pd.Series({"A0": 0.5, "A1": -0.2, "A2": 0.3})
    pc = PortfolioConstruction(PortfolioConfig(method="ic_weighted", long_only=True))
    w = pc.construct(scores)
    assert w.min() >= 0
    assert abs(w.sum() - 1.0) < 1e-6


def test_portfolio_equal_weight():
    scores = pd.Series({"A0": 1.0, "A1": 2.0, "A2": 3.0})
    pc = PortfolioConstruction(PortfolioConfig(method="equal_weight", top_n=2))
    w = pc.construct(scores)
    assert len(w) == 2
    assert abs(w.sum() - 1.0) < 1e-6


def test_portfolio_overlay_halt():
    scores = pd.Series({"A0": 1.0, "A1": 1.0})

    class OverlayStub:
        halt_new_buys = True
        size_multiplier = 1.0
    w = PortfolioConstruction(PortfolioConfig()).construct(scores, overlay=OverlayStub())
    assert len(w) == 0


def test_portfolio_overlay_size():
    scores = pd.Series({"A0": 1.0, "A1": 1.0})

    class OverlayStub:
        halt_new_buys = False
        size_multiplier = 0.5
    w = PortfolioConstruction(PortfolioConfig(method="equal_weight")).construct(scores, overlay=OverlayStub())
    assert abs(w.sum() - 0.5) < 1e-6


def test_portfolio_risk_budget_via_tool():
    scores = pd.Series({f"A{i}": float(np.random.default_rng(i).normal()) for i in range(5)})
    w = construct_portfolio(scores, method="risk_budget")
    assert len(w) > 0
    assert w.min() >= -1e-6


def test_order_manager_generate():
    weights = pd.Series({"A0": 0.5, "A1": 0.5})
    prices = {"A0": 10.0, "A1": 20.0}
    om = OrderManager(OrderManagerConfig(default_lot=100, cash_buffer=0.0))
    orders = om.generate_orders(weights, equity=100000.0, prices=prices)
    # 应生成 2 笔买单
    assert len(orders) == 2
    assert all(o.side == OrderSide.BUY for o in orders)
    assert all(o.quantity % 100 == 0 for o in orders)


def test_order_manager_with_positions():
    weights = pd.Series({"A0": 0.05})  # 0.05 * 100k / 10 = 500 股
    prices = {"A0": 10.0}
    cur = {"A0": Position(symbol="A0", quantity=1000, avg_cost=9.0,
                          market_value=10000, unrealized_pnl=1000)}
    om = OrderManager(OrderManagerConfig(default_lot=100))
    orders = om.generate_orders(weights, equity=100000.0, prices=prices,
                               current_positions=cur)
    # 目标 500 股，当前 1000 股 → 卖出 500
    assert len(orders) == 1
    assert orders[0].side == OrderSide.SELL
    assert orders[0].quantity == 500


def test_quant_pipeline_with_scores_and_broker():
    # 用一个最小 paper broker 模拟
    from trader3.v2.live.paper_broker import PaperBroker

    panel = _make_panel()
    scores = pd.Series({f"A{i}": 0.1 * i for i in range(6)})

    async def run():
        b = PaperBroker(initial_cash=1_000_000)
        await b.connect()
        res = await run_quant_pipeline(
            panel, scores=scores, broker=b,
            config=QuantPipelineConfig(method="ic_weighted", factor_exprs={}),
        )
        return res

    res = asyncio.run(run())
    assert "weights" in res
    assert res["weights"].sum() <= 1.0 + 1e-6
    assert len(res["orders"]) > 0


def test_quant_pipeline_no_broker_scores_only():
    scores = pd.Series({"A0": 1.0, "A1": -0.5, "A2": 0.3})
    res = asyncio.run(run_quant_pipeline(None, scores=scores,
                                         config=QuantPipelineConfig(method="ic_weighted")))
    assert res["weights"].sum() <= 1.0 + 1e-6
    assert res["orders"] == []
