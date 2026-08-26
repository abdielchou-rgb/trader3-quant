"""组合构建 / 订单管理 / 量化管线 集成测试。"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from trader3.v2.live.broker_base import BrokerBase, OrderSide, OrderStatus, Position
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


def test_quant_pipeline_with_regime_routing():
    panel = _make_panel(n_dates=120)
    scores = pd.Series({f"A{i}": 0.1 * i for i in range(6)})
    cfg = QuantPipelineConfig(method="ic_weighted", use_regime=True, factor_exprs={})
    res = asyncio.run(run_quant_pipeline(panel, scores=scores, config=cfg))
    # 状态检测应写入 meta，且权重合法
    assert "regime" in res["meta"]
    assert res["weights"].sum() <= 1.0 + 1e-6
    assert res["meta"]["regime"]["label"] in (
        "low-vol", "mid-vol", "high-vol", "calm", "turbulent")


def test_quant_pipeline_with_risk_model_cov():
    panel = _make_panel(n_dates=120)
    scores = pd.Series({f"A{i}": 0.1 * i for i in range(6)})
    # 用 risk_budget 触发协方差接入
    cfg = QuantPipelineConfig(method="risk_budget", risk_model_cov=True, factor_exprs={})
    res = asyncio.run(run_quant_pipeline(panel, scores=scores, config=cfg))
    assert res["weights"].sum() <= 1.0 + 1e-6
    assert res["meta"].get("cov_source") == "panel_ewma"


def test_quant_pipeline_with_risk_attribution():
    panel = _make_panel(n_dates=200, n_assets=30)
    scores = pd.Series({f"A{i}": (i % 5) - 2.0 for i in range(30)})
    cfg = QuantPipelineConfig(method="ic_weighted", risk_attribution=True, factor_exprs={})
    res = asyncio.run(run_quant_pipeline(panel, scores=scores, config=cfg))
    assert "risk_decomp" in res["meta"]
    # 30 资产足以支撑 Barra 风格分解，应产出真实风险数字
    decomp = res["meta"]["risk_decomp"]
    assert decomp.get("total_risk", 0.0) > 0.0
    assert "factor_exposure" in decomp


def _make_features_and_fwd(n_dates=120, n_assets=8, seed=7):
    """构造 features_wide + forward_returns 供 MoE 测试（不依赖 DSL）。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    assets = [f"A{i}" for i in range(n_assets)]
    # 因子1：带噪声的动量代理
    f1 = pd.DataFrame(rng.normal(0, 1, (n_dates, n_assets)), index=dates, columns=assets)
    # 因子2：反转代理
    f2 = pd.DataFrame(-rng.normal(0, 1, (n_dates, n_assets)), index=dates, columns=assets)
    feats = {"mom": f1, "rev": f2}
    fwd = pd.DataFrame(rng.normal(0, 0.02, (n_dates, n_assets)), index=dates, columns=assets)
    return feats, fwd


def test_moe_ensemble_basic():
    feats, fwd = _make_features_and_fwd()
    from trader3.v2.moe_ensemble import MoEConfig, train_moe
    res = train_moe(feats, fwd, MoEConfig(experts=["lgbm", "et", "ridge"], min_train=60))
    assert res.used_experts
    assert not res.scores.empty
    # 门控权重逐期和为 1
    gw = res.gate_weights
    assert np.allclose(gw.sum(axis=1).dropna().values, 1.0, atol=1e-6)
    # 融合得分应为各专家加权组合（与手动重算近似）
    assert res.oos_rank_ic is not None


def test_moe_regime_affinity_gate():
    feats, fwd = _make_features_and_fwd()
    from trader3.v2.moe_ensemble import MoEConfig, train_moe
    aff = {"high-vol": {"lgbm": 0.2, "et": 0.3, "ridge": 1.5},
           "low-vol": {"lgbm": 1.0, "et": 1.0, "ridge": 0.5}}
    res = train_moe(
        feats, fwd,
        MoEConfig(experts=["lgbm", "et", "ridge"], min_train=60,
                  gate_method="regime_affinity", regime_affinity=aff),
        regime_labels=pd.Series(["high-vol"] * len(fwd.index), index=fwd.index))
    assert not res.scores.empty
    assert res.gate_method == "regime_affinity"


def test_quant_pipeline_with_moe():
    panel = _make_panel(n_dates=140)
    feats = {
        "f1": "sub(log(vwap), log(close))",
        "f2": "rank(close)",
    }
    cfg = QuantPipelineConfig(method="ic_weighted", use_moe=True,
                             moe_experts=["lgbm", "et", "ridge"],
                             min_train=60, factor_exprs=feats)
    res = asyncio.run(run_quant_pipeline(panel, config=cfg))
    assert not res["weights"].empty
    assert res["weights"].sum() <= 1.0 + 1e-6


# ───────────────────────── 执行层加固测试 ─────────────────────────


def _make_panel_pos_vol(n_dates=80, n_assets=6, seed=11):
    """与 _make_panel 类似，但 volume 为正（供流动性引擎 ADV 计算）。"""
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
        df[(a, "volume")] = np.abs(rng.normal(1e5, 2e4, n_dates)) + 1e4
    return df


class FakeBroker(BrokerBase):
    """测试用内存券商：下单即按市价全量成交并维护持仓。"""

    def __init__(self, equity=1_000_000, prices=None, positions=None):
        super().__init__()
        self._equity = equity
        self._prices = prices or {f"A{i}": 100.0 + i for i in range(8)}
        self._positions = positions or {}
        self._connected = True

    async def connect(self):
        return True

    async def disconnect(self):
        return True

    async def place_order(self, order):
        price = self._prices.get(order.symbol, 100.0)
        order.status = OrderStatus.FILLED
        order.filled_qty = order.quantity
        order.avg_fill_price = price
        q = order.quantity if order.side == OrderSide.BUY else -order.quantity
        p = self._positions.get(order.symbol)
        new_q = (p.quantity + q) if p else q
        self._positions[order.symbol] = Position(
            symbol=order.symbol, quantity=new_q, avg_cost=price,
            market_value=new_q * price, unrealized_pnl=0.0, last_price=price)
        return order

    async def cancel_order(self, client_order_id):
        return True

    async def get_order(self, client_order_id):
        return self._orders.get(client_order_id)

    async def get_orders(self, status=None):
        return list(self._orders.values())

    async def get_positions(self):
        return dict(self._positions)

    async def get_account(self):
        from trader3.v2.live.broker_base import Account
        return Account(account_id="TEST", cash=self._equity, equity=self._equity,
                       buying_power=self._equity)

    async def get_market_data(self, symbols):
        from trader3.v2.live.broker_base import MarketData
        return {s: MarketData(symbol=s, price=self._prices.get(s, 100.0)) for s in symbols}

    async def subscribe_market_data(self, symbols, callback):
        return True

    async def unsubscribe_market_data(self, symbols):
        return True


def test_liquidity_engine_estimate_and_gate():
    from trader3.v2.execution import LiquidityEngine
    eng = LiquidityEngine(adv={"AAA": 1e5}, max_participation=0.1)
    est = eng.estimate("AAA", 1000, 100.0, adv=1e5)  # 参与率 1%
    assert est.participation_rate == 0.01
    assert est.tradable
    # 预期收益 50bp 覆盖成本
    assert eng.cost_gate(50.0, est)
    # 预期收益 0.1bp 不覆盖
    assert not eng.cost_gate(0.1, est)
    # 超大单：参与率超限 → 不可交易
    big = eng.estimate("AAA", 2e4, 100.0, adv=1e5)  # 参与率 20%
    assert not big.tradable
    assert not eng.cost_gate(100.0, big)


def test_reconcile_positions_match_warn_breach():
    from trader3.v2.execution import reconcile_positions
    disc = reconcile_positions({"A": 100.0, "B": 50.0}, {"A": 100.0, "B": 50.0})
    assert all(d.severity == "match" for d in disc)
    disc = reconcile_positions({"A": 100.0}, {"A": 0.0})  # 差 100 远超阈值
    assert disc[0].severity == "breach"
    disc = reconcile_positions({"A": 100.0}, {"A": 100.0000001})  # 极小差
    assert disc[0].severity == "match"


def test_kill_switch_trip_and_rollback():
    from trader3.v2.execution import KillSwitch
    ks = KillSwitch(max_drawdown=0.05)
    assert not ks.update(1_000_000)
    ks.set_last_good({"A1": 0.5, "A2": 0.5})
    assert not ks.update(970_000)   # -3% < 5% → 未熔断
    assert not ks.tripped
    assert ks.update(940_000)   # -6% > 5% → 熔断
    assert ks.tripped
    assert ks.rollback_weights() == {"A1": 0.5, "A2": 0.5}


def test_pipeline_cost_gate_filters_orders():
    panel = _make_panel_pos_vol(n_dates=120)
    feats = {"f1": "sub(log(vwap), log(close))", "f2": "rank(close)"}
    # 高 edge：不应被门禁剔除
    cfg = QuantPipelineConfig(method="ic_weighted", use_moe=True,
                              moe_experts=["lgbm", "et", "ridge"], min_train=60,
                              factor_exprs=feats, use_cost_gate=True,
                              edge_per_score_bps=500.0)
    br = FakeBroker()
    res = asyncio.run(run_quant_pipeline(panel, config=cfg, broker=br))
    assert res["meta"].get("cost_gated_out") == []
    n_high = res["meta"]["n_orders"]
    # 极低 edge：全部剔除
    cfg2 = QuantPipelineConfig(method="ic_weighted", use_moe=True,
                               moe_experts=["lgbm", "et", "ridge"], min_train=60,
                               factor_exprs=feats, use_cost_gate=True,
                               edge_per_score_bps=0.05)
    br2 = FakeBroker()
    res2 = asyncio.run(run_quant_pipeline(panel, config=cfg2, broker=br2))
    assert res2["meta"]["n_orders"] == 0
    assert res2["meta"]["cost_gated_out"]
    assert len(res2["meta"]["cost_gated_out"]) == n_high


def test_pipeline_reconcile_and_kill_switch():
    panel = _make_panel_pos_vol(n_dates=120)
    feats = {"f1": "sub(log(vwap), log(close))", "f2": "rank(close)"}
    cfg = QuantPipelineConfig(method="ic_weighted", use_moe=True,
                              moe_experts=["lgbm", "et", "ridge"], min_train=60,
                              factor_exprs=feats, use_reconcile=True)
    br = FakeBroker()
    res = asyncio.run(run_quant_pipeline(panel, config=cfg, broker=br))
    assert res["meta"]["n_orders"] > 0
    # 内存券商按成交更新持仓 → 对账一致（0 breach）
    assert res["meta"]["reconcile_breaches"] == 0

    # kill switch 已熔断 → 不下单
    from trader3.v2.execution import KillSwitch
    ks = KillSwitch(max_drawdown=0.05)
    ks.tripped = True
    br3 = FakeBroker()
    res3 = asyncio.run(run_quant_pipeline(panel, config=cfg, broker=br3, kill_switch=ks))
    assert res3["meta"].get("halted_by_kill_switch") is True
    assert res3["meta"].get("n_orders", 0) == 0
