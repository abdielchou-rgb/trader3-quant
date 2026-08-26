"""生产加固：配置(env)、面板构建、影子券商(回测→实盘并行)。"""

from __future__ import annotations

import asyncio

import pandas as pd

from trader3.v2.config import Settings
from trader3.v2.live.broker_base import (
    Account,
    BrokerBase,
    MarketData,
    Order,
    OrderSide,
    OrderStatus,
    ShadowBroker,
)
from trader3.v2.panel_builder import FIELDS, build_panel


def test_settings_load_env():
    e = {"OPENROUTER_API_KEY": "k", "CTP_USER_ID": "u1", "PAPER_TRADE": "false"}
    s = Settings.load(e)
    assert s.openrouter_api_key == "k"
    assert s.ctp_user_id == "u1"
    assert s.paper_trade is False
    assert s.ctp_config()["ctp_user_id"] == "u1"
    assert s.has_llm() is True


def test_settings_no_key():
    s = Settings.load({"OPENROUTER_API_KEY": ""})
    assert s.has_llm() is False
    assert s.make_llm_fn() is None


def test_build_panel_synthetic():
    u = ["A0", "A1", "A2"]
    p = build_panel(u, lookback_days=120)
    assert list(p.columns.get_level_values(0).unique()) == u
    assert set(p.columns.get_level_values(1).unique()) == set(FIELDS)
    assert p.shape[1] == len(u) * len(FIELDS)
    assert p.shape[0] == 120


def test_build_panel_custom_source():
    def src(code, n, end):
        return pd.DataFrame(
            {"open": [1], "high": [2], "low": [1], "close": [1.5],
             "volume": [10], "vwap": [1.5], "amount": [15]},
            index=pd.date_range("2024-01-01", periods=1))
    p = build_panel(["X"], 1, source=src)
    assert p.shape == (1, 7)


class StubBroker(BrokerBase):
    """最小真实券商桩：下单即全成。"""

    def __init__(self):
        super().__init__()
        self._connected = True

    async def connect(self):
        return True

    async def disconnect(self):
        return True

    async def place_order(self, o):
        o.status = OrderStatus.FILLED
        o.filled_qty = o.quantity
        return o

    async def cancel_order(self, cid):
        return True

    async def get_order(self, cid):
        return None

    async def get_orders(self, s=None):
        return []

    async def get_positions(self):
        return {}

    async def get_account(self):
        return Account("A", cash=1e6, equity=1e6, buying_power=1e6)

    async def get_market_data(self, syms):
        return {s: MarketData(s, price=100.0) for s in syms}

    async def subscribe_market_data(self, syms, cb):
        return True

    async def unsubscribe_market_data(self, syms):
        return True


def test_shadow_broker_records_not_fills():
    real = StubBroker()
    sh = ShadowBroker(real)

    async def go():
        ro = await real.place_order(Order(symbol="A0", side=OrderSide.BUY, quantity=100))
        so = await sh.place_order(Order(symbol="A1", side=OrderSide.BUY, quantity=100))
        return ro, so

    ro, so = asyncio.run(go())
    assert ro.status == OrderStatus.FILLED
    assert so.metadata.get("shadow") is True
    assert so.status == OrderStatus.SUBMITTED
    assert len(sh.shadow_orders) == 1


def test_shadow_broker_with_pipeline():
    from trader3.v2.quant_pipeline import QuantPipelineConfig, run_quant_pipeline

    panel = build_panel([f"A{i}" for i in range(6)], lookback_days=140)
    sh = ShadowBroker(StubBroker())
    cfg = QuantPipelineConfig(method="ic_weighted", use_moe=True,
                              moe_experts=["lgbm", "et", "ridge"], min_train=60,
                              factor_exprs={"f1": "sub(log(vwap), log(close))",
                                             "f2": "rank(close)"})
    res = asyncio.run(run_quant_pipeline(panel, config=cfg, broker=sh))
    assert res["meta"]["n_orders"] > 0
    # 影子模式：所有订单标记 shadow，且真实券商无持仓变动
    assert all(o.metadata.get("shadow") for o in res["orders"])
    placed_real = asyncio.run(sh.wrapped.get_positions())
    assert placed_real == {}
