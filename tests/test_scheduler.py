"""统一调度器测试。"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from trader3.v2.live.broker_base import Account, BrokerBase, MarketData, OrderSide, OrderStatus, Position
from trader3.v2.scheduler import (
    QuantScheduler,
    SchedulerConfig,
    run_scheduler_step,
)


class MiniBroker(BrokerBase):
    """最小券商桩：下单即按市价全量成交。"""

    def __init__(self, equity=1_000_000, prices=None):
        super().__init__()
        self._equity = equity
        self._prices = prices or {f"A{i}": 100.0 + i for i in range(8)}
        self._positions: dict[str, Position] = {}
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

    async def cancel_order(self, cid):
        return True

    async def get_order(self, cid):
        return None

    async def get_orders(self, status=None):
        return []

    async def get_positions(self):
        return dict(self._positions)

    async def get_account(self):
        return Account(account_id="T", cash=self._equity, equity=self._equity,
                       buying_power=self._equity)

    async def get_market_data(self, symbols):
        return {s: MarketData(symbol=s, price=self._prices.get(s, 100.0)) for s in symbols}

    async def subscribe_market_data(self, symbols, cb):
        return True

    async def unsubscribe_market_data(self, symbols):
        return True


def _panel(n_dates=140, n_assets=6, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    assets = [f"A{i}" for i in range(n_assets)]
    fields = ["open", "high", "low", "close", "volume", "vwap", "amount"]
    cols = pd.MultiIndex.from_product([assets, fields])
    df = pd.DataFrame(rng.normal(size=(n_dates, len(cols))), index=dates, columns=cols)
    for a in assets:
        w = np.cumsum(rng.normal(0, 1, n_dates))
        df[(a, "close")] = w + 100
        df[(a, "vwap")] = df[(a, "close")]
        df[(a, "volume")] = abs(rng.normal(1e5, 2e4, n_dates)) + 1e4
    return df


def test_should_run_schedule_logic(tmp_path):
    cfg = SchedulerConfig(state_path=str(tmp_path / "s.json"))
    s = QuantScheduler(cfg)
    # 工作日盘中
    wed = datetime(2024, 1, 3, 10, 0)
    assert s.should_run(wed)
    # 周末
    sat = datetime(2024, 1, 6, 10, 0)
    assert not s.should_run(sat)
    # 非交易时段（夜间）
    night = datetime(2024, 1, 3, 22, 0)
    assert not s.should_run(night)
    # 同一天已运行 → 不重复
    s.state["last_run"] = wed.isoformat()
    assert not s.should_run(wed)
    # 隔日 → 可运行
    tue = datetime(2024, 1, 2, 10, 0)
    assert s.should_run(tue)


def test_scheduler_step_runs_quant_and_persists(tmp_path):
    panel = _panel()
    from trader3.v2.quant_pipeline import QuantPipelineConfig
    cfg = SchedulerConfig(state_path=str(tmp_path / "s.json"))
    qcfg = QuantPipelineConfig(method="ic_weighted", use_moe=True,
                               moe_experts=["lgbm", "et", "ridge"], min_train=60,
                               factor_exprs={"f1": "sub(log(vwap), log(close))",
                                             "f2": "rank(close)"})
    run = run_scheduler_step(panel=panel, broker=MiniBroker(),
                             kind="manual", config=cfg, quant_config=qcfg)
    assert not run.quant["weights"].empty
    assert run.state["run_count"] == 1
    assert (tmp_path / "s.json").exists()
    # 再跑一次（不同实例读同一状态文件）应累计
    run2 = run_scheduler_step(panel=panel, broker=MiniBroker(),
                              kind="manual", config=cfg, quant_config=qcfg)
    assert run2.state["run_count"] == 2
    assert run2.state["last_weights"]


def test_scheduler_factor_factory_refresh(tmp_path):
    panel = _panel(160)
    cfg = SchedulerConfig(state_path=str(tmp_path / "s.json"),
                          enable_factor_factory_refresh=True,
                          factor_factory_every_n=1,
                          factor_factory_path=str(tmp_path / "registry.json"))
    run = run_scheduler_step(panel=panel, kind="manual", config=cfg)
    assert run.factory_refreshed is True
    # 运行态中记录工厂刷新标记
    assert run.state["history"][-1]["factory_refreshed"] is True


def test_scheduler_legacy_off_by_default(tmp_path):
    panel = _panel()
    cfg = SchedulerConfig(state_path=str(tmp_path / "s.json"))
    run = run_scheduler_step(panel=panel, kind="manual", config=cfg)
    assert run.legacy is None


def test_scheduler_auto_panel(tmp_path):
    from trader3.v2.quant_pipeline import QuantPipelineConfig
    cfg = SchedulerConfig(state_path=str(tmp_path / "s.json"), auto_panel=True,
                          universe=[f"A{i}" for i in range(6)], lookback_days=140)
    qcfg = QuantPipelineConfig(method="ic_weighted", use_moe=True,
                               moe_experts=["lgbm", "et", "ridge"], min_train=60,
                               factor_exprs={"f1": "sub(log(vwap), log(close))",
                                             "f2": "rank(close)"})
    run = run_scheduler_step(kind="manual", config=cfg, quant_config=qcfg)
    assert not run.quant["weights"].empty
    assert run.state["run_count"] == 1

