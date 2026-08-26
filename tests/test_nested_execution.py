"""嵌套执行（组合层+执行层联合优化）测试。"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from trader3.v2.execution import LiquidityEngine
from trader3.v2.nested_execution import (
    ExecutionPlan,
    NestedExecutor,
    NestedExecutorConfig,
    TradeSlice,
    _almgren_chriss_rate,
    _mv_weights,
)
from trader3.v2.panel_builder import build_panel


def _cov(n=6, seed=3):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 0.01, (n, n))
    C = m @ m.T / n + np.eye(n) * 1e-4
    return pd.DataFrame(C, index=[f"A{i}" for i in range(n)],
                        columns=[f"A{i}" for i in range(n)])


def test_mv_weights_normalized():
    alpha = pd.Series({"A0": 0.05, "A1": -0.02, "A2": 0.1})
    cov = _cov(3)
    w = _mv_weights(alpha, cov, 0.5)
    assert w.abs().sum() <= 1.0 + 1e-6
    assert (w.abs() <= 0.5 + 1e-6).all()


def test_almgren_chriss_rate_bounds():
    # 高波动/高风险厌恶 → 更快（rate 大）；低波动 → 慢
    r_hi = _almgren_chriss_rate(10.0, 0.05, 30.0, 2.0, 5)
    r_lo = _almgren_chriss_rate(10.0, 0.005, 30.0, 2.0, 5)
    assert 0.0 <= r_hi <= 1.0 and 0.0 <= r_lo <= 1.0
    assert r_hi >= r_lo


def test_nested_solve_impact_adjusts_weights():
    n = 6
    # A0 预期收益低（被冲击成本反超），其余较高
    alpha = pd.Series([0.02] + [0.08] * (n - 1), index=[f"A{i}" for i in range(n)])
    cov = _cov(n, seed=7)
    prices = {f"A{i}": 100.0 + i for i in range(n)}
    # 给 A0 极小 ADV → 冲击极大（slip≈30bp > A0 的 20bp）
    adv = {f"A{i}": 1e6 for i in range(n)}
    adv["A0"] = 1e2
    cur = pd.Series(0.0, index=alpha.index)
    ne = NestedExecutor(NestedExecutorConfig(horizon=5))
    plan = ne.solve(alpha, cov, cur, 1_000_000.0, prices, adv)
    assert isinstance(plan, ExecutionPlan)
    assert plan.target_weights.abs().sum() <= 1.0 + 1e-6
    assert len(plan.schedule) > 0
    # 冲击最大的 A0 在可执行权重中占比应低于原始目标占比
    raw_share = plan.raw_target_weights["A0"] / plan.raw_target_weights.abs().sum()
    tgt_share = plan.target_weights["A0"] / plan.target_weights.abs().sum()
    assert tgt_share <= raw_share + 1e-9
    assert plan.expected_shortfall >= 0.0


def test_nested_slower_for_high_impact_asset():
    # 同一 alpha，冲击大的标的执行速率应更慢（切片更分散）
    n = 4
    alpha = pd.Series(0.03, index=[f"A{i}" for i in range(n)])
    cov = _cov(n, seed=1)
    prices = {f"A{i}": 100.0 for i in range(n)}
    adv_low = {f"A{i}": 1e6 for i in range(n)}
    adv_high = {f"A{i}": 1e3 for i in range(n)}  # 全部高冲击
    cur = pd.Series(0.0, index=alpha.index)
    ne = NestedExecutor(NestedExecutorConfig(horizon=5))
    p_low = ne.solve(alpha, cov, cur, 1e6, prices, adv_low)
    p_high = ne.solve(alpha, cov, cur, 1e6, prices, adv_high)
    # 高冲击下每期切片占比更小（交易更慢）—— 用 schedule 首期占标的总量衡量
    def first_period_frac(plan):
        total = sum(s.qty for s in plan.schedule)
        first = sum(s.qty for s in plan.schedule if s.period == 0)
        return first / total if total else 0.0
    assert first_period_frac(p_high) <= first_period_frac(p_low) + 1e-9


def test_pipeline_nested_execution_flag():
    from trader3.v2.live.broker_base import (
        Account, BrokerBase, MarketData, Order, OrderSide, OrderStatus, Position,
    )
    from trader3.v2.quant_pipeline import QuantPipelineConfig, run_quant_pipeline

    class Stub(BrokerBase):
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

        async def cancel_order(self, c):
            return True

        async def get_order(self, c):
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

    panel = build_panel([f"A{i}" for i in range(6)], lookback_days=140)
    cfg = QuantPipelineConfig(method="ic_weighted", use_moe=True,
                              moe_experts=["lgbm", "et", "ridge"], min_train=60,
                              factor_exprs={"f1": "sub(log(vwap), log(close))",
                                             "f2": "rank(close)"},
                              use_nested_execution=True, nested_horizon=5)
    res = asyncio.run(run_quant_pipeline(panel, config=cfg, broker=Stub()))
    assert "execution_plan" in res["meta"]
    assert res["meta"]["n_orders"] > 0
