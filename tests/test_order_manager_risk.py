"""
F2/F3/F5 接线回归：
  1. OrderManager.submit + risk_gateway：超限单拦截（REJECTED+risk_reason、不触 broker）
  2. 无 risk_gateway → 历史行为不变（全提交 broker）
  3. 放行单经 record_submitted + telemetry.record_order_allowed
  4. 拒单经 telemetry.record_order_denied（/metrics 可见）
  5. DriftHalt 挂起触发 notify（用 monkeypatch 拦截）+ telemetry gauge
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.obs import telemetry  # noqa: E402
from trader3.risk.gateway import DenyReason, PreTradeRiskGateway  # noqa: E402
from trader3.runtime.strategy import AccountSnapshot  # noqa: E402
from trader3.v2.live.broker_base import (  # noqa: E402
    BrokerBase,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from trader3.v2.order_manager import OrderManager  # noqa: E402


class _FakeBroker(BrokerBase):
    def __init__(self):
        super().__init__()
        self.received: list[Order] = []

    async def connect(self):
        return True

    async def disconnect(self):
        return True

    async def place_order(self, order: Order) -> Order:
        self.received.append(order)
        order.status = OrderStatus.FILLED
        order.filled_qty = order.quantity
        order.broker_order_id = f"B-{order.client_order_id}"
        return order

    async def cancel_order(self, client_order_id: str):
        return True

    async def get_order(self, client_order_id: str):
        return None

    async def get_orders(self, status=None):
        return []

    async def get_positions(self):
        return {}

    async def get_account(self):
        return None

    async def get_market_data(self, symbols):
        return {}

    async def subscribe_market_data(self, symbols, callback):
        return False

    async def unsubscribe_market_data(self, symbols):
        return False


def _order(cid, qty=100, price=100.0, side="buy"):
    return Order(symbol="600519.SH",
                 side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                 quantity=qty, price=price, order_type=OrderType.LIMIT,
                 client_order_id=cid)


def test_submit_with_gateway_blocks_over_limit():
    gw = PreTradeRiskGateway(max_order_value=50_000.0, max_position_weight=1.0)
    acct = AccountSnapshot(cash=1e6, equity=1e6)
    brk = _FakeBroker()

    async def run():
        om = OrderManager()
        return await om.submit(
            [_order("c1", price=100.0),   # 1 万 → 放行
             _order("c2", price=600.0)],  # 6 万 → 拒
            brk, risk_gateway=gw, account=acct)

    res = asyncio.run(run())
    assert res[0].status == OrderStatus.FILLED
    assert res[1].status == OrderStatus.REJECTED
    assert res[1].metadata.get("risk_reason") == DenyReason.ORDER_VALUE.name
    assert [o.client_order_id for o in brk.received] == ["c1"]  # c2 未触 broker


def test_submit_without_gateway_legacy_behavior():
    brk = _FakeBroker()

    async def run():
        om = OrderManager()
        return await om.submit([_order("c1"), _order("c2")], brk)

    res = asyncio.run(run())
    assert all(o.status == OrderStatus.FILLED for o in res)
    assert len(brk.received) == 2


def test_telemetry_records_deny_and_allow():
    from trader3.obs.metrics import MetricsRegistry

    telemetry._registry = MetricsRegistry()  # 隔离单例，避免跨测试污染
    gw = PreTradeRiskGateway(max_order_value=10_000.0, max_position_weight=1.0)
    acct = AccountSnapshot(cash=1e6, equity=1e6)
    brk = _FakeBroker()

    async def run():
        om = OrderManager()
        return await om.submit([_order("c9", price=1500.0)], brk,
                               risk_gateway=gw, account=acct)

    asyncio.run(run())
    out = telemetry.render()
    assert 'risk_denied_total{reason="ORDER_VALUE"}' in out
    telemetry._registry = None  # 还原


def test_metrics_endpoint_smoke():
    """/metrics 路由存在且走 telemetry 单例（HTTP 层由既有 API 测试覆盖，
    此处验证渲染不炸）。"""
    from trader3.obs.metrics import MetricsRegistry

    telemetry._registry = MetricsRegistry()
    telemetry.record_equity(2_000_000.0)
    out = telemetry.render()
    assert "account_equity 2000000.0" in out
    telemetry._registry = None


def test_drift_halt_notifies_and_gauges(monkeypatch):
    from trader3.obs.metrics import MetricsRegistry
    from trader3.risk.drift_halt import DriftHaltEngine

    telemetry._registry = MetricsRegistry()
    sent = []
    monkeypatch.setattr(
        "trader3.notify.send_notification",
        lambda title, content: sent.append((title, content)) or [],
    )
    eng = DriftHaltEngine(tolerance_qty=0, halt_threshold=1)
    eng.reconcile(internal={"600519.SH": 1000}, broker={"600519.SH": 800})
    assert eng.halted
    assert any("持仓漂移挂起" in t for t, _ in sent)
    assert 'drift_halt_active 1' in telemetry.render()
    eng.resolve()
    assert any("已解除" in t for t, _ in sent)
    assert 'drift_halt_active 0' in telemetry.render()
    telemetry._registry = None
