"""
G4：集中度风控存量口径升级 — 回归测试
痛点：既有实现只有"增量口径"（本单名义/权益），不校验已持有市值 ——
逐步加仓可绕过集中度上限（每单都 <10%，累计 50%）。
契约：
  1. AccountSnapshot 可选 position_values（symbol→市值）；提供时用**存量口径**：
     (已持市值 + 本单名义) / equity > max_position_weight → 拒
  2. position_values 缺失 → 回退增量口径（历史行为，诚实降级）
  3. 卖出方向不触发集中度（减仓）
  4. broker Order 适配（check_broker_order）同享存量口径
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.risk.gateway import DenyReason, PreTradeRiskGateway  # noqa: E402
from trader3.runtime.events import OrderIntent  # noqa: E402
from trader3.runtime.strategy import AccountSnapshot  # noqa: E402


def _snap(equity=1_000_000.0, position_values=None):
    return AccountSnapshot(cash=equity, equity=equity,
                           positions={s: 1 for s in (position_values or {})},
                           position_values=position_values or {})


def _intent(cid="c1", qty=100, price=100.0, side="buy", symbol="600519.SH"):
    return OrderIntent(client_order_id=cid, symbol=symbol, side=side,
                       qty=qty, price=price)


def test_cumulative_concentration_blocked_with_position_values():
    """已持 8% + 新增 5% = 13% > 10% → 拒（存量口径拦截累计绕过）。"""
    gw = PreTradeRiskGateway(max_position_weight=0.10)
    snap = _snap(equity=1_000_000.0,
                 position_values={"600519.SH": 80_000.0})
    ok, why = gw.check(_intent(qty=500, price=100.0), snap)  # 5 万 → 13%
    assert not ok and why == DenyReason.CONCENTRATION


def test_within_cap_with_position_values_passes():
    gw = PreTradeRiskGateway(max_position_weight=0.10)
    snap = _snap(position_values={"600519.SH": 30_000.0})
    ok, why = gw.check(_intent(qty=500, price=100.0), snap)  # 3% + 5% = 8% < 10%
    assert ok and why is None


def test_fallback_to_incremental_when_no_values():
    """position_values 缺失 → 增量口径（历史行为）。"""
    gw = PreTradeRiskGateway(max_position_weight=0.10)
    snap = _snap()  # 无 position_values
    snap.positions = {"600519.SH": 100000}  # 持仓股数很大但无市值信息
    ok, _ = gw.check(_intent(qty=500, price=100.0), snap)  # 增量 5% < 10% → 放行
    assert ok


def test_sell_never_concentration_gated():
    gw = PreTradeRiskGateway(max_position_weight=0.01)
    snap = _snap(position_values={"600519.SH": 900_000.0})
    ok, why = gw.check(_intent(side="sell", qty=100), snap)
    assert ok  # 减仓永远放行（集中度只约束开仓）


def test_broker_order_adapter_uses_cumulative_too():
    """Order 适配链路同享存量口径。"""

    class _O:
        client_order_id = "b1"
        symbol = "600519.SH"
        side = "buy"
        quantity = 500
        price = 100.0

    gw = PreTradeRiskGateway(max_position_weight=0.10)
    snap = _snap(position_values={"600519.SH": 80_000.0})
    ok, why = gw.check_broker_order(_O(), snap)
    assert not ok and why == DenyReason.CONCENTRATION
