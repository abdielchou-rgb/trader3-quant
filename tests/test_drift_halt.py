"""
E4：持仓漂移对账机器人（Reconciliation Engine）— 回归测试
契约：
  1. 本地 vs 券商持仓差异超容差 → DriftHalt 生效：开仓意图被风控网关拒
  2. 漂移恢复（人工/自动重同步后）→ resolve() 解除挂起
  3. 差异在容差内 → 不挂起（防误报）
  4. 挂起期间平仓放行（降风险优先）
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.risk.drift_halt import DriftHaltEngine  # noqa: E402
from trader3.risk.gateway import DenyReason, PreTradeRiskGateway  # noqa: E402
from trader3.runtime.events import OrderIntent  # noqa: E402
from trader3.runtime.strategy import AccountSnapshot  # noqa: E402


def _intent(cid, side="buy", qty=100, symbol="600519.SH"):
    return OrderIntent(client_order_id=cid, symbol=symbol, side=side,
                       qty=qty, price=100.0)


def test_drift_triggers_halt_and_gates_new_buys():
    eng = DriftHaltEngine(tolerance_qty=0, halt_threshold=0)
    gw = PreTradeRiskGateway()
    eng.bind_gateway(gw)
    snap = AccountSnapshot(cash=1e6, equity=1e6, positions={"600519.SH": 1000})

    # 对账：本地 1000 vs 券商 800 → 漂移 200
    report = eng.reconcile(
        internal={"600519.SH": 1000}, broker={"600519.SH": 800})
    assert report["drifted"] is True
    assert gw.is_halted is True or eng.halted  # 引擎挂起

    # 开仓被拒（DRIFT）
    ok, why = gw.check(_intent("n1"), snap)
    assert not ok and why == DenyReason.DRIFT_HALT
    # 平仓放行
    ok_close, _ = gw.check(_intent("c1", side="sell"), snap)
    assert ok_close


def test_drift_resolved_unhalts():
    eng = DriftHaltEngine(tolerance_qty=0, halt_threshold=0)
    gw = PreTradeRiskGateway()
    eng.bind_gateway(gw)
    eng.reconcile(internal={"600519.SH": 1000}, broker={"600519.SH": 800})
    assert eng.halted
    # 人工重同步后标记解决
    eng.resolve()
    assert not eng.halted
    snap = AccountSnapshot(cash=1e6, equity=1e6)
    ok, _ = gw.check(_intent("after"), snap)
    assert ok


def test_within_tolerance_no_halt():
    eng = DriftHaltEngine(tolerance_qty=100, halt_threshold=1)
    eng.reconcile(internal={"600519.SH": 1000}, broker={"600519.SH": 1050})
    assert not eng.halted  # 50 股差异 < 容差 100


def test_halt_threshold_multiple_breaches():
    """漂移标的数达到阈值才挂起（单票小幅差异不挂）。"""
    eng = DriftHaltEngine(tolerance_qty=10, halt_threshold=2)
    # 2 个标的小幅漂移（各超容差）→ 达阈值 → 挂起
    eng.reconcile(internal={"A": 1000, "B": 2000},
                  broker={"A": 980, "B": 1970})
    assert eng.halted
