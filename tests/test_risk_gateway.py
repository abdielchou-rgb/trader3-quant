"""
E3：前置硬风控网关（Pre-Trade Risk Engine）— 回归测试
痛点：风控写在策略内部，策略异常/死循环时风控失效。
契约：
  1. 独立中间件：check(intent, snapshot) → Allow / Deny(原因)；策略层零依赖
  2. 单笔限额：qty×price > max_order_value → 拒
  3. 集中度：成交后单票权重 > max_position_weight → 拒
  4. 自成交：账户已有该票反向挂单意图（本批次内）→ 拒
  5. OTR（撤单/成交比）：窗口内 orders>0 且 fills/orders < min_ratio → 熔断限开新单
  6. 全局回撤熔断：KillSwitch tripped → 拒一切新开仓（平仓放行）
  7. 急停按钮：panic() 后拒一切委托（包括平仓），unpanic 恢复
  8. 幂等单号：重复 client_order_id → 拒（防网络重试重复开仓）
  9. 确定性单号生成器：trading_day+strategy_id+seq 唯一且可复现
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.risk.gateway import (  # noqa: E402
    DenyReason,
    IdempotentOrderIdGen,
    PreTradeRiskGateway,
)
from trader3.runtime.events import OrderIntent  # noqa: E402
from trader3.runtime.strategy import AccountSnapshot  # noqa: E402
from trader3.v2.execution import KillSwitch  # noqa: E402


def _intent(cid="c1", symbol="600519.SH", side="buy", qty=100, price=100.0):
    return OrderIntent(client_order_id=cid, symbol=symbol, side=side,
                       qty=qty, price=price)


def _acct(equity=1_000_000.0, cash=1_000_000.0, positions=None):
    return AccountSnapshot(cash=cash, equity=equity, positions=positions or {})


def test_single_order_value_cap():
    gw = PreTradeRiskGateway(max_order_value=50_000.0)
    ok, why = gw.check(_intent(cid="a1", qty=100, price=400.0), _acct())   # 4 万 < 5 万
    assert ok and why is None
    deny, dwhy = gw.check(_intent(cid="a2", qty=100, price=600.0), _acct())  # 6 万 > 5 万
    assert not deny and dwhy == DenyReason.ORDER_VALUE


def test_concentration_cap():
    """单笔增量名义/权益 > 上限 → 拒（增量口径，保守不低估）。"""
    gw = PreTradeRiskGateway(max_position_weight=0.05)
    snap = _acct(equity=1_000_000.0, positions={"600519.SH": 600_000})
    # 10 手 × 100 元 = 1 万 = 1% < 5% → 放行
    ok, _ = gw.check(_intent(cid="c1", qty=100, price=100.0), snap)
    assert ok
    # 60 手 × 100 元 = 6 万 = 6% > 5% → 拒
    deny, why = gw.check(_intent(cid="c2", qty=600, price=100.0), snap)
    assert not deny and why == DenyReason.CONCENTRATION


def test_self_match_prevention():
    """同批先 buy 后 sell 同票、卖出量超持仓（对敲特征）→ 第二笔拒。"""
    gw = PreTradeRiskGateway()
    snap = _acct(positions={"600519.SH": 100})
    buy = _intent(cid="b1", symbol="600519.SH", side="buy", qty=100)
    sell = _intent(cid="s1", symbol="600519.SH", side="sell", qty=200)
    b_ok, _ = gw.check(buy, snap)
    s_ok, why = gw.check(sell, snap, batch_intents=[buy])
    assert b_ok
    assert not s_ok and why == DenyReason.SELF_MATCH
    # 卖出量不超持仓 = 正常平仓 → 放行
    close_ok, _ = gw.check(
        _intent(cid="s2", side="sell", qty=100), snap, batch_intents=[buy])
    assert close_ok


def test_otr_circuit_breaker():
    """窗口内大量拒单零成交 → OTR 熔断，拒开新仓、放行平仓。"""
    gw = PreTradeRiskGateway(otr_window=10, min_fill_ratio=0.5)
    snap = _acct(positions={"600519.SH": 500})
    for i in range(10):
        ok, _ = gw.check(_intent(cid=f"o{i}", qty=100), snap)
        assert ok
        gw.record_submitted(f"o{i}")
        gw.record_reject(f"o{i}")  # 全部被"交易所"拒绝（零成交）
    denied, why = gw.check(_intent(cid="new1", side="buy", qty=100), snap)
    assert not denied and why == DenyReason.OTR_TRIPPED
    close_ok, _ = gw.check(_intent(cid="close1", side="sell", qty=100), snap)
    assert close_ok


def test_drawdown_kill_switch_blocks_new_buys():
    gw = PreTradeRiskGateway(kill_switch=KillSwitch(max_drawdown=0.05))
    gw.kill_switch.update(1_000_000.0)
    tripped = gw.kill_switch.update(940_000.0)   # -6% 熔断
    assert tripped
    snap = _acct(equity=940_000.0, positions={"600519.SH": 500})
    buy_deny, why = gw.check(_intent(cid="b", side="buy"), snap)
    assert not buy_deny and why == DenyReason.KILL_SWITCH
    sell_ok, _ = gw.check(_intent(cid="s", side="sell"), snap)  # 减仓放行
    assert sell_ok


def test_panic_button_blocks_everything():
    gw = PreTradeRiskGateway()
    snap = _acct(positions={"600519.SH": 500})
    gw.panic()
    b, w = gw.check(_intent(side="buy"), snap)
    s, ws = gw.check(_intent(side="sell"), snap)
    assert not b and w == DenyReason.PANIC
    assert not s and ws == DenyReason.PANIC  # 急停下平仓也停（人工接管）
    gw.unpanic()
    ok, _ = gw.check(_intent(cid="after", side="buy"), _acct())
    assert ok


def test_idempotent_client_order_id():
    gw = PreTradeRiskGateway()
    ok1, _ = gw.check(_intent(cid="dup-1"), _acct())
    assert ok1
    gw.record_submitted("dup-1")
    dup, why = gw.check(_intent(cid="dup-1"), _acct())
    assert not dup and why == DenyReason.DUPLICATE_ID


def test_deterministic_order_id_generator():
    g1 = IdempotentOrderIdGen(trading_day="2026-09-08", strategy_id="alpha1")
    g2 = IdempotentOrderIdGen(trading_day="2026-09-08", strategy_id="alpha1")
    ids1 = [g1.next() for _ in range(5)]
    ids2 = [g2.next() for _ in range(5)]
    assert ids1 == ids2            # 同参数同序列（可复现）
    assert len(set(ids1)) == 5     # 唯一
    assert ids1[0] == "alpha1-2026-09-08-000001"
    # 不同交易日不冲突
    g3 = IdempotentOrderIdGen(trading_day="2026-09-09", strategy_id="alpha1")
    assert g3.next() not in ids1
