"""
E1+E2：统一交易事件流 + 双模同构运行时 — 回归测试

Train-Serving Skew 痛点：回测与实盘两套抽象，策略上线要重写。
契约：
  1. 事件模型：BarEvent/SignalEvent/OrderIntent/FillEvent 不可变 dataclass；
     FillEvent 带 exchange_ts（交易所时间）与 local_ts（本地接收）双时间戳
  2. 时间戳守卫：回放数据 exchange_ts > local_ts（未来事件）→ 拒绝推进
  3. 双模同构：同一策略（on_bar 纯函数）在 ReplayRuntime（历史流）与
     LiveRuntime（事件回调）下产出**逐单一致**的 OrderIntent 序列
  4. 策略只面向事件+快照，不触碰任何 broker SDK —— Live 与 Replay 的
     差异仅在事件来源，不在策略代码
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.runtime.events import (  # noqa: E402
    BarEvent,
    FillEvent,
    OrderIntent,
)
from trader3.runtime.replay import ReplayRuntime, TimestampGuard  # noqa: E402
from trader3.runtime.strategy import (  # noqa: E402
    AccountSnapshot,
    DualModeStrategy,
)


def _bars():
    return [
        BarEvent(symbol="600519.SH", close=1500.0, volume=1e6, ts=1000),
        BarEvent(symbol="600519.SH", close=1515.0, volume=1.2e6, ts=2000),
        BarEvent(symbol="600519.SH", close=1490.0, volume=0.9e6, ts=3000),
        BarEvent(symbol="600519.SH", close=1520.0, volume=1.1e6, ts=4000),
    ]


def test_event_models_immutable_and_typed():
    b = BarEvent(symbol="600519.SH", close=1500.0, volume=1e6, ts=1000)
    with pytest.raises((AttributeError, TypeError)):  # frozen dataclass
        b.close = 1600.0  # type: ignore[misc]
    f = FillEvent(symbol="600519.SH", qty=100, price=1500.0,
                  exchange_ts=1000, local_ts=1005, client_order_id="c1")
    assert f.exchange_ts <= f.local_ts  # 交易所时间不晚于本地接收


def test_timestamp_guard_rejects_future_data():
    """回放守卫：local_ts < exchange_ts（事件未到达就被消费=前视）→ 拒绝。"""
    g = TimestampGuard()
    assert g.check(exchange_ts=1000, local_ts=1005) is True
    assert g.check(exchange_ts=1010, local_ts=1005) is False  # 未来事件


def test_replay_runtime_drives_strategy_deterministically():
    """回放运行时：历史流推进，策略产出确定性意图序列。"""

    class Momentum(DualModeStrategy):
        def __init__(self):
            self.prev_close: dict[str, float] = {}
            self.seq = 0

        def on_bar(self, bar: BarEvent, account: AccountSnapshot) -> list[OrderIntent]:
            prev = self.prev_close.get(bar.symbol)
            self.prev_close[bar.symbol] = bar.close
            if prev is None:
                return []
            self.seq += 1
            side = "buy" if bar.close > prev else "sell"
            return [OrderIntent(
                client_order_id=f"t3-{self.seq:04d}",
                symbol=bar.symbol, side=side, qty=100, price=bar.close,
            )]

    rt = ReplayRuntime(local_ts_lag=5)
    strat = Momentum()
    intents = rt.run(strat, _bars())
    # 4 根 bar → 首根无前值，后 3 根各产 1 意图
    assert len(intents) == 3
    assert intents[0].side == "buy"    # 1500→1515 涨
    assert intents[1].side == "sell"   # 1515→1490 跌
    assert intents[2].side == "buy"    # 1490→1520 涨
    # 确定性：同数据重放 → 逐单一致
    rt2 = ReplayRuntime(local_ts_lag=5)
    assert [i.client_order_id for i in rt2.run(Momentum(), _bars())] == \
           [i.client_order_id for i in intents]


def test_live_runtime_same_strategy_same_intents():
    """实盘运行时：事件回调驱动，同一策略类产出与回放一致的意图序列。
    同构性 = 策略代码零改动，仅事件来源不同。"""
    from trader3.runtime.live_rt import LiveRuntime

    class Momentum(DualModeStrategy):
        def __init__(self):
            self.prev_close: dict[str, float] = {}
            self.seq = 0

        def on_bar(self, bar: BarEvent, account: AccountSnapshot) -> list[OrderIntent]:
            prev = self.prev_close.get(bar.symbol)
            self.prev_close[bar.symbol] = bar.close
            if prev is None:
                return []
            self.seq += 1
            side = "buy" if bar.close > prev else "sell"
            return [OrderIntent(
                client_order_id=f"t3-{self.seq:04d}",
                symbol=bar.symbol, side=side, qty=100, price=bar.close,
            )]

    live = LiveRuntime()
    # 模拟网关推送（正常时序：exchange_ts <= local_ts）
    for b in _bars():
        live.push_bar(b)
    intents = live.collect_intents(strategy=Momentum())
    assert len(intents) == 3
    assert [i.side for i in intents] == ["buy", "sell", "buy"]  # 与回放一致


def test_replay_guard_blocks_lookahead_stream():
    """回放流中混入 local_ts < exchange_ts 的越权事件 → 整批拒绝并标记。"""
    from trader3.runtime.replay import LookaheadError

    bad_bars = _bars()
    # 第 3 根 bar 的本地接收时间早于交易所时间（= 事件未发生就被消费）
    bad_bars[2] = BarEvent(symbol="600519.SH", close=1490.0, volume=0.9e6,
                           ts=3000, local_ts=2500)
    rt = ReplayRuntime(local_ts_lag=5)
    with pytest.raises(LookaheadError):
        rt.run(_Momentum(), bad_bars)


@dataclass
class _Momentum(DualModeStrategy):
    prev_close: dict = None

    def on_bar(self, bar: BarEvent, account: AccountSnapshot) -> list[OrderIntent]:
        prev = (self.prev_close or {}).get(bar.symbol)
        self.prev_close = {**(self.prev_close or {}), bar.symbol: bar.close}
        if prev is None:
            return []
        return [OrderIntent(
            client_order_id=f"c-{bar.ts}",
            symbol=bar.symbol,
            side="buy" if bar.close > prev else "sell",
            qty=100, price=bar.close,
        )]
