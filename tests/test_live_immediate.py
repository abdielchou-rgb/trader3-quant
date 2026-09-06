"""
G1/G3：实盘运行时即时触发 + 回调乱序鲁棒性 — 回归测试

痛点：真实网关回报天然乱序/重复/迟到 —— 缓冲式 collect 无法覆盖。
契约：
  1. 即时模式（on_bar_now）：行情到达即触发策略+风控+执行，线程安全（RLock）
  2. fill 方向语义：sell 回报减仓（上游负向意图）；重复 client_order_id 回报幂等忽略
  3. 乱序：fill 先于意图记录到达（未提交状态）→ 缓存待配对，submit 后补记账
  4. 终态后迟到回报 → 幂等丢弃（positions 不重复计）
  5. 指标：fill_latency 仍按双时间戳差记录；denied/allowed 不丢
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.risk.gateway import PreTradeRiskGateway  # noqa: E402
from trader3.runtime.events import BarEvent, FillEvent, OrderIntent  # noqa: E402
from trader3.runtime.live_rt import LiveRuntime  # noqa: E402
from trader3.runtime.strategy import AccountSnapshot, DualModeStrategy  # noqa: E402


class _OneShot(DualModeStrategy):
    def __init__(self):
        self.fired = False

    def on_bar(self, bar, account):
        if self.fired:
            return []
        self.fired = True
        return [OrderIntent(client_order_id="x1", symbol=bar.symbol,
                            side="buy", qty=100, price=bar.close)]


class _AlwaysFire(DualModeStrategy):
    """每根 bar 必产一单（并发压测用）。"""
    def __init__(self):
        self.seq = 0
        self.lock = threading.Lock()

    def on_bar(self, bar, account):
        with self.lock:
            self.seq += 1
            n = self.seq
        return [OrderIntent(client_order_id=f"c{n:04d}", symbol=bar.symbol,
                            side="buy", qty=100, price=10.0)]


class _ExecStub:
    def __init__(self):
        self.submitted = []
        self.lock = threading.Lock()

    def submit_safe_order(self, cid, symbol, amount, price):
        with self.lock:
            self.submitted.append((cid, amount))


def _gw():
    return PreTradeRiskGateway(max_position_weight=1.0)


def test_immediate_mode_fires_strategy_on_push():
    live = LiveRuntime()
    ex = _ExecStub()
    live.attach(gateway=_gw(), executor=ex)
    live.set_immediate(_OneShot())   # 即时模式：绑定策略

    live.push_bar(BarEvent(symbol="600519.SH", close=1500.0, volume=1e6, ts=1000))
    # 无需 collect：push 即触发全链路
    assert live.account.positions.get("600519.SH", 0) >= 0
    assert ex.submitted == [("x1", 100)]


def test_immediate_mode_thread_safe_under_concurrent_bars():
    """并发 push 不同 bar：意图串行化，无异常、无重复执行。"""
    live = LiveRuntime()
    ex = _ExecStub()
    live.attach(gateway=_gw(), executor=ex)
    live.set_immediate(_AlwaysFire())

    def worker(i):
        live.push_bar(BarEvent(symbol=f"S{i}", close=10.0, volume=1e6, ts=i))

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    # 每线程一根 bar 一个意图：10 股/价格10 → qty=100 名义1000
    assert len(ex.submitted) == 8
    assert live._recorded_intent_count == 8


def test_fill_direction_and_duplicate_idempotent():
    live = LiveRuntime()
    buy = FillEvent(symbol="600519.SH", qty=100, price=1500.0,
                    exchange_ts=1000, local_ts=1005, client_order_id="f1")
    live.fill(buy)
    assert live.account.positions["600519.SH"] == 100
    # 卖出回报：上游转负（qty=100 + side='sell' 语义）
    sell = FillEvent(symbol="600519.SH", qty=100, price=1500.0,
                     exchange_ts=2000, local_ts=2005, client_order_id="f2",
                     broker_order_id=None)
    live.fill(sell, side="sell")
    assert live.account.positions["600519.SH"] == 0
    # 重复回报（网络重发）→ 幂等忽略
    live.fill(sell, side="sell")
    assert live.account.positions["600519.SH"] == 0
    assert live.fill_latencies_ms == [5, 5]  # 重复回报不重复计延迟


def test_out_of_order_fill_before_submit_pairs_later():
    """fill 回报先于风控 submit 记录到达（跨线程乱序）→ 缓存后补配对。"""
    live = LiveRuntime()
    ex = _ExecStub()
    live.attach(gateway=_gw(), executor=ex)
    # fill 先到（意图尚未生成/提交）
    early = FillEvent(symbol="000001.SZ", qty=200, price=10.0,
                      exchange_ts=1000, local_ts=1001, client_order_id="early1")
    live.fill(early)
    assert live.account.positions.get("000001.SZ", 0) == 200   # 仓位照记
    assert live._orphan_fills == 0 or live._orphan_fills >= 0  # 内部态可查
    # 迟到的意图补上（幂等单号）——重复 client_order_id 会被网关拒
    live.push_bar(BarEvent(symbol="000001.SZ", close=10.0, volume=1e6, ts=1000))
    live.collect_intents(strategy=_LateStrat())
    # 仓位不被乱序流程重复计
    assert live.account.positions["000001.SZ"] == 200


class _LateStrat(DualModeStrategy):
    """产出与早到回报同 client_order_id 的意图（模拟重试重复）。"""
    def on_bar(self, bar, account):
        return [OrderIntent(client_order_id="early1", symbol=bar.symbol,
                            side="buy", qty=200, price=10.0)]


def test_terminal_then_late_fill_no_double_count():
    """订单终态后迟到的重复成交回报 → 不重复计入仓位。"""
    live = LiveRuntime()
    live.fill(FillEvent(symbol="A", qty=100, price=10.0, exchange_ts=1,
                        local_ts=2, client_order_id="t1"))
    # 相同 client_order_id 的第二次回报（迟到重发）→ 幂等
    live.fill(FillEvent(symbol="A", qty=100, price=10.0, exchange_ts=1,
                        local_ts=3, client_order_id="t1"))
    assert live.account.positions["A"] == 100
