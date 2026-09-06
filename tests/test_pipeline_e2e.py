"""
F1：端到端同构闭环 — 回归测试
链路：strategy → OrderIntent → PreTradeRiskGateway → WAL executor → Fill 回报 → 账户回流
契约：
  1. LiveRuntime.gate_intents：风控放行的意图转发 executor；被拒的意图不触网关
  2. 放行意图经 WAL 落盘（PENDING_SUBMIT 在前），SIM 撮合回 SUBMITTED
  3. 成交回报 fill() 回流：账户 positions/cash 更新、fill_latency 指标记录
  4. 风控被拒计数进 metrics（risk_denied_total{reason=...}）
  5. 急停后整链路拒绝新单（含 executor 前的网关层）
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.risk.gateway import PreTradeRiskGateway  # noqa: E402
from trader3.runtime.events import BarEvent, FillEvent, OrderIntent  # noqa: E402
from trader3.runtime.live_rt import LiveRuntime  # noqa: E402
from trader3.runtime.strategy import DualModeStrategy  # noqa: E402


class _SimpleStrat(DualModeStrategy):
    def __init__(self, cid="s1"):
        self.cid = cid
        self.fired = False

    def on_bar(self, bar, account):
        if self.fired:
            return []
        self.fired = True
        return [OrderIntent(client_order_id=self.cid, symbol=bar.symbol,
                            side="buy", qty=100, price=bar.close)]


def _wal_stub():
    class _Exec:
        def __init__(self):
            self.submitted = []
            self.wal = []

        def submit_safe_order(self, cid, symbol, amount, price):
            self.submitted.append(cid)
            self.wal.append({"state": "PENDING_SUBMIT"})
            self.wal.append({"state": "SUBMITTED", "broker_order_id": f"B-{cid}"})
            return {"client_order_id": cid, "state": "SUBMITTED",
                    "broker_order_id": f"B-{cid}"}

    return _Exec()


def test_pipeline_allowed_intent_reaches_executor_and_fills(tmp_path):
    live = LiveRuntime()
    # 放松集中度：单票 15% 敞口放行（默认 10% 会拦截 15 万名义/100 万权益）
    gw = PreTradeRiskGateway(max_position_weight=0.5)
    ex = _wal_stub()
    live.attach(gateway=gw, executor=ex)

    live.push_bar(BarEvent(symbol="600519.SH", close=1500.0, volume=1e6, ts=1000))
    strat = _SimpleStrat()
    live.collect_intents(strategy=strat)

    # 放行 → 转发 executor
    assert "s1" in ex.submitted
    # 成交回报回流
    live.fill(FillEvent(symbol="600519.SH", qty=100, price=1500.0,
                         exchange_ts=1000, local_ts=1005, client_order_id="s1"))
    assert live.account.positions["600519.SH"] == 100
    assert live.fill_latencies_ms == [5]  # local-exchange 双时间戳差


def test_pipeline_denied_intent_never_reaches_executor():
    live = LiveRuntime()
    gw = PreTradeRiskGateway(max_order_value=10_000.0)  # 15 万必拒
    ex = _wal_stub()
    live.attach(gateway=gw, executor=ex)
    live.push_bar(BarEvent(symbol="600519.SH", close=1500.0, volume=1e6, ts=1000))
    live.collect_intents(strategy=_SimpleStrat())
    assert ex.submitted == []            # 未触执行器
    assert gw.stats["denied"] == 1        # 拒绝计数


def test_pipeline_metrics_recorded(tmp_path):
    live = LiveRuntime()
    # 放松集中度让 ORDER_VALUE（而非 CONCENTRATION）成为拦截主因
    gw = PreTradeRiskGateway(max_order_value=10_000.0, max_position_weight=1.0)
    live.attach(gateway=gw, executor=_wal_stub())
    live.push_bar(BarEvent(symbol="600519.SH", close=1500.0, volume=1e6, ts=1000))
    live.collect_intents(strategy=_SimpleStrat())
    out = live.render_metrics()
    assert 'risk_denied_total{reason="ORDER_VALUE"}' in out


def test_panic_blocks_entire_pipeline():
    live = LiveRuntime()
    gw = PreTradeRiskGateway()
    ex = _wal_stub()
    live.attach(gateway=gw, executor=ex)
    gw.panic()
    live.push_bar(BarEvent(symbol="600519.SH", close=1500.0, volume=1e6, ts=1000))
    live.collect_intents(strategy=_SimpleStrat())
    assert ex.submitted == []
    assert gw.stats["denied"] == 1
