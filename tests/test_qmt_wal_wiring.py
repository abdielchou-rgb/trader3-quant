"""
Track3 回归：QMTBroker ↔ RobustQMTExecutor WAL 接线。
契约：
  1. wal_path 注入的 QMTBroker：place_order 走 WAL（先 PENDING_SUBMIT 落盘后撮合）
  2. 无 wal_path：历史直撮路径不变
  3. 整手不足：拒绝且不产生 WAL 行
  4. 崩溃重启：新 QMTBroker 实例（同 wal）重放后 active_orders 恢复
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.v2.live.broker_base import Order, OrderSide, OrderType  # noqa: E402
from trader3.v2.live.qmt_broker import QMTBroker, QMTConfig  # noqa: E402


def _broker(wal: Path) -> QMTBroker:
    return QMTBroker(QMTConfig(qmt_path="C:/qmt", account_id="12345"),
                     simulated=True, wal_path=str(wal))


def test_wal_wired_place_order(tmp_path):
    wal = tmp_path / "wal.jsonl"
    brk = _broker(wal)

    async def run():
        await brk.connect()
        o = Order(symbol="600519.SH", side=OrderSide.BUY, quantity=300,
                  order_type=OrderType.LIMIT, price=1500.0)
        return await brk.place_order(o)

    res = asyncio.run(run())
    assert res.status.value in ("submitted", "filled")
    entries = [json.loads(x) for x in wal.read_text(encoding="utf-8").splitlines()]
    assert entries[0]["state"] == "PENDING_SUBMIT"  # 先落盘
    assert entries[-1]["state"] == "SUBMITTED"
    assert res.broker_order_id is not None


def test_no_wal_keeps_legacy_path(tmp_path):
    brk = QMTBroker(QMTConfig(qmt_path="C:/qmt", account_id="12345"),
                    simulated=True)

    async def run():
        await brk.connect()
        o = Order(symbol="600519.SH", side=OrderSide.BUY, quantity=300,
                  order_type=OrderType.LIMIT, price=1500.0)
        return await brk.place_order(o)

    res = asyncio.run(run())
    assert res.status.value == "filled"
    assert res.metadata.get("simulated") is True


def test_odd_lot_rejected_no_wal_line(tmp_path):
    wal = tmp_path / "wal.jsonl"
    brk = _broker(wal)

    async def run():
        await brk.connect()
        o = Order(symbol="600519.SH", side=OrderSide.BUY, quantity=57,
                  order_type=OrderType.LIMIT, price=1500.0)
        return await brk.place_order(o)

    res = asyncio.run(run())
    assert res.status.value == "rejected"
    assert not wal.exists() or not wal.read_text(encoding="utf-8").strip()


def test_crash_restart_replays_wal(tmp_path):
    wal = tmp_path / "wal.jsonl"
    brk = _broker(wal)

    async def run():
        await brk.connect()
        o = Order(symbol="600519.SH", side=OrderSide.BUY, quantity=200,
                  order_type=OrderType.LIMIT, price=1500.0)
        return await brk.place_order(o)

    asyncio.run(run())
    # 模拟崩溃：丢弃对象，新实例同 WAL 重放
    brk2 = _broker(wal)
    assert len(brk2.wal_executor.active_orders) == 1
    entry = next(iter(brk2.wal_executor.active_orders.values()))
    assert entry["state"] in ("SUBMITTED", "FILLED")
