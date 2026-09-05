"""
模块三：确定性订单状态机 + WAL 断线对账 — 回归测试
核心契约：
  1. 状态机：非法转移直接拒绝（ValueError）；终态（FILLED/CANCELED/REJECTED）吸收一切再转移
  2. WAL：先写日志后发单（crash-safe：进程被杀后重启能重放恢复 active_orders）
  3. 断线重连对账：reconcile 用券商回报对冲本地态——远程已撤/已成 → 本地收敛；
     遥远单（本地非终态、远程无记录）→ FAILED_LOST 待人工/超时处理
  4. A股整手约束：不足 100 股拒绝下单且不落 WAL
  5. WAL 顺序完整性：每个动作恰好追加一行，JSON-Lines 可逐行解析
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.v2.live.order_state import (  # noqa: E402
    FAILED_LOST,
    VALID_TRANSITIONS,
    OrderState,
    can_transition,
)
from trader3.v2.live.robust_qmt_executor import RobustQMTExecutor  # noqa: E402


class FakeQMT:
    """离线 QMT 客户端桩：记录报单，模拟 ack/断线/成交回报。"""

    def __init__(self, fail_submit=False, delay_broker_id=True):
        self.submitted: list[dict] = []
        self.fail_submit = fail_submit
        self.delay_broker_id = delay_broker_id
        self.remote: dict[int, dict] = {}      # broker_order_id -> 回报
        self._seq = 1000

    def order_stock(self, symbol, amount, price):
        if self.fail_submit:
            raise ConnectionError("gateway unreachable")
        self._seq += 1
        bid = self._seq if not self.delay_broker_id else None
        self.submitted.append({"symbol": symbol, "amount": amount, "price": price,
                               "broker_order_id": bid})
        return bid

    def query_stock_orders(self):
        return dict(self.remote)


def test_state_machine_rejects_illegal_transitions():
    assert can_transition(OrderState.PENDING_SUBMIT, OrderState.SUBMITTED)
    assert can_transition(OrderState.SUBMITTED, OrderState.ACKNOWLEDGED)
    assert can_transition(OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED)
    # 非法：PENDING_SUBMIT 不能直接 FILLED（必须经过券商 ack）
    assert not can_transition(OrderState.PENDING_SUBMIT, OrderState.FILLED)
    # 非法：CANCELED 后不能复活
    assert not can_transition(OrderState.CANCELED, OrderState.FILLED)
    # 终态集合为空 → 任何转移都不合法
    assert VALID_TRANSITIONS[OrderState.FILLED] == set()
    assert VALID_TRANSITIONS[OrderState.REJECTED] == set()


def test_wal_write_before_submit_crash_safe(tmp_path):
    """提交前 WAL 已含 PENDING_SUBMIT —— 模拟发送瞬间崩溃，重启后可重放。"""
    wal = tmp_path / "wal.jsonl"
    qmt = FakeQMT()
    ex = RobustQMTExecutor(str(wal), qmt)
    ex.submit_safe_order("c1", "600519.SH", 250, 1500.0)

    lines = wal.read_text(encoding="utf-8").strip().splitlines()
    entries = [json.loads(x) for x in lines]
    # 第一行必须是 PENDING_SUBMIT（先落盘后发送）
    assert entries[0]["state"] == "PENDING_SUBMIT"
    assert entries[0]["amount"] == 200  # 整手取整 250→200
    states = [e["state"] for e in entries]
    assert states == ["PENDING_SUBMIT", "SUBMITTED"]


def test_wal_replay_restores_active_orders(tmp_path):
    """重启重放：崩前活跃订单从 WAL 恢复。"""
    wal = tmp_path / "wal.jsonl"
    qmt = FakeQMT()
    ex = RobustQMTExecutor(str(wal), qmt)
    ex.submit_safe_order("c1", "600519.SH", 200, 1500.0)
    # 模拟崩溃后新执行器（旧对象丢弃）
    ex2 = RobustQMTExecutor(str(wal), FakeQMT())
    assert "c1" in ex2.active_orders
    assert ex2.active_orders["c1"]["state"] == "SUBMITTED"


def test_lot_rounding_rejects_odd_lot_no_wal(tmp_path):
    wal = tmp_path / "wal.jsonl"
    ex = RobustQMTExecutor(str(wal), FakeQMT())
    ex.submit_safe_order("c1", "600519.SH", 57, 1500.0)   # <100 不成整手
    assert "c1" not in ex.active_orders
    assert not wal.exists() or not wal.read_text(encoding="utf-8").strip()


def test_submit_failure_marks_rejected_in_wal(tmp_path):
    wal = tmp_path / "wal.jsonl"
    ex = RobustQMTExecutor(str(wal), FakeQMT(fail_submit=True))
    ex.submit_safe_order("c1", "600519.SH", 200, 1500.0)
    assert ex.active_orders["c1"]["state"] == "REJECTED"
    assert "gateway unreachable" in ex.active_orders["c1"].get("error", "")


def test_reconcile_ghost_order_flagged_lost(tmp_path):
    """本地 SUBMITTED 但券商无记录（发送丢失/超时）→ FAILED_LOST。"""
    wal = tmp_path / "wal.jsonl"
    qmt = FakeQMT()
    ex = RobustQMTExecutor(str(wal), qmt)
    ex.submit_safe_order("c1", "600519.SH", 200, 1500.0)
    # 券商侧无该单（broker_order_id=None，query 返回空）
    ex.reconcile_with_exchange()
    assert ex.active_orders["c1"]["state"] == FAILED_LOST.name


def test_reconcile_syncs_remote_fill(tmp_path):
    """远程已成交 → 本地从 SUBMITTED 收敛到 FILLED，WAL 记录转移。"""
    wal = tmp_path / "wal.jsonl"
    qmt = FakeQMT(delay_broker_id=False)
    ex = RobustQMTExecutor(str(wal), qmt)
    ex.submit_safe_order("c1", "600519.SH", 200, 1500.0)
    bid = ex.active_orders["c1"]["broker_order_id"]
    qmt.remote[bid] = {"status": "FILLED", "filled": 200}
    ex.reconcile_with_exchange()
    assert ex.active_orders["c1"]["state"] == "FILLED"
    # WAL 尾部含 FILLED 行
    tail = [json.loads(x) for x in wal.read_text(encoding="utf-8").strip().splitlines()]
    assert tail[-1]["state"] == "FILLED"


def test_reconcile_ignores_terminal_states(tmp_path):
    """终态订单不参与对账（不重写 WAL）。"""
    wal = tmp_path / "wal.jsonl"
    qmt = FakeQMT(delay_broker_id=False)
    ex = RobustQMTExecutor(str(wal), qmt)
    ex.submit_safe_order("c1", "600519.SH", 200, 1500.0)
    bid = ex.active_orders["c1"]["broker_order_id"]
    qmt.remote[bid] = {"status": "FILLED", "filled": 200}
    ex.reconcile_with_exchange()
    n_lines_before = len(wal.read_text(encoding="utf-8").strip().splitlines())
    # 再对账一次：终态跳过，WAL 不增长
    ex.reconcile_with_exchange()
    n_lines_after = len(wal.read_text(encoding="utf-8").strip().splitlines())
    assert n_lines_before == n_lines_after
