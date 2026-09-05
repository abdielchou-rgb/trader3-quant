"""
带 WAL（预写日志）对账机制的 QMT 订单执行器。

机构级稳健三原则：
  1. **先日志后动作**（Write-Ahead）：任何状态变化先追加 WAL 并 fsync，
     再执行真实副作用。进程被杀后重启，重放 WAL 即可恢复全部活跃订单。
  2. **状态机闭环**：所有转移经 order_state.VALID_TRANSITIONS 校验，
     非法转移本地拒绝（防异步回报乱序产生伪状态）。
  3. **对账收敛**：reconcile_with_exchange 拉取券商回报对冲本地态：
     - 远程已成交/已撤 → 本地收敛到真实终态；
     - 本地非终态但远程无记录（幽灵单/发送丢失）→ FAILED_LOST，
       该态只允许人工裁决后 CANCELED，杜绝"下落不明的钱"被静默遗忘。

WAL 格式：JSON-Lines，每动作一行（client_order_id 维度追加）。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from trader3.v2.live.order_state import (
    TERMINAL_STATES,
    OrderState,
    can_transition,
)

LOT_SIZE = 100  # A股一手


class RobustQMTExecutor:
    def __init__(self, wal_path: str, qmt_client: Any):
        self.wal_path = wal_path
        self.qmt = qmt_client
        self.active_orders: dict[str, dict] = {}
        self._replay_wal()

    # ── WAL 基础设施 ──────────────────────────────

    def _replay_wal(self):
        """重启重放：按行序重建本地订单簿（后行覆盖前行，与状态机时序一致）。"""
        if not os.path.exists(self.wal_path):
            return
        with open(self.wal_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self.active_orders[entry["client_order_id"]] = entry

    def _append_wal(self, order_dict: dict) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.wal_path)), exist_ok=True)
        with open(self.wal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(order_dict, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _transition(order_data: dict, dst: OrderState, **extra) -> None:
        """状态机校验转移：非法直接抛错（确定性优先，绝不静默吞掉乱序回报）。"""
        src = OrderState[order_data["state"]]
        if not can_transition(src, dst):
            raise ValueError(
                f"非法状态转移 {src.name} -> {dst.name} "
                f"(order {order_data.get('client_order_id')})"
            )
        order_data["state"] = dst.name
        order_data.update(extra)

    # ── 下单 ──────────────────────────────────────

    def submit_safe_order(
        self,
        client_order_id: str,
        symbol: str,
        amount: int,
        price: float,
    ) -> None:
        """整手校验 → WAL 落 PENDING_SUBMIT → 网关发送 → WAL 记结果。"""
        lot_rounded = (int(amount) // LOT_SIZE) * LOT_SIZE
        if lot_rounded <= 0:
            return  # 不足一手：拒绝且不落 WAL（未产生任何副作用）

        order_data = {
            "client_order_id": client_order_id,
            "broker_order_id": None,
            "symbol": symbol,
            "amount": lot_rounded,
            "price": float(price),
            "state": OrderState.PENDING_SUBMIT.name,
            "ts": time.time(),
        }
        # 1. 严格先写日志落盘（crash-safe 锚点）
        self._append_wal(order_data)
        self.active_orders[client_order_id] = order_data

        # 2. 报单发送
        try:
            broker_order_id = self.qmt.order_stock(symbol, lot_rounded, price)
            self._transition(order_data, OrderState.SUBMITTED,
                              broker_order_id=broker_order_id, ts=time.time())
        except Exception as exc:  # noqa: BLE001 — 网关异常必须留痕
            self._transition(order_data, OrderState.REJECTED,
                              error=str(exc), ts=time.time())
        self._append_wal(order_data)

    # ── 对账 ──────────────────────────────────────

    def reconcile_with_exchange(self) -> list[str]:
        """拉取券商实际回报，对冲本地态。返回本轮发生状态变化的订单 id 列表。"""
        live_orders = self.qmt.query_stock_orders() or {}
        changed: list[str] = []

        for client_id, local in list(self.active_orders.items()):
            if OrderState[local["state"]] in TERMINAL_STATES:
                continue  # 终态幂等跳过

            b_id = local.get("broker_order_id")
            remote = live_orders.get(b_id) if b_id is not None else None

            if remote is None:
                # 本地非终态但券商无记录：幽灵单/发送丢失 → 隔离为 FAILED_LOST
                if OrderState[local["state"]] in (OrderState.SUBMITTED,
                                                  OrderState.PENDING_SUBMIT):
                    self._transition(local, OrderState.FAILED_LOST, ts=time.time())
                    self._append_wal(local)
                    changed.append(client_id)
                continue

            remote_status = str(remote.get("status", "")).upper()
            dst = {
                "FILLED": OrderState.FILLED,
                "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
                "CANCELED": OrderState.CANCELED,
                "REJECTED": OrderState.REJECTED,
            }.get(remote_status)
            if dst is None or OrderState[local["state"]] == dst:
                continue
            try:
                self._transition(local, dst, ts=time.time())
            except ValueError:
                # 状态机拒绝（如回报乱序）：保留 FAILED_LOST 待人工，绝不静默改史
                self._transition(local, OrderState.FAILED_LOST, ts=time.time())
            self._append_wal(local)
            changed.append(client_id)
        return changed
