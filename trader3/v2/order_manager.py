"""
订单管理（Order Manager）。

把目标权重 → 目标持仓 → 与当前持仓的差额 → 下单指令（Order）。

设计：
- 输入：目标权重 pd.Series（index=asset）、账户权益、最新价、当前持仓
- 计算目标股数 = round(weight * equity / price)，按整手取整
- 差额 = 目标股数 − 当前股数 → 生成买/卖 Order
- 支持最大持仓只数、单票上限、现金约束（买不超可用现金）
- 通过 BrokerBase 异步下单（paper / CTP 通用）
- 全部为模拟撮合（纸面）或实盘（CTP），由传入 broker 决定
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from trader3.v2.live.broker_base import (
    BrokerBase,
    Order,
    OrderSide,
    OrderType,
    Position,
)

logger = logging.getLogger("trader3.v2.order_manager")


@dataclass
class OrderManagerConfig:
    default_lot: int = 100             # A股一手
    max_positions: int = 20
    cash_buffer: float = 0.0          # 保留现金比例（不用于买入）
    allow_sells: bool = True          # 是否生成减仓/清仓卖单


class OrderManager:
    def __init__(self, config: OrderManagerConfig | None = None):
        self.config = config or OrderManagerConfig()

    def target_shares(self, weights: pd.Series, equity: float,
                      prices: dict[str, float]) -> dict[str, int]:
        """目标股数字典（已按整手取整）。"""
        out: dict[str, int] = {}
        for a, w in weights.items():
            px = prices.get(a)
            if px is None or px <= 0 or w <= 0:
                continue
            budget = equity * float(w)
            raw = budget // (px * self.config.default_lot)
            sh = int(raw) * self.config.default_lot
            if sh > 0:
                out[a] = sh
        return out

    def generate_orders(
        self,
        weights: pd.Series,
        equity: float,
        prices: dict[str, float],
        current_positions: dict[str, Position] | None = None,
        lot_sizes: dict[str, int] | None = None,
    ) -> list[Order]:
        """
        生成从当前持仓到目标权重所需的买卖指令。
        """
        cfg = self.config
        current = current_positions or {}
        target = self.target_shares(weights, equity, prices)

        # 限制持仓只数：按权重降序保留 top_n
        if len(target) > cfg.max_positions:
            keep = sorted(target.items(), key=lambda kv: -weights.get(kv[0], 0.0))[: cfg.max_positions]
            target = dict(keep)

        orders: list[Order] = []
        # 卖出：当前有持仓但目标无/更少
        for a, pos in current.items():
            cur_qty = int(pos.quantity)
            if cur_qty <= 0:
                continue
            tgt = target.get(a, 0)
            if tgt < cur_qty:
                qty = cur_qty - tgt
                if cfg.allow_sells and qty > 0:
                    orders.append(self._make_order(a, OrderSide.SELL, qty, prices.get(a), lot_sizes))
        # 买入：目标 > 当前
        avail_cash = equity * (1.0 - cfg.cash_buffer)
        for a, tgt in target.items():
            cur_qty = int(current[a].quantity) if a in current else 0
            if tgt > cur_qty:
                qty = tgt - cur_qty
                lot = (lot_sizes or {}).get(a, cfg.default_lot) or 1
                qty = (qty // lot) * lot
                if qty <= 0:
                    continue
                px = prices.get(a)
                if px and px > 0:
                    cost = qty * px
                    if cost <= avail_cash:
                        orders.append(self._make_order(a, OrderSide.BUY, qty, px, lot_sizes))
                        avail_cash -= cost
                    else:
                        # 现金不足：按可用现金尽量买
                        affordable = int((avail_cash // (px * lot)) * lot)
                        if affordable > 0:
                            orders.append(self._make_order(a, OrderSide.BUY, affordable, px, lot_sizes))
                            avail_cash -= affordable * px
        return orders

    def _make_order(self, symbol, side: OrderSide, qty: int,
                    price, lot_sizes) -> Order:
        lot = (lot_sizes or {}).get(symbol, self.config.default_lot) or 1
        qty = (qty // lot) * lot
        return Order(
            symbol=symbol, side=side, quantity=float(qty),
            order_type=OrderType.MARKET, price=price,
            metadata={"lot": lot, "source": "order_manager"},
        )

    async def submit(self, orders: list[Order], broker: BrokerBase) -> list[Order]:
        """异步下单，返回成交后的 Order 列表（含状态）。"""
        placed: list[Order] = []
        for o in orders:
            try:
                res = await broker.place_order(o)
                placed.append(res)
            except Exception as e:  # noqa: BLE001
                logger.warning("[order_manager] 下单失败 %s: %s", o.symbol, e)
                o.status = o.status.REJECTED if hasattr(o.status, "REJECTED") else o.status
                placed.append(o)
        return placed

    def reconcile(self, weights: pd.Series, filled: list[Order],
                  prices: dict[str, float], equity: float) -> pd.Series:
        """用已成交订单反推实际权重（供回测/审计）。"""
        actual = {}
        for o in filled:
            if "fill" not in str(getattr(o, "status", "")).lower():
                # 仅统计已成交
                continue
            qty = getattr(o, "filled_qty", 0.0) or 0.0
            if qty <= 0:
                continue
            sign = 1.0 if o.side == OrderSide.BUY else -1.0
            actual[o.symbol] = actual.get(o.symbol, 0.0) + sign * qty * (prices.get(o.symbol, 1.0) or 0.0)
        s = pd.Series(actual)
        tot = float(s.abs().sum())
        if tot <= 0:
            return pd.Series(dtype=float)
        return s / tot * (equity if equity > 0 else 1.0)
