import asyncio
import random
from datetime import datetime
from typing import Any

from trader3.v2.live.broker_base import (
    Account,
    BrokerBase,
    MarketData,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


class PaperBroker(BrokerBase):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        initial_cash: float = 1_000_000.0,
        commission_rate: float = 0.0003,
        slippage_bps: float = 1.0,
        fill_probability: float = 1.0,
        latency_ms: int = 10,
    ):
        super().__init__(config)
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.fill_probability = fill_probability
        self.latency_ms = latency_ms
        self._positions: dict[str, Position] = {}
        self._account = Account(
            account_id="paper_account",
            cash=initial_cash,
            equity=initial_cash,
            buying_power=initial_cash * 2,
            positions={},
        )
        self._market_data: dict[str, MarketData] = {}
        self._callbacks: list = []
        self._running = False

    async def connect(self) -> bool:
        await asyncio.sleep(0.01)
        self._connected = True
        self._running = True
        asyncio.create_task(self._simulate_market_updates())
        return True

    async def disconnect(self) -> bool:
        self._running = False
        await asyncio.sleep(0.01)
        self._connected = False
        return True

    async def place_order(self, order: Order) -> Order:
        await asyncio.sleep(self.latency_ms / 1000)

        if random.random() > self.fill_probability:
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = "Simulated rejection"
            self._update_order(order)
            return order

        order.status = OrderStatus.SUBMITTED
        self._update_order(order)

        if order.order_type == OrderType.MARKET:
            fill_price = await self._get_market_price(order.symbol)
            if fill_price is None:
                order.status = OrderStatus.REJECTED
                order.metadata["reject_reason"] = "No market data"
                self._update_order(order)
                return order

            fill_price = self._apply_slippage(fill_price, order.side)
            filled_qty = order.quantity
            commission = filled_qty * fill_price * self.commission_rate

            if order.side == OrderSide.BUY:
                total_cost = filled_qty * fill_price + commission
                if total_cost > self.cash:
                    order.status = OrderStatus.REJECTED
                    order.metadata["reject_reason"] = "Insufficient cash"
                    self._update_order(order)
                    return order
                self.cash -= total_cost
            else:
                pos = self._positions.get(order.symbol)
                if not pos or pos.quantity < filled_qty:
                    order.status = OrderStatus.REJECTED
                    order.metadata["reject_reason"] = "Insufficient position"
                    self._update_order(order)
                    return order
                self.cash += filled_qty * fill_price - commission

            order.filled_qty = filled_qty
            order.avg_fill_price = fill_price
            order.status = OrderStatus.FILLED
            order.broker_order_id = f"paper_{order.client_order_id[:8]}"
            self._update_order(order)
            self._update_position_after_fill(order)

        elif order.order_type == OrderType.LIMIT:
            order.status = OrderStatus.PENDING
            self._update_order(order)

        return order

    async def cancel_order(self, client_order_id: str) -> bool:
        order = self._orders.get(client_order_id)
        if not order:
            return False

        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
            return False

        order.status = OrderStatus.CANCELLED
        self._update_order(order)
        return True

    async def get_order(self, client_order_id: str) -> Order | None:
        return self._orders.get(client_order_id)

    async def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        orders = list(self._orders.values())
        if status:
            orders = [o for o in orders if o.status == status]
        return orders

    async def get_positions(self) -> dict[str, Position]:
        await self._update_positions_market_value()
        return self._positions.copy()

    async def get_account(self) -> Account:
        await self._update_positions_market_value()
        self._account.cash = self.cash
        self._account.equity = self.cash + sum(p.market_value for p in self._positions.values())
        self._account.buying_power = self.cash * 2
        self._account.positions = self._positions.copy()
        self._account.updated_at = datetime.now()
        return self._account

    async def get_market_data(self, symbols: list[str]) -> dict[str, MarketData]:
        result = {}
        for sym in symbols:
            if sym in self._market_data:
                result[sym] = self._market_data[sym]
            else:
                base_price = 10.0 + random.random() * 90
                result[sym] = MarketData(
                    symbol=sym,
                    price=base_price,
                    bid=base_price * 0.999,
                    ask=base_price * 1.001,
                    volume=random.randint(100000, 10000000),
                )
        return result

    async def subscribe_market_data(self, symbols: list[str], callback) -> bool:
        self._callbacks.append((symbols, callback))
        return True

    async def unsubscribe_market_data(self, symbols: list[str]) -> bool:
        self._callbacks = [
            (syms, cb) for syms, cb in self._callbacks if not any(s in syms for s in symbols)
        ]
        return True

    async def _get_market_price(self, symbol: str) -> float | None:
        data = await self.get_market_data([symbol])
        return data.get(symbol, MarketData(symbol=symbol, price=0)).price

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        slippage = price * (self.slippage_bps / 10000)
        if side == OrderSide.BUY:
            return price + slippage
        return price - slippage

    def _update_position_after_fill(self, order: Order) -> None:
        pos = self._positions.get(order.symbol)
        if pos:
            if order.side == OrderSide.BUY:
                new_qty = pos.quantity + order.filled_qty
                new_cost = (pos.quantity * pos.avg_cost + order.filled_qty * order.avg_fill_price) / new_qty
                pos.quantity = new_qty
                pos.avg_cost = new_cost
            else:
                pos.quantity -= order.filled_qty
                if pos.quantity <= 0:
                    del self._positions[order.symbol]
                    return
        else:
            if order.side == OrderSide.BUY:
                pos = Position(
                    symbol=order.symbol,
                    quantity=order.filled_qty,
                    avg_cost=order.avg_fill_price or 0,
                    market_value=order.filled_qty * (order.avg_fill_price or 0),
                    unrealized_pnl=0.0,
                )
                self._positions[order.symbol] = pos

        if pos:
            pos.market_value = pos.quantity * (order.avg_fill_price or pos.avg_cost)
            pos.unrealized_pnl = pos.quantity * ((order.avg_fill_price or pos.avg_cost) - pos.avg_cost)

    async def _update_positions_market_value(self) -> None:
        for sym, pos in self._positions.items():
            md = await self.get_market_data([sym])
            price = md[sym].price
            pos.last_price = price
            pos.market_value = pos.quantity * price
            pos.unrealized_pnl = pos.quantity * (price - pos.avg_cost)

    async def _simulate_market_updates(self) -> None:
        while self._running:
            await asyncio.sleep(1.0)
            for sym in list(self._market_data.keys()):
                md = self._market_data[sym]
                change = random.uniform(-0.005, 0.005)
                new_price = max(0.01, md.price * (1 + change))
                md.price = new_price
                md.bid = new_price * 0.999
                md.ask = new_price * 1.001
                md.timestamp = datetime.now()

            for symbols, callback in self._callbacks:
                try:
                    updates = {s: self._market_data.get(s) for s in symbols if s in self._market_data}
                    if updates:
                        await callback(updates)
                except Exception:
                    pass


class PaperBrokerSync:
    def __init__(self, config: dict[str, Any] | None = None):
        self._broker = PaperBroker(config)

    def connect(self) -> bool:
        return asyncio.run(self._broker.connect())

    def disconnect(self) -> bool:
        return asyncio.run(self._broker.disconnect())

    def place_order(self, order: Order) -> Order:
        return asyncio.run(self._broker.place_order(order))

    def cancel_order(self, client_order_id: str) -> bool:
        return asyncio.run(self._broker.cancel_order(client_order_id))

    def get_order(self, client_order_id: str) -> Order | None:
        return asyncio.run(self._broker.get_order(client_order_id))

    def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        return asyncio.run(self._broker.get_orders(status))

    def get_positions(self) -> dict[str, Position]:
        return asyncio.run(self._broker.get_positions())

    def get_account(self) -> Account:
        return asyncio.run(self._broker.get_account())

    def get_market_data(self, symbols: list[str]) -> dict[str, MarketData]:
        return asyncio.run(self._broker.get_market_data(symbols))
