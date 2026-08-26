import asyncio
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


class CTPEndpoint:
    def __init__(self, front: str, broker_id: str, user_id: str, password: str, app_id: str, auth_code: str):
        self.front = front
        self.broker_id = broker_id
        self.user_id = user_id
        self.password = password
        self.app_id = app_id
        self.auth_code = auth_code
        self._connected = False
        self._logged_in = False


class CTPSession:
    def __init__(self, endpoint: CTPEndpoint):
        self.endpoint = endpoint
        self._req_id = 0

    def next_req_id(self) -> int:
        self._req_id += 1
        return self._req_id


class CTPBroker(BrokerBase):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.endpoint = CTPEndpoint(
            front=config.get("ctp_front", "tcp://180.168.146.187:10130"),
            broker_id=config.get("ctp_broker_id", "9999"),
            user_id=config.get("ctp_user_id", ""),
            password=config.get("ctp_password", ""),
            app_id=config.get("ctp_app_id", "simnow_client_test"),
            auth_code=config.get("ctp_auth_code", "0000000000000000"),
        )
        self._session: CTPSession | None = None
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}
        self._account = Account(
            account_id=self.endpoint.user_id,
            cash=0.0,
            equity=0.0,
            buying_power=0.0,
            positions={},
        )

    async def connect(self) -> bool:
        try:
            await asyncio.sleep(0.5)
            self._session = CTPSession(self.endpoint)
            self._connected = True
            await self._on_front_connected()
            return True
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"CTP connect failed: {e}") from e

    async def _on_front_connected(self) -> bool:
        if not self._session:
            return False
        await asyncio.sleep(0.2)
        self._logged_in = True
        await self._query_account()
        await self._query_position()
        return True

    async def disconnect(self) -> bool:
        self._connected = False
        self._logged_in = False
        self._session = None
        return True

    async def place_order(self, order: Order) -> Order:
        if not self._connected or not self._logged_in:
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = "Not connected/logged in"
            self._update_order(order)
            return order

        ctp_order = self._convert_to_ctp_order(order)
        await asyncio.sleep(0.05)

        order.broker_order_id = f"ctp_{self._session.next_req_id()}"
        order.status = OrderStatus.SUBMITTED
        self._update_order(order)

        asyncio.create_task(self._simulate_ctp_fill(order, ctp_order))
        return order

    async def _simulate_ctp_fill(self, order: Order, ctp_order: dict) -> None:
        await asyncio.sleep(0.1)
        order.filled_qty = order.quantity
        order.avg_fill_price = ctp_order.get("price", 0) or 10.0
        order.status = OrderStatus.FILLED
        self._update_order(order)
        self._update_position_after_fill(order)

    def _convert_to_ctp_order(self, order: Order) -> dict:
        direction = "0" if order.side == OrderSide.BUY else "1"
        offset = "0"
        price_type = "1" if order.order_type == OrderType.MARKET else "2"
        return {
            "instrument_id": order.symbol,
            "direction": direction,
            "offset": offset,
            "volume": int(order.quantity),
            "price": order.price or 0,
            "price_type": price_type,
            "time_condition": "3",
            "volume_condition": "1",
        }

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

    async def cancel_order(self, client_order_id: str) -> bool:
        order = self._orders.get(client_order_id)
        if not order or order.status not in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
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
        return self._positions.copy()

    async def get_account(self) -> Account:
        return self._account

    async def get_market_data(self, symbols: list[str]) -> dict[str, MarketData]:
        return {}

    async def subscribe_market_data(self, symbols: list[str], callback) -> bool:
        return True

    async def unsubscribe_market_data(self, symbols: list[str]) -> bool:
        return True

    async def _query_account(self) -> None:
        self._account.cash = 1000000.0
        self._account.equity = 1200000.0
        self._account.buying_power = 2000000.0

    async def _query_position(self) -> None:
        pass
