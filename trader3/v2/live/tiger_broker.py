import hashlib
import hmac
import time
from typing import Any

import aiohttp

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


class TigerBroker(BrokerBase):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.tiger_id = config.get("tiger_id", "")
        self.private_key = config.get("private_key", "")
        self.account = config.get("account", "")
        self.server_url = config.get("server_url", "https://openapi.tigerfintech.com")
        self._session: aiohttp.ClientSession | None = None
        self._access_token: str | None = None
        self._token_expires: float = 0
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}

    async def connect(self) -> bool:
        self._session = aiohttp.ClientSession()
        success = await self._get_access_token()
        self._connected = success
        return success

    async def _get_access_token(self) -> bool:
        if not self._session:
            return False

        timestamp = str(int(time.time() * 1000))
        params = {
            "method": "oauth.getAccessToken",
            "tiger_id": self.tiger_id,
            "sign_type": "RSA",
            "timestamp": timestamp,
            "charset": "UTF-8",
            "version": "1.0",
        }

        sign_content = self._build_sign_content(params)
        signature = self._sign(sign_content)
        params["sign"] = signature

        try:
            async with self._session.post(f"{self.server_url}/v1/oauth/token", data=params) as resp:
                data = await resp.json()
                if data.get("code") == 0:
                    self._access_token = data["data"]["access_token"]
                    self._token_expires = time.time() + data["data"]["expires_in"] - 60
                    return True
        except Exception:
            pass
        return False

    def _build_sign_content(self, params: dict) -> str:
        sorted_params = sorted(params.items())
        return "&".join(f"{k}={v}" for k, v in sorted_params)

    def _sign(self, content: str) -> str:
        private_key = self.private_key.replace("\\n", "\n")
        signature = hmac.new(
            private_key.encode(),
            content.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def _ensure_token(self) -> bool:
        if self._access_token and time.time() < self._token_expires:
            return True
        return await self._get_access_token()

    async def _request(self, method: str, params: dict) -> dict | None:
        if not await self._ensure_token():
            return None

        params.update({
            "method": method,
            "tiger_id": self.tiger_id,
            "sign_type": "RSA",
            "timestamp": str(int(time.time() * 1000)),
            "charset": "UTF-8",
            "version": "1.0",
            "access_token": self._access_token,
        })

        sign_content = self._build_sign_content(params)
        params["sign"] = self._sign(sign_content)

        try:
            async with self._session.post(self.server_url, data=params) as resp:
                return await resp.json()
        except Exception:
            return None

    async def disconnect(self) -> bool:
        if self._session:
            await self._session.close()
        self._connected = False
        return True

    async def place_order(self, order: Order) -> Order:
        if not self._connected:
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = "Not connected"
            self._update_order(order)
            return order

        params = {
            "account": self.account,
            "contract": order.symbol,
            "action": order.side.value.upper(),
            "orderType": order.order_type.value.upper(),
            "quantity": str(order.quantity),
            "timeInForce": order.time_in_force.value.upper(),
        }

        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            params["limitPrice"] = str(order.price)
        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            params["stopPrice"] = str(order.stop_price)

        order.status = OrderStatus.SUBMITTED
        self._update_order(order)

        result = await self._request("trade.placeOrder", params)
        if result and result.get("code") == 0:
            order.broker_order_id = result["data"]["orderId"]
            order.status = OrderStatus.FILLED
            order.filled_qty = order.quantity
            order.avg_fill_price = float(result["data"].get("avgFillPrice", order.price or 0))
            self._update_order(order)
            self._update_position_after_fill(order)
        else:
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = result.get("msg", "Unknown error") if result else "Request failed"
            self._update_order(order)

        return order

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
        if not order or not order.broker_order_id:
            return False

        result = await self._request("trade.cancelOrder", {
            "account": self.account,
            "orderId": order.broker_order_id,
        })

        if result and result.get("code") == 0:
            order.status = OrderStatus.CANCELLED
            self._update_order(order)
            return True
        return False

    async def get_order(self, client_order_id: str) -> Order | None:
        return self._orders.get(client_order_id)

    async def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        orders = list(self._orders.values())
        if status:
            orders = [o for o in orders if o.status == status]
        return orders

    async def get_positions(self) -> dict[str, Position]:
        result = await self._request("trade.getPositions", {"account": self.account})
        if result and result.get("code") == 0:
            for pos_data in result["data"]:
                pos = Position(
                    symbol=pos_data["contract"],
                    quantity=float(pos_data["quantity"]),
                    avg_cost=float(pos_data["avgCost"]),
                    market_value=float(pos_data["marketValue"]),
                    unrealized_pnl=float(pos_data["unrealizedPnl"]),
                )
                self._positions[pos.symbol] = pos
        return self._positions.copy()

    async def get_account(self) -> Account:
        result = await self._request("trade.getAccount", {"account": self.account})
        if result and result.get("code") == 0:
            data = result["data"]
            return Account(
                account_id=self.account,
                cash=float(data.get("cash", 0)),
                equity=float(data.get("netLiquidation", 0)),
                buying_power=float(data.get("buyingPower", 0)),
                positions=await self.get_positions(),
            )
        return Account(account_id=self.account, cash=0, equity=0, buying_power=0, positions={})

    async def get_market_data(self, symbols: list[str]) -> dict[str, MarketData]:
        contracts = ",".join(symbols)
        result = await self._request("quote.getQuote", {"contracts": contracts})
        data = {}
        if result and result.get("code") == 0:
            for q in result["data"]:
                data[q["contract"]] = MarketData(
                    symbol=q["contract"],
                    price=float(q.get("latest", 0)),
                    bid=float(q.get("bid", 0)) or None,
                    ask=float(q.get("ask", 0)) or None,
                    volume=int(q.get("volume", 0)) or None,
                )
        return data

    async def subscribe_market_data(self, symbols: list[str], callback) -> bool:
        return True

    async def unsubscribe_market_data(self, symbols: list[str]) -> bool:
        return True
