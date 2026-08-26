import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    broker_order_id: str | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    last_price: float | None = None
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Account:
    account_id: str
    cash: float
    equity: float
    buying_power: float
    positions: dict[str, Position] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class MarketData:
    symbol: str
    price: float
    bid: float | None = None
    ask: float | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    volume: int | None = None
    timestamp: datetime = field(default_factory=datetime.now)


class BrokerBase(ABC):
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._connected = False
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}

    @property
    def connected(self) -> bool:
        return self._connected

    @abstractmethod
    async def connect(self) -> bool:
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        pass

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        pass

    @abstractmethod
    async def cancel_order(self, client_order_id: str) -> bool:
        pass

    @abstractmethod
    async def get_order(self, client_order_id: str) -> Order | None:
        pass

    @abstractmethod
    async def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        pass

    @abstractmethod
    async def get_positions(self) -> dict[str, Position]:
        pass

    @abstractmethod
    async def get_account(self) -> Account:
        pass

    @abstractmethod
    async def get_market_data(self, symbols: list[str]) -> dict[str, MarketData]:
        pass

    @abstractmethod
    async def subscribe_market_data(self, symbols: list[str], callback) -> bool:
        pass

    @abstractmethod
    async def unsubscribe_market_data(self, symbols: list[str]) -> bool:
        pass

    def _update_order(self, order: Order) -> None:
        order.updated_at = datetime.now()
        self._orders[order.client_order_id] = order

    def _update_position(self, position: Position) -> None:
        position.updated_at = datetime.now()
        self._positions[position.symbol] = position


class ShadowBroker(BrokerBase):
    """影子券商：委托只记录不下发，用于回测→实盘并行验证（影子模式）。

    除 place_order 拦截（标记为 shadow，不落地）外，其余接口委托给被包装的真实券商，
    以便获取真实账户/持仓/行情，从而让策略在真实市场数据下“空跑”，验证下单逻辑与
    风险覆盖，而不产生真实成交。
    """

    def __init__(self, wrapped: BrokerBase | None = None):
        super().__init__()
        self.wrapped = wrapped
        self.shadow_orders: list[Order] = []
        self._connected = True if wrapped else False
        self._seq = 0

    async def connect(self):
        if self.wrapped:
            return await self.wrapped.connect()
        self._connected = True
        return True

    async def disconnect(self):
        if self.wrapped:
            return await self.wrapped.disconnect()
        self._connected = False
        return True

    async def place_order(self, order: Order) -> Order:
        self._seq += 1
        order.status = OrderStatus.SUBMITTED
        order.broker_order_id = f"SHADOW-{self._seq}"
        order.metadata = {**(order.metadata or {}), "shadow": True}
        self.shadow_orders.append(order)
        self._update_order(order)
        return order

    async def cancel_order(self, client_order_id: str) -> bool:
        if self.wrapped:
            return await self.wrapped.cancel_order(client_order_id)
        return False

    async def get_order(self, client_order_id: str):
        if self.wrapped:
            return await self.wrapped.get_order(client_order_id)
        return self._orders.get(client_order_id)

    async def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        if self.wrapped:
            return await self.wrapped.get_orders(status)
        return list(self._orders.values())

    async def get_positions(self) -> dict[str, Position]:
        if self.wrapped:
            return await self.wrapped.get_positions()
        return {}

    async def get_account(self) -> Account:
        if self.wrapped:
            return await self.wrapped.get_account()
        return Account(account_id="SHADOW", cash=0.0, equity=0.0, buying_power=0.0)

    async def get_market_data(self, symbols: list[str]) -> dict[str, MarketData]:
        if self.wrapped:
            return await self.wrapped.get_market_data(symbols)
        return {}

    async def subscribe_market_data(self, symbols: list[str], callback) -> bool:
        if self.wrapped:
            return await self.wrapped.subscribe_market_data(symbols, callback)
        return False

    async def unsubscribe_market_data(self, symbols: list[str]) -> bool:
        if self.wrapped:
            return await self.wrapped.unsubscribe_market_data(symbols)
        return False
