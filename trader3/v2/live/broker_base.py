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
