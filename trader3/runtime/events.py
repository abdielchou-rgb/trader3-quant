"""
统一交易事件模型（E1）。

设计原则：
  1. **不可变**（frozen dataclass）：事件一旦产生不允许篡改 —— 事件溯源
     与回放确定性的基础。
  2. **双时间戳**：exchange_ts（交易所/数据源时刻）与 local_ts（本地接收时刻）。
     回放守卫依赖二者分离来杜绝前视（消费尚未到达的事件）。
  3. **策略面向抽象**：DualModeStrategy 只见事件 + 账户快照，
     不触碰任何 broker SDK（Train-Serving Skew 的根治）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BarEvent:
    """K 线事件（回放/实时同构的最小单元）。"""
    symbol: str
    close: float
    volume: float
    ts: int                      # exchange_ts（数据源时刻，unix 毫秒）
    local_ts: int | None = None  # 本地接收时刻；None = 回放中按 ts + lag 推定


@dataclass(frozen=True)
class TickEvent:
    """逐笔事件（预留 tick 级扩展；日频引擎当前不消费）。"""
    symbol: str
    price: float
    qty: float
    exchange_ts: int
    local_ts: int


@dataclass(frozen=True)
class SignalEvent:
    """策略信号事件（意图前置的标准化中间产物）。"""
    symbol: str
    score: float
    ts: int


@dataclass(frozen=True)
class OrderIntent:
    """策略产出的下单意图 —— 尚未过风控、未绑 broker 单号。"""
    client_order_id: str
    symbol: str
    side: str            # buy / sell
    qty: int
    price: float | None = None
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FillEvent:
    """成交回报（含双时间戳与幂等单号）。"""
    symbol: str
    qty: int
    price: float
    exchange_ts: int
    local_ts: int
    client_order_id: str
    broker_order_id: str | None = None
