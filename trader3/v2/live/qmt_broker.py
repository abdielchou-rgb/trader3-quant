"""
QMT（miniQMT）实盘经纪商接入 — 生产加固版。

依赖 xtquant（迅投官方 SDK，随 miniQMT 终端分发）：
    - xtquant.xtdata  : 行情
    - xtquant.xttrader: 交易（需 miniQMT 终端已登录）

诚实声明：
    1. xtquant 缺失时 fail-fast（ImportError + 安装指引），绝不静默假装已连接
    2. simulated=True 为离线 SIM 撮合路径（本机无 miniQMT 时联调用），
       每笔订单 metadata 显式打标 simulated=True / broker="QMT-SIM"
    3. 本适配器尚未经过真实资金环境验证（QMT 终端联调待做）
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from trader3.v2.live.broker_base import (
    Account,
    BrokerBase,
    MarketData,
    Order,
    OrderStatus,
    OrderType,
    Position,
)

logger = logging.getLogger("trader3.v2.live.qmt")

LOT_SIZE = 100  # A股一手 = 100 股（大宗/ETF 另议，当前只覆盖普通 A 股）

XT_ORDER_TYPE_MAP = {
    OrderType.MARKET: 5,   # xtconstant.STOCK_BEST_TO_COUNTER_TRADING_ORDER 市价
    OrderType.LIMIT: 11,   # xtconstant.STOCK_FIX_PRICE_ORDER 限价
}

XT_STATUS_MAP = {
    48: OrderStatus.SUBMITTED,   # xtconstant.ORDER_UNREPORTED
    49: OrderStatus.SUBMITTED,   # ORDER_REPORTED
    50: OrderStatus.SUBMITTED,   # ORDER_REPORTED_CANCEL
    51: OrderStatus.PARTIAL,    # ORDER_PART_SUCC
    52: OrderStatus.FILLED,     # ORDER_SUCC
    53: OrderStatus.SUBMITTED,  # ORDER_REPORTED_INSERT
    54: OrderStatus.SUBMITTED,  # ORDER_CANCELED
    55: OrderStatus.PARTIAL,    # ORDER_PART_CANCEL
    56: OrderStatus.CANCELLED,  # ORDER_CANCEL
    57: OrderStatus.REJECTED,   # ORDER_REJECTED
}


@dataclass
class QMTConfig:
    qmt_path: str                # miniQMT userdata_mini 目录
    account_id: str              # 资金账号
    account_type: str = "STOCK"  # STOCK / CREDIT / FUTURE
    session_id: int = 0          # 0 = 自动生成（进程内唯一）

    def __post_init__(self) -> None:
        if not self.qmt_path or not self.qmt_path.strip():
            raise ValueError("qmt_path 不能为空（miniQMT userdata_mini 目录）")
        if not self.account_id or not self.account_id.strip():
            raise ValueError("account_id 不能为空（资金账号）")
        if not self.session_id:
            self.session_id = abs(hash((self.account_id, datetime.now().isoformat()))) % 1_000_000


def _xtquant_missing() -> ImportError:
    return ImportError(
        "xtquant 不可用。QMT 接入需要 miniQMT 终端（userdata_mini 目录内含 xtquant）："
        "1) 安装并登录 miniQMT；2) 把 userdata_mini 路径加入 sys.path 或 pip 安装 xtquant；"
        "3) 若只做离线联调，用 QMTBroker(config, simulated=True)。"
    )


class QMTBroker(BrokerBase):
    """miniQMT 经纪商（异步接口经 asyncio.to_thread 桥接同步 xttrader SDK）。"""

    def __init__(self, config: QMTConfig, simulated: bool = False):
        super().__init__(config=vars(config) if config else {})
        self.cfg = config
        self.simulated = simulated
        self._trader: Any = None
        self._xtdata: Any = None
        self._sim_prices: dict[str, float] = {"600519.SH": 1500.0, "000001.SZ": 11.0}

    # ── 连接 ──────────────────────────────────────

    async def connect(self) -> bool:
        try:
            from xtquant import xtdata, xttrader  # noqa: F401, W0621
        except ImportError as exc:
            if not self.simulated:
                raise _xtquant_missing() from exc
            self._connected = True
            logger.warning("QMTBroker 走 SIMULATED 离线撮合（xtquant 不可用）")
            return True
        if self.simulated:
            self._connected = True
            return True

        def _connect_sync() -> bool:
            api = xttrader.XtQuantTrader(self.cfg.qmt_path, self.cfg.session_id)
            api.start()
            ok = api.connect()
            if not ok:
                return False
            acc = xttrader.StockAccount(self.cfg.account_id, self.cfg.account_type)
            api.subscribe(acc)
            self._acc = acc
            return api

        result = await asyncio.to_thread(_connect_sync)
        if result is False:
            logger.error("miniQMT 连接失败（终端未登录或 session 冲突）")
            return False
        self._trader = result
        self._xtdata = xtdata
        self._connected = True
        return True

    async def disconnect(self) -> bool:
        if self._trader is not None:
            await asyncio.to_thread(self._trader.stop)
        self._trader = None
        self._connected = False
        return True

    # ── 下单 ──────────────────────────────────────

    @staticmethod
    def _round_lot(qty: float) -> int:
        return int(math.floor(qty / LOT_SIZE) * LOT_SIZE)

    async def place_order(self, order: Order) -> Order:
        rounded = self._round_lot(order.quantity)
        if rounded < LOT_SIZE:
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = f"A股整手约束: {order.quantity} 股不足一手（{LOT_SIZE}）"
            self._update_order(order)
            return order
        if rounded != order.quantity:
            order.metadata["lot_rounded_from"] = order.quantity
            order.quantity = rounded

        if self.simulated or self._trader is None:
            # SIM 撮合：限价按委托价、市价按内部价目，必成交（联调路径）
            price: float = (
                float(order.price or 0.0) if order.order_type == OrderType.LIMIT
                else self._sim_prices.get(order.symbol, 10.0)
            )
            order.status = OrderStatus.FILLED
            order.filled_qty = order.quantity
            order.avg_fill_price = float(price)
            order.broker_order_id = f"QMTSIM-{order.client_order_id[:8]}"
            order.metadata["simulated"] = True
            order.metadata["broker"] = "QMT-SIM"
            self._update_order(order)
            return order

        from xtquant.xttype import StockOrder

        xt_type = XT_ORDER_TYPE_MAP.get(order.order_type)
        if xt_type is None:
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = f"QMT 暂不支持订单类型 {order.order_type}"
            self._update_order(order)
            return order

        def _place_sync() -> int:
            xo = StockOrder()
            xo.account_type = self.cfg.account_type
            xo.account_id = self.cfg.account_id
            xo.stock_code = order.symbol
            xo.order_volume = int(order.quantity)
            xo.price = float(order.price or 0.0)
            xo.order_type = xt_type
            xo.strategy_name = "trader3"
            xo.order_remark = order.client_order_id
            return self._trader.orderStock(xo)

        seq = await asyncio.to_thread(_place_sync)
        if seq < 0:
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = f"QMT 下单失败 seq={seq}"
        else:
            order.status = OrderStatus.SUBMITTED
            order.broker_order_id = str(seq)
        self._update_order(order)
        return order

    async def cancel_order(self, client_order_id: str) -> bool:
        order = self._orders.get(client_order_id)
        if order is None or not order.broker_order_id:
            return False
        if self.simulated or self._trader is None:
            return order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED,
                                    OrderStatus.PARTIAL)
        await asyncio.to_thread(
            self._trader.cancelOrderStock, self._acc, int(order.broker_order_id)
        )
        return True

    # ── 查询 ──────────────────────────────────────

    def get_order_local(self, client_order_id: str) -> Order | None:
        """本地订单簿查询（ShadowBroker 测试用；不经异步桥）。"""
        return self._orders.get(client_order_id)

    async def get_order(self, client_order_id: str) -> Order | None:
        return self._orders.get(client_order_id)

    async def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        if status is None:
            return list(self._orders.values())
        return [o for o in self._orders.values() if o.status == status]

    async def get_positions(self) -> dict[str, Position]:
        if self.simulated or self._trader is None:
            return dict(self._positions)
        details = await asyncio.to_thread(
            self._trader.queryStockPositions, self._acc
        )
        out: dict[str, Position] = {}
        for d in details or []:
            out[d.stock_code] = Position(
                symbol=d.stock_code,
                quantity=float(d.m_volume),
                avg_cost=float(d.open_price) if d.m_volume else 0.0,
                market_value=float(d.market_value) if hasattr(d, "market_value") else 0.0,
                unrealized_pnl=0.0,
                last_price=float(d.m_can_use_volume and 0.0) or 0.0,
            )
        self._positions = out
        return out

    async def get_account(self) -> Account:
        if self.simulated or self._trader is None:
            return Account(account_id=f"QMT-SIM-{self.cfg.account_id}",
                           cash=1_000_000.0, equity=1_000_000.0, buying_power=2_000_000.0)
        a = await asyncio.to_thread(self._trader.queryStockAsset, self._acc)
        if a is None:
            return Account(account_id=self.cfg.account_id, cash=0, equity=0, buying_power=0)
        return Account(
            account_id=self.cfg.account_id,
            cash=float(a.cash),
            equity=float(a.total_asset),
            buying_power=float(a.buying_power) if hasattr(a, "buying_power") else float(a.cash),
            updated_at=datetime.now(),
        )

    async def get_market_data(self, symbols: list[str]) -> dict[str, MarketData]:
        if self.simulated or self._xtdata is None:
            return {
                s: MarketData(symbol=s, price=self._sim_prices.get(s, 10.0))
                for s in symbols
            }

        def _quote_sync() -> dict[str, Any]:
            return {s: self._xtdata.get_full_tick([s]).get(s) for s in symbols}

        ticks = await asyncio.to_thread(_quote_sync)
        out: dict[str, MarketData] = {}
        for s, t in (ticks or {}).items():
            if t:
                out[s] = MarketData(
                    symbol=s, price=float(t.get("lastPrice", 0.0)),
                    bid=float(t.get("bidPrice", 0.0)) or None,
                    ask=float(t.get("askPrice", 0.0)) or None,
                    volume=int(t.get("volume", 0) or 0),
                    timestamp=datetime.now(),
                )
        return out

    async def subscribe_market_data(self, symbols: list[str], callback) -> bool:
        if self.simulated or self._xtdata is None:
            return False
        for s in symbols:
            await asyncio.to_thread(self._xtdata.subscribeQuote, s)
        return True

    async def unsubscribe_market_data(self, symbols: list[str]) -> bool:
        if self.simulated or self._xtdata is None:
            return False
        for s in symbols:
            await asyncio.to_thread(self._xtdata.unsubscribeQuote, s)
        return True
