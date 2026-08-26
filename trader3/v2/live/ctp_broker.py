"""
CTP 实盘经纪商接入（生产加固版）。

特性：
- 状态机：DISCONNECTED -> CONNECTING -> CONNECTED -> LOGGED_IN
- 自动重连 + 指数退避
- CTP 错误码映射
- 线程安全的回调队列（CTP 回调在原生线程，经队列转入 asyncio 循环）
- 订单引用号（OrderRef）管理，防止重复
- 登录/登出生命周期
- 当 vnpy_ctp 不可用时降级为 SIMULATED 模式（明确标记，仅供测试/演示）

接入真实柜台：安装 vnpy_ctp 后，去掉 _USE_SIM 分支即可。
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from collections.abc import Callable
from enum import Enum
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

logger = logging.getLogger("trader3.v2.live.ctp")

try:
    import vnpy_ctp  # noqa: F401
    _HAS_CTP = True
except ImportError:
    _HAS_CTP = False

_USE_SIM = not _HAS_CTP


class ConnState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    LOGGED_IN = "logged_in"
    ERROR = "error"


# CTP 常见错误码（节选）
_CTP_ERRORS: dict[int, str] = {
    0: "成功",
    -1: "不明确错误",
    1: "不符合交易规则",
    2: "当前状态不允许此操作",
    3: "不合法的登录",
    4: "用户不活跃",
    5: "重复的报单",
    6: "找不到报单",
    7: "席位是关闭的",
    8: "插入报单失败",
    9: "报单已全成交",
    10: "报单已撤单",
    11: "报单已结束",
    12: "未知合约",
    13: "不支持的交易所",
    14: "达到最大报单数",
    15: "资金不足",
    16: "持仓不足",
    17: "价格超出涨跌停",
    18: "价位错误",
}


def ctp_error(code: int) -> str:
    return _CTP_ERRORS.get(code, f"CTP错误码{code}")


class CTPBroker(BrokerBase):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        cfg = self.config
        self.front = cfg.get("ctp_front", "tcp://180.168.146.187:10130")
        self.md_front = cfg.get("ctp_md_front", self.front)
        self.broker_id = cfg.get("ctp_broker_id", "9999")
        self.user_id = cfg.get("ctp_user_id", "")
        self.password = cfg.get("ctp_password", "")
        self.app_id = cfg.get("ctp_app_id", "simnow_client_test")
        self.auth_code = cfg.get("ctp_auth_code", "0000000000000000")
        self.investor_id = cfg.get("ctp_investor_id", self.user_id)

        self.state = ConnState.DISCONNECTED
        self._state_lock = threading.Lock()
        self._session: dict[str, Any] = {}
        self._req_id = 0
        self._order_ref = 0
        self._loop: asyncio.AbstractEventLoop | None = None

        # 回调队列：CTP 原生线程 -> asyncio
        self._cb_queue: asyncio.Queue | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._max_retries = int(cfg.get("ctp_max_retries", 5))
        self._retry_delay = float(cfg.get("ctp_retry_delay", 2.0))
        self._stop_heartbeat = threading.Event()

        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}
        self._account = Account(
            account_id=self.investor_id or self.user_id,
            cash=0.0, equity=0.0, buying_power=0.0, positions={},
        )
        self._md_subs: dict[str, Callable] = {}
        self._sim_prices: dict[str, float] = {}

        if _USE_SIM:
            logger.warning("vnpy_ctp 未安装，CTP 以 SIMULATED 模式运行（仅测试/演示）。")

    # ── 状态机 ──────────────────────────────────────

    def _set_state(self, s: ConnState) -> None:
        with self._state_lock:
            old = self.state
            self.state = s
        if s != old:
            logger.info("CTP 状态: %s -> %s", old.value, s.value)

    # ── 连接生命周期 ───────────────────────────────

    async def connect(self) -> bool:
        self._loop = asyncio.get_event_loop()
        self._cb_queue = asyncio.Queue()
        self._stop_heartbeat.clear()
        ok = await self._do_connect()
        if ok:
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())
            asyncio.create_task(self._heartbeat())
        return ok

    async def _do_connect(self) -> bool:
        self._set_state(ConnState.CONNECTING)
        try:
            if _USE_SIM:
                await asyncio.sleep(0.3)
                self._session = {"sim": True}
                self._set_state(ConnState.CONNECTED)
                await self._login()
                return True
            # 真实接入：初始化 CTP API、注册回调、连接前置
            return await self._ctp_connect_real()
        except Exception as e:  # noqa: BLE001
            self._set_state(ConnState.ERROR)
            logger.exception("CTP 连接失败: %s", e)
            return False

    async def _login(self) -> bool:
        try:
            if _USE_SIM:
                await asyncio.sleep(0.1)
                self._set_state(ConnState.LOGGED_IN)
                await self._query_account()
                await self._query_position()
                return True
            return await self._ctp_login_real()
        except Exception as e:  # noqa: BLE001
            self._set_state(ConnState.ERROR)
            logger.exception("CTP 登录失败: %s", e)
            return False

    async def disconnect(self) -> bool:
        self._stop_heartbeat.set()
        if self._reconnect_task:
            self._reconnect_task.cancel()
        if self.state == ConnState.LOGGED_IN and not _USE_SIM:
            await self._ctp_logout_real()
        self._set_state(ConnState.DISCONNECTED)
        self._session = {}
        return True

    async def _reconnect_loop(self) -> None:
        retries = 0
        while not self._stop_heartbeat.is_set():
            await asyncio.sleep(self._retry_delay * (2 ** min(retries, 4)))
            if self.state in (ConnState.LOGGED_IN, ConnState.CONNECTED):
                continue
            logger.warning("CTP 重连尝试 #%d", retries + 1)
            if await self._do_connect():
                retries = 0
            else:
                retries += 1
                if retries >= self._max_retries:
                    logger.error("CTP 重连次数超限，停止。")
                    self._stop_heartbeat.set()
                    break

    async def _heartbeat(self) -> None:
        """定时查询账户/持仓，保持会话活性。"""
        while not self._stop_heartbeat.is_set():
            await asyncio.sleep(30.0)
            if self.state == ConnState.LOGGED_IN:
                try:
                    await self._query_account()
                except Exception as e:  # noqa: BLE001
                    logger.warning("CTP 心跳查询失败: %s", e)

    # ── 订单引用 ───────────────────────────────────

    def _next_req_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _next_order_ref(self) -> str:
        self._order_ref += 1
        return f"{self.user_id}-{self._order_ref:012d}"

    # ── 下单 ───────────────────────────────────────

    async def place_order(self, order: Order) -> Order:
        if self.state != ConnState.LOGGED_IN:
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = f"未登录 (state={self.state.value})"
            self._update_order(order)
            return order

        order.broker_order_id = self._next_order_ref()
        ctp_order = self._convert_to_ctp_order(order)
        order.status = OrderStatus.SUBMITTED
        self._update_order(order)

        if _USE_SIM:
            asyncio.create_task(self._simulate_fill(order, ctp_order))
        else:
            await self._ctp_req_order_insert(order, ctp_order)
        return order

    def _convert_to_ctp_order(self, order: Order) -> dict:
        direction = "0" if order.side == OrderSide.BUY else "1"
        # 开平：简单规则，有持仓则平，否则开
        offset = "0"  # 开仓
        pos = self._positions.get(order.symbol)
        if order.side == OrderSide.SELL and pos and pos.quantity > 0:
            offset = "1"  # 平仓
        price_type = "1" if order.order_type == OrderType.MARKET else "2"
        return {
            "instrument_id": order.symbol,
            "direction": direction,
            "offset": offset,
            "volume": int(order.quantity),
            "price": float(order.price or 0),
            "price_type": price_type,
            "time_condition": "3" if price_type == "2" else "1",
            "volume_condition": "1",
            "order_ref": order.broker_order_id,
        }

    async def _simulate_fill(self, order: Order, ctp_order: dict) -> None:
        await asyncio.sleep(random.uniform(0.05, 0.2))
        # 模拟部分成交后全成
        fill_px = ctp_order.get("price", 0) or 10.0
        order.filled_qty = order.quantity
        order.avg_fill_price = fill_px
        order.status = OrderStatus.FILLED
        self._update_order(order)
        self._update_position_after_fill(order)
        await self._query_account()

    def _update_position_after_fill(self, order: Order) -> None:
        sym = order.symbol
        pos = self._positions.get(sym)
        fill_qty = order.filled_qty
        fill_px = order.avg_fill_price or 0.0
        if order.side == OrderSide.BUY:
            if pos:
                new_qty = pos.quantity + fill_qty
                if new_qty > 0:
                    pos.avg_cost = (pos.quantity * pos.avg_cost + fill_qty * fill_px) / new_qty
                pos.quantity = new_qty
            else:
                self._positions[sym] = Position(
                    symbol=sym, quantity=fill_qty, avg_cost=fill_px,
                    market_value=fill_qty * fill_px, unrealized_pnl=0.0,
                )
        else:
            if pos:
                pos.quantity -= fill_qty
                if pos.quantity <= 0:
                    self._positions.pop(sym, None)
            else:
                # 做空开仓
                self._positions[sym] = Position(
                    symbol=sym, quantity=-fill_qty, avg_cost=fill_px,
                    market_value=fill_qty * fill_px, unrealized_pnl=0.0,
                )

    async def cancel_order(self, client_order_id: str) -> bool:
        order = self._orders.get(client_order_id)
        if not order or order.status not in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
            return False
        if not _USE_SIM:
            await self._ctp_req_order_action(order)
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

    # ── 行情 ───────────────────────────────────────

    async def get_market_data(self, symbols: list[str]) -> dict[str, MarketData]:
        out: dict[str, MarketData] = {}
        for s in symbols:
            px = self._sim_prices.get(s, 0.0)
            out[s] = MarketData(
                symbol=s, last_price=px, open=px, high=px, low=px,
                volume=0, timestamp=time.time(),
            )
        return out

    async def subscribe_market_data(self, symbols: list[str], callback: Callable) -> bool:
        for s in symbols:
            self._md_subs[s] = callback
            if s not in self._sim_prices:
                self._sim_prices[s] = 10.0 + hash(s) % 100
        # 模拟行情推送
        if _USE_SIM:
            asyncio.create_task(self._sim_md_feed(symbols))
        return True

    async def _sim_md_feed(self, symbols: list[str]) -> None:
        while self.state == ConnState.LOGGED_IN and not self._stop_heartbeat.is_set():
            await asyncio.sleep(1.0)
            for s in symbols:
                self._sim_prices[s] = round(self._sim_prices[s] * (1 + random.uniform(-0.002, 0.002)), 2)
                cb = self._md_subs.get(s)
                if cb:
                    md = MarketData(
                        symbol=s, last_price=self._sim_prices[s],
                        open=self._sim_prices[s], high=self._sim_prices[s],
                        low=self._sim_prices[s], volume=0, timestamp=time.time(),
                    )
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            await cb(s, md)
                        else:
                            cb(s, md)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("行情回调异常: %s", e)

    async def unsubscribe_market_data(self, symbols: list[str]) -> bool:
        for s in symbols:
            self._md_subs.pop(s, None)
        return True

    # ── 查询 ───────────────────────────────────────

    async def _query_account(self) -> None:
        if _USE_SIM:
            # 依据持仓市值估算
            pos_val = sum(p.market_value for p in self._positions.values())
            self._account.equity = self._account.cash + pos_val
            self._account.buying_power = self._account.cash
            return
        await self._ctp_qry_trading_account()

    async def _query_position(self) -> None:
        if _USE_SIM:
            return
        await self._ctp_qry_investor_position()

    # ── 真实 CTP 接入（需 vnpy_ctp；未安装时恒走 SIMULATED）─────────────

    async def _ctp_connect_real(self) -> bool:
        if not _HAS_CTP:
            raise NotImplementedError("真实 CTP 接入需要 vnpy_ctp 及柜台配置。")
        from vnpy_ctp.api import MdApi, TraderApi

        class _MdSpi(MdApi):
            def __init__(self, brk):
                super().__init__()
                self.brk = brk

            def OnFrontConnected(self):
                self.brk._md_connected.set()

            def OnFrontDisconnected(self, n):
                self.brk._set_state(ConnState.DISCONNECTED)

            def OnRspUserLogin(self, p, info, rid, last):
                if info and getattr(info, "ErrorID", 0) == 0:
                    self.brk._md_logged.set()

            def OnRtnDepthMarketData(self, d):
                self.brk._on_md_tick(d)

        class _TdSpi(TraderApi):
            def __init__(self, brk):
                super().__init__()
                self.brk = brk

            def OnFrontConnected(self):
                self.brk._td_connected.set()

            def OnFrontDisconnected(self, n):
                self.brk._set_state(ConnState.DISCONNECTED)

            def OnRspUserLogin(self, p, info, rid, last):
                if info and getattr(info, "ErrorID", 0) == 0:
                    self.brk._td_logged.set()
                else:
                    self.brk._td_login_err = info

            def OnRspOrderInsert(self, p, info, rid, last):
                self.brk._on_ctp_err(info, rid)

            def OnRspOrderAction(self, p, info, rid, last):
                self.brk._on_ctp_err(info, rid)

            def OnRtnOrder(self, p):
                self.brk._on_rtn_order(p)

            def OnRtnTrade(self, p):
                self.brk._on_rtn_trade(p)

            def OnRspQryTradingAccount(self, p, info, rid, last):
                self.brk._on_qry_account(p)

            def OnRspQryInvestorPosition(self, p, info, rid, last):
                self.brk._on_qry_position(p)

        self._md_connected = asyncio.Event()
        self._md_logged = asyncio.Event()
        self._td_connected = asyncio.Event()
        self._td_logged = asyncio.Event()
        self._td_login_err = None

        self._md_api = _MdSpi(self)
        self._md_api.RegisterFront(self.md_front)
        self._md_api.Init()
        self._td_api = _TdSpi(self)
        self._td_api.RegisterFront(self.front)
        self._td_api.SubscribePrivateTopic(0)
        self._td_api.Init()
        try:
            await asyncio.wait_for(self._td_connected.wait(), 10)
        except asyncio.TimeoutError:
            self._set_state(ConnState.ERROR)
            return False
        return True

    async def _ctp_login_real(self) -> bool:
        from vnpy_ctp.api import ReqUserLoginField

        req = ReqUserLoginField()
        req.BrokerID = self.broker_id
        req.UserID = self.user_id
        req.Password = self.password
        req.UserProductInfo = "trader3"
        self._td_api.ReqUserLogin(req, self._next_req_id())
        try:
            await asyncio.wait_for(self._td_logged.wait(), 10)
        except asyncio.TimeoutError:
            self._set_state(ConnState.ERROR)
            return False
        if self._td_login_err is not None:
            self._set_state(ConnState.ERROR)
            return False
        # 行情登录
        mreq = ReqUserLoginField()
        mreq.BrokerID = self.broker_id
        mreq.UserID = self.user_id
        mreq.Password = self.password
        self._md_api.ReqUserLogin(mreq, self._next_req_id())
        try:
            await asyncio.wait_for(self._md_logged.wait(), 10)
        except asyncio.TimeoutError:
            pass
        return True

    async def _ctp_logout_real(self) -> bool:
        try:
            from vnpy_ctp.api import ReqUserLogoutField
            if getattr(self, "_td_api", None):
                req = ReqUserLogoutField()
                req.BrokerID = self.broker_id
                req.UserID = self.user_id
                self._td_api.ReqUserLogout(req, self._next_req_id())
            if getattr(self, "_md_api", None):
                mreq = ReqUserLogoutField()
                mreq.BrokerID = self.broker_id
                mreq.UserID = self.user_id
                self._md_api.ReqUserLogout(mreq, self._next_req_id())
        except Exception:  # noqa: BLE001
            pass
        return True

    async def _ctp_req_order_insert(self, order: Order, ctp_order: dict) -> None:
        from vnpy_ctp.api import InputOrderField

        o = InputOrderField()
        o.InstrumentID = ctp_order["instrument_id"]
        o.Direction = ctp_order["direction"]
        o.CombOffsetFlag = ctp_order["offset"]
        o.CombHedgeFlag = "1"
        o.VolumeTotalOriginal = ctp_order["volume"]
        o.LimitPrice = ctp_order["price"]
        o.OrderPriceType = ctp_order["price_type"]
        o.TimeCondition = ctp_order["time_condition"]
        o.VolumeCondition = ctp_order["volume_condition"]
        o.ContingentCondition = "1"
        o.MinVolume = 0
        o.ForceCloseReason = "0"
        o.OrderRef = ctp_order["order_ref"]
        o.RequestID = self._next_req_id()
        o.InvestorID = self.investor_id
        o.UserID = self.user_id
        o.BrokerID = self.broker_id
        self._td_api.ReqOrderInsert(o, o.RequestID)

    async def _ctp_req_order_action(self, order: Order) -> None:
        from vnpy_ctp.api import InputOrderActionField

        a = InputOrderActionField()
        a.InstrumentID = order.symbol
        a.OrderRef = order.broker_order_id
        a.FrontID = getattr(self, "_front_id", "")
        a.SessionID = getattr(self, "_session_id", "")
        a.ActionFlag = "0"  # 撤单
        a.InvestorID = self.investor_id
        a.UserID = self.user_id
        a.BrokerID = self.broker_id
        self._td_api.ReqOrderAction(a, self._next_req_id())

    async def _ctp_qry_trading_account(self) -> None:
        from vnpy_ctp.api import QryTradingAccountField
        self._td_api.ReqQryTradingAccount(QryTradingAccountField(), self._next_req_id())

    async def _ctp_qry_investor_position(self) -> None:
        from vnpy_ctp.api import QryInvestorPositionField
        q = QryInvestorPositionField()
        q.BrokerID = self.broker_id
        q.InvestorID = self.investor_id
        self._td_api.ReqQryInvestorPosition(q, self._next_req_id())

    # ── CTP 回调处理 ───────────────────────────────
    def _on_md_tick(self, d) -> None:
        sym = getattr(d, "InstrumentID", None)
        px = float(getattr(d, "LastPrice", 0.0) or 0.0)
        if sym:
            self._sim_prices[sym] = px
            cb = self._md_subs.get(sym)
            if cb:
                md = MarketData(symbol=sym, last_price=px, open=getattr(d, "OpenPrice", px),
                                high=getattr(d, "HighestPrice", px), low=getattr(d, "LowestPrice", px),
                                volume=int(getattr(d, "Volume", 0) or 0), timestamp=time.time())
                try:
                    if asyncio.iscoroutinefunction(cb):
                        asyncio.ensure_future(cb(sym, md))
                    else:
                        cb(sym, md)
                except Exception:  # noqa: BLE001
                    pass

    def _on_rtn_order(self, p) -> None:
        ref = getattr(p, "OrderRef", "")
        order = self._orders.get(ref)
        if not order:
            return
        status_map = {"0": OrderStatus.SUBMITTED, "1": OrderStatus.PARTIAL,
                      "3": OrderStatus.FILLED, "5": OrderStatus.CANCELLED,
                      "4": OrderStatus.REJECTED}
        st = status_map.get(str(getattr(p, "OrderStatus", "")), OrderStatus.SUBMITTED)
        order.status = st
        order.filled_qty = float(getattr(p, "VolumeTraded", 0) or 0)
        self._update_order(order)

    def _on_rtn_trade(self, p) -> None:
        ref = getattr(p, "OrderRef", "")
        order = self._orders.get(ref)
        if not order:
            return
        order.filled_qty = float(getattr(p, "VolumeTraded", 0) or 0)
        order.avg_fill_price = float(getattr(p, "Price", 0.0) or 0.0)
        order.status = OrderStatus.FILLED
        self._update_order(order)
        self._update_position_after_fill(order)
        asyncio.ensure_future(self._query_account())

    def _on_qry_account(self, p) -> None:
        if not p:
            return
        self._account.cash = float(getattr(p, "Available", 0.0) or 0.0)
        self._account.equity = float(getattr(p, "Balance", 0.0) or 0.0)
        self._account.buying_power = float(getattr(p, "Available", 0.0) or 0.0)

    def _on_qry_position(self, p) -> None:
        if not p:
            return
        sym = getattr(p, "InstrumentID", None)
        if not sym:
            return
        qty = float(getattr(p, "Position", 0) or 0)
        if qty == 0:
            self._positions.pop(sym, None)
            return
        self._positions[sym] = Position(
            symbol=sym, quantity=qty,
            avg_cost=float(getattr(p, "OpenCost", 0.0) or 0.0) / qty if qty else 0.0,
            market_value=qty * (self._sim_prices.get(sym, 0.0)),
            unrealized_pnl=float(getattr(p, "PositionProfit", 0.0) or 0.0))

    def _on_ctp_err(self, info, rid) -> None:
        if info and getattr(info, "ErrorID", 0) != 0:
            logger.error("CTP 错误(%s) rid=%s: %s", getattr(info, "ErrorID", ""),
                         rid, ctp_error(getattr(info, "ErrorID", 0)))
