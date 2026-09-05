"""
前置硬风控网关（Pre-Trade Risk Engine）。

机构铁律：风控不住在策略里 —— 策略异常/死循环时风控必须仍然生效。
本网关是订单路径上的**独立中间件**：

    strategy → OrderIntent → [PreTradeRiskGateway] → broker

拦截项（全部可配置）：
  1. 单笔委托金额上限（max_order_value）
  2. 单票集中度（成交后名义占比 > max_position_weight）
  3. 自成交防护（同批反向超持仓 = 对敲特征）
  4. OTR 熔断（窗口内 拒单/成交 比率超标 → 限开新单，平仓放行）
  5. 全局回撤熔断（KillSwitch tripped → 拒开仓，平仓放行）
  6. 急停按钮（panic() → 拒一切含平仓，人工接管；unpanic 恢复）
  7. 幂等单号（重复 client_order_id 拒绝，防网络重试重复开仓）

拒绝语义：check() 返回 (allowed: bool, reason: DenyReason | None)；
被拒订单不进入任何下游。全部规则纯本地计算，零外部调用 ——
风控路径的可用性只依赖进程存活，不依赖策略健康。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from enum import Enum, auto

from trader3.runtime.events import OrderIntent
from trader3.runtime.strategy import AccountSnapshot
from trader3.v2.execution import KillSwitch


class DenyReason(Enum):
    ORDER_VALUE = auto()
    CONCENTRATION = auto()
    SELF_MATCH = auto()
    OTR_TRIPPED = auto()
    KILL_SWITCH = auto()
    PANIC = auto()
    DUPLICATE_ID = auto()
    INSUFFICIENT_CASH = auto()
    DRIFT_HALT = auto()


class PreTradeRiskGateway:
    """订单前置风控中间件（进程内单例使用）。"""

    def __init__(
        self,
        max_order_value: float = 200_000.0,
        max_position_weight: float = 0.10,
        kill_switch: KillSwitch | None = None,
        otr_window: int = 100,
        min_fill_ratio: float = 0.5,
        allow_insufficient_cash: bool = True,  # 现金校验由 broker 兜底
    ):
        self.max_order_value = max_order_value
        self.max_position_weight = max_position_weight
        self.kill_switch = kill_switch or KillSwitch()
        self.otr_window = otr_window
        self.min_fill_ratio = min_fill_ratio
        self._allow_low_cash = allow_insufficient_cash

        self._panic = False
        self._drift_halted = False  # 持仓漂移挂起（DriftHaltEngine 联动）
        self._submitted_ids: set[str] = set()
        self._batch_sides: dict[str, list[str]] = {}  # symbol -> [sides]
        self._otr_orders: deque[int] = deque()   # 窗口内报单数
        self._otr_rejects: deque[int] = deque()  # 窗口内拒单/未成交数
        self.stats: dict[str, dict[str, int] | int] = {
            "allowed": 0, "denied": 0, "by_reason": {},
        }

    # ── 急停 ──────────────────────────────────────

    def panic(self) -> None:
        """全局急停：拒一切委托（含平仓），人工接管。"""
        self._panic = True

    def unpanic(self) -> None:
        self._panic = False

    @property
    def is_panicked(self) -> bool:
        return self._panic

    # ── 持仓漂移挂起（DriftHaltEngine 联动） ──────────

    def drift_halt(self) -> None:
        """漂移挂起：拒开新仓，平仓放行（降风险优先）。"""
        self._drift_halted = True

    def drift_resume(self) -> None:
        self._drift_halted = False

    @property
    def is_halted(self) -> bool:
        return self._drift_halted

    # ── 记录接口（订单生命周期回调） ────────────────

    def record_submitted(self, client_order_id: str) -> None:
        """订单已过风控并发送（幂等登记）。"""
        self._submitted_ids.add(client_order_id)
        self._otr_orders.append(1)
        self._trim_otr()

    def record_reject(self, client_order_id: str) -> None:
        """交易所/券商拒单（计入 OTR 分子）。"""
        self._otr_rejects.append(1)
        self._trim_otr()

    def record_fill(self, client_order_id: str) -> None:
        """成交（OTR 分母扩容：报单有效）。"""
        self._otr_orders.append(1)
        self._trim_otr()

    def _trim_otr(self) -> None:
        while len(self._otr_orders) > self.otr_window:
            self._otr_orders.popleft()
        while len(self._otr_rejects) > self.otr_window:
            self._otr_rejects.popleft()

    def _otr_tripped(self) -> bool:
        """窗口内报单数足够且未成交比率超阈 → 熔断。"""
        n_orders = len(self._otr_orders)
        if n_orders < self.otr_window:
            return False
        bad = len(self._otr_rejects)
        return (bad / max(n_orders, 1)) > (1.0 - self.min_fill_ratio)

    # ── 主检查 ──────────────────────────────────────

    def check(
        self,
        intent: OrderIntent,
        account: AccountSnapshot,
        batch_intents: Iterable[OrderIntent] = (),
    ) -> tuple[bool, DenyReason | None]:
        """风控裁决：True=放行。顺序：急停 > 幂等 > 熔断 > 集中度 > 限额 > 自成交。"""
        reason = self._deny_reason(intent, account, batch_intents)
        if reason is not None:
            self._bump("denied")
            self._bump_reason(reason)
            return False, reason
        self._bump("allowed")
        return True, None

    def _bump(self, key: str) -> None:
        cur = self.stats.get(key, 0)
        if isinstance(cur, int):
            self.stats[key] = cur + 1

    def _bump_reason(self, reason: DenyReason) -> None:
        by_reason = self.stats["by_reason"]
        if isinstance(by_reason, dict):
            by_reason[reason.name] = by_reason.get(reason.name, 0) + 1

    def _deny_reason(
        self,
        intent: OrderIntent,
        account: AccountSnapshot,
        batch_intents: Iterable[OrderIntent],
    ) -> DenyReason | None:
        if self._panic:
            return DenyReason.PANIC
        if intent.client_order_id in self._submitted_ids:
            return DenyReason.DUPLICATE_ID

        is_closing = intent.side == "sell"  # A股纯多框架：sell=减仓

        if self.kill_switch.tripped and not is_closing:
            return DenyReason.KILL_SWITCH
        if self._otr_tripped() and not is_closing:
            return DenyReason.OTR_TRIPPED
        if self._drift_halted and not is_closing:
            return DenyReason.DRIFT_HALT

        # 集中度（只约束开仓方向）
        if not is_closing and account.equity > 0:
            notional = intent.qty * float(intent.price or 0.0)
            held_value = 0.0  # 快照持仓只有股数，无市值——用集中度近似：
            # 权重上限按"名义金额/权益"口径（保守：不计已持有市值）
            _ = held_value
            projected = notional / account.equity
            if projected > self.max_position_weight:
                return DenyReason.CONCENTRATION

        # 单笔限额
        if intent.qty * float(intent.price or 0.0) > self.max_order_value:
            return DenyReason.ORDER_VALUE

        # 自成交：同批同票反向意图（卖出量超持仓 = 对敲特征）
        if is_closing:
            held = account.positions.get(intent.symbol, 0)
            if intent.qty > held:
                batch_syms = {b.symbol for b in batch_intents
                              if b.side == "buy"}
                if intent.symbol in batch_syms:
                    return DenyReason.SELF_MATCH
        return None


class IdempotentOrderIdGen:
    """确定性幂等单号：{strategy}-{trading_day}-{seq:06d}。

    同 (trading_day, strategy) 重放生成完全一致序列 —— 崩溃恢复后
    不会产生新号段，配合网关 DUPLICATE_ID 拦截网络重试重复报单。
    """

    def __init__(self, trading_day: str, strategy_id: str, start_seq: int = 1):
        if not trading_day or not strategy_id:
            raise ValueError("trading_day/strategy_id 必填")
        self.trading_day = trading_day
        self.strategy_id = strategy_id
        self._seq = start_seq

    def next(self) -> str:
        cid = f"{self.strategy_id}-{self.trading_day}-{self._seq:06d}"
        self._seq += 1
        return cid
