"""
实盘运行时（E2 实盘侧 + F1 闭环 + G1/G3 即时与乱序鲁棒）。

双模式：
  - 缓冲模式（默认，联调/回放兼容）：push_bar 入缓冲，collect_intents(strategy) 统一消化。
  - 即时模式（set_immediate(strategy)，生产）：push_bar 当场触发
    策略→风控→执行全链路，RLock 保证回调串行化（行情线程并发安全）。

回调乱序鲁棒性（G3，真实网关时序）：
  - fill 幂等：相同 client_order_id 的回报只记一次（网络重发防重复计仓）
  - fill 方向：sell 回报由上游 fill(..., side='sell') 转负
  - fill 先于意图到达（跨线程乱序）：仓位照记（券商事实优先），
    迟到意图若与已记回报同单号 → gateway DUPLICATE_ID 拦截，不重复执行
"""

from __future__ import annotations

import threading

from trader3.runtime.events import BarEvent, FillEvent, OrderIntent
from trader3.runtime.strategy import AccountSnapshot, DualModeStrategy


class LiveRuntime:
    """实盘事件驱动运行时。"""

    def __init__(self, account: AccountSnapshot | None = None):
        self.account = account or AccountSnapshot()
        self._pending: list[BarEvent] = []
        # F1 闭环组件（可选注入）
        self._gateway = None      # PreTradeRiskGateway
        self._executor = None     # RobustQMTExecutor / 兼容桩
        self._metrics = None      # MetricsRegistry
        self.fill_latencies_ms: list[float] = []
        # G1/G3 即时模式与乱序状态
        self._immediate_strategy: DualModeStrategy | None = None
        self._lock = threading.RLock()
        self._seen_fill_ids: set[str] = set()      # 幂等：已记账回报单号
        self._recorded_intent_count = 0
        self._orphan_fills = 0                      # 先于意图在途记录的回报数

    # ── 组件装配 ──────────────────────────────────

    def attach(self, gateway=None, executor=None, metrics=None) -> None:
        """装配生产链路：风控网关 + WAL 执行器 + 指标注册表。"""
        with self._lock:
            if gateway is not None:
                self._gateway = gateway
            if executor is not None:
                self._executor = executor
            if metrics is not None:
                self._metrics = metrics
            elif metrics is None and gateway is not None:
                # 风控在位但未显式给 registry → 懒建，确保拒绝/放行计数不丢
                self._ensure_metrics()

    def set_immediate(self, strategy: DualModeStrategy | bool) -> None:
        """即时模式：传策略对象开启（push_bar 当场触发全链路）；
        传 False 关闭回缓冲模式。"""
        if strategy is False:
            self._immediate_strategy = None
            return
        if not isinstance(strategy, DualModeStrategy):
            raise TypeError("set_immediate 需传 DualModeStrategy 实例或 False")
        self._immediate_strategy = strategy

    # ── 行情注入 ──────────────────────────────────

    def push_bar(self, bar: BarEvent) -> None:
        """行情网关注入（真实路径：broker 行情回调）。"""
        if self._immediate_strategy is not None:
            with self._lock:
                self.account.last_update_ts = bar.local_ts or bar.ts
                intents = self._immediate_strategy.on_bar(bar, self.account)
                self._gate_and_execute(intents)
            return
        with self._lock:
            self._pending.append(bar)

    # ── 缓冲模式批量消化 ────────────────────────────

    def collect_intents(
        self,
        strategy: DualModeStrategy,
    ) -> list[OrderIntent]:
        """消化缓冲事件 → 策略意图 → （装配时）风控裁决 → 执行器。

        即时模式下缓冲恒空（已即时消化），保留方法兼容旧调用。
        返回**被放行**的意图列表（生产语义：放行即已交执行器）。
        未装配 gateway 时透传（联调/回放兼容）。
        """
        with self._lock:
            pending, self._pending = self._pending, []
        intents: list[OrderIntent] = []
        for bar in pending:
            self.account.last_update_ts = bar.local_ts or bar.ts
            intents.extend(strategy.on_bar(bar, self.account))
        return self._gate_and_execute(intents)

    # ── 内部：风控 + 执行 ────────────────────────

    def _gate_and_execute(self, intents: list[OrderIntent]) -> list[OrderIntent]:
        """风控裁决 + 执行器转发 + 计数。无 gateway 透传。"""
        if self._gateway is None:
            self._recorded_intent_count += len(intents)
            return intents
        allowed: list[OrderIntent] = []
        for it in intents:
            ok, reason = self._gateway.check(it, self.account,
                                             batch_intents=intents)
            if not ok:
                self._record_deny(reason)
                continue
            self._record_allow()
            self._gateway.record_submitted(it.client_order_id)
            if self._executor is not None:
                px = float(it.price or 0.0)
                self._executor.submit_safe_order(
                    it.client_order_id, it.symbol, it.qty, px)
            allowed.append(it)
        self._recorded_intent_count += len(allowed)
        return allowed

    # ── 成交回报回流 ──────────────────────────────

    def fill(self, ev: FillEvent, side: str = "buy") -> None:
        """成交回报：幂等 + 方向 + 账户更新 + 延迟指标。

        Parameters
        ----------
        ev : 回报事件
        side : 'buy' 加仓 / 'sell' 减仓（上游意图方向；回报 qty 恒正）
        """
        with self._lock:
            if ev.client_order_id in self._seen_fill_ids:
                return  # 网络重发 → 幂等丢弃
            self._seen_fill_ids.add(ev.client_order_id)
            pos = dict(self.account.positions)
            cur = pos.get(ev.symbol, 0.0)
            delta = ev.qty if side == "buy" else -ev.qty
            pos[ev.symbol] = cur + delta
            self.account.positions = pos
            self.account.last_update_ts = ev.local_ts
            lag = float(max(0, ev.local_ts - ev.exchange_ts))
            self.fill_latencies_ms.append(lag)
            if self._gateway is not None:
                self._gateway.record_fill(ev.client_order_id)
            self._orphan_fills += 1  # 保守计数：回报到达时未校验意图在途
        reg = self._ensure_metrics()
        reg.histogram("fill_latency_ms").observe(lag)

    # ── 指标 ──────────────────────────────────────

    def _ensure_metrics(self):
        if self._metrics is None:
            from trader3.obs.metrics import build_default_registry
            self._metrics = build_default_registry()
        return self._metrics

    def _record_deny(self, reason) -> None:
        reg = self._ensure_metrics()
        if reason is not None:
            reg.counter("risk_denied_total").inc(reason=reason.name)

    def _record_allow(self) -> None:
        self._ensure_metrics().counter("risk_allowed_total").inc()

    def render_metrics(self) -> str:
        """输出 Prometheus 文本（未装配 registry 时懒建默认集）。"""
        return self._ensure_metrics().render()
