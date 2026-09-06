"""
实盘运行时（E2 实盘侧 + F1 闭环）：网关回调驱动同一套 DualModeStrategy。

与 ReplayRuntime 的唯一差异是事件来源：push_bar 由行情网关注入
（真实场景接 QMT/CTP 回调），意图收集后交前置风控网关（E3）拦截。

F1 闭环：attach(gateway, executor) 后，collect_intents 全链路 =
策略意图 → 风控裁决 → WAL 执行器落盘发送；fill() 回流成交回报，
更新账户快照并记录双时间戳延迟。策略代码零改动。
"""

from __future__ import annotations

from trader3.runtime.events import BarEvent, FillEvent, OrderIntent
from trader3.runtime.strategy import AccountSnapshot, DualModeStrategy


class LiveRuntime:
    """实盘事件驱动运行时（缓冲 push，collect 时统一喂策略）。"""

    def __init__(self, account: AccountSnapshot | None = None):
        self.account = account or AccountSnapshot()
        self._pending: list[BarEvent] = []
        # F1 闭环组件（可选注入）
        self._gateway = None      # PreTradeRiskGateway
        self._executor = None     # RobustQMTExecutor / 兼容桩
        self._metrics = None      # MetricsRegistry
        self.fill_latencies_ms: list[float] = []

    # ── 组件装配 ──────────────────────────────────

    def attach(self, gateway=None, executor=None, metrics=None) -> None:
        """装配生产链路：风控网关 + WAL 执行器 + 指标注册表。"""
        if gateway is not None:
            self._gateway = gateway
        if executor is not None:
            self._executor = executor
        if metrics is not None:
            self._metrics = metrics
        elif metrics is None and gateway is not None:
            # 风控在位但未显式给 registry → 懒建，确保拒绝/放行计数不丢
            self._ensure_metrics()

    # ── 行情注入 ──────────────────────────────────

    def push_bar(self, bar: BarEvent) -> None:
        """行情网关注入（真实路径：broker 行情回调）。"""
        self._pending.append(bar)

    # ── 意图收集 + 风控 + 执行 ────────────────────

    def collect_intents(
        self,
        strategy: DualModeStrategy,
    ) -> list[OrderIntent]:
        """消化缓冲事件 → 策略意图 → （装配时）风控裁决 → 执行器。

        返回**被放行**的意图列表（生产语义：放行即已交执行器）。
        未装配 gateway 时透传（联调/回放兼容）。
        """
        intents: list[OrderIntent] = []
        for bar in self._pending:
            self.account.last_update_ts = bar.local_ts or bar.ts
            intents.extend(strategy.on_bar(bar, self.account))
        self._pending.clear()

        if self._gateway is None:
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
        return allowed

    # ── 成交回报回流 ──────────────────────────────

    def fill(self, ev: FillEvent) -> None:
        """成交回报：更新账户 + 双时间戳延迟指标 + gateway 记账。"""
        if self._gateway is not None:
            self._gateway.record_fill(ev.client_order_id)
        pos = dict(self.account.positions)
        cur = pos.get(ev.symbol, 0.0)
        # 语义：qty 恒为正，方向由意图决定；此处简化为加仓（卖出回报由上游转负）
        pos[ev.symbol] = cur + ev.qty
        self.account.positions = pos
        self.account.last_update_ts = ev.local_ts
        self.fill_latencies_ms.append(float(max(0, ev.local_ts - ev.exchange_ts)))
        if self._metrics is not None:
            for lag in self.fill_latencies_ms[-1:]:
                self._metrics.histogram("fill_latency_ms").observe(lag)

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
