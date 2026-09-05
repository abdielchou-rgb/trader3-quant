"""
实盘运行时（E2 实盘侧）：网关回调驱动同一套 DualModeStrategy。

与 ReplayRuntime 的唯一差异是事件来源：push_bar 由行情网关注入
（真实场景接 QMT/CTP 回调），意图收集后交前置风控网关（E3）拦截。
策略代码零改动 —— Train-Serving Skew 的结构性根治。
"""

from __future__ import annotations

from trader3.runtime.events import BarEvent, OrderIntent
from trader3.runtime.strategy import AccountSnapshot, DualModeStrategy


class LiveRuntime:
    """实盘事件驱动运行时（缓冲 push，collect 时统一喂策略）。"""

    def __init__(self, account: AccountSnapshot | None = None):
        self.account = account or AccountSnapshot()
        self._pending: list[BarEvent] = []

    def push_bar(self, bar: BarEvent) -> None:
        """行情网关注入（真实路径：broker 行情回调）。"""
        self._pending.append(bar)

    def collect_intents(
        self,
        strategy: DualModeStrategy,
    ) -> list[OrderIntent]:
        """消化缓冲事件 → 策略意图（生产环境可改为逐事件即时触发）。"""
        intents: list[OrderIntent] = []
        for bar in self._pending:
            self.account.last_update_ts = bar.local_ts or bar.ts
            intents.extend(strategy.on_bar(bar, self.account))
        self._pending.clear()
        return intents
