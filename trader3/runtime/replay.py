"""
回放运行时 + 时间戳守卫（E2 回测侧）。

ReplayRuntime：按 exchange_ts 顺序推进历史 bar 流，经 TimestampGuard
校验因果合法性（local_ts >= exchange_ts），驱动 DualModeStrategy 产出
确定性意图序列。

前视根治：守卫拒绝任何"本地接收时刻早于交易所时刻"的事件 —— 即
事件在现实中尚未到达就被策略消费，构成未来函数。
"""

from __future__ import annotations

from collections.abc import Iterable

from trader3.runtime.events import BarEvent, OrderIntent
from trader3.runtime.strategy import AccountSnapshot, DualModeStrategy


class LookaheadError(ValueError):
    """回放流含未来事件（local_ts < exchange_ts）。"""


class TimestampGuard:
    """因果守卫：exchange_ts（源时刻）必须不晚于 local_ts（接收时刻）。"""

    def __init__(self, max_lag_ms: int = 10_000):
        self.max_lag_ms = max_lag_ms
        self.rejected = 0

    def check(self, exchange_ts: int, local_ts: int) -> bool:
        ok = local_ts >= exchange_ts and (local_ts - exchange_ts) <= self.max_lag_ms
        if not ok:
            self.rejected += 1
        return ok


class ReplayRuntime:
    """历史流驱动（回测运行时）。"""

    def __init__(self, local_ts_lag: int = 5, account: AccountSnapshot | None = None):
        # local_ts_lag：回放时本地接收时刻 = exchange_ts + lag（模拟传输延迟）
        self.lag = local_ts_lag
        self.account = account or AccountSnapshot()
        self.guard = TimestampGuard()
        self.events_consumed = 0

    def run(
        self,
        strategy: DualModeStrategy,
        bars: Iterable[BarEvent],
    ) -> list[OrderIntent]:
        intents: list[OrderIntent] = []
        for bar in bars:
            local_ts = bar.local_ts if bar.local_ts is not None else bar.ts + self.lag
            if not self.guard.check(exchange_ts=bar.ts, local_ts=local_ts):
                raise LookaheadError(
                    f"未来事件被拒绝消费: {bar.symbol} exchange_ts={bar.ts} "
                    f"local_ts={local_ts}（local_ts 不得早于 exchange_ts）"
                )
            self.account.last_update_ts = local_ts
            intents.extend(strategy.on_bar(bar, self.account))
            self.events_consumed += 1
        return intents
