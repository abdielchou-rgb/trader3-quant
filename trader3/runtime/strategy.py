"""
双模策略基类与账户快照（E2 的策略侧契约）。

DualModeStrategy：策略逻辑只依赖 on_bar(bar, account) → list[OrderIntent]。
同一份代码：
  - ReplayRuntime（历史流推进）驱动 = 回测
  - LiveRuntime（网关回调）驱动   = 实盘
两个运行时的差异只在事件来源与撮合去向，策略零感知。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from trader3.runtime.events import BarEvent, OrderIntent


@dataclass
class AccountSnapshot:
    """策略可见的账户快照（只读视图，运行时维护）。"""
    cash: float = 1_000_000.0
    equity: float = 1_000_000.0
    positions: dict[str, float] = field(default_factory=dict)  # symbol -> qty
    last_update_ts: int = 0


class DualModeStrategy(ABC):
    """双模同构策略接口。"""

    @abstractmethod
    def on_bar(self, bar: BarEvent, account: AccountSnapshot) -> list[OrderIntent]:
        """收到一根 bar → 返回 0..N 个下单意图（纯函数式：不直接下单）。"""
        raise NotImplementedError
