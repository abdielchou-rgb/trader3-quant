"""
执行层加固（Execution Hardening）。

对标 Aether Quant 的实盘防护三件套：
1. 流动性 / 市场冲击引擎：方根市场冲击模型估计滑点，预期收益不覆盖成本则拒单
   （expected-cost gate）。
2. 持仓对账：内部簿记 vs 券商实际持仓，差异告警。
3. Kill Switch / 自动回滚：权益回撤超阈即熔断，可回滚到上一已知良好权重。

全部为纯函数 / 轻量状态，可离线测试，无外部依赖。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger("trader3.v2.execution")


@dataclass
class LiquidityEstimate:
    symbol: str
    qty: float
    price: float
    adv: float                         # 日均成交量（股）
    participation_rate: float         # 订单量 / ADV
    slippage_bps: float               # 预估滑点（方根冲击）
    commission_bps: float             # 佣金（买卖单边）
    expected_cost_bps: float          # slippage + commission
    tradable: bool                    # 冲击是否可控（participation 未超限）


class LiquidityEngine:
    """方根市场冲击模型：slippage ≈ k * sqrt(participation) * vol_bps。"""

    def __init__(self,
                 adv: dict[str, float] | None = None,
                 vol_bps: float = 30.0,
                 impact_k: float = 1.0,
                 commission_bps: float = 2.0,
                 max_participation: float = 0.1):
        self.adv = adv or {}
        self.vol_bps = vol_bps
        self.impact_k = impact_k
        self.commission_bps = commission_bps
        self.max_participation = max_participation

    def estimate(self, symbol: str, qty: float, price: float,
                 adv: float | None = None) -> LiquidityEstimate:
        adv_v = adv if adv is not None else self.adv.get(symbol, np.nan)
        if not np.isfinite(adv_v) or adv_v <= 0:
            # 无 ADV 信息：保守假设不可交易 / 高冲击
            return LiquidityEstimate(
                symbol=symbol, qty=qty, price=price, adv=float(adv_v) if np.isfinite(adv_v) else 0.0,
                participation_rate=1.0, slippage_bps=self.vol_bps * 3,
                commission_bps=self.commission_bps,
                expected_cost_bps=self.vol_bps * 3 + self.commission_bps,
                tradable=False)
        part = abs(qty) / max(adv_v, 1e-9)
        slippage = self.impact_k * np.sqrt(min(part, 1.0)) * self.vol_bps
        cost = slippage + self.commission_bps
        tradable = part <= self.max_participation and np.isfinite(slippage)
        return LiquidityEstimate(
            symbol=symbol, qty=qty, price=price, adv=adv_v,
            participation_rate=float(part), slippage_bps=float(slippage),
            commission_bps=self.commission_bps, expected_cost_bps=float(cost),
            tradable=bool(tradable))

    def cost_gate(self, expected_edge_bps: float, est: LiquidityEstimate) -> bool:
        """预期收益（bp）是否覆盖交易成本 + 安全边际。"""
        if not est.tradable:
            return False
        return expected_edge_bps >= est.expected_cost_bps


@dataclass
class PositionDiscrepancy:
    symbol: str
    internal_qty: float
    broker_qty: float
    diff: float
    severity: str          # "match" | "warn" | "breach"


def reconcile_positions(
    internal: dict[str, float],
    broker: dict[str, float],
    tol: float = 1e-6,
    warn_threshold: float = 1e-6,
) -> list[PositionDiscrepancy]:
    """逐标的对账内部簿记 vs 券商实际持仓。"""
    out: list[PositionDiscrepancy] = []
    symbols = set(internal) | set(broker)
    for s in sorted(symbols):
        iq = float(internal.get(s, 0.0))
        bq = float(broker.get(s, 0.0))
        diff = iq - bq
        if abs(diff) <= tol:
            sev = "match"
        elif abs(diff) <= warn_threshold + tol:
            sev = "warn"
        else:
            sev = "breach"
        out.append(PositionDiscrepancy(symbol=s, internal_qty=iq, broker_qty=bq,
                                       diff=diff, severity=sev))
    return out


class KillSwitch:
    """权益回撤熔断 + 可回滚到最后已知良好权重。"""

    def __init__(self, max_drawdown: float = 0.05,
                 last_good_weights: Any | None = None):
        self.max_drawdown = max_drawdown
        self.peak: float | None = None
        self.tripped: bool = False
        self.breach_log: list[dict] = []
        self._last_good = last_good_weights
        self._history: list[tuple[float, float]] = []  # (equity, drawdown)

    def update(self, equity: float) -> bool:
        """喂入最新权益，返回是否处于熔断态。"""
        if self.peak is None or equity > self.peak:
            self.peak = equity
        dd = (self.peak - equity) / self.peak if self.peak and self.peak > 0 else 0.0
        self._history.append((equity, dd))
        if dd > self.max_drawdown:
            if not self.tripped:
                self.tripped = True
                self.breach_log.append({
                    "equity": equity, "peak": self.peak,
                    "drawdown": round(float(dd), 4),
                })
                logger.warning("[kill_switch] 回撤 %.2f%% 超阈 %.2f%%，熔断！",
                               dd * 100, self.max_drawdown * 100)
            return True
        return self.tripped

    def set_last_good(self, weights: Any) -> None:
        self._last_good = weights

    def rollback_weights(self) -> Any | None:
        """返回最后已知良好权重（用于自动回滚）。"""
        return self._last_good

    def reset(self) -> None:
        self.tripped = False

    @property
    def current_drawdown(self) -> float:
        if not self._history:
            return 0.0
        return self._history[-1][1]
