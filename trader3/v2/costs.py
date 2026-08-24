"""3号交易员 v2.1 — 可插拔交易费用模型（backtrader CommissionInfo 风格）

吸收 backtrader CommissionInfo 设计：费用参数集中为一个对象，按
「印花税（卖出单边） + 佣金（双边） + 冲击成本（双边）」合成总成本。

默认值对齐原 backtest.py 固定费率：
    stamp_tax_bp=10 / commission_bp=2 / slippage_bp=5  →  TOTAL_COST_BP=17bp

用法：
    from trader3.v2.costs import CommissionInfo, DEFAULT_COSTS
    cm = CommissionInfo(commission_bp=2, stamp_tax_bp=5, slippage_bp=3)
    cm.total_bp()                 # -> 10
    cm.turnover_cost(turnover)    # turnover(换手率) -> 成本(收益扣减)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CommissionInfo:
    """交易费用参数（bp = 万分之 1）"""

    commission_bp: float = 2.0   # 佣金（双边）
    stamp_tax_bp: float = 10.0   # 印花税（卖出单边）
    slippage_bp: float = 5.0     # 冲击成本（双边）
    min_commission: float = 5.0  # 最低佣金（元，单笔）

    def total_bp(self) -> float:
        """单笔双边总成本 bp"""
        return self.commission_bp + self.stamp_tax_bp + self.slippage_bp

    def turnover_cost(self, turnover: float) -> float:
        """按换手率（双边，如 0.35 表示 35%）计算成本（组合日收益扣减）"""
        return turnover * self.total_bp() / 10000.0

    def as_dict(self) -> dict:
        return {
            "commission_bp": self.commission_bp,
            "stamp_tax_bp": self.stamp_tax_bp,
            "slippage_bp": self.slippage_bp,
            "total_bp": self.total_bp(),
        }

    def describe(self) -> str:
        return (
            f"费用模型[佣金{self.commission_bp:.0f}bp + 印花税{self.stamp_tax_bp:.0f}bp"
            f" + 冲击{self.slippage_bp:.0f}bp = {self.total_bp():.0f}bp/笔]"
        )


# 默认费用（与原 backtest.py 17bp 一致）
DEFAULT_COSTS = CommissionInfo()


def build_commission(
    commission_bp: Optional[float] = None,
    stamp_tax_bp: Optional[float] = None,
    slippage_bp: Optional[float] = None,
) -> CommissionInfo:
    """按需覆盖默认费用；None 保持默认值"""
    return CommissionInfo(
        commission_bp=DEFAULT_COSTS.commission_bp if commission_bp is None else commission_bp,
        stamp_tax_bp=DEFAULT_COSTS.stamp_tax_bp if stamp_tax_bp is None else stamp_tax_bp,
        slippage_bp=DEFAULT_COSTS.slippage_bp if slippage_bp is None else slippage_bp,
    )
