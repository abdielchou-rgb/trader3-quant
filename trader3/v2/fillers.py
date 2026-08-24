"""3号交易员 v2.1 — 成交量填充器（backtrader Fillers 风格）

吸收 backtrader fillers 思想：成交需受「当根 Bar 流动性」约束，
避免在涨停一字板（无卖盘）买入、跌停一字板（无买盘）卖出时假设全额成交。

实现：
- FixedSizeFiller：固定按请求量成交（默认，向后兼容）
- BarVolumeFiller：按当根 Bar 成交量限制实际成交量
  - 涨停（price >= prev_close*(1+limit)）：买入量 ≤ 可卖供给（通常为 0 / 少量）
  - 跌停（price <= prev_close*(1-limit)）：卖出量 ≤ 可买承接（通常为 0 / 少量）
  - 未触及涨跌停：按 bar.volume 的 capacity_ratio 限制（如 10%）

用法：
    from trader3.v2.fillers import BarVolumeFiller, FixedSizeFiller
    f = BarVolumeFiller(limit_pct=0.10)
    filled = f.fill("buy", 1000, price=12.6, prev_close=11.5, bar_volume=8000)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FixedSizeFiller:
    """固定按请求量成交（不使用流动性约束）"""

    def fill(self, action: str, requested: float, price: float,
             prev_close: Optional[float] = None, limit_pct: Optional[float] = None,
             bar_volume: Optional[float] = None) -> float:
        return max(requested, 0.0)


@dataclass
class BarVolumeFiller:
    """按当根 Bar 成交量 / 涨跌停状态限制成交量（backtrader Fillers 风格）"""

    limit_pct: float = 0.10          # 涨跌停幅度（10% / 20% / 30%）
    tolerance: float = 1e-4          # 价格判定容差
    capacity_ratio: float = 0.10     # 未涨跌停时：成交量 × ratio 作为可成交上限
    limit_ratio: float = 0.001       # 涨跌停时：成交量 × ratio（一字板近似为 0）

    def _is_limit_up(self, price: float, prev_close: float) -> bool:
        return price >= prev_close * (1 + self.limit_pct) - self.tolerance

    def _is_limit_down(self, price: float, prev_close: float) -> bool:
        return price <= prev_close * (1 - self.limit_pct) + self.tolerance

    def fill(self, action: str, requested: float, price: float,
             prev_close: Optional[float] = None, limit_pct: Optional[float] = None,
             bar_volume: Optional[float] = None) -> float:
        """
        action: 'buy' / 'sell'
        requested: 请求量（股）
        price: 委托价
        prev_close: 昨收价（用于涨跌停判定；None 则跳过）
        limit_pct: 涨跌停幅度覆盖（None 用 self.limit_pct）
        bar_volume: 当根 Bar 成交量（用于流动性约束；None 则全额成交）
        """
        if requested <= 0:
            return 0.0
        limit = self.limit_pct if limit_pct is None else limit_pct

        # 1) 涨跌停拦截：涨停买入 / 跌停卖出 无对手盘
        if prev_close is not None and prev_close > 0:
            if action == "buy" and self._is_limit_up(price, prev_close):
                cap = (bar_volume or 0.0) * self.limit_ratio
                return min(requested, cap)
            if action == "sell" and self._is_limit_down(price, prev_close):
                cap = (bar_volume or 0.0) * self.limit_ratio
                return min(requested, cap)

        # 2) 常规流动性约束
        if bar_volume is not None and bar_volume > 0:
            cap = bar_volume * self.capacity_ratio
            return min(requested, cap)

        # 3) 无流动性信息 → 全额成交
        return requested


def build_filler(name: Optional[str] = None, **kwargs) -> object:
    """按名称构建填充器；None / 'fixed' 返回 FixedSizeFiller"""
    if name in (None, "", "fixed", "FixedSizeFiller"):
        return FixedSizeFiller()
    if name in ("bar", "volume", "BarVolumeFiller"):
        return BarVolumeFiller(**kwargs)
    raise ValueError(f"未知填充器: {name}")
