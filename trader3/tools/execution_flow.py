"""
多日流动性约束执行引擎（Participation Cap & Slip Engine）。

痛点：WFA/回测通常在调仓首日一次性全额成交 —— 当调仓金额超过当日
真实成交量的一定比例时，这一假设严重低估冲击与延迟成本。机构做法：

  1. **参与率上限**（participation cap）：单日执行量 ≤ 日均量 × max_rate
     （默认 5%），超额部分**顺延**至后续交易日逐日消化；
  2. **非线性冲击**：Almgren-Chriss 瞬时冲击
        impact = eta × σ × sqrt(participation)
     买入上抬成交价、卖出压低（side 参数控制方向）；
  3. **整手约束**：每日可执行量向下取整到 A 股一手（100 股）；
  4. **诚实报数**：窗口结束仍未成交的量显式返回 unfilled_shares，
     completed=False —— 绝不静默假设"总会成交"。

输出供回测/WFA 将"执行延迟成本"显性化：调仓跨多日时，顺延部分
暴露在后续日价格漂移下（挂单漂移 Drift 风险），不再是首日幻觉。
"""

from __future__ import annotations

import numpy as np

LOT_SIZE = 100


class MultiDayExecutionModel:
    """多日参与率约束执行模拟器。"""

    def __init__(self, max_participation_rate: float = 0.05, eta: float = 0.142):
        """
        Parameters
        ----------
        max_participation_rate : 日成交量占比上限（默认 5%，超过顺延）
        eta : Almgren-Chriss 瞬时冲击系数
        """
        if not 0 < max_participation_rate <= 1.0:
            raise ValueError("max_participation_rate 须在 (0, 1]")
        if eta < 0:
            raise ValueError("eta 须非负")
        self.max_rate = max_participation_rate
        self.eta = eta

    def simulate_order_flow(
        self,
        target_shares: int,
        daily_volumes: np.ndarray,
        daily_prices: np.ndarray,
        daily_volatilities: np.ndarray,
        side: str = "buy",
    ) -> dict:
        """模拟流动性限制下的逐日执行轨迹、滑点与成交均价。

        Parameters
        ----------
        target_shares : 目标股数（正数；方向由 side 决定）
        daily_volumes/prices/volatilities : 后续交易日序列（等长）
        side : 'buy'（冲击上抬）或 'sell'（冲击压低）
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side 须为 buy/sell，got {side!r}")
        vols = np.asarray(daily_volumes, dtype=float)
        prices = np.asarray(daily_prices, dtype=float)
        sigmas = np.asarray(daily_volatilities, dtype=float)
        if not (len(vols) == len(prices) == len(sigmas)):
            raise ValueError("volumes/prices/volatilities 长度须一致")

        remaining = int(target_shares)
        executed_shares: list[int] = []
        realized_prices: list[float] = []
        slippage_costs: list[float] = []
        direction = 1.0 if side == "buy" else -1.0

        for t in range(len(vols)):
            if remaining <= 0:
                break
            vol_t = float(vols[t])
            px_t = float(prices[t])
            sigma_t = float(sigmas[t])
            if vol_t <= 0 or px_t <= 0:
                continue  # 停牌/零成交日跳过

            # 当日最大可执行股数（整手向下取整）
            cap_shares = int(vol_t * self.max_rate // LOT_SIZE) * LOT_SIZE
            fill_shares = min(remaining, cap_shares)
            if fill_shares <= 0:
                continue

            # 瞬时冲击：eta * sigma * sqrt(participation)，方向感知
            participation = fill_shares / (vol_t + 1e-8)
            impact_pct = self.eta * sigma_t * np.sqrt(participation)
            exec_price = px_t * (1.0 + direction * impact_pct)

            executed_shares.append(fill_shares)
            realized_prices.append(exec_price)
            slippage_costs.append(fill_shares * px_t * impact_pct)
            remaining -= fill_shares

        total_executed = int(sum(executed_shares))
        vwap_realized = (
            float(np.average(realized_prices, weights=executed_shares))
            if total_executed > 0 else 0.0
        )

        return {
            "completed": remaining == 0,
            "unfilled_shares": remaining,
            "vwap_price": vwap_realized,
            "total_slippage": float(sum(slippage_costs)),
            "execution_days": len(executed_shares),
            "executed_shares": total_executed,
        }
