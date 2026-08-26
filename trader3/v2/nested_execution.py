"""
嵌套执行（Nested Execution）—— 组合层与订单执行层联合优化。

对标 Qlib NestedExecutor / AlphaAgent 的执行优化：组合层给出目标权重后，执行层考虑
市场冲击与流动性，反将「实现缺款（implementation shortfall）」回灌到组合层，重新求解
冲击调整后的可执行权重，并给出跨多期的最优交易轨迹（Almgren-Chriss 直觉）。

两层耦合：
1. 组合层：以 alpha(得分) + 风险协方差 做均值-方差，得原始目标权重 w*。
2. 执行层：用 LiquidityEngine 估计各标的因换手产生的冲击成本（bp）。
3. 嵌套回灌：alpha 扣减冲击成本得 alpha_adj，重新做均值-方差 → 冲击调整后的可执行权重。
4. 执行轨迹：每标的按 Almgren-Chriss 最优交易速率在 horizon 期内切片（冲击∝速率²，
   风险∝慢速），生成 ExecutionPlan（含 schedule）。

全流程离线、纯数值，可单测。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger("trader3.v2.nested_execution")


@dataclass
class TradeSlice:
    symbol: str
    side: str            # "buy" | "sell"
    qty: float
    period: int          # 第几期（0..horizon-1）


@dataclass
class ExecutionPlan:
    target_weights: pd.Series       # 冲击调整后的可执行目标权重
    raw_target_weights: pd.Series  # 组合层原始目标（未计冲击）
    schedule: list[TradeSlice] = field(default_factory=list)
    expected_cost_bps: float = 0.0
    expected_shortfall: float = 0.0
    participation: dict[str, float] = field(default_factory=dict)


@dataclass
class NestedExecutorConfig:
    horizon: int = 5                 # 执行跨度（期）
    risk_aversion: float = 1.0
    impact_coef: float = 30.0       # 冲击系数（bp per √participation）
    max_single: float = 0.2
    alpha_to_bps: float = 1000.0    # 得分→bp 的经验缩放（使冲击成本可比）


def _mv_weights(alpha: pd.Series, cov: pd.DataFrame, max_single: float) -> pd.Series:
    """均值-方差解析解 w ∝ Σ⁻¹·α，截断+归一。"""
    idx = list(cov.index)
    a = alpha.reindex(idx).fillna(0.0).values.astype(float)
    C = cov.reindex(index=idx, columns=idx).fillna(0.0).values.astype(float)
    C = (C + C.T) / 2 + np.eye(len(idx)) * 1e-6
    try:
        inv = np.linalg.inv(C)
    except Exception:  # noqa: BLE001
        inv = np.eye(len(idx))
    w = inv @ a
    w = np.clip(w, 0.0, None)  # 多头约束，与管线其余层一致
    s = w.sum()
    if s <= 1e-12:
        w = np.ones(len(idx)) / len(idx)
    else:
        w = w / s
    # 投影到单票上限 max_single（迭代 redistribut，保证合计=1）
    for _ in range(20):
        over = w > max_single
        if not over.any():
            break
        excess = float((w[over] - max_single).sum())
        w[over] = max_single
        under = ~over
        if under.any() and w[under].sum() > 1e-12:
            w[under] = w[under] + excess * (w[under] / w[under].sum())
        elif under.any():
            w[under] = w[under] + excess / under.sum()
        else:
            w = np.full(len(idx), max_single)
            break
    return pd.Series(w, index=idx)


def _almgren_chriss_rate(alpha_bps: float, volatility: float,
                         impact_coef: float, risk_aversion: float,
                         horizon: int) -> float:
    """最优常交易速率（每期换手比例）。

    权衡：冲击成本 ∝ rate²·impact_coef·horizon；延迟风险 ∝ (1-rate)·volatility·risk_aversion。
    d/drate = 2·rate·impact_coef·horizon - volatility·risk_aversion = 0
             → rate* = volatility·risk_aversion / (2·impact_coef·horizon)。
    """
    denom = 2 * impact_coef * horizon
    rate = (risk_aversion * max(volatility, 0.0)) / (denom + 1e-9)
    return float(min(1.0, max(0.0, rate)))


class NestedExecutor:
    """组合层 + 执行层联合优化器。"""

    def __init__(self, config: NestedExecutorConfig | None = None, liquidity=None):
        from trader3.v2.execution import LiquidityEngine
        self.config = config or NestedExecutorConfig()
        self.liquidity = liquidity or LiquidityEngine()

    def solve(self, alpha: pd.Series, cov: pd.DataFrame,
              current_weights: pd.Series, equity: float,
              prices: dict[str, float], adv: dict[str, float]) -> ExecutionPlan:
        cfg = self.config
        alpha = alpha.fillna(0.0)
        raw = _mv_weights(alpha, cov, cfg.max_single)

        # 换手 + 冲击成本
        cur = current_weights.reindex(raw.index).fillna(0.0)
        turnover = (raw - cur).abs()
        slip: dict[str, float] = {}
        part: dict[str, float] = {}
        total_shortfall = 0.0
        for s in raw.index:
            notional = abs(turnover[s]) * equity
            px = prices.get(s, float("nan"))
            if not np.isfinite(px) or px <= 0:
                slip[s] = 0.0
                part[s] = 0.0
                continue
            qty = notional / px if px > 0 else 0.0
            est = self.liquidity.estimate(s, qty, px, adv.get(s))
            slip[s] = est.slippage_bps
            part[s] = est.participation_rate
            total_shortfall += notional * est.slippage_bps / 1e4

        expected_cost_bps = (total_shortfall / equity * 1e4) if equity > 0 else 0.0

        # 嵌套回灌：alpha 扣冲击（同量纲 bp）
        alpha_bps = alpha * cfg.alpha_to_bps
        cost_bps = pd.Series({s: slip[s] for s in raw.index})
        adj = alpha_bps - cost_bps
        target = _mv_weights(adj, cov, cfg.max_single)

        # 执行轨迹：每标的最优速率切片
        schedule: list[TradeSlice] = []
        for s in target.index:
            delta = target[s] - cur[s]
            if abs(delta) <= 1e-9:
                continue
            px = prices.get(s, float("nan"))
            if not np.isfinite(px) or px <= 0:
                continue
            shares = abs(delta) * equity / px
            if shares <= 0:
                continue
            vol = float(np.sqrt(cov.loc[s, s])) if s in cov.index else 0.0
            rate = _almgren_chriss_rate(abs(adj.get(s, 0.0)), vol,
                                        cfg.impact_coef, cfg.risk_aversion, cfg.horizon)
            rate = max(rate, 1.0 / max(cfg.horizon, 1))  # 至少 horizon 期内平摊完
            side = "buy" if delta > 0 else "sell"
            remaining = shares
            for p in range(cfg.horizon):
                q = min(remaining, shares * rate)
                if q <= 1e-9:
                    break
                schedule.append(TradeSlice(symbol=s, side=side, qty=q, period=p))
                remaining -= q
                if remaining <= 1e-9:
                    break

        return ExecutionPlan(
            target_weights=target, raw_target_weights=raw, schedule=schedule,
            expected_cost_bps=round(float(expected_cost_bps), 4),
            expected_shortfall=round(float(total_shortfall), 4),
            participation=part,
        )
