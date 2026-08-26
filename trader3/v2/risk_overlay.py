"""
风险覆盖层（Risk Overlay）— 每日管线开仓前的组合级总闸。

串联上一轮新建的三个模块：
- regime.detect_regimes   → 市场状态门（turbulent 减仓/停新仓）
- tools.stress            → 尾部门（最差情景 PnL 低于阈值降杠杆）
- 已实现波动 / VaR        → 波动门（组合波动超限降规模）

输出统一的 size_multiplier ∈ [0,1] 与 halt_new_buys 开关，
供 daily_pipeline.run_paper_trades 在下单前调整 PAPER_TRADE_PCT。

设计原则：
- 纯函数、无副作用，输入缺失时逐级降级（fail-open 到中性乘数=1，
  但记录 reason）；任何异常不阻断主链路。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger("trader3.v2.risk_overlay")


@dataclass
class OverlayConfig:
    """覆盖层参数（全部可调，默认保守）。"""
    enabled: bool = True
    # 状态门
    turbulent_multiplier: float = 0.5     # turbulent 时仓位乘数
    calm_confidence_min: float = 0.60     # 判定状态所需最低置信度
    # 波动门
    vol_target_annual: float = 0.20       # 组合年化波动目标
    vol_lookback: int = 60                # 已实现波动回看天数
    max_vol_multiplier_cut: float = 0.25  # 波动门最多砍到 0.25×
    # 尾部门
    stress_pnl_floor: float = -0.15       # 最差情景 PnL 低于此值触发降杠杆
    stress_floor_multiplier: float = 0.6  # 触发后的乘数
    var95_daily_limit: float = 0.03       # 单日 95% VaR 上限（3%）
    var_breach_multiplier: float = 0.7


@dataclass
class OverlayResult:
    """覆盖层决策结果。"""
    size_multiplier: float                 # 最终仓位乘数 [0,1]
    halt_new_buys: bool                    # 是否暂停全部新买入
    gates: dict[str, str]                  # 各门的判定说明
    contributions: dict[str, float] = field(default_factory=dict)  # 各门乘数贡献


def realized_vol(returns: pd.Series | np.ndarray, lookback: int = 60,
                 periods_per_year: int = 252) -> float:
    """年化已实现波动。"""
    r = pd.Series(np.asarray(returns, dtype=float)).dropna().tail(lookback)
    if len(r) < max(10, lookback // 3):
        return float("nan")
    return float(r.std(ddof=0) * np.sqrt(periods_per_year))


def historical_var(returns: pd.Series | np.ndarray, confidence: float = 0.95,
                   lookback: int = 250) -> float:
    r = pd.Series(np.asarray(returns, dtype=float)).dropna().tail(lookback)
    if len(r) < 30:
        return float("nan")
    return float(-np.quantile(r, 1 - confidence))


def compute_risk_overlay(
    portfolio_returns: pd.Series | np.ndarray | None,
    config: OverlayConfig | None = None,
    regime_label: str | None = None,
    regime_confidence: float | None = None,
    worst_scenario_pnl: float | None = None,
) -> OverlayResult:
    """
    计算当日风险覆盖层决策。

    参数：
    - portfolio_returns: 组合日收益序列（波动门/VaR 门必需；None 则跳过这两门）
    - regime_label / regime_confidence: detect_regimes + current_regime 的输出；
      为 None 时现场计算（若 returns 可用）
    - worst_scenario_pnl: run_stress_suite 结果的最差 pnl；None 时现场计算
    """
    cfg = config or OverlayConfig()
    gates: dict[str, str] = {}
    contribs: dict[str, float] = {}
    mult = 1.0
    halt = False

    if not cfg.enabled:
        return OverlayResult(size_multiplier=1.0, halt_new_buys=False,
                             gates={"overlay": "disabled"}, contributions={})

    rets = pd.Series(np.asarray(portfolio_returns, dtype=float)).dropna() \
        if portfolio_returns is not None else pd.Series(dtype=float)

    # ── 门1：市场状态 ──────────────────────────────
    try:
        label, conf = regime_label, regime_confidence
        if (label is None or conf is None) and len(rets) >= 100:
            from trader3.v2.regime import current_regime, detect_regimes
            res = detect_regimes(rets.values, n_states=2, random_state=42)
            label, conf = current_regime(res)
        if label is not None and conf is not None:
            if label == "turbulent" and conf >= cfg.calm_confidence_min:
                mult *= cfg.turbulent_multiplier
                contribs["regime"] = cfg.turbulent_multiplier
                gates["regime"] = f"turbulent(conf={conf:.0%}) → ×{cfg.turbulent_multiplier}"
            else:
                gates["regime"] = f"{label}(conf={conf:.0%}) → pass"
        else:
            gates["regime"] = "insufficient data → pass(fail-open)"
    except Exception as e:  # noqa: BLE001 — 覆盖层不阻断主链路
        gates["regime"] = f"error({e}) → pass(fail-open)"

    # ── 门2：已实现波动 vs 目标 ────────────────────
    try:
        rv = realized_vol(rets, cfg.vol_lookback)
        if np.isfinite(rv):
            if rv > cfg.vol_target_annual:
                raw = cfg.vol_target_annual / rv
                cut = max(raw, cfg.max_vol_multiplier_cut)
                mult *= cut
                contribs["vol"] = cut
                gates["vol"] = f"rv={rv:.1%} > target={cfg.vol_target_annual:.0%} → ×{cut:.2f}"
            else:
                gates["vol"] = f"rv={rv:.1%} ≤ target → pass"
        else:
            gates["vol"] = "insufficient data → pass"
    except Exception as e:  # noqa: BLE001
        gates["vol"] = f"error({e}) → pass"

    # ── 门3：VaR 上限 ──────────────────────────────
    try:
        v95 = historical_var(rets, 0.95)
        if np.isfinite(v95):
            if v95 > cfg.var95_daily_limit:
                mult *= cfg.var_breach_multiplier
                contribs["var"] = cfg.var_breach_multiplier
                gates["var"] = f"VaR95={v95:.2%} > {cfg.var95_daily_limit:.0%} → ×{cfg.var_breach_multiplier}"
            else:
                gates["var"] = f"VaR95={v95:.2%} ≤ limit → pass"
        else:
            gates["var"] = "insufficient data → pass"
    except Exception as e:  # noqa: BLE001
        gates["var"] = f"error({e}) → pass"

    # ── 门4：压力情景尾部 ─────────────────────────
    try:
        worst = worst_scenario_pnl
        if worst is None and len(rets) >= 120:
            from trader3.tools.stress import run_stress_suite
            suite = run_stress_suite(rets, include_historical=True,
                                     include_hypothetical=True)
            worst = suite[0].portfolio_pnl_pct if suite else None
        if worst is not None:
            if worst < cfg.stress_pnl_floor:
                mult *= cfg.stress_floor_multiplier
                contribs["stress"] = cfg.stress_floor_multiplier
                gates["stress"] = (
                    f"worst={worst:+.1%} < floor={cfg.stress_pnl_floor:+.0%} "
                    f"→ ×{cfg.stress_floor_multiplier}")
            else:
                gates["stress"] = f"worst={worst:+.1%} ≥ floor → pass"
        else:
            gates["stress"] = "insufficient data → pass"
    except Exception as e:  # noqa: BLE001
        gates["stress"] = f"error({e}) → pass"

    # 极端保护：乘数被压得过低时直接熔断新开仓
    if mult <= 0.25:
        halt = True
        gates["circuit_breaker"] = f"multiplier={mult:.2f} ≤ 0.25 → HALT new buys"
    else:
        gates["circuit_breaker"] = "pass"

    logger.info("[overlay] mult=%.2f halt=%s gates=%s", mult, halt, gates)
    return OverlayResult(size_multiplier=float(min(max(mult, 0.0), 1.0)),
                         halt_new_buys=halt, gates=gates, contributions=contribs)


def apply_to_trade_pct(base_pct: float, overlay: OverlayResult) -> float:
    """把覆盖层结果应用到每单资金比例上。halt 时返回 0（不开新仓）。"""
    if overlay.halt_new_buys:
        return 0.0
    return float(base_pct * overlay.size_multiplier)


def summarize(overlay: OverlayResult) -> str:
    """人类可读摘要（写进每日提醒）。"""
    lines = [f"- 仓位乘数 ×{overlay.size_multiplier:.2f}"
             + ("｜[HALT] 暂停新开仓" if overlay.halt_new_buys else "")]
    for name, desc in overlay.gates.items():
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)
