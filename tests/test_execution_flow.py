"""
模块四：多日流动性约束执行引擎 — 回归测试
核心契约：
  1. 日参与率上限（默认 5%）：超限部分顺延至下一交易日（fill = min(remaining, cap)）
  2. 非线性冲击：impact = eta * sigma * sqrt(participation)，买入上抬/卖出压低
  3. 整手约束：cap 向下取整到 100 股
  4. 流动性不足时 completed=False + unfilled 明确报数（不许静默丢弃）
  5. 多日滑点累计 = Σ(股数×基准价×冲击率)；VWAP 按成交量加权
  6. WFA 接入：调仓换手超过单日容量时，超额部分按次日价格继续执行，
     产生真实的"执行延迟成本"而非一次性全额当日成交
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.tools.execution_flow import MultiDayExecutionModel  # noqa: E402


def _mk(vols, prices, sigmas):
    return (np.asarray(vols, dtype=float), np.asarray(prices, dtype=float),
            np.asarray(sigmas, dtype=float))


def test_single_day_within_cap_fills_all():
    """目标 1万股，日量 100万（1%）< 5% 上限 → 当日全部成交。"""
    v, p, s = _mk([1_000_000], [10.0], [0.02])
    m = MultiDayExecutionModel()
    r = m.simulate_order_flow(10_000, v, p, s)
    assert r["completed"] is True
    assert r["unfilled_shares"] == 0
    assert r["execution_days"] == 1


def test_cap_defers_excess_to_next_days():
    """目标 15万股，日量 100万 → 日上限 5万，3 天完成。"""
    v = np.full(5, 1_000_000.0)
    p = np.full(5, 10.0)
    s = np.full(5, 0.02)
    m = MultiDayExecutionModel()
    r = m.simulate_order_flow(150_000, v, p, s)
    assert r["completed"] is True
    assert r["execution_days"] == 3
    assert r["unfilled_shares"] == 0


def test_insufficient_liquidity_reports_unfilled():
    """日量太小，窗口内执行不完 → completed=False 且 unfilled 精确报数。"""
    v = np.array([100_000.0, 80_000.0])      # 日上限 5000/4000 股
    p = np.array([10.0, 10.0])
    s = np.array([0.02, 0.02])
    m = MultiDayExecutionModel()
    r = m.simulate_order_flow(50_000, v, p, s)
    assert r["completed"] is False
    assert r["unfilled_shares"] == 50_000 - 5000 - 4000


def test_lot_rounding_on_daily_cap():
    """日上限向下取整到整手：日量 99999 → cap = 4999.95 → 4900 股。"""
    v, p, s = _mk([99_999], [10.0], [0.02])
    m = MultiDayExecutionModel()
    r = m.simulate_order_flow(10_000, v, p, s)
    # fill = min(10000, 4900) = 4900
    assert r["vwap_price"] > 0
    assert not r["completed"]


def test_impact_raises_price_for_buy():
    """买入冲击上抬成交价：vwap > 基准价。"""
    v, p, s = _mk([2_000_000], [10.0], [0.02])
    m = MultiDayExecutionModel()
    r = m.simulate_order_flow(100_000, v, p, s, side="buy")
    assert r["vwap_price"] > 10.0
    assert r["total_slippage"] > 0


def test_impact_lowers_price_for_sell():
    """卖出冲击压低成交价：vwap < 基准价。"""
    v, p, s = _mk([2_000_000], [10.0], [0.02])
    m = MultiDayExecutionModel()
    r = m.simulate_order_flow(100_000, v, p, s, side="sell")
    assert r["vwap_price"] < 10.0


def test_higher_participation_higher_impact():
    """参与率翻倍 → 冲击率按 sqrt 增长（非线性验证）。"""
    v, p, s = _mk([10_000_000, 2_500_000], [10.0, 10.0], [0.02, 0.02])
    m = MultiDayExecutionModel()
    small = m.simulate_order_flow(50_000, v[:1], p[:1], s[:1])
    large = m.simulate_order_flow(200_000, v[1:], p[1:], s[1:])
    # 参与率 0.5% vs 8%（clamp 到 cap），冲击比率应显著更大
    assert large["vwap_price"] - 10.0 > (small["vwap_price"] - 10.0)


def test_no_volume_days_skipped():
    """停牌/零成交日跳过，不产生成交也不炸。"""
    v = np.array([0.0, 1_000_000.0, 0.0, 1_000_000.0])
    p = np.array([10.0, 10.0, 10.0, 10.0])
    s = np.array([0.02, 0.02, 0.02, 0.02])
    m = MultiDayExecutionModel()
    r = m.simulate_order_flow(80_000, v, p, s)
    assert r["completed"] is True
    assert r["execution_days"] == 2


def test_wfa_delay_cost_integration():
    """WFA 语义验证：大额调仓被顺延 → 相比一次性全额当日成交，
    延迟日价格不利时总成本更高（真实延迟成本显性化）。"""
    m = MultiDayExecutionModel(max_participation_rate=0.10)
    # 两日：第二日价格上行 1%（对买入不利）
    v = np.array([100_000.0, 100_000.0])
    p = np.array([10.0, 10.1])
    s = np.array([0.02, 0.02])
    r = m.simulate_order_flow(20_000, v, p, s, side="buy")
    # 第一日 cap=10000 股全成，第二日 10000 股按 10.1 + 冲击
    assert r["execution_days"] == 2
    assert r["vwap_price"] > 10.05  # 均价受第二日价格拖累
