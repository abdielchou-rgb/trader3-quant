"""
Track4 回归：WFA 多日参与率执行约束接入。
契约：
  1. daily_volumes=None → 历史行为不变（首日一次性全额成交）
  2. daily_volumes 提供 → 首日调仓超 5% 日量的部分顺延至次日按次日收益执行；
     OOS 净收益应低于无约束对照（冲击/延迟显性化）
  3. 顺延不超过 OOS 窗口末；窗口内未完成部分按现金处理
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.tools.backtest import _run_wfa_rolling  # noqa: E402


def _make_case(T=400, N=20, seed=11):
    """动量可学的合成面板。"""
    rng = np.random.default_rng(seed)
    mom = rng.normal(0, 0.01, size=(T, N))
    close = 100.0 * np.cumprod(1 + mom, axis=0)
    rets = np.zeros_like(close)
    rets[1:] = close[1:] / close[:-1] - 1.0
    # 因子 = 20 日动量（前 20 行 NaN）
    fs = np.full_like(close, np.nan)
    for t in range(20, T):
        fs[t] = close[t] / close[t - 20] - 1.0
    return rets, fs


def test_wfa_unchanged_without_volumes():
    rets, fs = _make_case()
    r1 = _run_wfa_rolling(rets, fs, train_window=120, test_window=40, step=40)
    # 同参再跑一遍：确定性
    r2 = _run_wfa_rolling(rets, fs, train_window=120, test_window=40, step=40)
    assert len(r1[0]) == len(r2[0])
    np.testing.assert_allclose(r1[6], r2[6], rtol=1e-12)  # oos_concat = 索引 6


def test_wfa_participation_cap_reduces_oos():
    """巨量调仓 + 小日成交额 → 参与率截断使 OOS 净收益低于无约束对照。"""
    rets, fs = _make_case(T=400, N=10)
    T, N = rets.shape
    # 极小日成交额：5% 上限 = 50 元/股/日 → 目标 100 万资金的建仓需求远超容量
    tiny_amounts = np.full((T, N), 1000.0)

    unconstrained = _run_wfa_rolling(
        rets, fs, train_window=120, test_window=40, step=40)
    constrained = _run_wfa_rolling(
        rets, fs, train_window=120, test_window=40, step=40,
        daily_volumes=tiny_amounts, capital=1_000_000.0,
    )
    oos_unc = float(np.nanmean(unconstrained[6]))
    oos_con = float(np.nanmean(constrained[6]))
    # 强约束下 OOS 更差（敞口大幅缩水至现金 + 成本照提）
    assert oos_con < oos_unc


def test_wfa_generous_volumes_match_unconstrained():
    """巨量日成交额（无约束）→ 与无 volumes 路径结果一致（上限不触发）。"""
    rets, fs = _make_case(T=400, N=10)
    T, N = rets.shape
    # 每股日成交额 1 亿：5% = 500 万 >> 单股目标 10 万（100万×10%）
    huge = np.full((T, N), 1e8)

    unc = _run_wfa_rolling(rets, fs, train_window=120, test_window=40, step=40)
    con = _run_wfa_rolling(
        rets, fs, train_window=120, test_window=40, step=40,
        daily_volumes=huge, capital=1_000_000.0,
    )
    np.testing.assert_allclose(unc[6][-1], con[6][-1], rtol=1e-9, atol=1e-12)
