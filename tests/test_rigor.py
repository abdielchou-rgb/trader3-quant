"""
R3 回测严谨性模块 — 回归测试

契约（AFML / Bailey-López de Prado）：
  1. probabilistic_sharpe_ratio：单夏普的显著概率（用偏度/峰度）
  2. deflated_sharpe_ratio：惩罚"试验次数 N"后夏普是否仍显著 ——
     试过 N 次后最佳夏普的期望值本身 >0，DSR 用它在零假设
  3. minimum_track_record_length：多少年数据才让该夏普"值得信"
  4. 试验次数越大 DSR 越低（诚实化）；正态收益无偏时退化为常规
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.research.rigor import (  # noqa: E402
    deflated_sharpe_ratio,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
)


def _norm_returns(n=500, mu=0.0005, sigma=0.01, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, n)


def test_psr_normal_returns_near_standard():
    """零收益（去均值）噪声：PSR≈0.5（不显著）。"""
    rets = _norm_returns(n=1000, mu=0.0, sigma=0.01)
    rets = rets - rets.mean()  # 强制零均值 → 期夏普≈0
    psr = probabilistic_sharpe_ratio(rets, sr_benchmark=0.0)
    assert 0.4 < psr < 0.6


def test_psr_positive_for_good_sharpe():
    """真实高夏普 → PSR 接近 1。"""
    rets = _norm_returns(n=2000, mu=0.001, sigma=0.01)
    psr = probabilistic_sharpe_ratio(rets, sr_benchmark=0.0)
    assert psr > 0.95


def test_dsr_decreases_with_trials():
    """试验次数越多 DSR 越低（多重检验惩罚）。"""
    rets = _norm_returns(n=2000, mu=0.0005, sigma=0.01, seed=3)
    dsr_1 = deflated_sharpe_ratio(rets, n_trials=1)
    dsr_100 = deflated_sharpe_ratio(rets, n_trials=100)
    assert dsr_1 > dsr_100


def test_dsr_with_noise_signal_not_significant():
    """纯噪声挖出的"最佳"夏普在大量试验后 DSR 应 < 0.95（不显著）。"""
    # 模拟：10 组噪声中挑最好的一组
    best = None
    for i in range(20):
        r = _norm_returns(n=500, mu=0.0, sigma=0.02, seed=100 + i)
        s = r.mean() / r.std() * np.sqrt(252)
        if best is None or s > best[1]:
            best = (r, s)
    rets, sr = best
    dsr = deflated_sharpe_ratio(rets, n_trials=20)
    assert dsr < 0.9  # 20 次试验后噪声最佳应基本不显著


def test_mtrl_positive_and_finite():
    """最小可信记录长度为正有限；高夏普需更少年份。"""
    rets = _norm_returns(n=1000, mu=0.0008, sigma=0.01)
    mtrl = minimum_track_record_length(rets, prob=0.95)
    assert np.isfinite(mtrl) and mtrl > 0
    # 更高夏普 → 更短 track record
    rets2 = _norm_returns(n=1000, mu=0.002, sigma=0.01)
    mtrl2 = minimum_track_record_length(rets2, prob=0.95)
    assert mtrl2 < mtrl


def test_dsr_consistent_with_psr_at_zero_trials():
    """n_trials=1 时 DSR 应接近同参数 PSR（退化为 PSR）。"""
    rets = _norm_returns(n=800, mu=0.0006, sigma=0.01, seed=7)
    dsr1 = deflated_sharpe_ratio(rets, n_trials=1)
    psr = probabilistic_sharpe_ratio(rets, sr_benchmark=0.0)
    assert abs(dsr1 - psr) < 0.05
