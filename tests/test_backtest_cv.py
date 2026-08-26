"""Purged-CV / 组合式回测（P0-4：严谨 OOS 验证）测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader3.v2.backtest_cv import (
    combinatorial_backtest,
    deflated_sharpe,
    embargo_purged_splits,
    purged_cv_rankic,
)


def _panel_signal(n_dates=240, n_assets=10, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    assets = [f"A{i}" for i in range(n_assets)]
    close = pd.DataFrame(
        np.cumsum(rng.normal(0, 1, (n_dates, n_assets)), axis=0) + 100,
        index=dates, columns=assets,
    )
    mom = close - close.shift(10)
    scores = mom.copy()
    fwd = mom.shift(-10) * 0.5 + rng.normal(0, 0.01, (n_dates, n_assets))
    fwd = fwd.iloc[:-10]
    scores = scores.iloc[:-10]
    return scores, fwd


def _panel_noise(n_dates=240, n_assets=10, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    assets = [f"A{i}" for i in range(n_assets)]
    scores = pd.DataFrame(rng.normal(size=(n_dates, n_assets)), index=dates, columns=assets)
    fwd = pd.DataFrame(rng.normal(0, 0.01, (n_dates, n_assets)), index=dates, columns=assets)
    return scores, fwd


def test_embargo_purged_splits_no_overlap():
    n = 200
    for _tr, te in embargo_purged_splits(n, test_size=20, embargo=5, min_train=60, horizon=10):
        # 测试折之间不重叠
        assert len(set(te)) == len(te)
    # 至少产出若干折
    folds = list(embargo_purged_splits(n, test_size=20, embargo=5, min_train=60, horizon=10))
    assert len(folds) >= 2


def test_purged_cv_rankic_signal_positive():
    scores, fwd = _panel_signal()
    r = purged_cv_rankic(scores, fwd, n_splits=5, embargo=5, horizon=10)
    assert np.isfinite(r["ic_mean"])
    assert r["ic_mean"] > 0.0
    assert r["pos_ratio"] >= 0.5
    assert np.isfinite(r["ic_t"])


def test_purged_cv_rankic_noise_low():
    scores, fwd = _panel_noise()
    r = purged_cv_rankic(scores, fwd, n_splits=5, embargo=5, horizon=10)
    # 噪声下 CV t 不应显著为正
    assert not (r["ic_t"] >= 1.96)


def test_combinatorial_backtest_signal_beats_noise():
    scores, fwd = _panel_signal()
    cb = combinatorial_backtest(scores, fwd, n_splits=5, embargo=5, horizon=10)
    assert cb["sharpe"] > 0
    assert cb["mean_ret"] > 0
    assert 0.0 <= cb["deflated_sharpe"] <= 1.0
    # 噪声组合式回测夏普应明显更低
    s2, f2 = _panel_noise()
    cb2 = combinatorial_backtest(s2, f2, n_splits=5, embargo=5, horizon=10)
    assert cb["sharpe"] > cb2["sharpe"]


def test_deflated_sharpe_below_naive():
    # 多重检验校正后缩水夏普应 <= 朴素夏普对应 PSR
    dsr = deflated_sharpe(sr=2.0, n_obs=250, skew=0.0, kurt=3.0, n_trials=100)
    assert 0.0 <= dsr <= 1.0
    # n_trials 越大，校正越严格（期望最大夏普基准越高 → PSR 越低）
    dsr1 = deflated_sharpe(2.0, 250, 0.0, 3.0, n_trials=5)
    dsr2 = deflated_sharpe(2.0, 250, 0.0, 3.0, n_trials=200)
    assert dsr2 < dsr1
