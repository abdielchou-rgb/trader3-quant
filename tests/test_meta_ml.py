"""
Meta-label 分类选股（R8）— 回归测试

解剖结论（mlfinlab/AFML meta-labeling）：主模型定方向，元模型判"这笔
是否值得真开仓"（吸收成本/噪声）。A股多头框架下，meta 标签 =
"次日扣成本后收益为正 → 1，否则 0"。

契约：
  1. build_meta_dataset：因子面板 → 二分类样本（含 cost_bps 标签、embargo 切分）
  2. fit_meta_classifier → 输出 p(win)
  3. 合成含可学信号数据：p(win) 高的子集命中率显著 > 全样本基础命中率
  4. 纯噪声数据：分类器 AUC ≈ 0.5（诚实）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.research.factor_library import compute_alpha_factors  # noqa: E402
from trader3.research.meta_ml import (  # noqa: E402
    build_meta_dataset,
    fit_meta_classifier,
    hit_rate_top,
)


def _mk_panel(T=280, N=24, seed=21):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.02, size=(T, N))
    close = 100.0 * np.cumprod(1 + rets, axis=0)
    open_ = close * (1 + rng.normal(0, 0.002, (T, N)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, (T, N))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, (T, N))))
    vol = np.abs(rng.normal(1e6, 2e5, size=(T, N))) + 1e5
    vwap = (high + low + close) / 3
    # 可学：真实 win 概率随"最近动量"单调升
    win_p = np.full((T, N), 0.5)
    for t in range(20, T):
        mom = close[t - 1] / close[t - 21] - 1.0
        # 动量越强 win 概率越高（上限 0.8），制造可学结构
        win_p[t] = np.clip(0.5 + 2.0 * mom, 0.2, 0.8)
    fwd = np.full_like(close, np.nan)
    for t in range(20, T - 1):
        fwd[t] = np.where(rng.random(N) < win_p[t],
                          rng.uniform(0.001, 0.03, N),      # win
                          rng.uniform(-0.03, -0.001, N))    # loss
    return {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": vol, "vwap": vwap, "amount": vol * vwap,
    }, fwd


def test_build_meta_dataset_labels_and_split():
    """标签为 0/1（扣成本后正负），训练/测试按 embargo 切分。"""
    panel, fwd = _mk_panel(T=200, N=12)
    facs = compute_alpha_factors(panel)
    ds = build_meta_dataset(facs, fwd, train_end=120, embargo=10, cost_bps=0.0)
    assert set(ds.keys()) >= {"X_train", "y_train", "X_test", "y_test"}
    assert set(np.unique(ds["y_train"])) <= {0, 1}
    assert ds["X_train"].shape[1] == len(facs)


def test_meta_classifier_improves_hit_rate():
    """meta 模型选出的 top 子集命中率应显著 > 基础命中率。"""
    panel, fwd = _mk_panel(T=240, N=24)
    facs = compute_alpha_factors(panel)
    ds = build_meta_dataset(facs, fwd, train_end=140, embargo=8, cost_bps=5.0)
    # 基础命中率（测试段 y=1 比例）
    base = float(ds["y_test"].mean())
    clf = fit_meta_classifier(ds["X_train"], ds["y_train"])
    top_hr, top_n = hit_rate_top(clf, ds["X_test"], ds["y_test"], top_frac=0.3)
    assert top_n >= 20
    assert top_hr > base + 0.08, \
        f"meta top 命中率 {top_hr:.3f} 应显著高于基础 {base:.3f}"


def test_cost_shifts_labels():
    """成本 >0 时部分微利样本标签翻 0（成本意识）。"""
    panel, fwd = _mk_panel(T=200, N=12)
    facs = compute_alpha_factors(panel)
    ds0 = build_meta_dataset(facs, fwd, train_end=110, embargo=5, cost_bps=0.0)
    ds1 = build_meta_dataset(facs, fwd, train_end=110, embargo=5, cost_bps=50.0)
    # 高成本下 y=1 比例应不高于零成本（部分微利变负）
    assert ds1["y_train"].mean() <= ds0["y_train"].mean()


def test_noise_data_auc_around_half():
    """噪声数据：分类器命中率提升应很小（≈基础）。"""
    rng = np.random.default_rng(7)
    T, N = 180, 16
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.02, (T, N)), axis=0)
    open_ = close.copy()
    panel = {
        "open": open_, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.full_like(close, 1e6),
        "vwap": close, "amount": close * 1e6,
    }
    fwd = np.full_like(close, np.nan)
    fwd[:-1] = close[1:] / close[:-1] - 1.0
    fwd[-1] = np.nan
    facs = compute_alpha_factors(panel)
    ds = build_meta_dataset(facs, fwd, train_end=100, embargo=5, cost_bps=0.0)
    clf = fit_meta_classifier(ds["X_train"], ds["y_train"])
    top_hr, top_n = hit_rate_top(clf, ds["X_test"], ds["y_test"], top_frac=0.3)
    base = float(ds["y_test"].mean())
    assert top_n >= 10
    assert top_hr < base + 0.15  # 噪声数据提升有限
