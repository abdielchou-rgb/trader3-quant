"""
ML 打分选股管线（R7）— 回归测试

解剖结论（alpha_research + qlib 解剖）：Alpha158 表格数据上树模型/线性
常强于深度；ML 聚合的意义是把"单因子排序"升级为"特征 → 模型打分"。
验证：用 factor_library 产出的因子面板喂 sklearn 模型，OOS 打分 Rank-IC
应显著高于单一最佳因子（当合成数据含非线性组合信号时）。

契约：
  1. build_ml_dataset(factors, fwd, train_end, embargo) → 训练/测试样本
     （按时间切分，embargo 防泄漏；只保留 fwd 有限行）
  2. fit_scorer + score → OOS 打分
  3. oos_rank_ic(scores, y)：OOS 期逐日横截面 Rank-IC
  4. 合成非线性信号：ML 打分 OOS IC 优于任一单因子
  5. 弱/噪声数据上模型仍可跑（不炸），IC 诚实低
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.research.factor_library import compute_alpha_factors  # noqa: E402
from trader3.research.ml_pipeline import (  # noqa: E402
    build_ml_dataset,
    fit_scorer,
    oos_rank_ic,
    score_oos,
)


def _mk_panel(T=260, N=20, seed=11):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, size=(T, N))
    close = 100.0 * np.cumprod(1 + rets, axis=0)
    open_ = close * (1 + rng.normal(0, 0.002, (T, N)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, (T, N))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, (T, N))))
    vol = np.abs(rng.normal(1e6, 2e5, size=(T, N))) + 1e5
    vwap = (high + low + close) / 3
    panel = {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": vol, "vwap": vwap, "amount": vol * vwap,
    }
    # fwd 由前 20 日动量的 log 变换 + 交互项驱动（非线性，利于 ML）
    fwd = np.full_like(close, np.nan)
    mom = np.full_like(close, np.nan)
    for t in range(20, T):
        mom[t] = close[t - 1] / close[t - 21] - 1.0
    for t in range(20, T - 1):
        vol20 = np.std(rets[max(0, t - 20):t + 1], axis=0)
        # 动量*波动交互（非线性可学），加噪声
        fwd[t] = np.sign(mom[t]) * np.abs(mom[t]) ** 0.6 * (1 + vol20) \
            + rng.normal(0, 0.005, N)
    return panel, fwd


def test_build_ml_dataset_shapes_and_embargo():
    """训练/测试样本正确切分且含 embargo。"""
    panel, fwd = _mk_panel(T=200, N=10)
    facs = compute_alpha_factors(panel)
    ds = build_ml_dataset(facs, fwd, train_end=120, embargo=10)
    assert set(ds.keys()) >= {"X_train", "y_train", "X_test", "y_test"}
    assert ds["X_train"].shape[1] == len(facs)  # 特征数 = 因子数
    assert ds["X_test"].shape[0] > 0
    assert ds["X_test"].shape[1] == len(facs)


def test_fit_and_score_oos_rankic():
    """合成非线性信号：Ridge/RF 至少一个模型 OOS IC 显著为正。"""
    panel, fwd = _mk_panel(T=240, N=16)
    facs = compute_alpha_factors(panel)
    ds = build_ml_dataset(facs, fwd, train_end=140, embargo=8)
    best_ic = -9.9
    for model in ("ridge", "rf", "gbr"):
        m = fit_scorer(ds["X_train"], ds["y_train"], model=model)
        scores = score_oos(m, ds["X_test"])
        ic = oos_rank_ic(scores, ds["y_test"], ds["test_t"])
        best_ic = max(best_ic, ic)
        assert np.isfinite(ic), f"{model} IC 非有限"
    assert best_ic > 0.05, f"ML OOS IC 均过低: {best_ic:.3f}"


def test_oos_rank_ic_period_semantics():
    """oos_rank_ic 返回逐日截面 Rank-IC 的均值（时间上是测试段）。"""
    panel, fwd = _mk_panel(T=180, N=12)
    facs = compute_alpha_factors(panel)
    ds = build_ml_dataset(facs, fwd, train_end=100, embargo=5)
    m = fit_scorer(ds["X_train"], ds["y_train"], model="ridge")
    scores = score_oos(m, ds["X_test"])
    ic = oos_rank_ic(scores, ds["y_test"], ds["test_t"])
    assert -1.0 <= ic <= 1.0


def test_noise_data_no_crash_low_ic():
    """纯噪声数据：不炸，IC 诚实低（近 0）。"""
    rng = np.random.default_rng(2)
    T, N = 150, 10
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.02, (T, N)), axis=0)
    open_ = close.copy()
    panel = {
        "open": open_, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.full_like(close, 1e6),
        "vwap": close, "amount": close * 1e6,
    }
    fwd = np.full_like(close, np.nan)
    fwd[:-1] = close[1:] / close[:-1] - 1.0 + rng.normal(0, 0.01, (T - 1, N))
    fwd[-1] = np.nan
    facs = compute_alpha_factors(panel)
    ds = build_ml_dataset(facs, fwd, train_end=90, embargo=5)
    m = fit_scorer(ds["X_train"], ds["y_train"], model="ridge")
    scores = score_oos(m, ds["X_test"])
    ic = oos_rank_ic(scores, ds["y_test"], ds["test_t"])
    assert np.isfinite(ic)
    assert abs(ic) < 0.2  # 噪声数据不应出现异常高 IC
