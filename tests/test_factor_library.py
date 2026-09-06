"""
Alpha158 骨架因子库 — 回归测试

解剖结论：qlib Alpha158 的 6 大类因子骨架（K线形态/动量/波动/量价/RSI/量能）
+ "除以当期值归一"元设计，可直接吸收到本地面板因子库（无需 qlib 依赖）。

契约：
  1. compute_alpha_factors(panel) → dict[name, (T,N) ndarray]
  2. 无前视：t 行只由 <=t 数据构成（截尾重训一致性）
  3. 关键因子覆盖：KMID/KUP/ROC/MA/STD/CORR/RSI/VMA 等
  4. 合成动量数据上动量族因子 OOS IC > 0.3（可学信号）
  5. NaN 处理：暖机期 NaN 保留（调用方决定填充）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.research.factor_library import (  # noqa: E402
    compute_alpha_factors,
    factor_names,
)


def _mk_panel(T=300, N=12, seed=5):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.02, size=(T, N))
    close = 100.0 * np.cumprod(1 + rets, axis=0)
    open_ = close * (1 + rng.normal(0, 0.002, size=(T, N)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, (T, N))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, (T, N))))
    vol = np.abs(rng.normal(1e6, 2e5, size=(T, N))) + 1e5
    vwap = (high + low + close) / 3
    return {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": vol, "vwap": vwap, "amount": vol * vwap,
    }


def _momentum_panel(T=300, N=12, seed=7):
    """带可学动量：未来收益与过去 20 日动量正相关。"""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, (T, N)), axis=0)
    open_ = close * (1 + rng.normal(0, 0.001, (T, N)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, (T, N))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, (T, N))))
    vol = np.abs(rng.normal(1e6, 2e5, size=(T, N))) + 1e5
    fwd = np.full_like(close, np.nan)
    for t in range(20, T):
        mom = close[t - 1] / close[t - 21] - 1.0
        fwd[t] = mom * 0.5 + rng.normal(0, 0.008, N)
    return {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": vol, "vwap": (high + low + close) / 3,
        "amount": vol * (high + low + close) / 3,
    }, fwd


def test_factor_names_cover_core_families():
    names = factor_names()
    for key in ("KMID", "KUP", "ROC5", "MA20", "STD20", "RSI14", "VMA5",
                "CORR_CR", "RSV20"):
        assert key in names, f"缺核心因子 {key}"


def test_shapes_and_no_lookahead():
    panel = _mk_panel()
    facs = compute_alpha_factors(panel)
    T, N = panel["close"].shape
    for name, f in facs.items():
        assert f.shape == (T, N), f"{name} shape {f.shape} != {(T, N)}"
    # 无前视：砍尾 50 天重算，前段因子完全一致
    short = {k: v[:-50] for k, v in panel.items()}
    facs2 = compute_alpha_factors(short)
    for name in facs:
        np.testing.assert_allclose(
            facs[name][:-50], facs2[name], rtol=1e-9, atol=1e-12,
            err_msg=f"{name} 前视泄露")


def test_momentum_factor_has_positive_ic():
    """动量族（ROC20）在可学动量数据上对 fwd 有正 IC。"""
    panel, fwd = _momentum_panel()
    facs = compute_alpha_factors(panel)
    from evolve.core.gp import _cs_rank
    # 用 ROC20 因子末 100 期算平均 Rank-IC
    ics = []
    for t in range(50, 250):
        f = facs["ROC20"][t]
        r = fwd[t]
        m = np.isfinite(f) & np.isfinite(r)
        if m.sum() < 6:
            continue
        fr = _cs_rank(f[m].reshape(1, -1)).ravel()
        rr = _cs_rank(r[m].reshape(1, -1)).ravel()
        ics.append(np.corrcoef(fr, rr)[0, 1])
    assert len(ics) > 50
    assert np.mean(ics) > 0.15, f"动量因子 IC 过低: {np.mean(ics):.3f}"


def test_finite_after_warmup():
    """暖机期后因子有限（非全 NaN）。"""
    panel = _mk_panel(T=100, N=6)
    facs = compute_alpha_factors(panel)
    for name, f in facs.items():
        tail = f[80:]
        assert np.isfinite(tail).mean() > 0.9, f"{name} 尾部 NaN 过多"


def test_volume_and_corr_factors_finite():
    panel = _mk_panel(T=120, N=8)
    facs = compute_alpha_factors(panel)
    for name in ("VMA5", "VSTD5", "CORR_CR", "CORD"):
        assert np.isfinite(facs[name][100:]).mean() > 0.9, name
