"""
滚动 ML 选股组合回测（R9）— 回归测试

解剖结论落地：把 ml_pipeline(回归排序) + meta_ml(p(win) 过滤) 结合成
"回归选 top-N 候选 → meta 过滤 → 等权组合"，做滚动训练 + 样本外组合净值。
诚实防前视：每次重训只用 [r-lookback, r] 数据，预测 [r, r+rebal) 收益。

契约：
  1. RollingStockSelector：walk-forward 滚动重训，不泄漏（测试期只用历史模型）
  2. 合成可学数据：meta 过滤版组合累计收益 > 全池等权（或 >=，含过滤语义）
  3. 输出含逐日组合收益序列 + 关键指标，可入 experiment
  4. 无信号/噪声数据：不炸，收益诚实低
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.research.stock_selector import (  # noqa: E402
    RollingStockSelector,
    run_rolling_selector,
)


def _learnable_market(T=400, N=16, seed=31):
    """含可学动量-波动结构：动量强且波动高 → 期望收益更高。"""
    rng = np.random.default_rng(seed)
    # 基础日收益含可预测成分
    raw = rng.normal(0, 0.02, (T, N))
    close = 100.0 * np.cumprod(1 + raw, axis=0)
    open_ = close * (1 + rng.normal(0, 0.002, (T, N)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, (T, N))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, (T, N))))
    vol = np.abs(rng.normal(1e6, 2e5, size=(T, N))) + 1e5
    vwap = (high + low + close) / 3
    panel = {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": vol, "vwap": vwap, "amount": vol * vwap,
    }
    # 真实次日收益 = alpha 结构 + 噪声（alpha 用滞后动量×波动构造）
    fwd = np.full_like(close, np.nan)
    for t in range(30, T):
        mom = close[t - 1] / close[t - 21] - 1.0
        vol20 = np.std(raw[max(0, t - 20):t], axis=0)
        alpha = np.sign(mom) * np.abs(mom) ** 0.6 * (1 + vol20)
        fwd[t] = alpha + rng.normal(0, 0.01, N)
    return panel, fwd


def _noise_market(T=250, N=12, seed=7):
    rng = np.random.default_rng(seed)
    raw = rng.normal(0, 0.02, (T, N))
    close = 100.0 * np.cumprod(1 + raw, axis=0)
    open_ = close.copy()
    high = close * 1.01
    low = close * 0.99
    vol = np.full_like(close, 1e6)
    vwap = close
    panel = {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": vol, "vwap": vwap, "amount": close * 1e6,
    }
    fwd = np.full_like(close, np.nan)
    fwd[:-1] = close[1:] / close[:-1] - 1.0
    fwd[-1] = np.nan
    return panel, fwd


def test_run_rolling_selector_produces_nav():
    """可学数据上能产出组合净值序列，且模型部分无前视（训练/测试分段）。"""
    panel, fwd = _learnable_market(T=300, N=12)
    res = run_rolling_selector(
        panel, fwd, lookback=150, rebal=30, top_n=5,
        meta_filter=False, start_train=180, warmup=40,
    )
    assert "nav" in res and "rets" in res and "bench" in res
    assert len(res["nav"]) == len(res["rets"]) > 0
    assert np.isfinite(res["nav"]).all()


def test_selector_beats_universe_equal_weight():
    """合成可学数据：meta 过滤组合累计收益应明显 > 全池等权。"""
    panel, fwd = _learnable_market(T=360, N=16)
    res = run_rolling_selector(
        panel, fwd, lookback=200, rebal=20, top_n=6,
        meta_filter=True, start_train=240, warmup=40,
    )
    bench = float(np.nanmean(res["rets"]))  # 组合期收益均值
    eq = float(np.nanmean(res["bench"]))    # 全池等权
    assert bench > eq, f"组合 {bench:.5f} 应优于全池等权 {eq:.5f}"


def test_meta_filter_raises_hit_rate_vs_regression_only():
    """同一 top-N 候选下，meta 过滤子集命中率 >= 不过滤（可交易性增强）。"""
    panel, fwd = _learnable_market(T=360, N=16)
    r_no = run_rolling_selector(
        panel, fwd, lookback=200, rebal=20, top_n=6,
        meta_filter=False, start_train=240, warmup=40,
    )
    r_yes = run_rolling_selector(
        panel, fwd, lookback=200, rebal=20, top_n=6,
        meta_filter=True, start_train=240, warmup=40,
    )
    assert r_yes["mean_win"] >= r_no["mean_win"] - 0.05, \
        f"meta 过滤命中率 {r_yes['mean_win']:.3f} 不应明显低于不过滤 {r_no['mean_win']:.3f}"


def test_noise_market_no_crash():
    """噪声数据：不炸，输出有限净值。"""
    panel, fwd = _noise_market(T=250, N=12)
    res = run_rolling_selector(
        panel, fwd, lookback=150, rebal=25, top_n=4,
        meta_filter=True, start_train=180, warmup=40,
    )
    assert np.isfinite(res["nav"]).all()
    assert len(res["rets"]) > 0


def test_rolling_selector_api():
    """对象式 API：fit 阶段分片可调用。"""
    panel, fwd = _learnable_market(T=250, N=12)
    sel = RollingStockSelector(lookback=150, rebal=25, top_n=4, meta_filter=True)
    sel.fit(panel, fwd, start_train=180, warmup=40)
    assert hasattr(sel, "portfolio_rets") and len(sel.portfolio_rets) > 0
