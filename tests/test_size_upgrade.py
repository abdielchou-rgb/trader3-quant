"""
G5：风格暴露真实市值注入 — 回归测试
痛点：size 列用"对数成交额代理"（非真实流通市值），共线剥离有效性打折。
契约：
  1. build_style_exposures(panel, shares=...)：提供股本（symbol→总股本）时
     size 列 = log(close × shares)（真实市值），仍标准化
  2. shares 缺失 → 成交额代理回退（历史行为，caveat 语义不变）
  3. 有真实市值时：与市值代理的秩相关性可以显著不同（构造可分辨用例）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "evolve"))

from core.style_exposures import build_style_exposures  # noqa: E402


def _panel(T=240, N=12, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, size=(T, N))
    close = 100.0 * np.cumprod(1 + rets, axis=0)
    codes = [f"S{i:02d}" for i in range(N)]
    return {
        "close": close,
        "volume": rng.uniform(1e6, 5e6, size=(T, N)),
        "amount": close * rng.uniform(1e6, 5e6, size=(T, N)),
        "codes": codes,
    }


def test_shares_injection_switches_size_to_real_market_cap():
    panel = _panel()
    codes = panel.pop("codes")
    shares = {c: 1e8 for c in codes}       # 人为差异化股本
    # 无 shares：代理口径
    x_proxy = build_style_exposures(panel)
    # 有 shares：真实市值口径
    x_real = build_style_exposures(panel, shares=shares)
    # size 列（index 1）应有差异（同样本下两种口径秩序不同）
    assert not np.allclose(x_proxy[:, 1], x_real[:, 1])
    # 标准化性质保持
    np.testing.assert_allclose(x_real.mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(x_real.std(axis=0), 1, atol=1e-6)


def test_shapes_and_finite_with_shares():
    panel = _panel()
    codes = panel.pop("codes")
    shares = {c: 5e7 + i * 1e6 for i, c in enumerate(codes)}
    X = build_style_exposures(panel, shares=shares)
    assert X.shape == (12, 3)
    assert np.isfinite(X).all()


def test_make_orthogonal_selector_accepts_shares():
    from core.style_exposures import make_orthogonal_selector

    panel = _panel()
    codes = panel.pop("codes")
    close = panel["close"]
    fwd = np.full_like(close, np.nan)
    fwd[:-1] = close[1:] / close[:-1] - 1.0
    fwd[-1] = np.nan
    sel = make_orthogonal_selector(panel, fwd, shares={c: 1e8 for c in codes})
    assert sel._orth_eval is not None
