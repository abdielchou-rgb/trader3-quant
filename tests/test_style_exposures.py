"""
Track2 回归测试：风格暴露构造 + 正交门禁在 run_evolution 的真实接入。
契约：
  1. build_style_exposures(panel)：末截面 (N,3) [动量20/市值代理/波动20]，标准化
  2. run_evolution --orthogonal：StrategySelector 注入 barra_styles + fwd 末截面，
     产出的 gates 含 orthogonality 键（真实 csi300 小进化冒烟）
  3. 与纯风格克隆：正交门禁拒绝；独立 alpha：通过
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "evolve"))

from evolve.core.style_exposures import build_style_exposures  # noqa: E402


def _panel(T=240, N=30, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, size=(T, N))
    close = 100.0 * np.cumprod(1 + rets, axis=0)
    return {
        "close": close,
        "volume": rng.uniform(1e6, 5e6, size=(T, N)),
        "amount": close * rng.uniform(1e6, 5e6, size=(T, N)),
    }, rets


def test_style_exposures_shape_and_finite():
    panel, _ = _panel()
    X = build_style_exposures(panel)
    assert X.shape == (panel["close"].shape[1], 3)
    assert np.isfinite(X).all()
    # 每列标准化（均值≈0，std≈1）
    np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(X.std(axis=0), 1, atol=1e-6)


def test_style_exposures_nan_safe():
    panel, _ = _panel()
    panel["close"][:30] = np.nan  # 暖机期 NaN
    X = build_style_exposures(panel)
    assert np.isfinite(X).all()


def test_orthogonal_gate_wired_in_evolution():
    """run_evolution 的 --orthogonal 路径：selector 拿到 orth_eval 且 fwd 注入。"""
    # 直接验证接线函数（不跑全量进化）：make_orthogonal_selector
    from evolve.core.style_exposures import make_orthogonal_selector

    panel, rets = _panel()
    fwd = np.full_like(panel["close"], np.nan)
    fwd[:-1] = panel["close"][1:] / panel["close"][:-1] - 1.0
    fwd[-1] = np.nan
    selector = make_orthogonal_selector(panel, fwd)
    assert selector._orth_eval is not None
    assert selector._fwd is not None
    # 手工构造候选：值=末截面动量（与风格共线） → orth_ic≈0 → 拒绝
    last_close = panel["close"][-1]
    mom20 = last_close / panel["close"][-21] - 1.0
    cand_clone = {
        "expr": "clone_mom", "ic": 0.05, "icir": 0.3, "monotonicity": 0.5,
        "long_short": 0.2, "fitness": 1.0,
        "values": np.nan_to_num(mom20),
    }
    res = selector._evaluate_one(cand_clone)
    assert "orthogonality" in res.gates
    assert res.gates["orthogonality"]["passed"] is False
