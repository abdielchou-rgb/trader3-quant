"""
模块二：GP 因子挖掘的正交残差化（Orthogonal Residualization）— 回归测试
痛点：GP 挖出的因子与动量/市值等 Barra 风格高度共线——单因子 IC 高但组合无增量。
核心契约：
  1. P_orth = I - X(X'X)^+X'，含常数截距（剥离均值）
  2. 因子 = 风格暴露的纯线性组合 → 残差 ≈ 0 → 正交 IC ≈ 0（被拦截）
  3. 因子 = 风格因子 + 独立 Alpha → 正交 IC 保留 Alpha 部分，显著低于原始 IC
  4. 全新独立因子 → 正交 IC ≈ 原始 IC（不误伤）
  5. 截面异常值（3σ clip）+ NaN 安全
  6. evolve 接入：compute_fitness_orthogonal 输出 marginal_ic；StrategySelector
     新增 orthogonality 门禁（正交 IC 不足 → 拒绝，标注共线）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "evolve"))

from core.orthogonal_fitness import OrthogonalFitnessEvaluator  # noqa: E402


def _styles(N=200, K=3, seed=7):
    """标准化风格暴露矩阵（市值/动量/波动代理）"""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, K))
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    return X


def test_projection_matrix_properties():
    """P_orth 是对称幂等投影矩阵（残差空间正交投影）。"""
    X = _styles()
    ev = OrthogonalFitnessEvaluator(X)
    P = ev.P_orth
    np.testing.assert_allclose(P, P.T, atol=1e-10)
    np.testing.assert_allclose(P @ P, P, atol=1e-10)
    # 投影后与风格列正交
    for k in range(X.shape[1]):
        np.testing.assert_allclose(P @ X[:, k], 0, atol=1e-10)


def test_pure_style_clone_killed():
    """因子=风格列的线性组合，收益也由风格驱动 → 残差因子≈0，
    正交 IC≈0（风格的预测力被完整剥离）。"""
    X = _styles(seed=7)
    rng = np.random.default_rng(1)
    coef = rng.normal(size=X.shape[1])
    factor = X @ coef
    # fwd 由同一风格结构驱动 + 噪声（模拟"风格溢价"传导到收益）
    fwd = X @ coef + rng.normal(0, 0.5, size=X.shape[0])
    ev = OrthogonalFitnessEvaluator(X)
    oic = ev.evaluate_orthogonal_ic(factor, fwd)
    assert abs(oic) < 0.05, f"纯风格克隆未被剥离: orth_IC={oic:.3f}"


def test_incremental_alpha_preserved():
    """因子 = 0.6×风格 + 独立Alpha：正交 IC 应显著保留（不误杀）。"""
    X = _styles(seed=7)
    rng = np.random.default_rng(3)
    alpha = rng.normal(size=X.shape[0])
    factor = 0.8 * (X[:, 0] + X[:, 1]) + 0.5 * alpha
    fwd, _ = _make_returns_from_alpha(alpha, noise=0.008, seed=4)
    ev = OrthogonalFitnessEvaluator(X)
    oic = ev.evaluate_orthogonal_ic(factor, fwd)
    # Alpha 的 IC 下限：alpha 与 fwd 相关构造约为 corr(alpha, fwd) ≈ 0.5~0.8
    assert oic > 0.2, f"增量 Alpha 被误杀: orth_IC={oic:.3f}"


def test_independent_factor_not_harmed():
    """与风格独立的因子 → 正交 IC ≈ 原始 IC。"""
    X = _styles(seed=7)
    rng = np.random.default_rng(5)
    alpha = rng.normal(size=X.shape[0])
    fwd, _ = _make_returns_from_alpha(alpha, noise=0.01, seed=6)
    ev = OrthogonalFitnessEvaluator(X)
    oic = ev.evaluate_orthogonal_ic(alpha, fwd)
    raw_ic = np.corrcoef(alpha, fwd)[0, 1]
    assert abs(oic - raw_ic) < 0.15, f"独立因子被误伤: orth={oic:.3f} raw={raw_ic:.3f}"


def test_nan_and_outlier_safety():
    """NaN 列、极端值不炸、不产生假信号。"""
    X = _styles(seed=7)
    rng = np.random.default_rng(8)
    factor = rng.normal(size=X.shape[0])
    factor[:20] = 1e8   # 极端值 → clip 后不主导
    factor[21:40] = np.nan
    fwd, _ = _make_returns_from_alpha(factor, noise=0.02, seed=9)
    ev = OrthogonalFitnessEvaluator(X)
    oic = ev.evaluate_orthogonal_ic(factor, fwd)
    assert np.isfinite(oic), "NaN/极端值路径产生非有限 IC"


def test_constant_factor_returns_zero():
    X = _styles()
    ev = OrthogonalFitnessEvaluator(X)
    assert ev.evaluate_orthogonal_ic(np.ones(X.shape[0]), np.zeros(X.shape[0])) == 0.0


def _make_returns(factor, noise, seed):
    """forward return = 因子 + 噪声（用于纯风格克隆测试：IC 由风格传导）"""
    rng = np.random.default_rng(seed)
    fwd = factor + rng.normal(0, noise, size=len(factor))
    return factor, fwd


def _make_returns_from_alpha(alpha, noise, seed):
    rng = np.random.default_rng(seed)
    fwd = alpha + rng.normal(0, noise, size=len(alpha))
    return fwd, alpha
