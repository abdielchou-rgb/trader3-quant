"""
协同因子评估（S1 落地）— 回归测试

解剖结论：GP 进化从"挖单因子"升级为"挖有协同的集合"。核心度量 = 边际贡献：
加入新因子后组合 Rank-IC 的增量（惩罚与已选因子共线 → 增量小/负则无价值）。

契约：
  1. evaluate_marginal_contribution(new_factor, existing_factors, fwd)
     → 新因子加入后的组合 IC 增量
  2. 与现有因子完全共线的克隆 → 边际贡献 ≈ 0（不增加信息）
  3. 独立有效新因子 → 边际贡献 > 0（确实增强组合）
  4. 组合 IC 用等权 zscore 合成 + 截面 Rank-IC
  5. greedy_select：从候选集里按边际贡献贪心选 top-k，去冗余
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.research.collaborative import (  # noqa: E402
    combo_rank_ic,
    evaluate_marginal_contribution,
    greedy_select,
)


def _data(T=300, N=30, seed=3):
    rng = np.random.default_rng(seed)
    # alpha1: 强独立信号
    alpha1 = rng.normal(size=(T, N))
    # alpha2: 与 alpha1 独立的弱信号
    alpha2 = rng.normal(size=(T, N))
    # fwd 由 alpha1 + 0.5*alpha2 + 噪声 驱动（两因子都有预测力且独立）
    fwd = np.full_like(alpha1, np.nan)
    for t in range(T):
        fwd[t] = alpha1[t] + 0.5 * alpha2[t] + rng.normal(0, 2, N)
    # clone = alpha1 的精确副本（共线）
    clone = alpha1.copy()
    # 噪声因子（无预测力）
    noise = rng.normal(size=(T, N))
    return {"alpha1": alpha1, "alpha2": alpha2, "clone": clone,
            "noise": noise}, fwd


def test_combo_rank_ic_positive_for_signal():
    """有效信号组合的 IC 为正。"""
    facs, fwd = _data()
    ic = combo_rank_ic([facs["alpha1"], facs["alpha2"]], fwd)
    assert ic > 0.05, f"信号组合 IC 过低: {ic:.3f}"


def test_clone_marginal_contribution_near_zero():
    """克隆（共线）边际贡献 ≈0：不加信息。"""
    facs, fwd = _data()
    base_ic = combo_rank_ic([facs["alpha1"]], fwd)
    new_ic = combo_rank_ic([facs["alpha1"], facs["clone"]], fwd)
    # 加入克隆 IC 几乎不变
    assert abs(new_ic - base_ic) < 1e-6, f"克隆改变了 IC: {base_ic}→{new_ic}"
    mc = evaluate_marginal_contribution(facs["clone"], [facs["alpha1"]], fwd)
    assert abs(mc) < 1e-6, f"克隆边际贡献应≈0，got {mc:.4f}"


def test_independent_factor_positive_marginal():
    """独立有效因子的边际贡献 > 0。"""
    facs, fwd = _data()
    mc = evaluate_marginal_contribution(facs["alpha2"], [facs["alpha1"]], fwd)
    assert mc > 0.005, f"alpha2 边际贡献应>0，got {mc:.4f}"


def test_noise_not_positive_marginal():
    """噪声因子边际贡献不应为正（稀释信号 → ≈0 或负）。"""
    facs, fwd = _data()
    mc_noise = evaluate_marginal_contribution(facs["noise"], [facs["alpha1"]], fwd)
    mc_sig = evaluate_marginal_contribution(facs["alpha2"], [facs["alpha1"]], fwd)
    # 真实独立信号贡献为正；噪声稀释不增 IC
    assert mc_sig > 0, f"alpha2 贡献应>0: {mc_sig:.4f}"
    assert mc_noise <= 0.0005, f"噪声贡献应≤0: {mc_noise:.4f}"


def test_greedy_select_picks_signal_not_noise():
    """贪心选择应选 alpha1+alpha2，排除 clone 与 noise。"""
    facs, fwd = _data()
    names = ["alpha1", "alpha2", "clone", "noise"]
    selected = greedy_select([facs[n] for n in names], fwd,
                             max_factors=2, names=names)
    assert "clone" not in selected, "clone 被选中（去冗余失败）"
    assert "noise" not in selected, "noise 被选中"
    assert "alpha1" in selected
