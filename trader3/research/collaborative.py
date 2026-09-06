"""
协同因子评估（S1 落地）—— 从"挖单因子"到"挖协同集合"。

解剖结论（alpha_research/dissection.md S1）：GP 进化最核心落差 =
适应度只看单因子 IC，忽视与已有因子/Barra 风格的共线 —— 单因子 IC 高
但组合无增量。

本模块提供组合级度量：
  - combo_rank_ic：多因子等权 zscore 合成 → 截面 Rank-IC（与单因子同口径）
  - evaluate_marginal_contribution：新因子加入组合后的 IC 增量
      （与已选因子共线 → 增量≈0 → 无价值，应被贪心排除）
  - greedy_select：从候选池按边际贡献贪心选因子（去冗余）

用途：GP 进化适应度可改为「加入精英池后的边际贡献」而非裸 IC；
候选入库前用 greedy_select 验证其确实增强组合。
"""

from __future__ import annotations

import numpy as np


def _cs_rank_1d(a: np.ndarray) -> np.ndarray:
    """横截面 rank（沿 0 轴，返回 [0,1]）。"""
    order = np.argsort(a)
    ranks = np.argsort(order)
    n = len(a)
    return ranks / (n - 1) if n > 1 else ranks


def _combo_signal(factors: list[np.ndarray]) -> np.ndarray:
    """多因子等权 zscore 合成（逐列 zscore 后平均）。"""
    stack = []
    for f in factors:
        arr = np.asarray(f, dtype=np.float64)
        # 横截面 zscore（每行）
        mu = np.nanmean(arr, axis=1, keepdims=True)
        sd = np.nanstd(arr, axis=1, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        stack.append(np.nan_to_num((arr - mu) / sd, nan=0.0))
    combo = np.mean(stack, axis=0)
    return combo


def combo_rank_ic(factors: list[np.ndarray],
                  forward_returns: np.ndarray) -> float:
    """组合信号（等权 zscore 合成）与 fwd 的平均截面 Rank-IC。

    Parameters
    ----------
    factors : 因子面板列表，各 (T,N)
    forward_returns : (T,N) 前瞻收益

    Returns
    -------
    平均 Rank-IC（非 NaN 期）
    """
    if not factors:
        return 0.0
    combo = _combo_signal(factors)
    fwd = np.asarray(forward_returns, dtype=np.float64)
    T, N = combo.shape
    ics: list[float] = []
    for t in range(T):
        s = combo[t]
        r = fwd[t]
        mask = np.isfinite(r)
        if mask.sum() < max(5, N // 3):
            continue
        s, r = s[mask], r[mask]
        if np.std(s) < 1e-10 or np.std(r) < 1e-10:
            continue
        sr = _cs_rank_1d(s)
        rr = _cs_rank_1d(r)
        ics.append(float(np.corrcoef(sr, rr)[0, 1]))
    return float(np.mean(ics)) if len(ics) > 5 else 0.0


def evaluate_marginal_contribution(
    new_factor: np.ndarray,
    existing_factors: list[np.ndarray],
    forward_returns: np.ndarray,
) -> float:
    """新因子加入现有组合后的 IC 增量（边际贡献）。

    共线克隆 → 增量≈0；独立有效信号 → 增量>0；噪声 → ≈0。
    """
    base_ic = combo_rank_ic(existing_factors, forward_returns)
    new_ic = combo_rank_ic(existing_factors + [new_factor], forward_returns)
    return float(new_ic - base_ic)


def greedy_select(
    candidates: list[np.ndarray],
    forward_returns: np.ndarray,
    max_factors: int = 5,
    names: list[str] | None = None,
    mc_min: float = 1e-5,
) -> list[str]:
    """从候选池按边际贡献贪心选因子（去冗余）。

    Parameters
    ----------
    candidates : 因子面板列表
    names : 可选候选名（返回这些名）
    max_factors : 最多选几个
    mc_min : 边际贡献门槛（低于视为不加信息）

    Returns
    -------
    被选因子名（无 names 时返回索引序号 "f0"..）
    """
    if not candidates:
        return []
    labels = names if names else [f"f{i}" for i in range(len(candidates))]
    selected_idx: list[int] = []
    selected_factors: list[np.ndarray] = []
    remaining = list(range(len(candidates)))
    for _ in range(min(max_factors, len(candidates))):
        best_mc = -np.inf
        best_i = -1
        for i in remaining:
            mc = evaluate_marginal_contribution(
                candidates[i], selected_factors, forward_returns)
            if mc > best_mc:
                best_mc = mc
                best_i = i
        if best_i < 0 or best_mc < mc_min:
            break
        selected_idx.append(best_i)
        selected_factors.append(candidates[best_i])
        remaining.remove(best_i)
    return [labels[i] for i in selected_idx]
