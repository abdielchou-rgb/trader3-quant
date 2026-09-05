"""
GP 因子挖掘的正交残差化评估器（Orthogonal Residualization）。

痛点：GP 挖出的因子大量与 Barra 风格（动量/市值/波动）高度共线 ——
单因子 IC 极高，但对组合没有增量收益（风格溢价可由指数增强/期货多头
更廉价地获取）。

解法：在适应度评估处做截面回归剥离 ——
    residual = P_orth @ factor,  P_orth = I - X(X'X)^+ X'
其中 X = [1, 风格暴露]。残差 IC（Spearman 秩相关）才是该因子的
**边际增量 Alpha**；显著低于原始 IC 的候选被 StrategySelector 的
orthogonality 门禁拒绝。

性能：P_orth 在构造时一次性预计算（投影算子复用），逐因子评估只剩
一个矩阵乘法，GP 大种群迭代下开销可忽略。
"""

from __future__ import annotations

import numpy as np


class OrthogonalFitnessEvaluator:
    """截面正交残差评估器（Barra 风格剥离）。"""

    def __init__(self, barra_styles: np.ndarray):
        """
        Parameters
        ----------
        barra_styles : (N_assets, K_factors)
            预先标准化的行业/风格暴露矩阵。加常数截距后构造正交投影算子。
        """
        X = np.asarray(barra_styles, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"barra_styles 须为 2D (N, K)，got {X.shape}")
        if X.shape[0] < X.shape[1] + 2:
            raise ValueError("样本数须多于风格因子数（否则残差空间退化）")
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        # pinv：风格间高度共线（如市值与流通市值）时仍数值稳定
        XtX_inv = np.linalg.pinv(X.T @ X)
        self.P_orth = np.eye(X.shape[0]) - X @ XtX_inv @ X.T
        self.n_assets = X.shape[0]

    def residualize(self, raw_factor_values: np.ndarray) -> np.ndarray:
        """标准化 → 投影剥离 → 残差 clip（顺序关键：先投影后 clip，
        clip 先于投影会破坏投影几何，让纯风格克隆产生伪残差）。"""
        f = np.asarray(raw_factor_values, dtype=np.float64).ravel()
        if f.shape[0] != self.n_assets:
            raise ValueError(
                f"因子截面长度 {f.shape[0]} 与暴露矩阵 {self.n_assets} 不一致"
            )
        mu = np.nanmean(f)
        sd = np.nanstd(f)
        if not np.isfinite(mu) or sd < 1e-8:
            return np.zeros_like(f)
        norm = (f - mu) / sd
        norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
        residual = self.P_orth @ norm
        # 残差异常值抑制（投影后的自身离群，非风格泄漏）
        rstd = np.std(residual)
        if rstd > 1e-12:
            residual = np.clip(residual / rstd, -3.0, 3.0)
        return residual

    def evaluate_orthogonal_ic(
        self,
        raw_factor_values: np.ndarray,
        forward_returns: np.ndarray,
    ) -> float:
        """残差 Alpha 与前向收益的截面 Spearman 秩 IC（边际增量）。

        无未来函数：残差化只用当日截面（横截面回归），不触碰时间维度。
        """
        residual = self.residualize(raw_factor_values)
        if np.std(residual) < 1e-12:  # 常量/全 NaN 截面 → 无信息
            return 0.0
        r = np.asarray(forward_returns, dtype=np.float64).ravel()
        if r.shape[0] != self.n_assets:
            raise ValueError(
                f"前向收益长度 {r.shape[0]} 与暴露矩阵 {self.n_assets} 不一致"
            )
        mask = np.isfinite(r) & np.isfinite(residual)
        if mask.sum() < 5:
            return 0.0
        res_rank = np.argsort(np.argsort(residual[mask]))
        ret_rank = np.argsort(np.argsort(r[mask]))
        if np.std(res_rank) < 1e-10 or np.std(ret_rank) < 1e-10:
            return 0.0
        corr = float(np.corrcoef(res_rank, ret_rank)[0, 1])
        return 0.0 if np.isnan(corr) else corr
