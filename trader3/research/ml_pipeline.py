"""
ML 打分选股管线（R7）—— 用 sklearn 把因子面板聚合成横截面打分。

解剖结论验证：Alpha158 类表格数据上，线性/树模型常强于单因子排序甚至
深度模型。本模块把 factor_library 的因子面板 → 训练样本（时间切分 +
embargo）→ sklearn 打分器 → OOS 逐日截面 Rank-IC。

关键纪律（防泄漏）：
  - 训练/测试按时间切（train_end 前训练，之后 +embargo 测试）
  - 每行样本 = (t, asset)；特征 = 该日横截面 zscore 后的因子值
  - 只保留 fwd 有限的行；评分后 OOS 逐日算截面 Rank-IC（非合并相关，
    与 GP 的 IC 口径一致，避免跨期 pooling 虚高）
"""

from __future__ import annotations

import numpy as np


def _cross_sectional_zscore(X: np.ndarray) -> np.ndarray:
    """每行（每个时间 t）横截面 zscore。输入 (rows, features)。"""
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)


def _t_index_of(rows: int, n_assets: int) -> np.ndarray:
    return np.repeat(np.arange(rows // n_assets), n_assets)


def build_ml_dataset(
    factors: dict[str, np.ndarray],
    forward_returns: np.ndarray,
    train_end: int,
    embargo: int = 5,
    warmup: int = 40,
) -> dict:
    """因子面板 → 训练/测试样本。

    Parameters
    ----------
    factors : {name: (T,N)}
    forward_returns : (T,N) fwd 收益
    train_end : 时间行切分点（<=train_end 训练；>train_end+embargo 测试）
    embargo : 训练与测试间的净空期（防标签窗口泄漏）
    warmup : 因子暖机期（此前行丢弃，因子 NaN）

    Returns
    -------
    {X_train, y_train, X_test, y_test} 各为 (rows, F)/(rows,)
    每行 = 一个 (t, asset) 样本。
    """
    names = list(factors.keys())
    F = np.stack([factors[n] for n in names], axis=-1)  # (T,N,F)
    T, N, _ = F.shape
    y = np.asarray(forward_returns, dtype=np.float64)
    test_start = train_end + embargo

    def _rows(t0: int, t1: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
        Xs, ys, ts = [], [], []
        for t in range(t0, t1):
            fv = F[t]  # (N, F)
            yv = y[t]  # (N,)
            mask = np.isfinite(yv)
            if t < warmup:
                mask &= np.all(np.isfinite(fv), axis=1)
            if mask.sum() < max(5, N // 3):
                continue
            fv_z = _cross_sectional_zscore(fv)[mask]
            Xs.append(fv_z)
            ys.append(yv[mask])
            ts.extend([t] * int(mask.sum()))
        X = np.vstack(Xs) if Xs else np.zeros((0, F.shape[-1]))
        yf = np.concatenate(ys) if ys else np.zeros(0)
        return X, yf, ts

    X_train, y_train, _ = _rows(0, train_end)
    X_test, y_test, test_ts = _rows(test_start, T)
    return {"X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test, "test_t": np.array(test_ts)}


def fit_scorer(X_train: np.ndarray, y_train: np.ndarray, model: str = "ridge",
               seed: int = 42):
    """训练打分器。model ∈ {ridge, rf, gbr}（轻量、低算力）。"""
    if X_train.shape[0] < 30:
        raise ValueError(f"训练样本不足: {X_train.shape[0]}")
    if model == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0).fit(X_train, y_train)
    if model == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=60, max_depth=6,
                                     n_jobs=1, random_state=seed).fit(
            X_train, y_train)
    if model == "gbr":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=80, max_depth=3,
                                         learning_rate=0.05,
                                         random_state=seed).fit(X_train, y_train)
    raise ValueError(f"未知模型 {model}")


def score_oos(model, X_test: np.ndarray) -> np.ndarray:
    """OOS 打分。"""
    return np.asarray(model.predict(X_test), dtype=np.float64)


def _cs_rank_1d(a: np.ndarray) -> np.ndarray:
    n = len(a)
    if n < 2:
        return np.zeros_like(a)
    ranks = np.argsort(np.argsort(a))
    return ranks / (n - 1)


def oos_rank_ic(scores: np.ndarray, y_test: np.ndarray,
                day_of_row: np.ndarray | None = None) -> float:
    """OOS 逐日截面 Rank-IC 均值。

    scores/y_test 为拼好的样本行；day_of_row 提供每行归属的日期 t 时按日分组，
    否则把全部视为单日截面（仅当传入单日样本时适用）。
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y_test, dtype=np.float64)
    if len(s) != len(y) or len(s) < 5:
        return 0.0
    if day_of_row is None:
        return _single_day_ic(s, y)
    ics: list[float] = []
    for t in np.unique(day_of_row):
        m = day_of_row == t
        if m.sum() < 5:
            continue
        ic = _single_day_ic(s[m], y[m])
        ics.append(ic)
    return float(np.mean(ics)) if len(ics) > 3 else 0.0


def _single_day_ic(s: np.ndarray, y: np.ndarray) -> float:
    if np.std(s) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    sr = _cs_rank_1d(s)
    yr = _cs_rank_1d(y)
    return float(np.corrcoef(sr, yr)[0, 1])
