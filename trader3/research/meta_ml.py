"""
Meta-label 分类选股（R8）—— 把"该不该开仓"作为独立学习目标。

解剖结论（mlfinlab/AFML meta-labeling）：主模型定方向，元模型吸收成本/
噪声判断"这笔是否值得真执行"。A股多头框架（无做空）下 meta 化为：
  标签 y = 1  ⟺ 该股次日扣成本后收益 > 0（值得做多）
特征 = 因子面板横截面 zscore；分类器输出 p(win)。

与回归打分（ml_pipeline）的区别：
  - 回归：学"收益多高"（排序精度）
  - meta 分类：学"这笔是否值得进组合"（命中率/可交易性）
两者可互补：先用回归排序选候选，再用 p(win) 过滤/赋权。

复用 ml_pipeline 的逐日横截面 zscore + embargo 时间切分工具。
"""

from __future__ import annotations

import numpy as np

from trader3.research.ml_pipeline import _cross_sectional_zscore


def build_meta_dataset(
    factors: dict[str, np.ndarray],
    forward_returns: np.ndarray,
    train_end: int,
    embargo: int = 5,
    warmup: int = 40,
    cost_bps: float = 0.0,
) -> dict:
    """因子面板 → 二分类样本。

    y = 1 ⟺ fwd 扣双边成本 > 0；只保留 fwd 有限行。
    其余切分语义与 ml_pipeline.build_ml_dataset 一致。
    """
    names = list(factors.keys())
    F = np.stack([factors[n] for n in names], axis=-1)  # (T,N,F)
    T, N, _ = F.shape
    y = np.asarray(forward_returns, dtype=np.float64)
    cost = cost_bps / 10_000.0
    test_start = train_end + embargo

    def _rows(t0: int, t1: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
        Xs, ys, ts = [], [], []
        for t in range(t0, t1):
            fv = F[t]
            yv = y[t]
            mask = np.isfinite(yv)
            if t < warmup:
                mask &= np.all(np.isfinite(fv), axis=1)
            if mask.sum() < max(5, N // 3):
                continue
            fv_z = _cross_sectional_zscore(fv)[mask]
            ylab = (yv[mask] - 2 * cost > 0).astype(np.int8)
            Xs.append(fv_z)
            ys.append(ylab)
            ts.extend([t] * int(mask.sum()))
        X = np.vstack(Xs) if Xs else np.zeros((0, F.shape[-1]))
        yf = np.concatenate(ys) if ys else np.zeros(0, dtype=np.int8)
        return X, yf, ts

    X_train, y_train, _ = _rows(0, train_end)
    X_test, y_test, test_ts = _rows(test_start, T)
    return {"X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test, "test_t": np.array(test_ts)}


def fit_meta_classifier(X_train: np.ndarray, y_train: np.ndarray,
                        seed: int = 42):
    """训练 meta 分类器（梯度提升树，轻量）。"""
    if X_train.shape[0] < 30:
        raise ValueError(f"训练样本不足: {X_train.shape[0]}")
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(
        n_estimators=60, max_depth=3, learning_rate=0.1,
        random_state=seed).fit(X_train, y_train)


def meta_win_probability(clf, X_test: np.ndarray) -> np.ndarray:
    """输出 p(win)（正类概率）。"""
    return np.asarray(clf.predict_proba(X_test)[:, 1], dtype=np.float64)


def hit_rate_top(clf, X_test: np.ndarray, y_test: np.ndarray,
                 top_frac: float = 0.3) -> tuple[float, int]:
    """按 p(win) 取 top_frac 子集，返回该子集命中率与样本数。"""
    p = meta_win_probability(clf, X_test)
    y = np.asarray(y_test, dtype=np.int64)
    n_top = max(int(len(p) * top_frac), 1)
    idx = np.argsort(p)[::-1][:n_top]
    return float(np.mean(y[idx] == 1)), int(n_top)
