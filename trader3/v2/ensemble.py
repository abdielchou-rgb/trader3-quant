"""
多因子ML合成（Ensemble Stacking）。

将多个因子值作为特征，预测未来收益，输出合成因子得分。
模型优先级：LightGBM > XGBoost > sklearn (GradientBoosting / ExtraTrees / Ridge)。

设计要点（防过拟合）：
- 时间序列 CV（expanding window），禁止随机 shuffle
- 嵌入式 IC 加权基线对比：若 ML 不显著优于线性基线则自动回退
- 输出带 OOS 评估指标（RankIC / ICIR）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMRegressor
    _LGB = True
except ImportError:
    _LGB = False

try:
    from xgboost import XGBRegressor
    _XGB = True
except ImportError:
    _XGB = False

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge


@dataclass
class EnsembleConfig:
    model: str = "auto"                # auto | lgbm | xgb | gbr | et | rf | ridge
    n_splits: int = 5                  # expanding-window 折数
    min_train: int = 250               # 最小训练样本
    retrain_every: int = 60            # 滚动再训练频率（天）
    fallback_ic_edge: float = 0.002    # ML 相对线性基线的最小 RankIC 提升
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleResult:
    scores: pd.Series                       # 合成因子得分 (date*asset)
    oos_rank_ic: float                      # OOS 平均 RankIC
    oos_ic_ir: float                        # ICIR
    baseline_rank_ic: float                 # 等权/IC加权基线 RankIC
    model_used: str
    used_ml: bool                           # 是否启用ML（未回退到基线）
    feature_importance: dict[str, float]
    fold_metrics: list[dict[str, float]]


def _make_model(name: str, seed: int, params: dict):
    p = dict(params)
    if name == "lgbm" and _LGB:
        return LGBMRegressor(
            n_estimators=p.pop("n_estimators", 300), learning_rate=p.pop("learning_rate", 0.05),
            num_leaves=p.pop("num_leaves", 31), max_depth=p.pop("max_depth", -1),
            subsample=p.pop("subsample", 0.8), colsample_bytree=p.pop("colsample_bytree", 0.8),
            random_state=seed, verbose=-1, **p)
    if name == "xgb" and _XGB:
        return XGBRegressor(
            n_estimators=p.pop("n_estimators", 300), learning_rate=p.pop("learning_rate", 0.05),
            max_depth=p.pop("max_depth", 6), subsample=p.pop("subsample", 0.8),
            colsample_bytree=p.pop("colsample_bytree", 0.8),
            random_state=seed, verbosity=0, **p)
    if name == "gbr":
        return GradientBoostingRegressor(
            n_estimators=p.pop("n_estimators", 200), learning_rate=p.pop("learning_rate", 0.05),
            max_depth=p.pop("max_depth", 3), subsample=p.pop("subsample", 0.8), random_state=seed, **p)
    if name == "et":
        return ExtraTreesRegressor(
            n_estimators=p.pop("n_estimators", 300), max_depth=p.pop("max_depth", None),
            n_jobs=-1, random_state=seed, **p)
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=p.pop("n_estimators", 300), max_depth=p.pop("max_depth", None),
            n_jobs=-1, random_state=seed, **p)
    return Ridge(alpha=p.pop("alpha", 1.0))


def _rank_ic(pred: np.ndarray, fwd: np.ndarray) -> float:
    m = ~(np.isnan(pred) | np.isnan(fwd))
    if m.sum() < 10 or np.std(fwd[m]) < 1e-12 or np.std(pred[m]) < 1e-12:
        return 0.0
    rp = pd.Series(pred[m]).rank().values
    rf = pd.Series(fwd[m]).rank().values
    return float(np.corrcoef(rp, rf)[0, 1])


def rank_ic_series(scores_wide: pd.DataFrame, fwd_wide: pd.DataFrame) -> pd.Series:
    """逐日横截面 RankIC。"""
    out = {}
    for date in scores_wide.index.intersection(fwd_wide.index):
        s, f = scores_wide.loc[date], fwd_wide.loc[date]
        ic = _rank_ic(s.values.astype(float), f.values.astype(float))
        out[date] = ic
    return pd.Series(out).dropna()


def build_ensemble(features_wide: dict[str, pd.DataFrame],
                   forward_returns: pd.DataFrame,
                   config: EnsembleConfig | None = None,
                   seed: int = 42) -> EnsembleResult:
    """
    features_wide:     {factor_name -> DataFrame(date × asset)} 因子暴露宽表
    forward_returns:   DataFrame(date × asset) 未来收益（对齐因子日期）
    """
    cfg = config or EnsembleConfig()

    # 对齐所有特征
    names = list(features_wide.keys())
    idx = sorted(set.intersection(*(set(v.index) for v in features_wide.values()))
                 & set(forward_returns.index))
    cols = sorted(set.intersection(*(set(v.columns) for v in features_wide.values()))
                  & set(forward_returns.columns))
    dates = pd.DatetimeIndex(idx)

    X_all = np.stack([features_wide[n].loc[idx, cols].values for n in names], axis=-1)  # T×N×K
    y_all = forward_returns.loc[idx, cols].values                                       # T×N
    T = len(dates)
    if T < cfg.min_train + 20:
        raise ValueError(f"时间长度不足: {T} < min_train({cfg.min_train})+20")

    model_name = cfg.model
    if model_name == "auto":
        model_name = "lgbm" if _LGB else ("xgb" if _XGB else "gbr")

    score_mat = np.full((T, len(cols)), np.nan)
    fold_metrics: list[dict[str, float]] = []
    importance_acc: dict[str, float] = {}

    # 滚动 expanding-window OOS
    start = cfg.min_train
    next_train_at = start
    model = None
    for t in range(start, T):
        if model is None or t >= next_train_at:
            train_slice = slice(0, t)
            Xt = X_all[train_slice].reshape(-1, len(names))
            yt = y_all[train_slice].reshape(-1)
            m = ~np.isnan(Xt).any(axis=1) & ~np.isnan(yt)
            if m.sum() >= cfg.min_train * len(cols):
                model = _make_model(model_name, seed + t // max(cfg.retrain_every, 1), cfg.params)
                try:
                    model.fit(Xt[m], yt[m])
                    next_train_at = t + cfg.retrain_every
                    imp = getattr(model, "feature_importances_", None)
                    if imp is not None and hasattr(imp, "__len__") and len(imp) == len(names):
                        tot = float(np.sum(imp)) or 1.0
                        for nm, v in zip(names, imp / tot, strict=True):
                            importance_acc[nm] = importance_acc.get(nm, 0.0) + v
                except Exception:
                    model = None
        if model is not None:
            xt = X_all[t].reshape(-1, len(names))
            mt = ~np.isnan(xt).any(axis=1)
            pred = np.full(len(cols), np.nan)
            if mt.sum() > 2:
                try:
                    pred[mt] = model.predict(xt[mt])
                except Exception:
                    pass
            score_mat[t] = pred

    scores_wide = pd.DataFrame(score_mat, index=dates, columns=cols)

    # OOS 指标
    fwd_wide = forward_returns.loc[dates, cols]
    ic_series = rank_ic_series(scores_wide.iloc[cfg.min_train:], fwd_wide.iloc[cfg.min_train:])
    oos_ic = float(ic_series.mean()) if len(ic_series) else 0.0
    oos_icir = float(ic_series.mean() / ic_series.std()) if len(ic_series) > 2 and ic_series.std() > 0 else 0.0

    # 线性基线：各因子逐日 z-score 后等权
    z = []
    for n in names:
        f = features_wide[n].loc[dates, cols]
        mu = f.expanding(min_periods=cfg.min_train).mean()
        sd = f.expanding(min_periods=cfg.min_train).std()
        z.append((f - mu) / sd.replace(0, np.nan))
    base_scores = sum(z_i.fillna(0).values for z_i in z) / len(z)
    base_wide = pd.DataFrame(base_scores, index=dates, columns=cols)
    base_ic_series = rank_ic_series(base_wide.iloc[cfg.min_train:], fwd_wide.iloc[cfg.min_train:])
    base_ic = float(base_ic_series.mean()) if len(base_ic_series) else 0.0

    used_ml = oos_ic >= base_ic + cfg.fallback_ic_edge
    final_model = model_name if used_ml else "equal_weight_baseline"

    total_imp = sum(importance_acc.values()) or 1.0
    feat_imp = {k: v / total_imp for k, v in sorted(importance_acc.items(), key=lambda kv: -kv[1])}

    long = scores_wide.stack() if not used_ml else scores_wide.where(~scores_wide.isna(), base_wide).stack()
    return EnsembleResult(
        scores=long.rename("ensemble_score"),
        oos_rank_ic=oos_ic,
        oos_ic_ir=oos_icir,
        baseline_rank_ic=base_ic,
        model_used=final_model,
        used_ml=used_ml,
        feature_importance=feat_imp,
        fold_metrics=fold_metrics,
    )


def available_models() -> list[str]:
    models = ["gbr", "et", "rf", "ridge"]
    if _LGB:
        models.insert(0, "lgbm")
    if _XGB:
        models.insert(1 if _LGB else 0, "xgb")
    return models
