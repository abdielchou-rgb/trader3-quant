"""
MoE 专家集成（Mixture-of-Experts Ensemble）。

把多个专家模型（如 lgbm / et / ridge / lstm / transformer）的预测，
用门控网络（gating）逐期动态加权融合，产出组合得分。

门控策略：
- ic_softmax    : 各专家近窗 RankIC 的 EWMA 做 softmax 权重（动态模型融合）
- regime_affinity: 在 IC 权重基础上，乘以「状态→专家」亲和力（状态路由）

OOS 严谨性：专家得分由 build_ensemble 的 expanding-window OOS 产出，
门控权重只用历史信息计算，避免前视。

用法：
    res = train_moe(features_wide, fwd, MoEConfig(experts=["lgbm","et","ridge"]))
    combined = res.scores
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trader3.v2.ensemble import EnsembleConfig, build_ensemble

logger = logging.getLogger("trader3.v2.moe")


@dataclass
class MoEConfig:
    experts: list[str] = field(default_factory=lambda: ["lgbm", "et", "ridge"])
    min_train: int = 250
    ic_window: int = 20                  # RankIC 滚动窗口
    ic_ewma_lambda: float = 0.9         # IC 指数加权衰减
    gate_method: str = "ic_softmax"      # ic_softmax | regime_affinity
    regime_affinity: dict[str, dict[str, float]] | None = None  # label -> {expert: w}
    ensemble_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class MoEResult:
    scores: pd.Series                     # 融合后的组合得分 (date, asset)
    expert_scores: dict[str, pd.Series]   # 各专家 OOS 得分
    gate_weights: pd.DataFrame            # date × expert 权重
    oos_rank_ic: float
    used_experts: list[str]
    gate_method: str


def _rank_ic(pred: np.ndarray, fwd: np.ndarray) -> float:
    m = ~(np.isnan(pred) | np.isnan(fwd))
    if m.sum() < 10:
        return 0.0
    rp = pd.Series(pred[m].astype(float)).rank().values
    rf = pd.Series(fwd[m].astype(float)).rank().values
    if np.std(rp) < 1e-10 or np.std(rf) < 1e-10:
        return 0.0
    return float(np.corrcoef(rp, rf)[0, 1])


def build_expert_scores(
    features_wide: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    experts: list[str],
    min_train: int = 250,
    ensemble_kwargs: dict | None = None,
) -> dict[str, pd.Series]:
    """逐专家调用 build_ensemble，返回 {expert: OOS 堆叠得分 (date, asset)}。"""
    out: dict[str, pd.Series] = {}
    for mdl in experts:
        try:
            res = build_ensemble(
                features_wide, forward_returns,
                config=EnsembleConfig(model=mdl, min_train=min_train, **(ensemble_kwargs or {})),
            )
            out[mdl] = res.scores
        except Exception as e:  # noqa: BLE001
            logger.warning("专家 %s 训练失败: %s", mdl, e)
    return out


def _rolling_expert_ic(
    expert_scores: dict[str, pd.Series],
    forward_returns: pd.DataFrame,
    window: int,
    lam: float,
) -> pd.DataFrame:
    """各专家逐期 RankIC 的 EWMA（截面对齐 forward）。"""
    experts = list(expert_scores.keys())
    dates = None
    for s in expert_scores.values():
        dates = s.index.get_level_values(0).unique() if isinstance(s.index, pd.MultiIndex) else s.index
        break
    dates = pd.DatetimeIndex(dates)
    fwd = forward_returns.reindex(index=dates)
    ic_records: dict[str, list[float]] = {k: [] for k in experts}
    for t in dates:
        for k in experts:
            s = expert_scores[k]
            if isinstance(s.index, pd.MultiIndex):
                sc = s.xs(t, level=0) if t in s.index.get_level_values(0) else None
            else:
                sc = s.loc[[t]] if t in s.index else None
            if sc is None:
                ic_records[k].append(np.nan)
                continue
            sc = sc.reindex(fwd.columns).astype(float)
            if t in fwd.index:
                fw = fwd.loc[t].reindex(sc.index).astype(float)
            else:
                fw = pd.Series(np.nan, index=sc.index)
            ic_records[k].append(_rank_ic(sc.values, fw.values))
    raw = pd.DataFrame(ic_records, index=dates)
    # EWMA（按时间正向）
    ewma = raw.copy()
    for k in experts:
        vals = raw[k].values.astype(float)
        out = np.full_like(vals, np.nan, dtype=float)
        cur = np.nan
        for i, v in enumerate(vals):
            if np.isnan(v):
                out[i] = cur
                continue
            cur = v if np.isnan(cur) else lam * cur + (1 - lam) * v
            out[i] = cur
        ewma[k] = out
    return ewma


def _gate_weights(
    ewma_ic: pd.DataFrame,
    method: str,
    regime_affinity: dict | None,
    regime_labels: pd.Series | None,
) -> pd.DataFrame:
    """由 EWMA IC 计算逐期 softmax 权重；regime_affinity 做状态修正。

    每个交易日（行）对专家做 softmax，权重在该行内求和为 1。
    """
    experts = list(ewma_ic.columns)
    w = ewma_ic.copy().fillna(0.0)
    for dt in w.index:
        row = w.loc[dt].values.astype(float)
        if method == "regime_affinity" and regime_affinity and regime_labels is not None:
            lab = regime_labels.get(dt)
            aff = regime_affinity.get(lab) if lab else None
            if aff:
                for j, k in enumerate(experts):
                    row[j] = row[j] * aff.get(k, 1.0)
        mx = np.max(row) if len(row) else 0.0
        ex = np.exp(np.clip(row - mx, -30, 30))
        s = ex.sum()
        w.loc[dt] = (ex / s) if s > 0 else np.full(len(experts), 1.0 / len(experts))
    return w


def train_moe(
    features_wide: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    config: MoEConfig | None = None,
    regime_labels: pd.Series | None = None,
) -> MoEResult:
    """训练 MoE：专家 OOS 得分 → 门控权重 → 融合得分。"""
    cfg = config or MoEConfig()
    expert_scores = build_expert_scores(
        features_wide, forward_returns, cfg.experts, cfg.min_train, cfg.ensemble_kwargs)
    if not expert_scores:
        raise RuntimeError("所有专家训练失败，MoE 无法构建")
    experts = list(expert_scores.keys())

    ewma_ic = _rolling_expert_ic(expert_scores, forward_returns, cfg.ic_window, cfg.ic_ewma_lambda)
    gate = _gate_weights(ewma_ic, cfg.gate_method, cfg.regime_affinity, regime_labels)

    template = next(iter(expert_scores.values()))
    is_multi = isinstance(template.index, pd.MultiIndex)
    all_dates = (template.index.get_level_values(0).unique()
                 if is_multi else template.index)

    fused_rows = []
    for t in all_dates:
        if t not in gate.index:
            continue
        gw = gate.loc[t]
        acc: dict[Any, float] = {}
        for k in experts:
            s = expert_scores[k]
            if is_multi:
                if t not in s.index.get_level_values(0):
                    continue
                sc = s.xs(t, level=0)
            else:
                if t not in s.index:
                    continue
                sc = s.loc[t]
            for a, v in sc.items():
                if np.isnan(v):
                    continue
                acc[a] = acc.get(a, 0.0) + float(gw[k]) * float(v)
        for a, v in acc.items():
            fused_rows.append((t, a, v))

    scores = pd.Series(
        [v for _, _, v in fused_rows],
        index=pd.MultiIndex.from_tuples([(d, a) for d, a, _ in fused_rows]),
    )

    ic_vals = []
    for t in all_dates:
        if t not in gate.index:
            continue
        if is_multi:
            if t not in scores.index.get_level_values(0):
                continue
            sc = scores.xs(t, level=0)
        else:
            if t not in scores.index:
                continue
            sc = scores.loc[t]
        if t not in forward_returns.index:
            continue
        fw = forward_returns.loc[t].reindex(sc.index)
        sc = sc.reindex(fw.index)
        ic_vals.append(_rank_ic(sc.values, fw.values))
    oos_ic = float(np.nanmean(ic_vals)) if ic_vals else 0.0

    return MoEResult(
        scores=scores, expert_scores=expert_scores, gate_weights=gate,
        oos_rank_ic=oos_ic, used_experts=experts, gate_method=cfg.gate_method,
    )
