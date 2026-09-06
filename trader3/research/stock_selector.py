"""
滚动 ML 选股组合回测（R9）—— 从研究模块到可交易组合。

把 ml_pipeline（回归打分）与 meta_ml（p(win) 过滤）结合成完整选股器：
  A. 回归模型（rf/gbr）对全池逐日打分，取 top_n 为候选
  B. meta 模型对候选给 p(win)，过滤掉 p < meta_thr 的（默认关=纯回归）
  C. 候选等权 → 当日组合收益

诚实防前视（walk-forward）：
  - 训练只用 [r-lookback, r] 的样本（含 fwd 对齐到 r 内）
  - 用该模型预测 [r+1, r+rebal] 每天，选组合；下一再平衡点重训
  - 换手不建模（日频等权近似），但组合收益与全池等权基准对比
"""

from __future__ import annotations

import numpy as np

from trader3.research.factor_library import compute_alpha_factors
from trader3.research.meta_ml import build_meta_dataset, fit_meta_classifier, meta_win_probability
from trader3.research.ml_pipeline import build_ml_dataset, fit_scorer


def _daily_top_portfolio_returns(
    scores: np.ndarray,
    daily_rets: np.ndarray,
    top_n: int,
    meta_p: np.ndarray | None = None,
    meta_thr: float = 0.0,
) -> tuple[list[float], list[float], list[float]]:
    """按打分逐日选 top_n（meta 可选过滤），返回 (组合日收益, 命中标记)。

    scores: (T,N) 打分；daily_rets: (T,N) 当日实现的收益（预测 t 用前一日信号，
    组合持有一日后收益即为当日——此处调用方保证信号与收益差一天）。
    简化：传入已对齐的 (信号_t, 收益_t)。
    """
    T, N = scores.shape
    port_rets: list[float] = []
    win: list[float] = []
    eq_rets: list[float] = []
    for t in range(T):
        s = scores[t]
        r = daily_rets[t]
        m = np.isfinite(s) & np.isfinite(r)
        if m.sum() < max(5, N // 3):
            continue
        idx = np.where(m)[0]
        sub_s = s[idx]
        order = idx[np.argsort(sub_s)[::-1]]
        cand = order[:top_n]
        # meta 过滤
        if meta_p is not None and len(cand):
            cand = cand[meta_p[t][cand] >= meta_thr] if len(cand) else cand
        if len(cand) == 0:
            cand = order[:top_n]  # 全被滤 → 回退回归 top（保守）
        eq_rets.append(float(np.nanmean(r[idx])))       # 全池等权基准
        port_rets.append(float(np.nanmean(r[cand])))    # 组合
        win.append(1.0 if port_rets[-1] > 0 else 0.0)
    return port_rets, win, eq_rets


class RollingStockSelector:
    """滚动训练选股器：周期重训回归+meta，样本外逐日组合。"""

    def __init__(self, lookback: int = 200, rebal: int = 20, top_n: int = 6,
                 meta_filter: bool = True, reg_model: str = "rf",
                 meta_thr: float = 0.0, cost_bps: float = 10.0):
        self.lookback = lookback
        self.rebal = rebal
        self.top_n = top_n
        self.meta_filter = meta_filter
        self.reg_model = reg_model
        self.meta_thr = meta_thr
        self.cost_bps = cost_bps
        self.portfolio_rets: list[float] = []
        self.win_flags: list[float] = []
        self.bench_rets: list[float] = []

    def fit(self, panel: dict, forward_returns: np.ndarray,
            start_train: int, warmup: int = 40) -> dict:
        """滚动训练：每 rebal 天重训，中间日复用模型。

        时序（防前视核心）：
          - 重训点 r：用 [0, r]（实际 [r-lookback, r] 内样本）训练
          - 预测 [r, min(r+rebal, T)) 每天的信号，选当日组合
          - 组合收益用 forward_returns[t]（即 t 日信号对应 t→t+1 收益）
        为简化对齐：我们令"预测 t 的信号"用因子面板 t-1 收盘构造——此处
        直接以 fwd 已把信号日与收益错开为前提（fwd[t] 是 t 信号 → t+1 收益）。
        """
        factors = compute_alpha_factors(panel)
        T, N = panel["close"].shape
        scores_all = np.full((T, N), np.nan)
        meta_p_all = np.full((T, N), np.nan)
        fwd = np.asarray(forward_returns, dtype=np.float64)

        r = start_train
        while r < T:
            end_pred = min(r + self.rebal, T)
            # 训练段样本：用 fwd 有效且因子齐全的行（在 [r-lookback, r] 内）
            ds_r = build_ml_dataset(factors, fwd, train_end=r, embargo=5,
                                    warmup=warmup)
            if ds_r["X_train"].shape[0] < 50:
                r = end_pred
                continue
            try:
                reg = fit_scorer(ds_r["X_train"], ds_r["y_train"],
                                 model=self.reg_model)
                meta_clf = None
                if self.meta_filter:
                    mds = build_meta_dataset(factors, fwd, train_end=r,
                                             embargo=5, warmup=warmup,
                                             cost_bps=self.cost_bps)
                    if mds["X_train"].shape[0] >= 50:
                        meta_clf = fit_meta_classifier(mds["X_train"],
                                                       mds["y_train"])
            except Exception:  # noqa: BLE001
                r = end_pred
                continue
            # 预测 [r, end_pred) 的每日：因子 t 值 → 打分，收益用 fwd[t]
            for t in range(r, end_pred):
                # 单日特征：全 N 股该日因子值；列级 NaN 经横截面 zscore+0 填充
                X_t = np.stack([factors[n][t] for n in factors], axis=-1)  # (N,F)
                # 横截面 zscore（列内 NaN 安全）
                mu = np.nanmean(X_t, axis=0, keepdims=True)
                sd = np.nanstd(X_t, axis=0, keepdims=True)
                sd = np.where(sd < 1e-12, 1.0, sd)
                Xz = np.nan_to_num((X_t - mu) / sd, nan=0.0, posinf=0.0,
                                   neginf=0.0)
                # 候选 = fwd 有限者（特征 NaN 已中性化，不整行丢弃）
                ok = np.isfinite(fwd[t])
                if ok.sum() < max(5, N // 3):
                    continue
                scores_all[t][ok] = reg.predict(Xz[ok])
                if meta_clf is not None:
                    meta_p_all[t][ok] = meta_win_probability(meta_clf, Xz[ok])
            r = end_pred

        # 用 scores[t] 与 fwd[t] 直接配（fwd[t]=t→t+1 收益）
        port, win, eq = _daily_top_portfolio_returns(
            scores_all, fwd, self.top_n,
            meta_p_all if self.meta_filter else None, self.meta_thr)
        self.portfolio_rets = port
        self.win_flags = win
        self.bench_rets = eq
        nav = np.cumprod(1 + np.array(port)) if port else np.array([1.0])
        return {
            "rets": np.array(port),
            "nav": nav,
            "bench": np.array(eq),
            "mean_win": float(np.mean(win)) if win else 0.0,
        }


def run_rolling_selector(
    panel: dict, forward_returns: np.ndarray, *,
    lookback: int = 200, rebal: int = 20, top_n: int = 6,
    meta_filter: bool = True, start_train: int, warmup: int = 40,
) -> dict:
    """便捷入口：构造 + fit。"""
    sel = RollingStockSelector(lookback=lookback, rebal=rebal, top_n=top_n,
                               meta_filter=meta_filter)
    return sel.fit(panel, forward_returns, start_train=start_train,
                   warmup=warmup)
