"""
Barra 风格暴露构造器（正交门禁的输入侧）。

从量价面板构造末截面三因子标准暴露（机构最小集）：
  1. mom_20 : 20 日动量（close/close[t-20]-1）
  2. size   : 市值代理（对数成交额 20 日均值；无股本数据时的诚实替代，
              与真实市值秩相关度高，用于共线剥离足够）
  3. vol_20 : 20 日收益波动（日收益 std）

全部只用 <=t 数据（末截面），无前视；横截面 zscore 标准化，NaN 安全。
接入点：run_evolution --orthogonal → make_orthogonal_selector。
"""

from __future__ import annotations

import numpy as np

from core.selection import StrategySelector

STYLE_WINDOW = 20


def _last_valid(vec: np.ndarray, w: int) -> np.ndarray:
    """末截面回看 w 期窗口：NaN（暖机/停牌）行剔除后取最后 w 个有效行。"""
    valid_rows = ~np.all(np.isnan(vec), axis=1)
    idx = np.where(valid_rows)[0]
    if len(idx) < 2:
        return np.zeros((0, vec.shape[1]))
    return vec[idx[-w - 1:]]


def build_style_exposures(panel: dict[str, np.ndarray]) -> np.ndarray:
    """面板 → 末截面 (N, 3) 标准化风格暴露 [mom_20, size, vol_20]。"""
    close = np.asarray(panel["close"], dtype=np.float64)
    amount = np.asarray(panel.get("amount", close * 1e6), dtype=np.float64)
    if close.ndim != 2:
        raise ValueError("panel 字段须为 2D (T, N)")
    T, N = close.shape

    # 1. 动量：末值 / 回看 20 期 - 1
    w = min(STYLE_WINDOW, T - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mom = close[-1] / close[-1 - w] - 1.0

    # 2. 市值代理：对数成交额 20 日均值（末截面）
    with np.errstate(invalid="ignore"):
        log_amt = np.log(np.where(amount > 0, amount, np.nan))
    tail = _last_valid(log_amt, STYLE_WINDOW)
    size = np.nanmean(tail, axis=0) if tail.shape[0] else np.zeros(N)

    # 3. 波动：日收益末 20 期 std
    with np.errstate(invalid="ignore"):
        rets = close[1:] / close[:-1] - 1.0
    rtail = _last_valid(rets, STYLE_WINDOW)
    vol = np.nanstd(rtail, axis=0) if rtail.shape[0] else np.zeros(N)

    X = np.vstack([
        np.nan_to_num(mom, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(size, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0),
    ]).T  # (N, 3)

    # 横截面 zscore（每列），退化列（std≈0）置 0
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd_safe = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd_safe


def make_orthogonal_selector(
    panel: dict[str, np.ndarray],
    forward_returns: np.ndarray,
    **selector_kwargs,
) -> StrategySelector:
    """构造带正交门禁的 StrategySelector（evolve 管线入口）。

    fwd 取末截面（与因子值末截面对齐）——残差化评估只用横截面信息。
    """
    X = build_style_exposures(panel)
    fwd = np.asarray(forward_returns, dtype=np.float64)
    # 末截面 fwd：最后一个全有限行（或直接末行 NaN→0 交由评估器 mask）
    last = fwd[-1].copy()
    return StrategySelector(barra_styles=X, forward_returns=last,
                            **selector_kwargs)
