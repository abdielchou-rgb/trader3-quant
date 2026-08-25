"""
3号交易员 — Combinatorial Purged Cross-Validation (CPCV)

实现 López de Prado《Advances in Financial Machine Learning》ch.12 的
组合净化交叉验证：把全样本均分为 n_blocks 个连续块，枚举全部
C(n_blocks, test_blocks) 个测试块组合；每组合用其余块（剔除与测试块相邻
purge 日）训练因子均值 → 等权 top-k，固定权重应用于该组合的各测试块。

与单点 WFA 不同，CPCV 给策略 OOS 表现一个**分布**：
- sr_daily_list 覆盖所有组合路径的日度夏普；
- 中位数 / p05 / p95 刻画分布位置与宽度；
- prob_negative（日 SR<0 占比）是"策略实际为负"的非参数概率估计。

交易现实性与 WFA 完全同口径（复用 backtest 引擎既有函数）：
- 各测试块首日视为调仓执行日：_apply_execution_constraints 涨跌停拦截，
  按 commission（缺省 DEFAULT_COSTS）对实际成交换手扣单边建仓成本；
- 段内不调仓；被拦截资金按现金处理（收益 0）。
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from trader3.tools.backtest import (
    DEFAULT_PRICE_LIMIT,
    TRADING_DAYS_PER_YEAR,
    _apply_execution_constraints,
    _price_limit_ratio,
)
from trader3.v2.costs import DEFAULT_COSTS, CommissionInfo


def split_blocks(T: int, n_blocks: int) -> list[tuple[int, int]]:
    """把 [0, T) 均分为 n_blocks 个连续块，返回 [(start, end), ...]（end 不含）。

    T 不能整除时块长相差至多 1（np.array_split 口径）。
    """
    idx = np.array_split(np.arange(T), n_blocks)
    return [(int(b[0]), int(b[-1]) + 1) for b in idx]


def iter_test_combos(n_blocks: int, test_blocks: int) -> list[tuple[int, ...]]:
    """枚举全部 C(n_blocks, test_blocks) 个测试块组合（块索引升序元组）。"""
    return list(combinations(range(n_blocks), test_blocks))


def train_keep_mask(T: int, blocks: list[tuple[int, int]],
                    combo: tuple[int, ...], purge: int) -> np.ndarray:
    """组合的训练样本掩码：其余块中剔除与任一测试块相邻 purge 日的样本。

    测试块本身及其前/后各 purge 天一律置 False —— 训练段任何观测的
    信息窗口都不与测试段重叠（purge + embargo 防泄漏）。
    """
    keep = np.ones(T, dtype=bool)
    for bi in combo:
        s, e = blocks[bi]
        keep[max(0, s - purge):min(T, e + purge)] = False
    return keep


def run_cpcv(
    stock_returns: np.ndarray,
    factor_scores: np.ndarray,
    codes: list[str] | None = None,
    n_blocks: int = 6,
    test_blocks: int = 2,
    purge: int = 5,
    commission: CommissionInfo | None = None,
    top_k: int | None = None,
) -> dict:
    """
    Combinatorial Purged CV（AFML ch.12）—— OOS 表现分布而非单点。

    Parameters
    ----------
    stock_returns : (T, N) 日收益面板
    factor_scores : (T, N) 因子得分面板（NaN/inf 观测忽略，与 WFA 同语义）
    codes : 可选，逐股涨跌停幅度按板块判定；None 统一主板 DEFAULT_PRICE_LIMIT
    n_blocks : 块数 N
    test_blocks : 每组合留作测试的块数 k（须 1 <= k < N）
    purge : 测试块两侧各剔除的相邻训练日数（embargo）
    commission : 可选费用模型；None 用 DEFAULT_COSTS
    top_k : 等权持仓数；None 用 max(N//5, 10)（与 _run_wfa_rolling 同语义）

    Returns
    -------
    dict(
      n_combos        : C(n_blocks, test_blocks)
      paths           : AFML 公式 C(N,k)*k/N（组合重排出的独立回测路径数）
      sr_daily_list   : 各组合 OOS 路径日度夏普列表
      sr_ann_median   : 年化夏普中位数
      sr_ann_p05/p95  : 年化夏普 5%/95% 分位
      prob_negative   : 日 SR<0 的组合占比
      oos_days_total  : 全部组合的测试日观测总数
    )
    """
    stock_returns = np.asarray(stock_returns, dtype=np.float64)
    factor_scores = np.asarray(factor_scores, dtype=np.float64)
    if stock_returns.ndim != 2 or factor_scores.shape != stock_returns.shape:
        raise ValueError(
            f"面板形状不一致: returns {stock_returns.shape} vs factors {factor_scores.shape}"
        )
    T, N = stock_returns.shape
    if not (2 <= n_blocks <= T):
        raise ValueError(f"n_blocks={n_blocks} 需在 [2, T={T}] 内")
    if not (1 <= test_blocks < n_blocks):
        raise ValueError(f"test_blocks={test_blocks} 需在 [1, n_blocks-1={n_blocks - 1}] 内")
    if purge < 0:
        raise ValueError(f"purge={purge} 不能为负")

    k = int(top_k) if top_k else max(N // 5, 10)
    k = min(k, N)
    cm = commission if commission is not None else DEFAULT_COSTS

    if codes is not None:
        limit_ratios = np.array(
            [_price_limit_ratio(c) for c in codes], dtype=np.float64
        )
    else:
        limit_ratios = np.full(N, DEFAULT_PRICE_LIMIT, dtype=np.float64)

    blocks = split_blocks(T, n_blocks)
    combos = iter_test_combos(n_blocks, test_blocks)
    sqrt_ann = math.sqrt(TRADING_DAYS_PER_YEAR)

    sr_daily_list: list[float] = []
    oos_days_total = 0
    for combo in combos:
        # ── 训练段因子均值 → 等权 top-k（与 _run_wfa_rolling 同语义）──
        keep = train_keep_mask(T, blocks, combo, purge)
        fs_block = factor_scores[keep]
        finite_mask = np.isfinite(fs_block)
        obs_counts = finite_mask.sum(axis=0)
        obs_sums = np.where(finite_mask, fs_block, 0.0).sum(axis=0)
        is_signal = np.full(N, -np.inf, dtype=np.float64)
        np.divide(obs_sums, obs_counts, out=is_signal, where=obs_counts > 0)

        ranked = np.argsort(is_signal)[::-1]
        weights = np.zeros(N, dtype=np.float64)
        weights[ranked[:k]] = 1.0 / k

        # ── 固定权重应用于该组合的全部测试块（首日=执行日，同 WFA 口径）──
        chunks: list[np.ndarray] = []
        for bi in combo:
            s, e = blocks[bi]
            effective, cash_weight, _n_up, _n_down = _apply_execution_constraints(
                np.zeros(N, dtype=np.float64),
                weights,
                stock_returns[s],
                limit_ratios,
                0.0,
            )
            executed_delta = float(np.sum(np.abs(effective)))  # 空仓建仓：|eff - 0|
            day0_cost = cm.turnover_cost(executed_delta)

            seg = stock_returns[s:e] @ effective
            seg[0] = float(effective @ stock_returns[s]) + cash_weight * 0.0 - day0_cost
            chunks.append(seg)
            oos_days_total += e - s

        path_rets = np.concatenate(chunks)
        p_std = float(np.std(path_rets, ddof=1))
        sr_daily_list.append(
            float(np.mean(path_rets)) / p_std if p_std > 1e-10 else 0.0
        )

    sr_daily = np.asarray(sr_daily_list, dtype=np.float64)
    sr_ann = sr_daily * sqrt_ann
    n_combos = len(combos)

    return {
        "n_combos": n_combos,
        # AFML ch.12: 独立回测路径数 = C(N,k)*k/N（N=6,k=2 时为 5 条）
        "paths": int(round(n_combos * test_blocks / n_blocks)),
        "sr_daily_list": [float(x) for x in sr_daily],
        "sr_ann_median": float(np.percentile(sr_ann, 50)),
        "sr_ann_p05": float(np.percentile(sr_ann, 5)),
        "sr_ann_p95": float(np.percentile(sr_ann, 95)),
        "prob_negative": float(np.mean(sr_daily < 0.0)),
        "oos_days_total": int(oos_days_total),
    }
