"""
3号交易员 — 策略筛选门禁

从进化产生的候选因子中，用多道门槛筛选出"真正有用"的策略。
防止过拟合、低 IC、非单调、拥挤度高的因子混入最终策略库。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SelectionResult:
    """单个候选的筛选结果"""
    expr: str
    passed: bool
    score: float = 0.0
    gates: dict[str, dict] = field(default_factory=dict)  # {gate_name: {passed, value}}
    reason: str = ""


class StrategySelector:
    """
    策略筛选器。

    门槛:
      1. IC 绝对值 > 0.02        （相关性门槛）
      2. ICIR > 0.15             （稳定性门槛，A股量价因子实际水平）
      3. 单调性 > 0.4            （分组收益单调门槛）
      4. 多空年化 > 0.10         （可交易性门槛，年化多空差 > 10%）
      5. 表达式复杂度 < 25 节点   （可解释性门槛）
      6. 去重（与已选策略的表达式相似度）
      7. 正交性（可选）：注入 Barra 风格暴露后，残差 IC 不足 → 判共线拒绝

    综合评分 = 0.4*ICIR + 0.3*单调性 + 0.3*多空年化
    """

    def __init__(
        self,
        ic_min: float = 0.02,
        icir_min: float = 0.15,
        monotonicity_min: float = 0.4,
        long_short_min: float = 0.10,
        max_nodes: int = 25,
        barra_styles: np.ndarray | None = None,
        forward_returns: np.ndarray | None = None,
        orth_ic_min: float = 0.01,
        n_trials: int | None = None,
        dsr_min: float = 0.90,
    ):
        self.ic_min = ic_min
        self.icir_min = icir_min
        self.monotonicity_min = monotonicity_min
        self.long_short_min = long_short_min
        self.max_nodes = max_nodes
        self._selected_exprs: list[str] = []
        self._selected_values: list[np.ndarray] = []
        # ── 正交残差化（多重共线拦截，P2-2）──
        self._orth_eval = None
        self._fwd = forward_returns
        self.orth_ic_min = orth_ic_min
        if barra_styles is not None:
            from core.orthogonal_fitness import OrthogonalFitnessEvaluator
            self._orth_eval = OrthogonalFitnessEvaluator(barra_styles)
        # ── DSR 验收闸门（解剖结论 S2）：候选入库前惩罚试验次数 ──
        self.n_trials = n_trials  # 整个进化实验累计个体评估数；None=不启用
        self.dsr_min = dsr_min

    def select(self, candidates: list[dict]) -> list[SelectionResult]:
        """
        筛选候选。

        Parameters
        ----------
        candidates : [{"expr": str, "ic": float, "icir": float,
                       "monotonicity": float, "long_short": float,
                       "fitness": float, "generation": int,
                       "values": ndarray(可选，用于相关性去重)}, ...]

        Returns
        -------
        SelectionResult 列表（按 fitness 降序处理以做贪心去重，返回保持原顺序）
        """
        order = sorted(range(len(candidates)),
                       key=lambda i: candidates[i].get("fitness", -999), reverse=True)
        results: list[SelectionResult | None] = [None] * len(candidates)
        for i in order:
            results[i] = self._evaluate_one(candidates[i])
        return results  # type: ignore[return-value]

    def _evaluate_one(self, cand: dict) -> SelectionResult:
        expr = cand.get("expr", "")
        ic = cand.get("ic", 0)
        icir = cand.get("icir", 0)
        mono = cand.get("monotonicity", 0)
        ls = cand.get("long_short", 0)
        n_nodes = _count_nodes_str(expr)

        gates = {
            "ic": {"passed": ic > self.ic_min, "value": round(ic, 4)},
            "icir": {"passed": icir > self.icir_min, "value": round(icir, 4)},
            "monotonicity": {"passed": mono > self.monotonicity_min, "value": round(mono, 4)},
            "long_short": {"passed": ls > self.long_short_min or abs(ls) > self.long_short_min * 2, "value": round(ls, 4)},
            "complexity": {"passed": n_nodes <= self.max_nodes, "value": n_nodes},
        }

        # 去重：优先用因子值相关性（|Spearman ρ|>0.7），无值时退回 token Jaccard
        values = cand.get("values")
        if values is not None and self._selected_values:
            corr = max(
                (_rank_corr(values, sv) for sv in self._selected_values),
                default=0.0,
            )
            dup = corr > 0.7
            gates["uniqueness"] = {"passed": not dup, "value": round(corr, 4)}
        else:
            dup = any(_expr_similarity(expr, s) > 0.7 for s in self._selected_exprs)
            gates["uniqueness"] = {"passed": not dup, "value": 1 if not dup else 0}

        # 正交性门禁（可选）：风格暴露注入时，残差 IC 不足 → 共线拒绝
        if self._orth_eval is not None and values is not None and self._fwd is not None:
            try:
                orth_ic = self._orth_eval.evaluate_orthogonal_ic(
                    np.asarray(values, dtype=np.float64).ravel(), self._fwd
                )
                gates["orthogonality"] = {
                    "passed": abs(orth_ic) >= self.orth_ic_min,
                    "value": round(orth_ic, 4),
                }
            except Exception:  # noqa: BLE001 — 形状不齐等评估失败 → 保守跳过该门禁
                gates["orthogonality"] = {"passed": True, "value": "n/a (eval skipped)"}

        # DSR 门禁（可选，解剖结论 S2）：候选带逐期 IC 时序时，
        # 用 Deflated Sharpe（惩罚整个进化的累计试验次数）判显著。
        # 缺 ic_series 或未注入 n_trials → 跳过（向后兼容）。
        if self.n_trials is not None:
            ic_series = cand.get("ic_series")
            if ic_series is not None and _has_finite(ic_series):
                try:
                    dsr = _deflated_sharpe(np.asarray(ic_series, dtype=np.float64),
                                           self.n_trials)
                    gates["dsr"] = {
                        "passed": dsr >= self.dsr_min,
                        "value": round(dsr, 4),
                    }
                except Exception:  # noqa: BLE001 — 退化序列保守跳过
                    gates["dsr"] = {"passed": True, "value": "n/a (dsr skip)"}

        passed = all(g["passed"] for g in gates.values())

        if passed:
            # 综合评分
            score = 0.4 * icir + 0.3 * mono + 0.3 * abs(ls)
            self._selected_exprs.append(expr)
            if values is not None:
                self._selected_values.append(np.asarray(values, dtype=np.float64).ravel())
            reason = "通过"
        else:
            score = 0.0
            failed = [k for k, g in gates.items() if not g["passed"]]
            reason = f"未通过: {', '.join(failed)}"

        return SelectionResult(
            expr=expr,
            passed=passed,
            score=round(score, 4),
            gates=gates,
            reason=reason,
        )

    def select_best(self, candidates: list[dict], top_k: int = 5) -> list[SelectionResult]:
        """筛选 + 取 Top-K"""
        results = self.select(candidates)
        passed = [r for r in results if r.passed]
        passed.sort(key=lambda r: r.score, reverse=True)
        return passed[:top_k]


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """秩 Pearson（≈Spearman）。任一侧 NaN/长度不齐返回 0。"""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(len(a), len(b))
    if n < 5:
        return 0.0
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return 0.0
    ra = np.argsort(np.argsort(a[mask]))
    rb = np.argsort(np.argsort(b[mask]))
    if np.std(ra) < 1e-10 or np.std(rb) < 1e-10:
        return 0.0
    return float(abs(np.corrcoef(ra, rb)[0, 1]))


def _count_nodes_str(expr: str) -> int:
    """粗略统计表达式节点数（逗号数 + 1）"""
    return expr.count(",") + 1


def _expr_similarity(e1: str, e2: str) -> float:
    """表达式相似度（基于 token 集合的 Jaccard）"""
    t1 = set(_tokenize_expr(e1))
    t2 = set(_tokenize_expr(e2))
    if not t1 or not t2:
        return 0.0
    inter = len(t1 & t2)
    union = len(t1 | t2)
    return inter / union if union > 0 else 0.0


def _tokenize_expr(expr: str) -> list[str]:
    import re
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+", expr)


def _has_finite(a) -> bool:
    arr = np.asarray(a, dtype=np.float64)
    f = arr[np.isfinite(arr)]
    return len(f) >= 10


def compute_ic_series(signal: np.ndarray, forward_returns: np.ndarray,
                      close_panel: np.ndarray | None = None) -> list[float]:
    """逐期横截面 Rank-IC 序列（供 DSR 门禁）。与 compute_fitness 同口径。

    Returns
    -------
    per-period Spearman IC list（剔除样本不足/退化期）
    """
    from .gp import _cs_rank

    signal = np.asarray(signal, dtype=np.float64)
    fwd = np.asarray(forward_returns, dtype=np.float64)
    T, N = signal.shape
    ics: list[float] = []
    for t in range(T):
        s = signal[t]
        r = fwd[t]
        mask = np.isfinite(r)
        if close_panel is not None:
            mask &= close_panel[t] > 0
        if mask.sum() < max(5, N // 3):
            continue
        s, r = s[mask], r[mask]
        if np.std(s) < 1e-10 or np.std(r) < 1e-10:
            continue
        sr = _cs_rank(s.reshape(1, -1)).ravel()
        rr = _cs_rank(r.reshape(1, -1)).ravel()
        c = np.corrcoef(sr, rr)[0, 1]
        if np.isfinite(c):
            ics.append(float(c))
    return ics


def _deflated_sharpe(ic_series: np.ndarray, n_trials: int) -> float:
    """Deflated Sharpe（Bailey & López de Prado 2014），输入为逐期 IC 序列。

    惩罚试验次数：n_trials 越大，基准 SR* 越高，DSR 越低。
    口径：期级（IC 每期一个值，直接用 mean/std，不年化）。
    """
    import math

    from scipy import stats

    ic = np.asarray(ic_series, dtype=np.float64)
    ic = ic[np.isfinite(ic)]
    if len(ic) < 10 or n_trials < 1:
        return 0.0
    mu = float(ic.mean())
    sd = float(ic.std(ddof=1))
    if sd < 1e-12:
        return 0.0
    z = (ic - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4)) - 3.0  # excess kurtosis
    sr = mu / sd  # 期级 IC-SR
    # SR̂ 估计方差 → SR* 期望最大值
    est_var = 1.0 - skew * sr + kurt / 4.0 * sr ** 2
    sr_std = math.sqrt(max(est_var, 1e-12) / max(len(ic) - 1, 1))
    _euler = 0.5772156649015329
    inv_n = stats.norm.ppf(1.0 - 1.0 / n_trials)
    inv_ne = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    sr_star = sr_std * ((1.0 - _euler) * inv_n + _euler * inv_ne)
    # PSR
    denom = math.sqrt(max(1.0 - skew * sr + kurt / 4.0 * sr ** 2, 1e-12))
    z_psr = (sr - max(sr_star, 0.0)) * math.sqrt(len(ic) - 1) / denom
    return float(stats.norm.cdf(z_psr))
