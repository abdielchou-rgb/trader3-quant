"""
3号交易员 — 策略筛选门禁

从进化产生的候选因子中，用多道门槛筛选出"真正有用"的策略。
防止过拟合、低 IC、非单调、拥挤度高的因子混入最终策略库。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SelectionResult:
    """单个候选的筛选结果"""
    expr: str
    passed: bool
    score: float = 0.0
    gates: Dict[str, dict] = field(default_factory=dict)  # {gate_name: {passed, value}}
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

    综合评分 = 0.4*ICIR + 0.3*单调性 + 0.3*多空年化
    """

    def __init__(
        self,
        ic_min: float = 0.02,
        icir_min: float = 0.15,
        monotonicity_min: float = 0.4,
        long_short_min: float = 0.10,
        max_nodes: int = 25,
    ):
        self.ic_min = ic_min
        self.icir_min = icir_min
        self.monotonicity_min = monotonicity_min
        self.long_short_min = long_short_min
        self.max_nodes = max_nodes
        self._selected_exprs: List[str] = []
        self._selected_values: List[np.ndarray] = []

    def select(self, candidates: List[dict]) -> List[SelectionResult]:
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
        results: List[Optional[SelectionResult]] = [None] * len(candidates)
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

    def select_best(self, candidates: List[dict], top_k: int = 5) -> List[SelectionResult]:
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


def _tokenize_expr(expr: str) -> List[str]:
    import re
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+", expr)