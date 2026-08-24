"""
3号交易员 — 策略进化工厂 (GP 遗传编程挖因子)

用于 marvis 自动执行的策略进化系统。
基于 alphagen 思想，用遗传编程(GP)自动生成公式化因子表达式，
用 IC/ICIR/分组收益做适应度筛选，进化出有用的交易策略。

核心设计:
  1. 表达式树（Expression Tree）表示因子
  2. 种群进化：选择 → 交叉 → 变异 → 评估
  3. 适应度 = 加权 ICIR + 带符号 IC + 单调性 + 多空收益（负 IC 因子整体低分）
  4. 精英保留 + 复杂度惩罚

CPU 友好：纯 numpy 实现，串行评估。
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# ═══════════════════════════════════════════
# 表达式树
# ═══════════════════════════════════════════

# 操作符集合: (名称, 元数, 函数)
OPS: dict[str, tuple[int, Callable]] = {
    "add": (2, lambda a, b: a + b),
    "sub": (2, lambda a, b: a - b),
    "mul": (2, lambda a, b: a * b),
    "div": (2, lambda a, b: np.divide(a.astype(np.float64), b.astype(np.float64),
                                      out=np.zeros_like(a, dtype=np.float64),
                                      where=np.abs(b) > 1e-10)),
    "neg": (1, lambda a: -a),
    "abs": (1, lambda a: np.abs(a)),
    "sqrt": (1, lambda a: np.sqrt(np.abs(a))),
    "log": (1, lambda a: np.sign(a) * np.log1p(np.abs(a))),
    "rank": (1, lambda a: _cs_rank(a)),
    "zscore": (1, lambda a: _cs_zscore(a)),
    "delay": (2, lambda a, d: _delay(a, int(d))),
    "ts_mean": (2, lambda a, w: _ts_mean(a, int(w))),
    "ts_std": (2, lambda a, w: _ts_std(a, int(w))),
    "ts_max": (2, lambda a, w: _ts_max(a, int(w))),
    "ts_min": (2, lambda a, w: _ts_min(a, int(w))),
    "ts_corr": (3, lambda a, b, w: _ts_corr(a, b, int(w))),
}

# 叶子节点: 价格/成交量字段
FIELDS = ["open", "high", "low", "close", "volume", "vwap", "amount"]
# 常数叶子
CONSTANTS = [1.0, -1.0, 0.5, 2.0, 5.0, 10.0, 20.0]


@dataclass
class Node:
    """表达式树节点"""
    op: str                    # 操作符或字段名或常数
    children: list[Node] = field(default_factory=list)
    value: float | None = None   # 常数叶子的值

    def to_str(self) -> str:
        if self.op in FIELDS:
            return self.op
        if self.op == "const":
            return f"{self.value:g}"
        if self.op in ("ts_mean", "ts_std", "ts_max", "ts_min", "delay"):
            param = _param_value(self.children[1])
            return f"{self.op}({self.children[0].to_str()}, {param})"
        if self.op == "ts_corr":
            param = _param_value(self.children[2])
            return f"ts_corr({self.children[0].to_str()}, {self.children[1].to_str()}, {param})"
        args = ", ".join(c.to_str() for c in self.children)
        return f"{self.op}({args})"


def _param_value(node: Node) -> int:
    """提取窗口/延迟参数（const 值或退化求值）"""
    if node.op == "const":
        return int(node.value)
    return 5  # 兜底默认窗口


def _cs_rank(a: np.ndarray) -> np.ndarray:
    """横截面 rank（沿最后轴）"""
    order = a.argsort(axis=-1)
    ranks = order.argsort(axis=-1)
    return ranks / (a.shape[-1] - 1 if a.shape[-1] > 1 else 1)


def _cs_zscore(a: np.ndarray) -> np.ndarray:
    """横截面 zscore"""
    with np.errstate(all="ignore"):
        mean = np.nanmean(a, axis=-1, keepdims=True)
        std = np.nanstd(a, axis=-1, keepdims=True)
        return np.where(std > 1e-10, (a - mean) / (std + 1e-10), 0.0)


def _delay(a: np.ndarray, d: int) -> np.ndarray:
    """时序 shift（d 期前）。负 d 会引用未来数据，构成未来函数，禁止。"""
    if d < 0:
        raise ValueError("delay 仅允许非负整数（负延迟=未来函数）")
    if d == 0:
        return a
    out = np.full_like(a, np.nan)
    out[d:, :] = a[:-d, :]
    return out


def _ts_mean(a: np.ndarray, w: int) -> np.ndarray:
    return _rolling(a, w, lambda x: np.nanmean(x))


def _ts_std(a: np.ndarray, w: int) -> np.ndarray:
    return _rolling(a, w, lambda x: np.nanstd(x))


def _ts_max(a: np.ndarray, w: int) -> np.ndarray:
    return _rolling(a, w, lambda x: np.nanmax(x))


def _ts_min(a: np.ndarray, w: int) -> np.ndarray:
    return _rolling(a, w, lambda x: np.nanmin(x))


def _rolling(a: np.ndarray, w: int, func: Callable) -> np.ndarray:
    """沿时间轴(axis=0)滚动窗口"""
    w = max(w, 2)
    out = np.full_like(a, np.nan)
    T = a.shape[0]
    for t in range(w - 1, T):
        out[t, :] = func(a[t - w + 1:t + 1, :])
    return out


def _ts_corr(a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    """两序列滚动相关系数"""
    w = max(w, 2)
    out = np.full_like(a, np.nan)
    T = a.shape[0]
    for t in range(w - 1, T):
        x = a[t - w + 1:t + 1, :]
        y = b[t - w + 1:t + 1, :]
        xm = x - np.nanmean(x, axis=0, keepdims=True)
        ym = y - np.nanmean(y, axis=0, keepdims=True)
        num = np.nansum(xm * ym, axis=0)
        den = np.sqrt(np.nansum(xm ** 2, axis=0) * np.nansum(ym ** 2, axis=0))
        out[t, :] = np.where(den > 1e-10, num / (den + 1e-10), 0.0)
    return out


# ═══════════════════════════════════════════
# 表达式求值
# ═══════════════════════════════════════════

def evaluate(node: Node, data: dict[str, np.ndarray]) -> np.ndarray:
    """递归求值。data: {field: (T, N) 数组}"""
    op = node.op
    if op in FIELDS:
        arr = data.get(op, np.zeros(data.get("close", np.zeros((1, 1))).shape))
        return np.asarray(arr, dtype=np.float64)
    if op == "const":
        shape = data.get("close", np.zeros((1, 1))).shape
        return np.full(shape, float(node.value), dtype=np.float64)
    if op in OPS:
        # 特殊处理：时序操作符的窗口/延迟参数是标量
        if op in ("delay", "ts_mean", "ts_std", "ts_max", "ts_min"):
            arr = evaluate(node.children[0], data)
            param = _extract_scalar(node.children[1], data)
            return OPS[op][1](arr, param)
        if op == "ts_corr":
            a = evaluate(node.children[0], data)
            b = evaluate(node.children[1], data)
            w = _extract_scalar(node.children[2], data)
            return OPS[op][1](a, b, w)
        vals = [np.asarray(evaluate(c, data), dtype=np.float64) for c in node.children]
        return OPS[op][1](*vals)
    raise ValueError(f"未知操作符: {op}")


def _extract_scalar(node: Node, data: dict[str, np.ndarray]) -> float:
    """提取标量参数（const 节点或常数表达式）"""
    if node.op == "const":
        return float(node.value)
    # 如果是纯常数表达式（如 add(const(1), const(2))），求值后取标量
    arr = evaluate(node, data)
    if hasattr(arr, "item"):
        return float(np.nanmean(arr))
    return float(arr)


# ═══════════════════════════════════════════
# 随机生成表达式（ramped half-and-half）
# ═══════════════════════════════════════════

def random_node(max_depth: int = 4, rng: random.Random | None = None) -> Node:
    rng = rng or random
    return _random_node(max_depth, rng, 0)


def _random_node(max_depth: int, rng: random.Random, depth: int) -> Node:
    # 叶子
    if depth >= max_depth or rng.random() < 0.3:
        if rng.random() < 0.15:
            return Node(op="const", value=rng.choice(CONSTANTS))
        return Node(op=rng.choice(FIELDS))
    # 内部节点
    op = rng.choice(list(OPS.keys()))
    n_args = OPS[op][0]
    children = [_random_node(max_depth, rng, depth + 1) for _ in range(n_args)]
    # 处理常数参数（delay/ts_* 的第二三参数）
    node = Node(op=op, children=children)
    _fix_constant_args(node, rng)
    return node


def _fix_constant_args(node: Node, rng: random.Random) -> None:
    """确保时序操作符的参数是常数"""
    window_ops = {"ts_mean", "ts_std", "ts_max", "ts_min"}
    if node.op in window_ops and len(node.children) == 2:
        node.children[1] = Node(op="const", value=rng.choice([3, 5, 10, 20, 30, 60]))
    if node.op == "delay" and len(node.children) == 2:
        node.children[1] = Node(op="const", value=rng.choice([1, 2, 3, 5, 10]))
    if node.op == "ts_corr" and len(node.children) == 3:
        node.children[2] = Node(op="const", value=rng.choice([5, 10, 20, 30]))


# ═══════════════════════════════════════════
# 遗传操作
# ═══════════════════════════════════════════

def crossover(parent1: Node, parent2: Node, rng: random.Random | None = None) -> Node:
    """子树交叉：交换两棵树的随机子树"""
    rng = rng or random
    node1 = _random_subtree(parent1, rng)
    node2 = _random_subtree(parent2, rng)
    # 用 node2 替换 node1 的位置 —— 复制实现
    return _replace_subtree(parent1, node1, node2, rng)


def _random_subtree(node: Node, rng: random.Random) -> Node:
    """随机选一个子树（含自身）"""
    nodes = _collect_nodes(node)
    return rng.choice(nodes)


def _collect_nodes(node: Node) -> list[Node]:
    nodes = [node]
    for c in node.children:
        nodes.extend(_collect_nodes(c))
    return nodes


def _replace_subtree(root: Node, target: Node, replacement: Node, rng: random.Random) -> Node:
    """复制 root，把 target 替换成 replacement 的副本"""
    if root is target:
        return _clone(replacement)
    new = Node(op=root.op, children=[_replace_subtree(c, target, replacement, rng) for c in root.children], value=root.value)
    return new


def mutate(node: Node, max_depth: int = 4, rng: random.Random | None = None) -> Node:
    """点变异：随机替换一个子树"""
    rng = rng or random
    target = _random_subtree(node, rng)
    new_sub = _random_node(max_depth, rng, 0)
    return _replace_subtree(node, target, new_sub, rng)


def _clone(node: Node) -> Node:
    return Node(op=node.op, children=[_clone(c) for c in node.children], value=node.value)


def normalize(node: Node) -> Node:
    """
    修复表达式树的不变量：
    时序操作符(delay/ts_*)的窗口/延迟参数必须是 const 节点。
    在交叉/变异后调用，保证表达式可求值、可序列化。
    """
    if node.op in ("ts_mean", "ts_std", "ts_max", "ts_min", "delay"):
        node.children[0] = normalize(node.children[0])
        param_node = node.children[1] if len(node.children) > 1 else Node(op="const", value=5)
        if param_node.op != "const":
            # 非 const 参数 → 退化求值取整数；失败则默认 5
            val = _try_eval_const(param_node)
            node.children[1] = Node(op="const", value=val)
        else:
            node.children[1] = Node(op="const", value=_coerce_window(param_node.value))
        return node
    if node.op == "ts_corr":
        node.children[0] = normalize(node.children[0])
        node.children[1] = normalize(node.children[1])
        param_node = node.children[2] if len(node.children) > 2 else Node(op="const", value=10)
        if param_node.op != "const":
            node.children[2] = Node(op="const", value=10)
        else:
            node.children[2] = Node(op="const", value=_coerce_window(param_node.value))
        return node
    node.children = [normalize(c) for c in node.children]
    return node


def _coerce_window(v) -> int:
    """窗口值合理范围钳制"""
    try:
        v = int(v)
        return min(max(v, 2), 120)
    except Exception:
        return 5


def _try_eval_const(node: Node) -> int:
    """尝试求一个纯常数表达式的值（无 data 依赖）"""
    try:
        if node.op == "const":
            return _coerce_window(node.value)
        # 递归：只有 FIELDS 之外的才算纯常数
        if node.op in FIELDS:
            return 5
        vals = [_try_eval_const(c) for c in node.children]
        # 简单二元
        if node.op == "add" and len(vals) >= 2:
            return _coerce_window(vals[0] + vals[1])
        if node.op == "sub" and len(vals) >= 2:
            return _coerce_window(vals[0] - vals[1])
        if node.op == "mul" and len(vals) >= 2:
            return _coerce_window(vals[0] * vals[1])
        return 5
    except Exception:
        return 5


# ═══════════════════════════════════════════
# 适应度评估
# ═══════════════════════════════════════════

def compute_fitness(
    expr: str,
    panel: dict[str, np.ndarray],
    forward_returns: np.ndarray,
    n_top: int = 5,
) -> dict[str, float]:
    """
    计算因子适应度。

    Parameters
    ----------
    expr : str — 表达式字符串（或 Node，或直接给 (T,N) 信号矩阵）
    panel : {field: (T,N)} 行情面板
    forward_returns : (T,N) 前瞻收益（T 期收益）
    n_top : 选取多头/空头分组的组数

    Returns
    -------
    {ic, icir, monotonicity, long_short, fitness}
    """
    from .parser import parse_expr

    if isinstance(expr, np.ndarray):
        signal = np.nan_to_num(expr, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        node = parse_expr(expr) if isinstance(expr, str) else expr
        signal = evaluate(node, panel)
        # 清理 NaN/Inf（0 = 中性秩）
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    # 有效性掩码：前向收益有限 且 当日价格真实（剔除停牌僵尸样本）
    close_panel = panel.get("close") if isinstance(panel, dict) else None
    T, N = signal.shape
    ic_list = []
    for t in range(T):
        s = signal[t]
        r = forward_returns[t]
        mask = np.isfinite(r)
        if close_panel is not None:
            mask &= close_panel[t] > 0
        if mask.sum() < max(5, N // 3):
            continue
        s = s[mask]
        r = r[mask]
        if np.std(s) < 1e-10 or np.std(r) < 1e-10:
            continue
        # rank transform
        s_rank = _cs_rank(s.reshape(1, -1)).ravel()
        r_rank = _cs_rank(r.reshape(1, -1)).ravel()
        ic_list.append(np.corrcoef(s_rank, r_rank)[0, 1])

    if len(ic_list) < 20:
        return {"ic": 0.0, "icir": 0.0, "monotonicity": 0.0, "long_short": 0.0, "fitness": -999.0}

    ic_arr = np.array(ic_list)
    ic = float(np.nanmean(ic_arr))
    icir = float(ic / (np.nanstd(ic_arr) + 1e-10)) if np.nanstd(ic_arr) > 1e-10 else 0.0

    # 分组收益（5分位）
    group_rets = _group_returns(signal, forward_returns)
    monotonicity = _monotonicity(group_rets)

    # 多空收益（top5% - bottom5%）
    long_short = float(group_rets[-1] - group_rets[0]) if len(group_rets) >= 2 else 0.0

    # 综合适应度：ICIR 为主，奖励单调性和多空收益，惩罚低 IC
    # 符号一致性：直接用带符号 IC（负 IC 因子整体低分，与门禁方向一致）
    fitness = icir * 0.5 + ic * 2.0 + monotonicity * 0.3 + long_short * 0.2
    if abs(ic) < 0.01:
        fitness -= 0.5  # IC 太弱重罚

    # alphagen 模式④：parsimony 惩罚（>20 token 直接 -1，进化期即压长树）
    if isinstance(expr, np.ndarray):
        expr_str = "<signal>"
    else:
        node_ = parse_expr(expr) if isinstance(expr, str) else expr
        expr_str = node_.to_str() if hasattr(node_, "to_str") else str(expr)
    n_ops = expr_str.count(",") + expr_str.count("(")
    if n_ops > 20:
        fitness -= 1.0
    elif n_ops > 12:
        fitness -= 0.3  # 中度复杂轻微惩罚

    return {
        "ic": round(ic, 4),
        "icir": round(icir, 4),
        "monotonicity": round(monotonicity, 4),
        "long_short": round(long_short, 4),
        "fitness": round(fitness, 4),
    }


def _group_returns(signal: np.ndarray, forward_returns: np.ndarray, n_groups: int = 5) -> np.ndarray:
    """按信号值分 n_groups 组，返回各组**年化**平均收益"""
    T, N = signal.shape
    group_rets = np.zeros(n_groups)
    counts = np.zeros(n_groups)
    for t in range(T):
        s = signal[t]
        r = forward_returns[t]
        valid = np.isfinite(s) & np.isfinite(r)
        if valid.sum() < n_groups:
            continue
        # 分位数分组
        order = np.argsort(np.argsort(s[valid]))
        frac = order / max(valid.sum() - 1, 1)
        group_idx = np.clip((frac * n_groups).astype(int), 0, n_groups - 1)
        for g in range(n_groups):
            mask = group_idx == g
            if mask.sum() > 0:
                group_rets[g] += np.nanmean(r[valid][mask])
                counts[g] += 1
    for g in range(n_groups):
        if counts[g] > 0:
            group_rets[g] /= counts[g]
    # 年化：日平均收益 × 252
    return group_rets * 252.0


def _monotonicity(group_rets: np.ndarray) -> float:
    """分组收益单调性（Q1<Q2<...<Qn 的单调对比例）"""
    n = len(group_rets)
    if n < 2:
        return 0.0
    monotone = 0
    total = 0
    for i in range(n - 1):
        total += 1
        if group_rets[i] < group_rets[i + 1]:
            monotone += 1
    return monotone / total
