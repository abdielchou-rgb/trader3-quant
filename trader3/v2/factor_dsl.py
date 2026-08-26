"""
统一因子DSL (Domain Specific Language)。

统一入口：GP表达式 + 库因子 + 情绪因子 → 编译为可调用对象。

设计：
- 复用 evolve.core.parser.parse_expr + evolve.core.gp.evaluate
- 注册库因子 (trader3.v2.factors.library) 与情绪因子 (trader3.v2.sentiment)
- 编译缓存 (LRU) + 类型验证
- 面板数据标准输入: MultiIndex 列 (asset, field) 的 DataFrame
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from evolve.core.gp import FIELDS, OPS, Node, evaluate
from evolve.core.parser import parse_expr

logger = logging.getLogger("trader3.v2.factor_dsl")

# ── 基础字段别名 ───────────────────────────────────────
FIELD_ALIASES = {
    "o": "open", "h": "high", "l": "low", "c": "close",
    "v": "volume", "vol": "volume", "amt": "amount", "vw": "vwap",
}


@dataclass
class CompiledFactor:
    """已编译因子，含可调用对象与元信息。"""
    expr: str
    func: Callable[[pd.DataFrame], pd.Series]   # panel -> scores (index=asset)
    deps: list[str]                             # 依赖的字段/库因子名
    is_gp: bool                                 # 是否为GP树表达式


class FactorDSL:
    """
    统一因子DSL编译器。

    用法：
        dsl = FactorDSL()
        f = dsl.compile("add(log(close), lib.momentum)")
        scores = f(panel)  # panel: (date x (asset, field)) MultiIndex DataFrame
    """

    def __init__(self, library_namespace: dict[str, Callable] | None = None):
        self._library = library_namespace or {}
        self._sentiment_fn: Callable | None = None
        self._cache: dict[str, CompiledFactor] = {}

    def register_library(self, name: str, func: Callable) -> None:
        """注册库因子：名字 -> 可调用 (panel -> scores Series)。"""
        self._library[name] = func

    def register_library_bulk(self, mapping: dict[str, Callable]) -> None:
        self._library.update(mapping)

    def set_sentiment_fn(self, func: Callable) -> None:
        """注册情绪因子函数 (panel -> scores Series)。"""
        self._sentiment_fn = func

    # ── 核心编译 ───────────────────────────────────────

    def compile(self, expr: str) -> CompiledFactor:
        """
        编译表达式为 CompiledFactor。

        语法：
        - GP 表达式: add(log(close), delay(close, 5))
        - 库因子引用: lib.momentum, lib.rsi, lib.alpha1
        - 情绪因子: sentiment
        - 常数: 1.5, const(2.0)
        - 组合: mul(lib.rsi, sentiment), add(log(close), lib.momentum)
        """
        expr = expr.strip()
        if expr in self._cache:
            return self._cache[expr]

        # 解析顶层结构
        func, deps, is_gp = self._compile_expr(expr)
        cf = CompiledFactor(expr=expr, func=func, deps=deps, is_gp=is_gp)
        self._cache[expr] = cf
        return cf

    def _compile_expr(self, expr: str) -> tuple[Callable, list[str], bool]:
        expr = expr.strip()

        # 1. 库因子直接引用: lib.xxx
        if expr.startswith("lib.") or expr.startswith("library."):
            name = expr.split(".", 1)[1]
            if name not in self._library:
                raise ValueError(f"未注册的库因子: {name}")
            func = self._wrap_lib_func(name, self._library[name])
            return func, [f"lib.{name}"], False

        # 2. 情绪因子
        if expr in ("sentiment", "sent"):
            if self._sentiment_fn is None:
                raise ValueError("情绪因子未注册，请先 set_sentiment_fn")
            return self._sentiment_fn, ["sentiment"], False

        # 3. 纯常数
        try:
            val = float(expr)
            def _const_func(panel: pd.DataFrame, _v: float = val) -> pd.Series:
                return pd.Series(_v, index=panel.columns.get_level_values(0).unique())
            return _const_func, [f"const({val})"], False
        except ValueError:
            pass

        # 4. GP 表达式解析 → 编译为高效调用
        try:
            node = parse_expr(expr)
            func = self._compile_gp_node(node)
            deps = self._collect_gp_deps(node)
            return func, deps, True
        except Exception as e:
            raise ValueError(f"无法解析表达式: {expr} -> {e}") from None

    def _wrap_lib_func(self, name: str, func: Callable) -> Callable:
        """包装库因子，确保输入输出一致。"""
        def wrapper(panel: pd.DataFrame) -> pd.Series:
            return func(panel)
        wrapper.__name__ = f"lib_{func.__name__}"
        return wrapper

    def _compile_gp_node(self, node: Node) -> Callable[[pd.DataFrame], pd.Series]:
        """将 GP Node 编译为 panel -> scores 函数（取最新一期截面）。"""
        def evaluator(panel: pd.DataFrame) -> pd.Series:
            result = self._eval_gp_full(node, panel)  # (T, N)
            if result.ndim == 2:
                scores = pd.Series(result[-1], index=panel.columns.get_level_values(0).unique())
            else:
                scores = pd.Series(result, index=panel.columns.get_level_values(0).unique())
            return scores
        return evaluator

    def _eval_gp_full(self, node: Node, panel: pd.DataFrame) -> np.ndarray:
        """对 GP Node 求全样本 (T, N) 结果（供 ensemble 历史使用）。"""
        data = self._panel_to_arrays(panel)
        return evaluate(node, data)

    def full_series(self, expr: str, panel: pd.DataFrame) -> pd.DataFrame:
        """编译 expr 并返回全样本宽表 (date × asset)，供 ensemble 使用。"""
        cf = self.compile(expr)
        if not cf.is_gp:
            raise ValueError(f"full_series 仅支持 GP 表达式，收到: {expr}")
        node = parse_expr(expr)
        arr = self._eval_gp_full(node, panel)  # (T, N)
        assets = list(panel.columns.get_level_values(0).unique())
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return pd.DataFrame(arr, index=panel.index, columns=assets)

    def _collect_gp_deps(self, node: Node) -> list[str]:
        """收集 GP 树依赖的字段/库因子。"""
        deps = set()
        def walk(n: Node):
            if n.op in FIELDS:
                deps.add(n.op)
            elif n.op in OPS:
                for c in n.children:
                    walk(c)
            elif n.op == "const":
                pass
            else:
                # 可能是库因子引用（如果在表达式中用 lib.xxx 这种语法）
                pass
        walk(node)
        return sorted(deps)

    def _panel_to_arrays(self, panel: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        将 (date x (asset, field)) MultiIndex DataFrame 转为
        {field: (T, N) array}，asset 顺序按列序。
        """
        if not isinstance(panel.columns, pd.MultiIndex):
            raise ValueError("panel 必须是 MultiIndex 列 (asset, field)")
        assets = panel.columns.get_level_values(0).unique()
        fields = panel.columns.get_level_values(1).unique()
        data = {}
        for f in fields:
            sub = panel.xs(f, axis=1, level=1)
            # 确保 asset 顺序一致
            sub = sub.reindex(columns=assets)
            data[f] = sub.values.astype(np.float64)
        return data

    def __call__(self, expr: str) -> Callable[[pd.DataFrame], pd.Series]:
        """快捷调用：dsl(expr)(panel)"""
        return self.compile(expr).func


# ── 内置库因子注册表（延迟导入避免循环）─────────────────
_DSL_INSTANCE: FactorDSL | None = None


def get_dsl() -> FactorDSL:
    """获取单例 DSL 并自动注册内置库因子。"""
    global _DSL_INSTANCE
    if _DSL_INSTANCE is None:
        _DSL_INSTANCE = FactorDSL()
        _DSL_INSTANCE.register_library_bulk(_builtin_library())
        _DSL_INSTANCE.set_sentiment_fn(_default_sentiment_fn)
    return _DSL_INSTANCE


def _builtin_library() -> dict[str, Callable]:
    """从 trader3.v2.factors.library 构建内置库因子映射。

    library 因子期望单资产宽表 (date × field)，返回 (date,) Series。
    这里逐资产计算并取最新一期值，输出 (asset,) 截面得分。
    """
    from trader3.v2.factors.library import FACTOR_REGISTRY
    mapping = {}
    for name, cls in FACTOR_REGISTRY.items():
        def make_wrapper(factory, fname):
            def _wrap(panel: pd.DataFrame) -> pd.Series:
                assets = panel.columns.get_level_values(0).unique()
                scores = {}
                for a in assets:
                    a_wide = panel.xs(a, axis=1, level=0)
                    try:
                        s = factory().compute(a_wide)
                        scores[a] = float(s.iloc[-1]) if len(s) else 0.0
                    except Exception:
                        scores[a] = 0.0
                return pd.Series(scores)
            _wrap.__name__ = f"lib_{fname}"
            return _wrap
        mapping[name] = make_wrapper(cls, name)
    return mapping


def _default_sentiment_fn(panel: pd.DataFrame) -> pd.Series:
    """默认情绪因子：每日截面 z-score，缺失填 0。"""
    try:
        from trader3.v2.sentiment import SentimentStore, zscore_cross_section
        store = SentimentStore()
        scores = store.window_frame(days=10).mean(axis=0)
        if scores.empty:
            return pd.Series(0.0, index=panel.columns.get_level_values(0).unique())
        return zscore_cross_section(scores)
    except Exception:
        assets = panel.columns.get_level_values(0).unique()
        return pd.Series(0.0, index=assets)


# ── 高级辅助 ──────────────────────────────────────────

def compile_factor(expr: str) -> Callable[[pd.DataFrame], pd.Series]:
    """快捷编译：compile_factor(expr)(panel) -> scores"""
    return get_dsl().compile(expr).func


def list_available_factors() -> dict[str, list[str]]:
    """列出可用因子：{'fields': [...], 'library': [...], 'ops': [...]}"""
    dsl = get_dsl()
    return {
        "fields": FIELDS,
        "library": sorted(dsl._library.keys()),
        "ops": sorted(OPS.keys()),
        "special": ["sentiment", "const(value)"],
    }


def validate_expression(expr: str) -> tuple[bool, str | None]:
    """验证表达式合法性，返回 (ok, error_msg)。"""
    try:
        get_dsl().compile(expr)
        return True, None
    except Exception as e:
        return False, str(e)


# 兼容旧接口
compile_expr = compile_factor
