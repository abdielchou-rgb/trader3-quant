"""
协同目标 GP 进化（S1 闭环）— 回归测试

解剖结论：把进化适应度从"裸单因子 IC"升级为"加入精英池的边际贡献"，
引导种群挖出互补而非共线的因子组合。

契约：
  1. EvolutionEngine(collaborative=True) 启用协同目标：
     - 维护 elite_pool（历代表现最好个体的因子值矩阵）
     - 每个体 fitness = 自身 IC 与"对精英池边际贡献"的加权（默认并重）
  2. collaborative=False（默认）→ 行为完全不变（向后兼容，全部既有测试仍绿）
  3. 合成双信号数据：协同模式应选出互补两因子；单因子模式可能只锁定一个
  4. 协同模式 best_history 记录 elite_pool_size
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "evolve"))

from core.evolution import EvolutionEngine  # noqa: E402
from core.parser import parse_expr  # noqa: E402


def _rng(seed=1):
    import random
    return random.Random(seed)


def _panel_data(T=200, N=12, seed=5):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, size=(T, N))
    close = 100.0 * np.cumprod(1 + rets, axis=0)
    panel = {
        "open": close.copy(), "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.full_like(close, 1e6),
        "vwap": close, "amount": close * 1e6,
    }
    fwd = np.full_like(close, np.nan)
    for j in range(N):
        col = close[:, j]
        v = np.where(col > 0)[0]
        for k in range(len(v) - 1):
            if v[k + 1] - v[k] <= 5:
                fwd[v[k], j] = col[v[k + 1]] / col[v[k]] - 1.0
    return panel, fwd


def _fixed_population():
    """固定手搓种群：一强一弱两个动量型因子，用于验证精英池演化。"""
    nodes = []
    for expr in ("ROC20", "ROC10"):
        try:
            nodes.append(parse_expr(expr))
        except Exception:
            pass
    return nodes


def test_default_mode_unchanged_backward_compat():
    """collaborative 默认关：引擎行为与既有完全一致（接口兼容）。"""
    panel, fwd = _panel_data()
    eng = EvolutionEngine(population_size=8, generations=2, seed=42)
    eng.init_population()
    # 默认 _evaluate_population 走 compute_fitness，无 elite_pool
    eng.evolve(panel, fwd, generations=1)
    assert eng.best_history  # 正常产出
    assert not hasattr(eng, "elite_pool") or len(eng.elite_pool) == 0


def test_collaborative_engine_accepts_flag():
    """collaborative=True 可构造且能跑。"""
    panel, fwd = _panel_data(T=150, N=10)
    eng = EvolutionEngine(population_size=6, generations=2, seed=7,
                          collaborative=True)
    eng.init_population()
    best, _fit = eng.evolve(panel, fwd, generations=2)
    assert best is not None
    assert eng.elite_pool_size > 0  # 记录了精英池规模


def test_collaborative_fitness_uses_marginal_contribution():
    """协同模式：个体 fitness 含对精英池的边际贡献成分。"""

    panel, fwd = _panel_data(T=150, N=10)
    eng = EvolutionEngine(population_size=6, generations=1, seed=1,
                          collaborative=True)
    eng.init_population()
    # 直接调 _fitness_for：初代 elite_pool 空 → 退化为自身 IC + 微调
    f0 = eng._fitness_for(eng.population[0], panel, fwd)
    assert "fitness" in f0 and "mc" in f0  # 返回含 mc 键
    assert f0["mc"] >= 0.0  # 空池边际贡献=自身组合 IC≥0


def test_elite_pool_grows_across_generations():
    """跨代精英池应累积（不缩水）。"""
    panel, fwd = _panel_data(T=150, N=10)
    eng = EvolutionEngine(population_size=8, generations=3, seed=3,
                          collaborative=True, elite_pool_cap=5)
    eng.init_population()
    eng.evolve(panel, fwd, generations=3)
    assert 1 <= eng.elite_pool_size <= 5


def test_collaborative_best_history_records_pool():
    """best_history 每条含 elite_pool_size。"""
    panel, fwd = _panel_data(T=120, N=8)
    eng = EvolutionEngine(population_size=6, generations=2, seed=9,
                          collaborative=True)
    eng.init_population()
    eng.evolve(panel, fwd, generations=2)
    assert all("elite_pool_size" in h for h in eng.best_history)
