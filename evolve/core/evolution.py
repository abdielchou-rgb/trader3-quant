"""
3号交易员 — 遗传编程进化引擎

种群进化主循环：选择 → 交叉 → 变异 → 评估 → 精英保留。
支持多进程并行评估（CPU 笔记本友好）。
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .gp import Node, compute_fitness, crossover, mutate, normalize, random_node
from .parser import parse_expr

logger = logging.getLogger("evolve")


class EvolutionEngine:
    """GP 进化引擎"""

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 20,
        max_depth: int = 4,
        crossover_rate: float = 0.6,
        mutation_rate: float = 0.3,
        elitism: int = 3,
        seed: int = 42,
        n_workers: int = 1,
        log_dir: str = "",
    ):
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = min(elitism, population_size // 5)
        self.seed = seed
        self.n_workers = n_workers
        self.log_dir = log_dir or str(Path(__file__).resolve().parent.parent / "evolution_log")
        os.makedirs(self.log_dir, exist_ok=True)

        self.rng = random.Random(seed)
        self.population: List[Node] = []
        self.fitness_history: List[float] = []
        self.best_history: List[dict] = []

    # ── 初始化种群 ──

    def init_population(self) -> List[Node]:
        self.population = [random_node(self.max_depth, self.rng) for _ in range(self.population_size)]
        return self.population

    # ── 进化主循环 ──

    def evolve(
        self,
        panel: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
        generations: Optional[int] = None,
        progress_cb: Optional[callable] = None,
    ) -> Tuple[Node, dict]:
        """
        运行进化。

        Returns
        -------
        (best_node, best_fitness_dict)
        """
        gens = generations or self.generations
        if not self.population:
            self.init_population()

        for gen in range(gens):
            start = time.time()
            fitness_map = self._evaluate_population(panel, forward_returns)

            # 记录
            valid = {k: v for k, v in fitness_map.items() if v.get("fitness", -999) > -900}
            if valid:
                best_key = max(valid, key=lambda k: valid[k]["fitness"])
                best_fit = valid[best_key]
                avg_fit = float(np.mean([v["fitness"] for v in valid.values()]))
            else:
                best_key = None
                best_fit = {"fitness": -999.0}
                avg_fit = -999.0

            self.fitness_history.append(avg_fit)
            self.best_history.append({
                "generation": gen + 1,
                "expr": self.population[best_key].to_str() if best_key is not None else "",
                "fitness": best_fit.get("fitness", 0),
                "ic": best_fit.get("ic", 0),
                "icir": best_fit.get("icir", 0),
                "monotonicity": best_fit.get("monotonicity", 0),
                "long_short": best_fit.get("long_short", 0),
                "avg_fitness": round(avg_fit, 4),
                "elapsed": round(time.time() - start, 2),
            })

            logger.info(
                f"[Gen {gen+1}/{gens}] 最优: {self.best_history[-1]['expr'][:60]} | "
                f"fitness={best_fit.get('fitness',0):.3f} icir={best_fit.get('icir',0):.3f} "
                f"| avg={avg_fit:.3f} | {self.best_history[-1]['elapsed']}s"
            )

            if progress_cb:
                progress_cb(gen + 1, gens, self.best_history[-1])

            # 选择下一代
            if gen < gens - 1:
                self._next_generation(fitness_map)

            self._save_state(gen + 1)

        # 返回全局最优（重新评估一次）
        best_idx = 0
        best_fit_val = -999.0
        for i, node in enumerate(self.population):
            f = compute_fitness(node, panel, forward_returns)
            if f.get("fitness", -999) > best_fit_val:
                best_fit_val = f["fitness"]
                best_idx = i
                best_fit = f
        return self.population[best_idx], best_fit

    def _fitness_for(self, node: Node, panel, fwd) -> dict:
        return compute_fitness(node, panel, fwd)

    def _evaluate_population(self, panel, fwd) -> Dict[int, dict]:
        """评估整个种群（串行；panel 较大时进程间复制开销高于计算本身）"""
        if self.n_workers > 1:
            logger.info("当前版本为串行评估，n_workers=%s 被忽略", self.n_workers)
        return {i: compute_fitness(self.population[i], panel, fwd)
                for i in range(len(self.population))}

    def _next_generation(self, fitness_map: Dict[int, dict]) -> None:
        """选择 + 交叉 + 变异 → 下一代种群"""
        # 按 fitness 排序
        ranked = sorted(fitness_map.items(), key=lambda kv: kv[1].get("fitness", -999), reverse=True)
        # 精英保留
        elites = [self.population[i] for i, _ in ranked[:self.elitism]]
        new_pop = [_clone(n) for n in elites]

        # 轮盘赌选择（基于 fitness 加权）
        fit_vals = [f.get("fitness", 0) for _, f in ranked]
        fit_vals = [max(f, 0.001) for f in fit_vals]  # 避免负数
        total = sum(fit_vals)
        weights = [f / total for f in fit_vals]

        while len(new_pop) < self.population_size:
            r = self.rng.random()
            if r < self.crossover_rate and len(new_pop) + 1 < self.population_size:
                p1 = self._select(ranked, weights)
                p2 = self._select(ranked, weights)
                child = crossover(p1, p2, self.rng)
                child = normalize(child)
                new_pop.append(child)
            else:
                p = self._select(ranked, weights)
                if self.rng.random() < self.mutation_rate:
                    child = mutate(p, self.max_depth, self.rng)
                    child = normalize(child)
                else:
                    child = _clone(p)
                new_pop.append(child)

        self.population = new_pop[:self.population_size]

    def _select(self, ranked, weights) -> Node:
        """按权重选择"""
        r = self.rng.random()
        acc = 0.0
        for (i, _), w in zip(ranked, weights):
            acc += w
            if r <= acc:
                return self.population[i]
        return self.population[ranked[-1][0]]

    # ── 状态持久化 ──

    def _save_state(self, generation: int) -> None:
        """保存当前最优到 evolution_log/"""
        if not self.best_history:
            return
        best = self.best_history[-1]
        path = os.path.join(self.log_dir, f"gen_{generation:03d}_best.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(best, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def _clone(node: Node) -> Node:
    from .gp import Node as N
    return N(op=node.op, children=[_clone(c) for c in node.children], value=node.value)