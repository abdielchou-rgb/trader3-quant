#!/usr/bin/env python3
"""
run_evolution.py — 策略进化工厂主入口（marvis 调用）

用法:
    # 用 qlib 股票数据进化（无网）
    python evolve/run_evolution.py --source qlib --universe csi300 --gen 15 --pop 50

    # 用 ETF 数据进化（先 fetch_etf_data.py 拉数据）
    python evolve/run_evolution.py --source etf --etf-dir evolve/data/etf --gen 15 --pop 50

    # 只评估一个候选（调试）
    python evolve/run_evolution.py --eval "ts_mean(close,20)"

输出:
    evolve/evolution_log/          每代最优
    evolve/strategies/selected.json 最终筛选出的策略
    evolve/strategies/evolution_summary.json 进化摘要
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "evolve"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evolve")

from core.data_loader import load_etf_panel, load_qlib_panel  # noqa: E402
from core.evolution import EvolutionEngine  # noqa: E402
from core.gp import compute_fitness  # noqa: E402
from core.parser import parse_expr  # noqa: E402
from core.selection import StrategySelector  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="GP 策略进化工厂")
    parser.add_argument("--source", choices=["qlib", "etf"], default="qlib")
    parser.add_argument("--universe", default="csi300")
    parser.add_argument("--etf-dir", default="evolve/data/etf")
    parser.add_argument("--n-stocks", type=int, default=60)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2023-12-31",
                        help="训练区间末（验证必须晚于该日期，时间上不相交）")
    parser.add_argument("--gen", type=int, default=15, help="进化代数")
    parser.add_argument("--pop", type=int, default=50, help="种群大小")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval", default="", help="只评估单个表达式（调试）")
    parser.add_argument("--top-k", type=int, default=5, help="最终筛选 Top-K")
    args = parser.parse_args()

    # ── 加载数据 ──
    logger.info(f"加载数据: source={args.source}, {args.start}~{args.end}")
    if args.source == "qlib":
        panel, fwd = load_qlib_panel(
            universe=args.universe,
            n_stocks=args.n_stocks,
            start=args.start,
            end=args.end,
        )
        data_label = f"qlib/{args.universe}"
    else:
        panel, fwd = load_etf_panel(
            csv_dir=args.etf_dir,
            n_etfs=args.n_stocks,
            start=args.start,
            end=args.end,
        )
        data_label = f"etf/{args.etf_dir}"

    T, N = fwd.shape
    logger.info(f"面板: {T} 交易日 × {N} 标的")
    fields_available = {f: panel[f].shape for f in panel}

    # ── 单表达式评估（调试） ──
    if args.eval:
        from core.gp import normalize
        node = normalize(parse_expr(args.eval))
        fit = compute_fitness(node, panel, fwd)
        print(json.dumps({"expr": args.eval, **fit}, ensure_ascii=False, indent=2))
        return

    # ── 进化 ──
    logger.info(f"开始进化: {args.gen} 代 × {args.pop} 种群")
    engine = EvolutionEngine(
        population_size=args.pop,
        generations=args.gen,
        seed=args.seed,
        log_dir=str(_ROOT / "evolve" / "evolution_log"),
    )

    start_t = time.time()
    best_node, best_fit = engine.evolve(panel, fwd)
    elapsed = time.time() - start_t
    logger.info(f"进化完成: 耗时 {elapsed:.1f}s")
    logger.info(f"全局最优: {best_node.to_str()} fitness={best_fit['fitness']}")

    # ── 收集所有候选（从历史最优 + 最终种群），携带因子值用于相关性去重 ──
    from core.gp import evaluate as _eval_node

    def _signal_of(expr_or_node):
        node = parse_expr(expr_or_node) if isinstance(expr_or_node, str) else expr_or_node
        return _eval_node(node, panel)

    candidates = []
    # 历史每代最优
    for h in engine.best_history:
        if h.get("expr"):
            candidates.append({
                "expr": h["expr"],
                "ic": h.get("ic", 0),
                "icir": h.get("icir", 0),
                "monotonicity": h.get("monotonicity", 0),
                "long_short": h.get("long_short", 0),
                "fitness": h.get("fitness", 0),
                "generation": h.get("generation", 0),
                "source": data_label,
                "values": _signal_of(h["expr"]),
            })
    # 最终种群
    for node in engine.population:
        f = compute_fitness(node, panel, fwd)
        candidates.append({
            "expr": node.to_str(),
            "ic": f["ic"],
            "icir": f["icir"],
            "monotonicity": f["monotonicity"],
            "long_short": f["long_short"],
            "fitness": f["fitness"],
            "generation": args.gen,
            "source": data_label,
            "values": _eval_node(node, panel),
        })

    # ── 筛选 ──
    selector = StrategySelector()
    selected = selector.select_best(candidates, top_k=args.top_k)

    strategies_dir = _ROOT / "evolve" / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)

    selected_json = [
        {
            "expr": r.expr,
            "score": r.score,
            "gates": r.gates,
            "source": data_label,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for r in selected
    ]
    with open(strategies_dir / "selected.json", "w", encoding="utf-8") as f:
        json.dump(selected_json, f, ensure_ascii=False, indent=2)

    summary = {
        "source": data_label,
        "date_range": [args.start, args.end],
        "panel_shape": [T, N],
        "population_size": args.pop,
        "generations": args.gen,
        "elapsed_seconds": round(elapsed, 1),
        "candidates_considered": len(candidates),
        "selected_count": len(selected),
        "best_fitness": best_fit.get("fitness", 0),
        "selected": selected_json,
        "fitness_history": engine.fitness_history,
    }
    with open(strategies_dir / "evolution_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── 控制台输出 ──
    print("\n" + "=" * 70)
    print(f"📊 进化完成: {data_label} | {args.gen}代×{args.pop} | {elapsed:.0f}s")
    print(f"评估候选: {len(candidates)} | 筛选通过: {len(selected)}")
    print("=" * 70)
    for i, r in enumerate(selected):
        print(f"\n[{i+1}] 策略: {r.expr}")
        print(f"    评分: {r.score}")
        for g, info in r.gates.items():
            mark = "✅" if info["passed"] else "❌"
            print(f"    {mark} {g}: {info.get('value', '')}")
    print("\n" + "=" * 70)
    print(f"结果保存: {strategies_dir}/selected.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
