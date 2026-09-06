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
import subprocess
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
    # GBK 控制台兼容：emoji 汇总行不因编码崩溃（内容损失不影响 JSON 落盘）
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
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
    parser.add_argument("--deep", action="store_true",
                        help="追加 LSTM 深度打分器候选（torch 优先，缺失时 numpy 回退）")
    parser.add_argument("--deep-epochs", type=int, default=300, help="LSTM 训练轮数")
    parser.add_argument("--orthogonal", action="store_true",
                        help="启用正交残差化门禁（Barra 风格剥离，拒绝共线因子）")
    parser.add_argument("--dsr", action="store_true",
                        help="启用 Deflated Sharpe 验收闸门（惩罚累计试验次数，拒绝过拟合候选）")
    parser.add_argument("--collaborative", action="store_true",
                        help="启用协同目标（适应度含对精英池的边际贡献，挖互补因子集合）")
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
    {f: panel[f].shape for f in panel}

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
        collaborative=args.collaborative,
        log_dir=str(_ROOT / "evolve" / "evolution_log"),
    )
    if args.collaborative:
        logger.info("协同目标已启用：适应度=单因子IC×%.1f + 对精英池边际贡献×%.1f",
                    1.0 - engine.mc_weight, engine.mc_weight * 2.0)

    start_t = time.time()
    best_node, best_fit = engine.evolve(panel, fwd)
    elapsed = time.time() - start_t
    logger.info(f"进化完成: 耗时 {elapsed:.1f}s")
    logger.info(f"全局最优: {best_node.to_str()} fitness={best_fit['fitness']}")

    # ── 收集所有候选（从历史最优 + 最终种群），携带因子值用于相关性去重 ──
    from core.gp import evaluate as _eval_node

    from core.selection import compute_ic_series

    _close_p = panel.get("close")

    def _signal_of(expr_or_node):
        node = parse_expr(expr_or_node) if isinstance(expr_or_node, str) else expr_or_node
        return _eval_node(node, panel)

    def _ic_series_of(sig):
        return compute_ic_series(sig, fwd, _close_p)

    candidates = []
    # 历史每代最优
    for h in engine.best_history:
        if h.get("expr"):
            sig = _signal_of(h["expr"])
            candidates.append({
                "expr": h["expr"],
                "ic": h.get("ic", 0),
                "icir": h.get("icir", 0),
                "monotonicity": h.get("monotonicity", 0),
                "long_short": h.get("long_short", 0),
                "fitness": h.get("fitness", 0),
                "generation": h.get("generation", 0),
                "source": data_label,
                "values": sig,
                "ic_series": _ic_series_of(sig),
            })
    # 最终种群
    for node in engine.population:
        f = compute_fitness(node, panel, fwd)
        sig = _eval_node(node, panel)
        candidates.append({
            "expr": node.to_str(),
            "ic": f["ic"],
            "icir": f["icir"],
            "monotonicity": f["monotonicity"],
            "long_short": f["long_short"],
            "fitness": f["fitness"],
            "generation": args.gen,
            "source": data_label,
            "values": sig,
            "ic_series": _ic_series_of(sig),
        })

    # ── LSTM 深度打分器候选（与 GP 候选并列，同一筛选门禁） ──
    if args.deep:
        from core.deep_model import train_lstm_scorer

        deep_scores, deep_meta = train_lstm_scorer(
            panel, fwd, seed=args.seed, epochs=args.deep_epochs
        )
        deep_fit = compute_fitness(deep_scores, panel, fwd)
        candidates.append({
            "expr": f"LSTM(deep, backend={deep_meta['backend']}, "
                    f"epochs={args.deep_epochs})",
            "ic": deep_fit["ic"],
            "icir": deep_fit["icir"],
            "monotonicity": deep_fit["monotonicity"],
            "long_short": deep_fit["long_short"],
            "fitness": deep_fit["fitness"],
            "generation": -1,
            "source": f"{data_label}+lstm",
            "values": deep_scores,
            "ic_series": _ic_series_of(deep_scores),
        })
        logger.info(
            f"LSTM 候选: backend={deep_meta['backend']} "
            f"IC={deep_fit['ic']:.3f} ICIR={deep_fit['icir']:.3f} "
            f"({deep_meta['elapsed']}s)"
            + ("（torch 不可用，numpy 确定性回退）" if deep_meta["fallback_used"] else "")
        )

    # ── 筛选 ──
    # DSR 闸门：n_trials = 累计个体评估数（诚实上报试验次数，防过拟合）
    _n_trials = args.gen * args.pop * 2 if args.dsr else None
    if args.orthogonal:
        from core.style_exposures import make_orthogonal_selector

        selector = make_orthogonal_selector(panel, fwd, n_trials=_n_trials)
        logger.info("正交门禁已启用：Barra 风格暴露 [mom20/size/vol20] 残差化")
        if args.dsr:
            logger.info("DSR 闸门已启用：n_trials=%d（累计评估次数）", _n_trials)
    else:
        selector = StrategySelector(n_trials=_n_trials)
        if args.dsr:
            logger.info("DSR 闸门已启用：n_trials=%d（累计评估次数）", _n_trials)
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

    # ── 实验清单：向 evolve/experiments/index.jsonl 追加一行 manifest ──
    try:
        from trader3.shared_state import SharedState

        _dv_state = SharedState().read_json("data_version") or {}
        data_version = (_dv_state.get("versions") or {}).get("qlib_bin") or "unknown"
    except Exception:
        data_version = "unknown"

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_ROOT), stderr=subprocess.DEVNULL, text=True,
        ).strip() or "nogit"
    except Exception:
        git_sha = "nogit"

    manifest = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "expr_best": best_node.to_str(),
        "fitness_best": best_fit.get("fitness", 0),
        "ic": best_fit.get("ic", 0),
        "icir": best_fit.get("icir", 0),
        "mono": best_fit.get("monotonicity", 0),
        "ls": best_fit.get("long_short", 0),
        "universe": args.universe,
        "n_stocks": N,
        "gen": args.gen,
        "pop": args.pop,
        "train_start": args.start,
        "train_end": args.end,
        "data_version": data_version,
        "selected_count": len(selected),
        "git_sha": git_sha,
    }
    experiments_dir = _ROOT / "evolve" / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    with open(experiments_dir / "index.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False) + "\n")
    logger.info(f"manifest 已追加: {experiments_dir / 'index.jsonl'}")


if __name__ == "__main__":
    main()
