#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_evolved_factors.py — 进化因子回测验证

用 qlib 真实数据验证 evolve/strategies/selected.json 里进化出的因子：
1. 计算因子信号（core.gp.evaluate）
2. 月频调仓 Top-N 组合回测（含交易成本）
3. 与动量基线 + CSI300 基准对比
4. 报告年化/夏普/回撤/换手

用法:
    python evolve/validate_evolved_factors.py --n-stocks 60 --top-n 10
    python evolve/validate_evolved_factors.py --expr "ts_corr(volume, close, 10)" --name myfactor
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "evolve"))

from core.data_loader import load_qlib_panel  # noqa: E402
from core.gp import evaluate  # noqa: E402
from core.parser import parse_expr  # noqa: E402

TRADING_COST = 0.0015  # 单边交易成本（佣金+滑点+冲击），A股实际水平


def load_factors(path: Path) -> list:
    """从 selected.json 读取因子列表"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(d["expr"], d.get("score", 0)) for d in data]


def monthly_rebalance(signal: np.ndarray, close: np.ndarray, n_hold: int,
                      cost: float = TRADING_COST) -> tuple:
    """
    月频调仓 Top-N 等权组合回测（信号 t 日收盘产生，t+1 起生效并计成本）。

    Returns (equity_curve, rets, total_turnover, rebalance_days)
    """
    T, N = signal.shape
    weights = np.zeros(N)
    equity = np.ones(T)
    rets = np.zeros(T)
    total_turnover = 0.0
    rebalance_days = 0

    # 日收益
    daily_ret = np.zeros((T, N))
    valid = close > 0
    daily_ret[1:] = np.where(valid[1:] & valid[:-1],
                             close[1:] / close[:-1] - 1.0, 0.0)

    pending = None
    for t in range(T):
        day_cost = 0.0
        if pending is not None:
            turnover = float(np.abs(pending - weights).sum() / 2)
            total_turnover += turnover
            day_cost = turnover * cost * 2  # 单边成本 × 双边换手
            weights = pending
            pending = None
            rebalance_days += 1

        if t < T - 1 and ((t == 0) or ((t + 1) % 21 == 0)):
            s = signal[t]
            valid_today = np.isfinite(s) & (close[t] > 0)
            if valid_today.sum() >= n_hold:
                scores = np.where(valid_today, s, -np.inf)
                ranked = np.argsort(scores)[::-1][:n_hold]
                target = np.zeros(N)
                target[ranked] = 1.0 / n_hold
                pending = target  # 次日生效

        # 组合收益（生效日扣交易成本）
        rets[t] = np.dot(weights, daily_ret[t]) - day_cost
        equity[t] = equity[t - 1] * (1 + rets[t]) if t > 0 else 1.0 + rets[t]

    return equity, rets, total_turnover, rebalance_days


def annualize(equity: np.ndarray) -> dict:
    """从净值曲线计算指标"""
    T = len(equity)
    rets = np.diff(equity) / equity[:-1]
    years = T / 252
    total_ret = equity[-1] / equity[0] - 1
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    ann_vol = np.std(rets) * np.sqrt(252) if len(rets) > 1 else 0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    # 最大回撤
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()
    return {
        "年化收益": ann_ret,
        "夏普比": sharpe,
        "最大回撤": max_dd,
        "总收益": total_ret,
    }


def baseline_momentum(signal: np.ndarray) -> np.ndarray:
    """动量基线：20日对数收益"""
    T, N = signal.shape
    mom = np.zeros_like(signal)
    for j in range(N):
        col = signal[:, j]
        for i in range(20, T):
            if col[i] > 0 and col[i - 20] > 0:
                mom[i, j] = np.log(col[i] / col[i - 20])
    return mom


def main():
    parser = argparse.ArgumentParser(description="进化因子回测验证")
    parser.add_argument("--universe", default="csi300")
    parser.add_argument("--n-stocks", type=int, default=60)
    parser.add_argument("--start", default="2024-01-01",
                        help="验证区间起点（必须晚于训练区间末，保证样本外）")
    parser.add_argument("--end", default="2025-07-31")
    parser.add_argument("--train-end", default="2023-12-31",
                        help="训练区间末（用于校验窗口不相交）")
    parser.add_argument("--top-n", type=int, default=10, help="持仓数量")
    parser.add_argument("--json", default="evolve/strategies/selected.json")
    parser.add_argument("--expr", default="", help="单表达式验证")
    parser.add_argument("--name", default="custom")
    parser.add_argument("--combine", action="store_true", help="组合验证（前3因子等权）")
    args = parser.parse_args()

    if args.start <= args.train_end:
        parser.error(f"验证起点 {args.start} 必须晚于训练末 {args.train_end}（样本外要求）")

    # ── 加载数据 ──
    print(f"加载数据: {args.universe} {args.start}~{args.end} ({args.n_stocks}只)")
    panel, fwd = load_qlib_panel(
        universe=args.universe, n_stocks=args.n_stocks,
        start=args.start, end=args.end,
    )
    close = panel["close"]
    T, N = close.shape
    print(f"面板: {T} 交易日 × {N} 标的\n")

    # ── 确定因子列表 ──
    factors = []
    if args.expr:
        factors = [(args.expr, 0)]
        names = [args.name]
    else:
        factors = load_factors(_ROOT / args.json)
        if not factors:
            print("no factors in", args.json); return
        names = [f"F{i+1}" for i in range(len(factors))]

    print("=" * 78)
    print(f"{'因子':<8} {'表达式':<52} {'年化':>7} {'夏普':>6} {'回撤':>8} {'换手':>6}")
    print("=" * 78)

    results = {}

    # ── 动量基线 ──
    years = T / 252
    mom = baseline_momentum(close)
    eq, rets, turn, _ = monthly_rebalance(mom, close, args.top_n)
    m = annualize(eq)
    results["动量基线"] = m
    print(f"{'动量基线':<8} {'log(close/close[-20])':<52} {m['年化收益']:>7.1%} "
          f"{m['夏普比']:>6.2f} {m['最大回撤']:>8.1%} {turn/max(years,1e-9):>6.1f}x/年")

    # ── 各因子 ──
    for name, (expr, _score) in zip(names, factors):
        try:
            from core.gp import normalize
            node = normalize(parse_expr(expr))
            sig = evaluate(node, panel)
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            print(f"{name}: 计算失败 {e}")
            continue
        eq, rets, turn, _ = monthly_rebalance(sig, close, args.top_n)
        m = annualize(eq)
        results[name] = m
        print(f"{name:<8} {expr[:52]:<52} {m['年化收益']:>7.1%} "
              f"{m['夏普比']:>6.2f} {m['最大回撤']:>8.1%} {turn/max(years,1e-9):>6.1f}x/年")

    # ── 组合验证（前3因子等权排名合并） ──
    if args.combine and len(factors) >= 3:
        print("\n组合验证（前3因子等权 zscore 合并）:")
        sigs = []
        from core.gp import normalize
        for expr, _ in factors[:3]:
            node = normalize(parse_expr(expr))
            s = evaluate(node, panel)
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            # zscore 标准化
            mu = np.nanmean(s, axis=1, keepdims=True)
            sd = np.nanstd(s, axis=1, keepdims=True)
            s = np.where(sd > 1e-10, (s - mu) / (sd + 1e-10), 0.0)
            sigs.append(s)
        combo = np.mean(sigs, axis=0)
        eq, rets, turn, _ = monthly_rebalance(combo, close, args.top_n)
        m = annualize(eq)
        results["组合"] = m
        print(f"{'组合':<8} {'前3因子等权':<52} {m['年化收益']:>7.1%} "
              f"{m['夏普比']:>6.2f} {m['最大回撤']:>8.1%} {turn/max(years,1e-9):>6.1f}x/年")

    # ── 总结 ──
    print("\n" + "=" * 78)
    print("对比总结（超越动量基线 = 因子有效）:")
    base = results.get("动量基线", {})
    base_ann = base.get("年化收益", 0)
    for name, m in results.items():
        if name == "动量基线":
            continue
        delta = m.get("年化收益", 0) - base_ann
        better = "✅ 超越" if delta > 0 else "❌ 落后"
        print(f"  {name}: {delta:+.1%} vs 基线 ({better})")
    print("=" * 78)
    print("⚠ 候选信号，非投资建议。需 WFA + 换窗验证后进策略库。")


if __name__ == "__main__":
    main()