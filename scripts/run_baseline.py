#!/usr/bin/env python3
"""
run_baseline.py — 基线基准卡跑批（可复现）

策略 × 区间矩阵 + WFA(DSR)，全部走修复后引擎统一语义：
次日生效 / 涨跌停拦截 / 现金跟踪 / as-of 成分 / 成本模型。

分层诚实声明：
  IS  = 2020-01-01 ~ 2023-12-31   （GP 训练区，仅参考）
  VAL = 2024-01-01 ~ 2025-07-31   （验证区——因子选择时看过该段结果）
  OOS = 2025-08-01 ~ 日历末        （从未参与任何决策的干净样本外）

输出:
  docs/baseline/baseline_results.json
  docs/baseline/baseline_<date>.md
"""

from __future__ import annotations

import json
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from trader3 import Trader3  # noqa: E402
from trader3.data_provider import QlibDataProvider  # noqa: E402

PERIODS = {
    "IS": ("2020-01-01", "2023-12-31"),
    "VAL": ("2024-01-01", "2025-07-31"),
    "OOS": ("2025-08-01", "2100-12-31"),  # end 取日历末
}

STRATEGIES = [
    {"key": "S1_momentum", "name": "动量20d", "kwargs": {}, "exprs": None},
    {"key": "S2_F1_vwap_gap", "name": "F1 vwap-gap",
     "kwargs": {"signal_expr": "sub(log(vwap), log(close))"},
     "exprs": ["sub(log(vwap), log(close))"]},
    {"key": "S3_F2_amt_open", "name": "F2 amount-open",
     "kwargs": {"signal_expr": "neg(sub(amount, open))"},
     "exprs": ["neg(sub(amount, open))"]},
    {"key": "S4_topk_combine", "name": "TopK合成(K=2)",
     "kwargs": {"signal_exprs": ["sub(log(vwap), log(close))",
                                 "neg(sub(amount, open))"]},
     "exprs": ["sub(log(vwap), log(close))", "neg(sub(amount, open))"]},
]

METRIC_KEYS = ["年化收益", "夏普比", "超额收益", "最大回撤", "t统计量"]


def bench_buy_hold(start: str, end: str) -> dict:
    dp = QlibDataProvider()
    cal = [d for d in dp.calendar() if start <= d <= end]
    close, dates = dp.load_stock("sh000300", "close")
    dmap = dict(zip(dates, close, strict=False))
    vals = [dmap[d] for d in cal if d in dmap and dmap[d] > 0]
    if len(vals) < 10:
        return {}
    years = len(vals) / 244.0
    total = vals[-1] / vals[0] - 1.0
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0.0
    daily = np_diff(vals)
    sharpe = (ann - 0.015) / (daily.std(ddof=1) * 244 ** 0.5) if daily.std() > 0 else 0.0
    peak, mdd = vals[0], 0.0
    for v in vals:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    return {"年化收益率": round(ann, 4), "夏普比": round(sharpe, 2),
            "最大回撤": round(mdd, 4), "总收益": round(total, 4)}


def np_diff(a):
    a = __import__("numpy").asarray(a)
    return a[1:] / a[:-1] - 1


def main() -> int:
    t3 = Trader3()
    out_dir = os.path.join(_ROOT, "docs", "baseline")
    os.makedirs(out_dir, exist_ok=True)
    results: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "periods": PERIODS, "strategies": {}, "wfa": {}}

    # 基准（每区间一次）
    results["benchmark_bh"] = {p: bench_buy_hold(s, e if e != "2100-12-31" else "2099-12-31")
                               for p, (s, e) in PERIODS.items()}
    # 修正 OOS 终点为真实日历末
    cal_last = QlibDataProvider().calendar()[-1]
    results["benchmark_bh"]["OOS"] = bench_buy_hold(PERIODS["OOS"][0], cal_last)
    results["calendar_last"] = cal_last

    total = len(STRATEGIES) * len(PERIODS)
    done = 0
    for strat in STRATEGIES:
        key = strat["key"]
        results["strategies"][key] = {"name": strat["name"], "periods": {}}
        for pname, (s, e) in PERIODS.items():
            kwargs = dict(strat["kwargs"])
            kwargs.update({"start_date": s,
                           "end_date": e if e != "2100-12-31" else cal_last})
            r = t3.run_backtest(**kwargs)
            km = r.key_metrics or {}
            picked = {k: km.get(k) for k in METRIC_KEYS}
            results["strategies"][key]["periods"][pname] = {
                "metrics": picked,
                "summary": r.summary[:120],
            }
            done += 1
            print(f"[{done}/{total}] {key} @ {pname}: "
                  f"ann={picked['年化收益']} sharpe={picked['夏普比']}")

    # WFA × 每策略（全历史滚动，含 PSR）；收集跨策略 DSR 所需日频 Sharpe
    wfa_daily_srs: list[float] = []
    for strat in STRATEGIES:
        kwargs = dict(strat["kwargs"])
        kwargs.update({"train_window": 252, "test_window": 63})
        r = t3.walk_forward_analysis(**kwargs)
        km = r.key_metrics or {}
        dsr_caveats = [c for c in (r.caveats or []) if "DSR" in c]
        oos_ann = km.get("样本外收益")
        oos_sr_ann = km.get("样本外夏普")
        if isinstance(oos_sr_ann, (int, float)):
            wfa_daily_srs.append(float(oos_sr_ann) / (252 ** 0.5))
        results["wfa"][strat["key"]] = {
            "name": strat["name"],
            "oos_annualized": oos_ann,
            "oos_sharpe": oos_sr_ann,
            "overfit_prob": km.get("过拟合概率"),
            "psr": km.get("dsr"),
            "dsr_caveat": dsr_caveats[0] if dsr_caveats else "",
            "summary": r.summary[:140],
        }
        print(f"[WFA] {strat['key']}: {results['wfa'][strat['key']]['summary']}")

    # 跨策略 Deflated Sharpe：n_trials=策略数，V[SR]=各策略 OOS 日频 Sharpe 方差
    K = len(wfa_daily_srs)
    if K >= 2:
        import math as _math

        import numpy as _np

        from trader3.tools.backtest import deflated_sharpe_ratio as _dsr

        best = max(
            ((s["key"], results["wfa"][s["key"]]["oos_sharpe"]) for s in STRATEGIES
             if isinstance(results["wfa"][s["key"]]["oos_sharpe"], (int, float))),
            key=lambda kv: kv[1], default=(None, None),
        )
        if best[0]:
            sr_var = float(_np.var(wfa_daily_srs, ddof=1))
            # OOS 期长度近似：日历总长 - 训练窗（WFA 拼接 OOS 的量级）
            n_periods = max(2, len(QlibDataProvider().calendar()) - 252)
            cross_dsr = _dsr(
                sharpe_observed=best[1] / _math.sqrt(252),
                n_trials=K,
                sr_variance=sr_var,
                tail_risk_adj=True,
                n_periods=n_periods,
            )
            results["wfa"]["_cross_strategy_dsr"] = {
                "n_trials": K, "best_strategy": best[0],
                "observed_oos_sharpe": best[1], "dsr": round(cross_dsr, 4),
                "n_periods": n_periods,
                "note": "跨策略多重比较校正（基线集合层面）",
            }
            print(f"[DSR 跨策略] N={K} best={best[0]} DSR={cross_dsr:.3f}")

    json_path = os.path.join(out_dir, "baseline_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("JSON:", json_path)

    # Markdown 骨架（分析段落由人工补写）
    lines = [
        "# 基线基准卡（引擎修复后首个完整基线）",
        "",
        f"- 生成时间: {results['generated_at']}　日历末: {cal_last}",
        "- 引擎语义: 次日生效 / 涨跌停拦截 / 现金跟踪 / as-of 成分 / 成本模型",
        "",
        "## 分层声明",
        "",
        "| 层 | 区间 | 性质 |",
        "|---|---|---|",
        "| IS | 2020-01-01 ~ 2023-12-31 | GP 训练区（仅参考） |",
        "| VAL | 2024-01-01 ~ 2025-07-31 | 验证区（因子选择看过该段，有选择偏差） |",
        f"| OOS | 2025-08-01 ~ {cal_last} | 干净样本外（从未参与决策） |",
        "",
    ]
    for pname in PERIODS:
        def _fmt(v):
            if v is None:
                return "-"
            if isinstance(v, float) and abs(v) <= 5:
                return f"{v:.1%}"
            return str(v)

        lines += [f"## {pname}", "",
                  "| 策略 | 年化 | 夏普 | 超额 | 最大回撤 | t值 |", "|---|---|---|---|---|---|"]
        bh = results["benchmark_bh"].get(pname, {})
        if bh:
            lines.append(f"| CSI300买入持有 | {bh.get('年化收益率', '-')} | "
                         f"{bh.get('夏普比', '-')} | - | {bh.get('最大回撤', '-')} | - |")
        for s in STRATEGIES:
            m = results["strategies"][s["key"]]["periods"][pname]["metrics"]
            lines.append(f"| {s['name']} | {_fmt(m['年化收益'])} | {m['夏普比']} | "
                         f"{_fmt(m['超额收益'])} | {_fmt(m['最大回撤'])} | {m['t统计量']} |")
        lines.append("")
    lines += ["## WFA（train=252/test=63 非重叠, 含成本与首日涨跌停）", "",
              "| 策略 | OOS年化 | OOS夏普 | 过拟合概率 | DSR |", "|---|---|---|---|---|"]
    for s in STRATEGIES:
        w = results["wfa"][s["key"]]
        lines.append(f"| {s['name']} | {w['oos_annualized']} | {w['oos_sharpe']} | "
                     f"{w['overfit_prob']} | {w.get('psr')} |")
    cross = results["wfa"].get("_cross_strategy_dsr")
    if cross:
        lines += ["", f"**跨策略 DSR（N={cross['n_trials']}，最优={cross['best_strategy']}）: "
                      f"{cross['dsr']}** —— 校正后仍 >0.95 才可认为最优夏普非运气。"]
    lines.append("")

    md_path = os.path.join(out_dir, f"baseline_{time.strftime('%Y%m%d')}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("MD:", md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
