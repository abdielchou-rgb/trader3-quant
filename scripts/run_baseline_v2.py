#!/usr/bin/env python3
"""
run_baseline_v2.py — 全量回测矩阵 v2

在 v1（csi300 抽样域）基础上扩展：
  - 策略 × {csi300, csi500, csi1000} × 三层区间 全矩阵（显式 as-of 成分，不抽样上限内全取）
  - 新增 S5：ICIR 加权合成（top_k_combine + weight_by=icir）
  - 每策略 WFA(mode=wfa) 与 CPCV(mode=cpcv) 双口径过拟合评估
    （注：WFA/CPCV 引擎固定 csi300 全历史面板，无 universe 参数——范围如实声明）

输出:
  docs/baseline/baseline_v2_results.json
  docs/baseline/baseline_v2_<date>.md   （含与 v1 的漂移对比表）
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from trader3 import Trader3  # noqa: E402
from trader3.data_provider import QlibDataProvider  # noqa: E402

sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from f1_robustness import resolve_universe_codes  # noqa: E402

PERIODS = {
    "IS": ("2020-01-01", "2023-12-31"),
    "VAL": ("2024-01-01", "2025-07-31"),
    "OOS": ("2025-08-01", None),  # None -> 日历末
}
UNIVERSES = ["csi300", "csi500", "csi1000"]

F1 = "sub(log(vwap), log(close))"
F2 = "neg(sub(amount, open))"

STRATEGIES = [
    {"key": "S1_momentum", "name": "动量20d", "bt_kwargs": {}, "wfa_kwargs": {}},
    {"key": "S2_F1", "name": "F1 vwap-gap",
     "bt_kwargs": {"signal_expr": F1}, "wfa_kwargs": {"signal_expr": F1}},
    {"key": "S3_F2", "name": "F2 amount-open",
     "bt_kwargs": {"signal_expr": F2}, "wfa_kwargs": {"signal_expr": F2}},
    {"key": "S4_combine_eq", "name": "等权合成(K=2)",
     "bt_kwargs": {"signal_exprs": [F1, F2]},
     "wfa_kwargs": {"signal_exprs": [F1, F2]}},
    {"key": "S5_combine_icir", "name": "ICIR加权合成(K=2)",
     "bt_kwargs": {"top_k_combine": True, "factor_from_selected": 2,
                   "weight_by": "icir"},
     "wfa_kwargs": None},  # WFA 不支持加权合成 —— 如实跳过
]

METRIC_KEYS = ["年化收益", "夏普比", "超额收益", "最大回撤", "t统计量"]


def bench_buy_hold(dp, codes: list[str], start: str, end: str) -> dict:
    cal = [d for d in dp.calendar() if start <= d <= end]
    closes = {}
    for code in codes[:80]:
        try:
            close, dates = dp.load_stock(code.lower(), "close")
        except Exception:
            continue
        dmap = dict(zip(dates, close, strict=False))
        seq = [dmap[d] for d in cal if dmap.get(d, 0) > 0]
        if len(seq) >= len(cal) * 0.9:
            closes[code] = seq
    if not closes:
        return {}
    n = min(len(v) for v in closes.values())
    mat = np.array([v[-n:] for v in closes.values()])
    eq = mat.mean(axis=0)
    years = n / 244.0
    total = eq[-1] / eq[0] - 1.0
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0.0
    daily = eq[1:] / eq[:-1] - 1.0
    sharpe = float(daily.mean() / daily.std(ddof=1) * np.sqrt(244)) if daily.std() > 0 else 0.0
    peak, mdd = eq[0], 0.0
    for v in eq:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    return {"年化收益": round(float(ann), 4), "夏普比": round(sharpe, 2),
            "最大回撤": round(float(mdd), 4)}


def main() -> int:
    t0 = time.time()
    t3 = Trader3()
    dp = QlibDataProvider()
    cal_last = dp.calendar()[-1]
    out_dir = os.path.join(_ROOT, "docs", "baseline")
    os.makedirs(out_dir, exist_ok=True)

    results: dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "calendar_last": cal_last,
        "periods": {k: [s, e or cal_last] for k, (s, e) in PERIODS.items()},
        "universes": UNIVERSES,
        "matrix": {},
        "wfa": {},
        "cpcv": {},
    }

    # 成分解析缓存（per universe per period-start；OOS 用日历末）
    codes_cache: dict[tuple[str, str], list[str]] = {}

    def get_codes(u: str, period_start: str) -> list[str]:
        key = (u, period_start)
        if key not in codes_cache:
            codes_cache[key] = resolve_universe_codes(dp, u, period_start)
        return codes_cache[key]

    total = len(STRATEGIES) * len(UNIVERSES) * len(PERIODS)
    done = 0
    for u in UNIVERSES:
        results["matrix"][u] = {}
        for strat in STRATEGIES:
            results["matrix"][u][strat["key"]] = {"name": strat["name"], "periods": {}}
            for pname, (ps, pe) in PERIODS.items():
                codes_u = get_codes(u, ps)
                kwargs = dict(strat["bt_kwargs"])
                kwargs.update({"universe": codes_u,
                               "start_date": ps,
                               "end_date": pe or cal_last})
                r = t3.run_backtest(**kwargs)
                km = r.key_metrics or {}
                picked = {k: km.get(k) for k in METRIC_KEYS}
                results["matrix"][u][strat["key"]]["periods"][pname] = {
                    "metrics": picked,
                    "summary": r.summary[:120],
                    "success": bool(r.success),
                }
                done += 1
                print(f"[{done}/{total}] {u} {strat['key']} @ {pname}: "
                      f"ann={picked['年化收益']} sharpe={picked['夏普比']}")

    # 基准 B&H（每宇宙每区间）
    results["benchmark_bh"] = {}
    for u in UNIVERSES:
        results["benchmark_bh"][u] = {}
        for pname, (ps, pe) in PERIODS.items():
            results["benchmark_bh"][u][pname] = bench_buy_hold(
                dp, get_codes(u, ps), ps, pe or cal_last)

    # WFA + CPCV（引擎固定 csi300 面板）
    for strat in STRATEGIES:
        if strat["wfa_kwargs"] is None:
            continue
        for mode in ("wfa", "cpcv"):
            kw = dict(strat["wfa_kwargs"])
            kw.update({"train_window": 252, "test_window": 63, "mode": mode})
            r = t3.walk_forward_analysis(**kw)
            km = r.key_metrics or {}
            entry = {
                "oos_annualized": km.get("样本外收益"),
                "oos_sharpe": km.get("样本外夏普"),
                "overfit_prob": km.get("过拟合概率"),
                "psr": km.get("dsr"),
                "summary": r.summary[:140],
            }
            if mode == "cpcv":
                entry.update({
                    "cpcv_median_sr": km.get("cpcv_median_sr"),
                    "cpcv_p05": km.get("cpcv_p05"),
                    "cpcv_p95": km.get("cpcv_p95"),
                    "cpcv_prob_negative": km.get("cpcv_prob_negative"),
                })
                results["cpcv"][strat["key"]] = {"name": strat["name"], **entry}
            else:
                results["wfa"][strat["key"]] = {"name": strat["name"], **entry}
            print(f"[{mode}] {strat['key']}: oos_ann={entry['oos_annualized']} "
                  f"oos_sharpe={entry['oos_sharpe']}")

    json_path = os.path.join(out_dir, "baseline_v2_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("JSON:", json_path, f"耗时 {time.time()-t0:.0f}s")

    # ---- Markdown 卡片骨架 ----
    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, (int, float)) and abs(v) <= 8:
            return f"{v:.1%}"
        return str(v)

    lines = [
        "# 基线基准卡 v2（全量矩阵）",
        "",
        f"- 生成: {results['generated_at']}　日历末: {cal_last}",
        "- 引擎: post-audit-7+（含成员段刷新/拼接修复后首次全量矩阵）",
        "- 策略: S1动量 / S2=F1 / S3=F2 / S4=等权合成 / S5=ICIR加权合成",
        "- WFA/CPCV 仅 csi300 引擎面板（无 universe 参数，范围如实声明）；S5 加权合成不可经 WFA 表达，跳过",
        "",
    ]
    for u in UNIVERSES:
        for pname in PERIODS:
            lines += [f"## {u} · {pname}", "",
                      "| 策略 | 年化 | 夏普 | 超额 | 最大回撤 |", "|---|---|---|---|---|"]
            bh = results["benchmark_bh"][u][pname]
            if bh:
                lines.append(f"| B&H等权基准 | {fmt(bh.get('年化收益'))} | "
                             f"{bh.get('夏普比')} | - | {fmt(bh.get('最大回撤'))} |")
            for s in STRATEGIES:
                m = results["matrix"][u][s["key"]]["periods"][pname]["metrics"]
                lines.append(f"| {s['name']} | {fmt(m['年化收益'])} | {m['夏普比']} | "
                             f"{fmt(m['超额收益'])} | {fmt(m['最大回撤'])} |")
            lines.append("")

    lines += ["## 过拟合双口径（csi300）", "",
              "| 策略 | WFA-OOS年化 | WFA过拟合概率 | PSR | CPCV中位SR | CPCV P05/P95 | P(SR<0) |",
              "|---|---|---|---|---|---|---|"]
    for s in STRATEGIES:
        k = s["key"]
        w = results["wfa"].get(k, {})
        c = results["cpcv"].get(k, {})
        if not w and not c:
            continue
        lines.append(
            f"| {s['name']} | {fmt(w.get('oos_annualized'))} "
            f"| {fmt(w.get('overfit_prob'))} | {fmt((w.get('psr') or {}).get('value') if isinstance(w.get('psr'), dict) else w.get('psr'))} "
            f"| {fmt(c.get('cpcv_median_sr'))} "
            f"| {c.get('cpcv_p05','-')} / {c.get('cpcv_p95','-')} "
            f"| {fmt(c.get('cpcv_prob_negative'))} |")
    cross = results["wfa"].get("_cross_strategy_dsr") or results["cpcv"].get("_cross_strategy_dsr")
    lines.append("")

    md_path = os.path.join(out_dir, f"baseline_v2_{time.strftime('%Y%m%d')}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("MD:", md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
