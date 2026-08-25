#!/usr/bin/env python3
"""
f1_robustness.py — F1 因子稳健性扫描器（可复现）

扫描维度:
- universes  = [csi300, csi500, csi1000]
  成分取各自在 IS 起点（2020-01-01）的 asof 名单（防幸存者偏差）；超过引擎上限
  （MAX_UNIVERSE=150）时按种子 42 确定性抽样，与引擎默认路径同规则；
  三层区间共用同一固定池，保证参数维度可比。
- rebalances = [10, 21, 42]
  通过注入模块常量 trader3.tools.backtest.REBALANCE_FREQ 实现（用后恢复）；
  回测指纹不含调仓频率，故每个 (universe, rebalance) 配置使用独立缓存目录，
  杜绝跨配置串缓存。
- holds = [10, 20, 30] —— 跳过该维度：
  引擎 execute 签名没有持仓数参数（n_hold 由池规模内部推导，
  min(max(M//5, 10), 50)，见 trader3/tools/backtest.py::_real_data_backtest），
  不可参数化，本脚本如实声明并在报告中注明。

表达式固定 F1 = sub(log(vwap), log(close))
区间固定三层（与 docs/baseline / scripts/run_baseline.py 一致）:
  IS  = 2020-01-01 ~ 2023-12-31   （GP 训练区，仅参考）
  VAL = 2024-01-01 ~ 2025-07-31   （验证区——因子选择看过该段结果）
  OOS = 2025-08-01 ~ 日历末        （从未参与任何决策的干净样本外）

输出:
- docs/baseline/f1_robustness_results.json （全部原始结果 + 汇总）
- docs/baseline/f1_robustness_<date>.md    （卡片：每 universe 的 OOS 年化/夏普/
  超额中位数与跨参数离散度 std/|mean|）

用法:
  py -3.11 scripts/f1_robustness.py           # 全量: 3 universe × 3 rebalance × 3 区间
  py -3.11 scripts/f1_robustness.py --quick   # 冒烟: 仅 csi500 × rb=21 × 3 区间

注意: 全量扫描共 27 次真实面板回测，耗时较长，由主线程择机执行；本脚本不做任何 git 操作。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

import trader3.tools.backtest as btmod  # noqa: E402
from trader3 import Trader3  # noqa: E402
from trader3.data_provider import QlibDataProvider  # noqa: E402

PERIODS = {
    "IS": ("2020-01-01", "2023-12-31"),
    "VAL": ("2024-01-01", "2025-07-31"),
    "OOS": ("2025-08-01", "2100-12-31"),  # end 取日历末
}
UNIVERSES = ["csi300", "csi500", "csi1000"]
REBALANCES = [10, 21, 42]
QUICK_REBALANCE = 21
F1_EXPR = "sub(log(vwap), log(close))"
METRIC_KEYS = ["年化收益", "夏普比", "超额收益"]

# 注入前的原始调仓频率（用后恢复，避免污染同进程其他调用方）
_ORIG_REBALANCE_FREQ = btmod.REBALANCE_FREQ


def resolve_universe_codes(dp: QlibDataProvider, universe: str, asof_date: str) -> list[str]:
    """universe 名 → 固定成分代码列表（asof 过滤 + 超上限种子 42 确定性抽样，同引擎规则）。"""
    codes = dp.instruments(universe, asof_date=asof_date) or []
    if not codes:
        raise RuntimeError(f"universe={universe} 在 {asof_date} 起点无可用成分股")
    if len(codes) > btmod.MAX_UNIVERSE:
        rng = np.random.default_rng(42)
        pick = rng.choice(len(codes), size=btmod.MAX_UNIVERSE, replace=False)
        return sorted(codes[i] for i in pick)
    return sorted(codes)


def run_one(
    t3: Trader3, codes: list[str], start: str, end: str,
    rebalance: int, cache_dir: str,
) -> dict:
    """单个 (universe, rebalance, 区间) 配置的一次回测。

    注入 REBALANCE_FREQ 并将工具缓存重定向到配置专属空目录（指纹不含调仓频率，
    共享缓存会让不同 rebalance 命中同一份旧结果）；用后恢复原值。
    """
    tool = t3._registry.get("run_backtest")
    if tool is None:
        raise RuntimeError("注册表中未找到 run_backtest 工具")
    prev_cache = tool._cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    btmod.REBALANCE_FREQ = int(rebalance)
    try:
        tool._cache_dir = cache_dir
        resp = t3.run_backtest(
            universe=codes, start_date=start, end_date=end, signal_expr=F1_EXPR,
        )
    finally:
        btmod.REBALANCE_FREQ = _ORIG_REBALANCE_FREQ
        tool._cache_dir = prev_cache

    km = resp.key_metrics or {}
    row: dict = {
        "success": bool(resp.success),
        "metrics": {k: km.get(k) for k in METRIC_KEYS},
        "summary": str(resp.summary)[:160],
    }
    if not resp.success:
        row["error"] = str(resp.summary)[:240]
    return row


def _stat_entry(values: list[float]) -> dict:
    """一组跨参数指标 → 中位数 / 均值 / 离散度 std(ddof=1)/|mean|。"""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    entry: dict = {"n": int(arr.size), "median": None, "dispersion_std_over_abs_mean": None}
    if arr.size == 0:
        return entry
    entry["median"] = round(float(np.median(arr)), 6)
    if arr.size >= 2:
        mean = float(np.mean(arr))
        if abs(mean) > 1e-12:
            entry["dispersion_std_over_abs_mean"] = round(
                float(np.std(arr, ddof=1) / abs(mean)), 4
            )
    return entry


def aggregate(runs_by_universe: dict) -> dict:
    """每 universe × 区间：对 rebalance 维度聚合 OOS/IS/VAL 的中位数与离散度。"""
    summary: dict = {}
    for uni, per_period in runs_by_universe.items():
        summary[uni] = {}
        for pname, rows in per_period.items():
            ok_rows = [r for r in rows if r["success"]]
            summary[uni][pname] = {
                "n_success": len(ok_rows),
                "n_total": len(rows),
                **{
                    key: _stat_entry([r["metrics"].get(key) for r in ok_rows])
                    for key in METRIC_KEYS
                },
            }
    return summary


def _fmt_pct(v) -> str:
    return "-" if v is None else f"{v:.2%}"


def _fmt_num(v) -> str:
    return "-" if v is None else f"{v:.2f}"


def _period_table(lines: list[str], uni: str, pname: str, rebalances: list[int],
                  per_period: dict, agg: dict) -> None:
    lines += [f"### {uni} · {pname}", "",
              "| rebalance | 年化 | 夏普 | 超额 |", "|---|---|---|---|"]
    for i, rb in enumerate(rebalances):
        m = per_period[pname][i]["metrics"]
        lines.append(
            f"| {rb} | {_fmt_pct(m.get('年化收益'))} | "
            f"{_fmt_num(m.get('夏普比'))} | {_fmt_pct(m.get('超额收益'))} |"
        )
    a = agg[uni][pname]
    lines += [
        "",
        f"- 中位数: 年化 {_fmt_pct(a['年化收益']['median'])}, "
        f"夏普 {_fmt_num(a['夏普比']['median'])}, "
        f"超额 {_fmt_pct(a['超额收益']['median'])}",
        f"- 跨参数离散度 std/|mean|: "
        f"年化 {_fmt_num(a['年化收益']['dispersion_std_over_abs_mean'])}, "
        f"夏普 {_fmt_num(a['夏普比']['dispersion_std_over_abs_mean'])}, "
        f"超额 {_fmt_num(a['超额收益']['dispersion_std_over_abs_mean'])}"
        f"（样本 n={a['年化收益']['n']}）",
        "",
    ]


def build_md(results: dict) -> str:
    """Markdown 卡片：每 universe 的 OOS 年化/夏普/超额中位数与跨参数离散度。"""
    rebalances = results["rebalances"]
    cal_last = results["calendar_last"]
    lines = [
        "# F1 因子稳健性扫描卡",
        "",
        f"- 生成时间: {results['generated_at']}　日历末: {cal_last}"
        f"{'　[QUICK 冒烟模式]' if results['quick'] else ''}",
        f"- 表达式: `{results['f1_expr']}`",
        f"- 扫描维度: universes={results['universes']}, rebalances={rebalances}",
        "- 引擎语义: 次日生效 / 涨跌停拦截 / 现金跟踪 / as-of 成分 / 成本模型",
        "",
        "## 方法学注记",
        "",
        "| 项目 | 说明 |",
        "|---|---|",
        "| holds 维度 | 已跳过——引擎 execute 无持仓数参数（n_hold 由池规模推导"
        " min(max(M//5,10),50)），不可参数化 |",
        "| 调仓频率 | 注入模块常量 REBALANCE_FREQ 实现；指纹不含该参数，"
        "每配置独立缓存目录防串缓存 |",
        "| 成分池 | 各 universe 在 IS 起点的 asof 名单（防幸存者偏差），"
        ">150 只按种子 42 确定性抽样；三层区间共用固定池保证参数可比 |",
        "| 离散度口径 | std(ddof=1)/|mean|，跨 rebalance 参数组计算 |",
        "",
        "## 区间分层（同 docs/baseline）",
        "",
        "| 层 | 区间 | 性质 |",
        "|---|---|---|",
        "| IS | 2020-01-01 ~ 2023-12-31 | GP 训练区（仅参考） |",
        "| VAL | 2024-01-01 ~ 2025-07-31 | 验证区（有选择偏差） |",
        f"| OOS | 2025-08-01 ~ {cal_last} | 干净样本外（从未参与决策） |",
        "",
    ]
    for uni, per_period in results["runs_by_universe"].items():
        summary = results["summary"]
        # OOS 卡片主体
        _period_table(lines, uni, "OOS", rebalances, per_period, summary)
        # IS /VAL 附于卡片尾部（完整数值见 JSON）
        _period_table(lines, uni, "IS", rebalances, per_period, summary)
        _period_table(lines, uni, "VAL", rebalances, per_period, summary)
    failed = [r for r in results["runs"] if not r["success"]]
    if failed:
        lines += ["## 失败记录", ""]
        for r in failed:
            lines.append(f"- {r['universe']} rb={r['rebalance']} {r['period']}: {r.get('error', '')}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="F1 因子稳健性扫描器")
    parser.add_argument(
        "--quick", action="store_true",
        help="冒烟模式: 仅 csi500 × 单参数组（rb=21）× 3 区间",
    )
    args = parser.parse_args()

    universes = ["csi500"] if args.quick else list(UNIVERSES)
    rebalances = [QUICK_REBALANCE] if args.quick else list(REBALANCES)

    t3 = Trader3()
    dp = QlibDataProvider()
    cal_last = dp.calendar()[-1]

    out_dir = os.path.join(_ROOT, "docs", "baseline")
    os.makedirs(out_dir, exist_ok=True)
    cache_root = tempfile.mkdtemp(prefix="f1_robust_cache_")

    results: dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "f1_expr": F1_EXPR,
        "periods": {k: list(v) for k, v in PERIODS.items()},
        "universes": universes,
        "rebalances": rebalances,
        "holds_requested": [10, 20, 30],
        "holds_note": (
            "跳过: 引擎 execute 签名无持仓数参数（n_hold 由池规模内部推导 "
            "min(max(M//5,10),50)），不可参数化"
        ),
        "quick": bool(args.quick),
        "calendar_last": cal_last,
        "runs": [],
        "runs_by_universe": {},
        "summary": {},
    }

    total = len(universes) * len(rebalances) * len(PERIODS)
    done = 0
    t0 = time.time()
    try:
        for uni in universes:
            codes = resolve_universe_codes(dp, uni, PERIODS["IS"][0])
            results["runs_by_universe"][uni] = {}
            print(f"[pool] {uni}: {len(codes)} 只（asof {PERIODS['IS'][0]}）")
            for pname, (p_start, p_end) in PERIODS.items():
                p_end_real = p_end if p_end != "2100-12-31" else cal_last
                period_rows = []
                for rb in rebalances:
                    cache_dir = os.path.join(cache_root, f"{uni}_rb{rb}")
                    row = run_one(t3, codes, p_start, p_end_real, rb, cache_dir)
                    results["runs"].append({
                        "universe": uni, "rebalance": rb, "period": pname, **row,
                    })
                    period_rows.append(row)
                    done += 1
                    m = row["metrics"]
                    print(
                        f"[{done}/{total}] {uni} rb={rb} @ {pname}: "
                        f"ann={m.get('年化收益')} sharpe={m.get('夏普比')}"
                        + ("" if row["success"] else " [FAILED]")
                        + f" ({time.time() - t0:.0f}s)"
                    )
                results["runs_by_universe"][uni][pname] = period_rows
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)

    results["summary"] = aggregate(results["runs_by_universe"])

    json_path = os.path.join(out_dir, "f1_robustness_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(out_dir, f"f1_robustness_{time.strftime('%Y%m%d')}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_md(results))

    print("JSON:", json_path)
    print("MD:", md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
