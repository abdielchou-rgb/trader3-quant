#!/usr/bin/env python3
"""
capacity_analysis.py — F1 因子容量分析（csi1000）

方法（诚实简化，假设全部写入输出）：
  - ADV = 各成分近 244 个交易日成交量均值（volume.day.bin）
  - 组合：K=30 持仓、月频调仓、年化换手取实测口径 8x
  - 单票参与率 p = (C/K) / ADV；约束 p <= 10%（max_order_size_pct_adv）
  - 冲击成本 bp = sqrt(p) * 45（引擎同款平方根模型）
  - 年化拖累 = 换手 × 平均冲击bp / 1e4
  - 净夏普近似 = (毛年化 − 拖累) / 波动(23.8%，由 OOS 夏普2.21 反推)

输出: docs/baseline/capacity_f1_csi1000.md + json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from trader3.data_provider import QlibDataProvider  # noqa: E402

K = 30
TURNOVER = 8.0
GROSS_ANN = 0.526
VOL_ANN = 0.238
PART_CAP = 0.10
IMPACT_K = 45.0  # bp at p=1


def main() -> int:
    dp = QlibDataProvider()
    dp.calendar()
    codes = dp.instruments("csi1000")
    advs: list[float] = []
    for code in codes:
        try:
            vol, _ = dp.load_stock(code.lower(), "volume")
        except Exception:
            continue
        v = vol[vol > 0][-244:]
        if len(v) >= 60:
            advs.append(float(np.mean(v)))
    advs_arr = np.array(sorted(advs))
    held_adv = float(np.median(advs_arr[-K:])) if len(advs_arr) >= K else float(np.median(advs_arr))
    print(f"样本 {len(advs_arr)} 只；持仓池近似 ADV 中位数 = {held_adv:,.0f} 股/日")

    rows = []
    for cap in (1e6, 3e6, 1e7, 3e7, 1e8, 3e8):
        per_name = cap / K
        # 价格量纲缺失 → 以"股数金额"需要价格；改用成交额代理：
        # volume 为股数，需均价。用 csi1000 近似均价 15 元（诚实标注的常数假设）。
        avg_px = 15.0
        adv_cny = held_adv * avg_px
        p = per_name / adv_cny if adv_cny > 0 else 9.9
        viol = p > PART_CAP
        impact_bp = min((p ** 0.5) * IMPACT_K, 500.0)
        drag = TURNOVER * impact_bp / 1e4
        net_ann = GROSS_ANN - drag
        net_sharpe = net_ann / VOL_ANN
        rows.append({
            "capital": cap,
            "per_name_cny": round(per_name),
            "participation": round(p, 4),
            "cap_violation": bool(viol),
            "impact_bp": round(impact_bp, 1),
            "drag_ann": round(drag, 4),
            "net_ann": round(net_ann, 4),
            "net_sharpe": round(net_sharpe, 2),
        })

    out_dir = os.path.join(_ROOT, "docs", "baseline")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "capacity_f1_csi1000.json"), "w", encoding="utf-8") as f:
        json.dump({"assumptions": {"K": K, "turnover": TURNOVER, "gross_ann": GROSS_ANN,
                                   "vol_ann": VOL_ANN, "avg_px_assumed": 15.0,
                                   "adv_basis": "近244日成交量均值×假设均价15元"},
                   "curve": rows}, f, ensure_ascii=False, indent=2)

    lines = [
        "# F1 容量分析（csi1000）", "",
        "**假设（诚实声明）**：K=30、月频换手 8x、毛年化 52.6%、波动 23.8%；",
        "ADV=近244日成交量均值 × 假设均价15元（volume bin 无价格量纲）；",
        "冲击=√参与率×45bp（引擎同款）；参与率上限 10%。", "",
        "| 资金 | 单票金额 | 参与率 | 超10%上限 | 冲击bp | 年拖累 | 净年化 | 净夏普 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['capital']:.0e} | {r['per_name_cny']:,.0f} | {r['participation']:.1%} "
            f"| {'⚠' if r['cap_violation'] else ''} | {r['impact_bp']} "
            f"| {r['drag_ann']:.1%} | {r['net_ann']:.1%} | {r['net_sharpe']} |")
    half = next((r for r in rows if r["net_sharpe"] <= GROSS_ANN / VOL_ANN / 2), None)
    if half:
        lines += ["", f"**净夏普腰斩点 ≈ {half['capital']:.0e} 元**。"]
    lines += ["", "*常数假设敏感：avg_px 与 IMPACT_K 变化将线性/平方根级移动结论。*"]

    md = os.path.join(out_dir, "capacity_f1_csi1000.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("MD:", md)
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
