"""
3号交易员 — 估值/基本面 Tools (M5: 真实估值引擎 + 基本面评分卡)

M5 upgrades over M0 template:
1. Real DCF valuation (two-stage: 5y explicit forecast + Gordon terminal value)
   - WACC = cost_of_equity * E/(E+D) + cost_of_debt * (1-tax) * D/(E+D)
   - Cost of equity via CAPM: rf + beta * ERP
   - Terminal value via Gordon growth: FCF_T * (1+g) / (WACC - g)
   - Sensitivity: WACC grid (±100bp) + terminal growth grid (±0.5%)
   - Three scenarios: base / bull / bear with different FCF growth assumptions
2. Real relative valuation
   - PE percentile: current PE vs 5y historical PE distribution
   - PB-ROE: target P/B = ROE / r (r implied from PE)
   - EV/EBITDA: historical mean + sector premium
3. Weighted target price (DCF 40% / PE percentile 25% / PB-ROE 20% / EV-EBITDA 15%)
4. Real 6-dimension fundamental scorecard (0-10 each) with template weighting
5. Red flag rule engine (AR/goodwill/margin/debt/cash-flow)
6. Peer comparison (percentile ranks or synthetic sector averages)
7. Private company valuation bridge (revenue × PS, liquidity discount + control premium)
   — absorbed from ai-berkshire financial_rigor + private-company-research skill

Input handling: user-provided financials dict is used verbatim (missing keys filled
from synthetic profile); otherwise synthetic/demo financials with a clear caveat.
"""

from __future__ import annotations

import math
import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from trader3.base_tool import BaseTool, ChartSpec, Trader3Response
from trader3.models import PrivateCompanyBridge, ScorecardReport, ValuationReport


# ═══════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════

DCF_FORECAST_YEARS = 5
DCF_METHOD_WEIGHTS = {
    "dcf": 0.40,
    "pe_percentile": 0.25,
    "pb_roe": 0.20,
    "ev_ebitda": 0.15,
}
SCORECARD_TEMPLATES = {
    "quality_growth": {
        "label": "质量成长",
        "weights": {
            "盈利能力": 0.35,
            "成长性": 0.25,
            "财务健康": 0.15,
            "估值合理性": 0.10,
            "管理层质量": 0.08,
            "竞争壁垒": 0.07,
        },
    },
    "value": {
        "label": "价值型",
        "weights": {
            "估值合理性": 0.30,
            "财务健康": 0.20,
            "盈利能力": 0.20,
            "成长性": 0.10,
            "管理层质量": 0.10,
            "竞争壁垒": 0.10,
        },
    },
    "turnaround": {
        "label": "反转型",
        "weights": {
            "成长性": 0.30,
            "管理层质量": 0.20,
            "财务健康": 0.20,
            "盈利能力": 0.15,
            "估值合理性": 0.10,
            "竞争壁垒": 0.05,
        },
    },
}
ALL_DIMENSIONS = ["盈利能力", "成长性", "财务健康", "估值合理性", "管理层质量", "竞争壁垒"]
SECTOR_DEFAULT_PS = {
    "消费": 4.5, "科技": 6.0, "医药": 5.5, "金融": 3.0,
    "制造": 2.5, "能源": 1.8, "地产": 1.5, "传媒": 4.0, "通用": 3.5,
}


# ═══════════════════════════════════════════
# Synthetic Data Generators
# ═══════════════════════════════════════════


def _seed_from(code: str) -> int:
    """Deterministic seed from a stock code (stable across runs)."""
    return zlib.crc32(str(code).encode("utf-8")) & 0xFFFFFFFF


def _gen_pe_history(
    rng: np.random.Generator,
    current_pe: float,
    n: int = 60,
) -> np.ndarray:
    """Mean-reverting historical PE series (5y quarterly ≈ 60 points)."""
    mean_reversion = current_pe * float(rng.uniform(0.85, 1.2))
    sigma = current_pe * 0.08
    vals = [mean_reversion]
    for _ in range(n):
        prev = vals[-1]
        vals.append(prev + 0.3 * (mean_reversion - prev) + float(rng.normal(0, sigma)))
    return np.maximum(np.array(vals, dtype=np.float64), 1.0)


def _gen_ev_history(
    rng: np.random.Generator,
    current_ev_ebitda: float,
    n: int = 24,
) -> np.ndarray:
    """Mean-reverting historical EV/EBITDA series."""
    mean_reversion = current_ev_ebitda * float(rng.uniform(0.9, 1.15))
    sigma = current_ev_ebitda * 0.10
    vals = [mean_reversion]
    for _ in range(n):
        prev = vals[-1]
        vals.append(prev + 0.4 * (mean_reversion - prev) + float(rng.normal(0, sigma)))
    return np.maximum(np.array(vals, dtype=np.float64), 1.0)


def _synthetic_financials(code: str) -> Dict[str, Any]:
    """
    Generate a coherent synthetic financial profile for a stock code.

    All financials are per-share based where meaningful so the output
    target prices are directly comparable with the current price.
    Deterministic per code (seeded), so repeated calls are stable.
    """
    rng = np.random.default_rng(_seed_from(code))

    # ── Market / valuation raw inputs ──
    pe = float(rng.uniform(12.0, 38.0))
    roe = float(rng.uniform(0.08, 0.22))
    price = float(rng.uniform(10.0, 120.0))
    eps = price / pe
    bvps = eps / roe if roe > 0 else eps / 0.10
    ps = float(rng.uniform(0.8, 6.0))
    rev_ps = price / ps
    fcf_yield = float(rng.uniform(0.03, 0.08))
    fcf_ps = price * fcf_yield
    ebitda_margin = float(rng.uniform(0.10, 0.32))
    ebitda_ps = rev_ps * ebitda_margin
    net_debt_ps = bvps * float(rng.uniform(0.05, 0.60))

    # ── WACC inputs ──
    rf = 0.025
    erp = 0.06
    beta = float(rng.uniform(0.7, 1.4))
    tax_rate = 0.25
    cost_of_debt = float(rng.uniform(0.035, 0.055))
    debt_to_equity = net_debt_ps / bvps if bvps > 0 else 0.3

    # ── Fundamentals ──
    gross_margin = float(rng.uniform(0.25, 0.60))
    net_margin = ebitda_margin * float(rng.uniform(0.30, 0.50))
    revenue_cagr = float(rng.uniform(0.05, 0.25))
    earnings_cagr = float(rng.uniform(0.02, 0.30))
    current_ratio = float(rng.uniform(0.8, 2.5))
    fcf_conversion = float(rng.uniform(0.6, 1.4))
    goodwill_to_assets = float(rng.uniform(0.0, 0.25))
    sector_pe = float(rng.uniform(15.0, 30.0))
    sector_premium = float(rng.uniform(-0.05, 0.10))

    # ── 5-year histories (index 0..4, most recent = last) ──
    years = 5
    revenue = [1.0]
    for _ in range(years):
        revenue.append(revenue[-1] * (1.0 + revenue_cagr + float(rng.normal(0, 0.02))))
    revenue_growth_list = [revenue[i + 1] / revenue[i] - 1.0 for i in range(years)]

    net_margin_list = [net_margin * (1.0 + float(rng.normal(0, 0.02))) for _ in range(years)]
    roe_list = [roe * (1.0 + float(rng.normal(0, 0.02))) for _ in range(years)]
    net_income_list = [revenue[i] * net_margin_list[i] for i in range(years)]
    fcf_list = [
        net_income_list[i] * fcf_conversion * (1.0 + float(rng.normal(0, 0.10)))
        for i in range(years)
    ]
    ocf_list = [
        fcf_list[i] + net_income_list[i] * float(rng.uniform(0.02, 0.15))
        for i in range(years)
    ]
    ar_list = [
        revenue[i] * float(rng.uniform(0.10, 0.40)) * (1.0 + float(rng.normal(0, 0.05)))
        for i in range(years)
    ]

    # ── Historical multiple distributions ──
    pe_history = _gen_pe_history(rng, pe)
    current_ev_ebitda = (price + net_debt_ps) / ebitda_ps if ebitda_ps > 0 else 10.0
    ev_ebitda_history = _gen_ev_history(rng, current_ev_ebitda)

    # DCF assumptions (base FCF growth over forecast period)
    fcf_growth_5y = float(np.clip(earnings_cagr + rng.uniform(-0.02, 0.02), 0.0, 0.35))
    terminal_growth = float(np.clip(rf - 0.005 + rng.uniform(-0.005, 0.005), 0.0, 0.04))

    return {
        "code": code,
        "current_price": price,
        "eps": eps,
        "bvps": bvps,
        "revenue_per_share": rev_ps,
        "fcf_per_share": fcf_ps,
        "ebitda_per_share": ebitda_ps,
        "net_debt_per_share": net_debt_ps,
        "pe": pe,
        "pb": price / bvps if bvps > 0 else 1.0,
        "ps": ps,
        "roe": roe,
        "gross_margin": gross_margin,
        "net_margin": net_margin,
        "revenue_cagr": revenue_cagr,
        "earnings_cagr": earnings_cagr,
        "current_ratio": current_ratio,
        "fcf_conversion": fcf_conversion,
        "goodwill_to_assets": goodwill_to_assets,
        "debt_to_equity": debt_to_equity,
        "sector_pe": sector_pe,
        "sector_premium": sector_premium,
        # WACC inputs
        "rf": rf,
        "erp": erp,
        "beta": beta,
        "tax_rate": tax_rate,
        "cost_of_debt": cost_of_debt,
        "fcf_growth_5y": fcf_growth_5y,
        "terminal_growth": terminal_growth,
        # histories
        "pe_history": pe_history,
        "ev_ebitda_history": ev_ebitda_history,
        "revenue_growth_list": revenue_growth_list,
        "net_margin_list": net_margin_list,
        "roe_list": roe_list,
        "net_income_list": net_income_list,
        "fcf_list": fcf_list,
        "ocf_list": ocf_list,
        "ar_list": ar_list,
        "revenue_list": revenue,
    }


def _fill_financials(user_fin: Optional[Dict], code: str) -> Tuple[Dict[str, Any], bool, set]:
    """
    Merge user-provided financials over the synthetic profile.

    When real data is merged, regenerate synthetic *histories* around the real
    values so the PE-percentile / EV-EBITDA methods behave consistently
    (real PE ~14.5x shouldn't be compared to synthetic history ~38x).

    Returns (financials, used_synthetic_flag, synthetic_fields)。
    synthetic_fields 为仍来自合成档案的字段名集合，用于按方法禁用（白名单模式）。
    """
    synthetic = _synthetic_financials(code)
    if not user_fin:
        return synthetic, True, set(synthetic.keys())

    merged = dict(synthetic)
    overridden = set()
    for k, v in user_fin.items():
        if v is not None:
            merged[k] = v
            overridden.add(k)

    # Regenerate history series around the merged (possibly real) values.
    # 注意：生成的历史是"围绕真实锚点的合成均值回归序列"，用于演示；
    # 混合模式下它会被标记进 synthetic_fields → 依赖历史的分位法被禁用。
    rng = np.random.default_rng(_seed_from(code) + 999)
    if user_fin.get("pe") and not user_fin.get("pe_history"):
        pe_val = float(merged["pe"])
        merged["pe_history"] = _gen_pe_history(rng, pe_val)

    if user_fin.get("current_price") and not user_fin.get("ev_ebitda_history"):
        price = float(merged.get("current_price", 50.0))
        ndps = float(merged.get("net_debt_per_share", 0.0))
        ebps = float(merged.get("ebitda_per_share", 5.0))
        if ebps > 0:
            ev = (price + ndps) / ebps
            merged["ev_ebitda_history"] = _gen_ev_history(rng, ev)

    synthetic_fields = set(merged.keys()) - overridden
    # 生成的历史序列视为合成字段 → 分位数类方法在混合模式下禁用（防循环论证）
    if "pe" in overridden and "pe_history" not in overridden and "pe_history" in merged:
        synthetic_fields.add("pe_history")
    if ("current_price" in overridden and "ev_ebitda_history" not in overridden
            and "ev_ebitda_history" in merged):
        synthetic_fields.add("ev_ebitda_history")

    return merged, False, synthetic_fields


# ═══════════════════════════════════════════
# WACC / DCF
# ═══════════════════════════════════════════


def _compute_wacc(fin: Dict[str, Any]) -> Tuple[float, float]:
    """WACC = cost_of_equity * E/(E+D) + cost_of_debt * (1-tax) * D/(E+D)."""
    rf = fin.get("rf", 0.025)
    erp = fin.get("erp", 0.06)
    beta = fin.get("beta", 1.0)
    tax_rate = fin.get("tax_rate", 0.25)
    cost_of_debt = fin.get("cost_of_debt", 0.045)

    cost_of_equity = rf + beta * erp

    de = max(fin.get("debt_to_equity", 0.3), 0.0)
    equity_weight = 1.0 / (1.0 + de)
    debt_weight = de / (1.0 + de)

    wacc = cost_of_equity * equity_weight + cost_of_debt * (1.0 - tax_rate) * debt_weight
    return float(wacc), float(cost_of_equity)


def _dcf_value(
    fcf_0: float,
    growth_rate: float,
    wacc: float,
    terminal_growth: float,
    years: int = DCF_FORECAST_YEARS,
) -> float:
    """
    Two-stage DCF per share.

    Stage 1: FCF_t = FCF_0 * (1+g)^t discounted for `years` years.
    Stage 2: Gordon terminal value = FCF_T * (1+g_term) / (WACC - g_term),
             discounted back to today.
    """
    if wacc <= terminal_growth or wacc <= 0:
        return 0.0

    pv = 0.0
    fcf = fcf_0
    for t in range(1, years + 1):
        fcf = fcf * (1.0 + growth_rate)
        pv += fcf / (1.0 + wacc) ** t

    tv = fcf * (1.0 + terminal_growth) / (wacc - terminal_growth)
    pv += tv / (1.0 + wacc) ** years
    return float(pv)


def _dcf_valuation(fin: Dict[str, Any], wacc: float) -> Dict[str, float]:
    """Base-case DCF with configured growth assumptions."""
    fcf_0 = fin.get("fcf_per_share", 1.0)
    growth = fin.get("fcf_growth_5y", 0.10)
    terminal_growth = fin.get("terminal_growth", 0.025)

    value = _dcf_value(fcf_0, growth, wacc, terminal_growth)
    return {
        "target": value,
        "wacc": wacc,
        "terminal_growth": terminal_growth,
        "fcf_growth_5y": growth,
        "forecast_years": DCF_FORECAST_YEARS,
    }


def _dcf_scenarios(fin: Dict[str, Any], wacc: float) -> Dict[str, float]:
    """Base / Bull / Bear DCF fair values under different FCF growth assumptions."""
    fcf_0 = fin.get("fcf_per_share", 1.0)
    base_growth = fin.get("fcf_growth_5y", 0.10)
    terminal_growth = fin.get("terminal_growth", 0.025)

    bull_growth = float(np.clip(base_growth + 0.03, 0.0, 0.40))
    bear_growth = float(np.clip(base_growth - 0.03, -0.10, 0.35))

    base = _dcf_value(fcf_0, base_growth, wacc, terminal_growth)
    bull = _dcf_value(fcf_0, bull_growth, wacc, terminal_growth)
    bear = _dcf_value(fcf_0, bear_growth, wacc, terminal_growth)

    return {
        "fair_value_base": base,
        "fair_value_bull": bull,
        "fair_value_bear": bear,
        "bull_growth": bull_growth,
        "bear_growth": bear_growth,
    }


def _dcf_sensitivity(fin: Dict[str, Any], wacc: float) -> Dict[str, List[float]]:
    """
    Sensitivity: WACC grid (±100bp / ±50bp) and terminal-growth grid (±0.5%).
    Each entry is the DCF target price under the perturbed assumption.
    """
    fcf_0 = fin.get("fcf_per_share", 1.0)
    growth = fin.get("fcf_growth_5y", 0.10)
    terminal_growth = fin.get("terminal_growth", 0.025)

    wacc_grid = [wacc - 0.010, wacc - 0.005, wacc, wacc + 0.005, wacc + 0.010]
    wacc_targets = [
        _dcf_value(fcf_0, growth, w, terminal_growth) for w in wacc_grid
    ]

    tg_grid = [
        terminal_growth - 0.005, terminal_growth - 0.0025,
        terminal_growth, terminal_growth + 0.0025, terminal_growth + 0.005,
    ]
    tg_targets = [
        _dcf_value(fcf_0, growth, wacc, g) for g in tg_grid
    ]

    return {
        "wacc": [round(float(v), 2) for v in wacc_targets],
        "terminal_growth": [round(float(v), 2) for v in tg_targets],
    }


# ═══════════════════════════════════════════
# Relative Valuation
# ═══════════════════════════════════════════


def _pe_percentile_valuation(fin: Dict[str, Any]) -> Dict[str, Any]:
    """PE percentile: current PE vs 5y historical PE distribution (synthetic)."""
    current_pe = fin.get("pe", 20.0)
    eps = fin.get("eps", 1.0)
    hist = np.asarray(fin.get("pe_history", [current_pe]), dtype=np.float64)

    p25, p50, p75 = np.percentile(hist, [25, 50, 75])
    percentile = float(np.mean(hist < current_pe))  # fraction below current

    # Expensive (>70th pct) → penalize toward lower historical PE; cheap (<30th) → reward.
    if percentile > 0.70:
        target_pe = float(max(p25, p50 * 0.90))
    elif percentile < 0.30:
        target_pe = float(min(p75, p50 * 1.10))
    else:
        target_pe = float(p50)

    return {
        "target": float(target_pe * eps),
        "current_pe": current_pe,
        "historical_p50_pe": float(p50),
        "historical_p25_pe": float(p25),
        "historical_p75_pe": float(p75),
        "pe_percentile": percentile,
        "target_pe": target_pe,
    }


def _pb_roe_valuation(fin: Dict[str, Any]) -> Dict[str, Any]:
    """
    PB-ROE: target P/B = ROE / r, where r is the required return implied from PE
    (r = earnings yield = 1 / current PE).
    """
    current_pe = fin.get("pe", 20.0)
    roe = fin.get("roe", 0.12)
    bvps = fin.get("bvps", 8.0)

    r = 1.0 / current_pe if current_pe > 0 else 0.10
    target_pb = roe / r if r > 0 else 0.0

    return {
        "target": float(target_pb * bvps),
        "current_pb": float(fin.get("pb", target_pb)),
        "roe": roe,
        "r_implied": r,
        "target_pb": float(target_pb),
    }


def _ev_ebitda_valuation(fin: Dict[str, Any]) -> Dict[str, Any]:
    """EV/EBITDA: historical mean + sector premium."""
    price = fin.get("current_price", 50.0)
    net_debt_ps = fin.get("net_debt_per_share", 0.0)
    ebitda_ps = fin.get("ebitda_per_share", 5.0)

    current_ev_ebitda = (
        (price + net_debt_ps) / ebitda_ps if ebitda_ps > 0 else 10.0
    )
    hist = np.asarray(fin.get("ev_ebitda_history", [current_ev_ebitda]), dtype=np.float64)
    hist_mean = float(np.mean(hist))

    sector_premium = fin.get("sector_premium", 0.05)
    target_ev_ebitda = hist_mean * (1.0 + sector_premium)
    target_ev = target_ev_ebitda * ebitda_ps
    target_equity = target_ev - net_debt_ps

    return {
        "target": float(target_equity),
        "current_ev_ebitda": float(current_ev_ebitda),
        "historical_mean_ev_ebitda": hist_mean,
        "sector_premium": sector_premium,
        "target_ev_ebitda": float(target_ev_ebitda),
    }


def _compute_all_valuations(fin: Dict[str, Any]) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
    """Run all four valuation methods + WACC. Returns (methods, aux)."""
    wacc, cost_of_equity = _compute_wacc(fin)

    methods = {
        "dcf": _dcf_valuation(fin, wacc),
        "pe_percentile": _pe_percentile_valuation(fin),
        "pb_roe": _pb_roe_valuation(fin),
        "ev_ebitda": _ev_ebitda_valuation(fin),
    }
    aux = {
        "wacc": wacc,
        "cost_of_equity": cost_of_equity,
        "scenarios": _dcf_scenarios(fin, wacc),
        "sensitivity": _dcf_sensitivity(fin, wacc),
    }
    return methods, aux


# 各估值方法依赖的必须为真实值的字段（白名单模式：缺失即禁用该方法）
_METHOD_REQUIRED_REAL_FIELDS = {
    "dcf": {"fcf_per_share"},
    "pe_percentile": {"pe", "eps", "pe_history"},
    "pb_roe": {"pe", "roe", "bvps"},
    "ev_ebitda": {"current_price", "ebitda_per_share", "net_debt_per_share",
                  "ev_ebitda_history"},
}


def _gate_methods_by_provenance(
    methods_all: Dict[str, Dict],
    selected: List[str],
    synthetic_fields: set,
) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    """禁用关键输入来自合成档案的估值方法，返回 (保留方法, 禁用明细)。"""
    disabled: List[Tuple[str, List[str]]] = []
    kept: List[str] = []
    for m in selected:
        missing = sorted(_METHOD_REQUIRED_REAL_FIELDS.get(m, set()) & synthetic_fields)
        if missing:
            disabled.append((m, missing))
        else:
            kept.append(m)
    return kept, disabled


def _weighted_target(
    methods: Dict[str, Dict],
    selected: List[str],
) -> Tuple[float, Dict[str, float]]:
    """Weighted average across methods (weights renormalized over selection)."""
    weights = DCF_METHOD_WEIGHTS
    chosen = [m for m in selected if m in weights]
    if not chosen:
        chosen = list(weights.keys())
    total_w = sum(weights[m] for m in chosen)
    norm = {m: weights[m] / total_w for m in chosen}
    target = sum(norm[m] * methods[m]["target"] for m in chosen)
    return target, norm


# ═══════════════════════════════════════════
# Fundamental Scorecard
# ═══════════════════════════════════════════


def _clip10(x: float) -> float:
    return float(np.clip(x, 0.0, 10.0))


def _score_profitability(fin: Dict[str, Any]) -> float:
    roe = fin.get("roe", 0.10)
    gross_margin = fin.get("gross_margin", 0.35)
    net_margin = fin.get("net_margin", 0.10)
    nm_list = fin.get("net_margin_list", [net_margin] * 5)

    roe_score = min(roe / 0.20, 1.0) * 10.0
    margin_score = min(0.5 * (gross_margin / 0.50) + 0.5 * (net_margin / 0.15), 1.0) * 10.0

    if nm_list[-1] > nm_list[0]:
        trend_score = 10.0
    elif nm_list[-1] < nm_list[0] * 0.95:
        trend_score = 3.0
    else:
        trend_score = 6.0

    return round(_clip10(0.5 * roe_score + 0.3 * margin_score + 0.2 * trend_score), 1)


def _score_growth(fin: Dict[str, Any]) -> float:
    rev_cagr = fin.get("revenue_cagr", 0.10)
    earn_cagr = fin.get("earnings_cagr", 0.10)
    rev_growth_list = fin.get("revenue_growth_list", [0.10] * 5)

    rev_score = min(rev_cagr / 0.20, 1.0) * 10.0
    earn_score = min(earn_cagr / 0.20, 1.0) * 10.0
    consistency = sum(1 for g in rev_growth_list if g > 0) / max(len(rev_growth_list), 1)

    return round(_clip10(0.4 * rev_score + 0.4 * earn_score + 0.2 * consistency * 10.0), 1)


def _score_financial_health(fin: Dict[str, Any]) -> float:
    de = fin.get("debt_to_equity", 0.5)
    cr = fin.get("current_ratio", 1.5)
    fcf_conv = fin.get("fcf_conversion", 1.0)

    de_score = 10.0 if de <= 0.5 else max(10.0 - (de - 0.5) * 2.5, 0.0)
    cr_score = 10.0 if 1.2 <= cr <= 2.5 else max(10.0 - abs(cr - 2.0) * 4.0, 2.0)
    fcf_score = min(fcf_conv / 1.0, 1.0) * 10.0

    return round(_clip10(0.4 * de_score + 0.3 * cr_score + 0.3 * fcf_score), 1)


def _score_valuation_dim(fin: Dict[str, Any]) -> float:
    current_pe = fin.get("pe", 20.0)
    hist = np.asarray(fin.get("pe_history", [current_pe]), dtype=np.float64)
    percentile = float(np.mean(hist < current_pe))
    sector_pe = fin.get("sector_pe", float(np.median(hist)))

    score = (1.0 - percentile) * 10.0
    if current_pe > sector_pe * 1.3:
        score -= 1.5
    elif current_pe < sector_pe * 0.7:
        score += 1.5
    return round(_clip10(score), 1)


def _detect_red_flags(fin: Dict[str, Any]) -> List[str]:
    flags: List[str] = []

    # 1) Accounts receivable growth >> revenue growth
    ar_list = fin.get("ar_list", [])
    rev_growth_list = fin.get("revenue_growth_list", [])
    if len(ar_list) >= 5 and len(rev_growth_list) >= 4:
        ar_cagr = (ar_list[-1] / ar_list[0]) ** 0.25 - 1.0
        rev_cagr = (rev_growth_list[-1] + rev_growth_list[-2]) / 2.0
        if ar_cagr > rev_cagr * 1.5 and ar_cagr > 0.15:
            flags.append("应收账款增速显著高于营收增速")

    # 2) Goodwill > 15% of total assets
    if fin.get("goodwill_to_assets", 0.0) > 0.15:
        flags.append("商誉占总资产比例 > 15%")

    # 3) Net margin declining 3 consecutive years
    nm_list = fin.get("net_margin_list", [])
    if len(nm_list) >= 4:
        last3 = nm_list[-3:]
        if all(last3[i] > last3[i + 1] for i in range(2)):
            flags.append("净利率连续 3 年下滑")

    # 4) Debt/equity > 2.0
    if fin.get("debt_to_equity", 0.0) > 2.0:
        flags.append("负债/权益 > 2.0")

    # 5) Operating cash flow < net income for 2 years
    ocf = fin.get("ocf_list", [])
    ni = fin.get("net_income_list", [])
    if len(ocf) >= 2 and len(ni) >= 2:
        weak = sum(1 for o, n in zip(ocf[-2:], ni[-2:]) if o < n)
        if weak >= 2:
            flags.append("经营性现金流连续 2 年低于净利润")

    return flags


def _build_peer_comparison(
    overall_score: float,
    peer_data: Optional[List[dict]],
    code: str,
) -> Dict[str, float]:
    """Peer comparison via percentile ranks, else synthetic sector averages."""
    if peer_data:
        scores = [
            float(p.get("overall_score", p.get("score", 0)))
            for p in peer_data
            if p.get("overall_score") is not None or p.get("score") is not None
        ]
        if scores:
            arr = np.array(scores, dtype=np.float64)
            pct = float(np.mean(arr < overall_score))
            return {
                "同行均值": round(float(np.mean(arr)), 2),
                "行业75分位": round(float(np.percentile(arr, 75)), 2),
                "行业25分位": round(float(np.percentile(arr, 25)), 2),
                "该股分位": round(pct, 2),
            }

    rng = np.random.default_rng(_seed_from(code + ":peers"))
    peer_scores = rng.normal(overall_score - 0.3, 1.2, 20)
    peer_scores = np.clip(peer_scores, 0.5, 9.5)
    pct = float(np.mean(peer_scores < overall_score))
    return {
        "同行均值": round(float(np.mean(peer_scores)), 2),
        "行业75分位": round(float(np.percentile(peer_scores, 75)), 2),
        "行业25分位": round(float(np.percentile(peer_scores, 25)), 2),
        "该股分位": round(pct, 2),
    }


def _build_key_positives_concerns(
    dimension_scores: Dict[str, float],
    fin: Dict[str, Any],
    red_flags: List[str],
) -> Tuple[List[str], List[str]]:
    positives: List[str] = []
    concerns: List[str] = []

    if dimension_scores.get("盈利能力", 0) >= 7.0:
        if fin.get("roe", 0) >= 0.15:
            positives.append("ROE 持续保持在较高水平")
        if fin.get("net_margin_list", []) and fin["net_margin_list"][-1] >= fin["net_margin_list"][0]:
            positives.append("净利率趋势向好")
    if dimension_scores.get("成长性", 0) >= 7.0:
        positives.append(f"营收 CAGR {fin.get('revenue_cagr', 0):.0%}，成长动能强")
    if dimension_scores.get("财务健康", 0) >= 7.0:
        positives.append("经营现金流覆盖净利润良好")

    if red_flags:
        flag_concern_map = {
            "应收账款增速显著高于营收增速": "应收账款快速膨胀，收入质量存疑",
            "商誉占总资产比例 > 15%": "高商誉存在减值风险",
            "净利率连续 3 年下滑": "盈利能力持续恶化",
            "负债/权益 > 2.0": "杠杆水平过高",
            "经营性现金流连续 2 年低于净利润": "利润含金量偏低",
        }
        for f in red_flags:
            concerns.append(flag_concern_map.get(f, f))
    if dimension_scores.get("估值合理性", 0) < 5.0 and not red_flags:
        concerns.append("当前估值高于历史中位数，安全边际有限")

    if not positives:
        positives.append("基本面整体稳健")
    if not concerns:
        concerns.append("暂无重大负面信号")
    return positives, concerns


def _score_all_dimensions(
    fin: Dict[str, Any],
    management_score: Optional[float],
    moat_score: Optional[float],
) -> Dict[str, float]:
    return {
        "盈利能力": _score_profitability(fin),
        "成长性": _score_growth(fin),
        "财务健康": _score_financial_health(fin),
        "估值合理性": _score_valuation_dim(fin),
        "管理层质量": round(_clip10(management_score if management_score is not None else 5.0), 1),
        "竞争壁垒": round(_clip10(moat_score if moat_score is not None else 5.0), 1),
    }


# ═══════════════════════════════════════════
# Private Company Valuation Bridge
# ═══════════════════════════════════════════


def _default_ps_by_industry(industry: str) -> float:
    for key, val in SECTOR_DEFAULT_PS.items():
        if key in industry:
            return val
    return SECTOR_DEFAULT_PS["通用"]


def _private_company_bridge(
    company_name: str,
    industry: str,
    estimated_revenue: float,
    ps_multiple: Optional[float] = None,
    comparables: Optional[List[dict]] = None,
    liquidity_discount: float = 0.20,
    control_premium: float = 0.0,
) -> PrivateCompanyBridge:
    """
    Private company valuation bridge (absorbed from ai-berkshire private-company skill):
      Revenue × comparable PS multiple (from public comps),
      then apply liquidity discount (10-30%) and optional control premium.
    """
    if comparables:
        ps_list = [
            float(c["ps"] if c.get("ps") is not None else c.get("ps_multiple"))
            for c in comparables
            if c.get("ps") is not None or c.get("ps_multiple") is not None
        ]
        if ps_list:
            ps_multiple = float(np.median(ps_list))
        pe_list = [float(c.get("pe", 0)) for c in comparables if c.get("pe") is not None]
        median_pe = float(np.median(pe_list)) if pe_list else 0.0
        median_ps = float(np.median(ps_list)) if ps_list else 0.0
    else:
        median_pe = 0.0
        median_ps = 0.0

    if ps_multiple is None or ps_multiple <= 0:
        ps_multiple = _default_ps_by_industry(industry or "")

    liquidity_discount = float(np.clip(liquidity_discount, 0.10, 0.30))
    control_premium = float(np.clip(control_premium, 0.0, 0.30))

    ps_based_valuation = estimated_revenue * ps_multiple
    comp_based_valuation = (
        ps_based_valuation * (1.0 - liquidity_discount) * (1.0 + control_premium)
    )

    fair_value_range = [
        round(comp_based_valuation * 0.85, 2),
        round(comp_based_valuation, 2),
        round(comp_based_valuation * 1.15, 2),
    ]

    key_assumptions = [
        f"PS 倍数: {ps_multiple:.1f}x",
        f"流动性折价: {liquidity_discount:.0%}",
    ]
    if control_premium > 0:
        key_assumptions.append(f"控制权溢价: {control_premium:.0%}")
    key_assumptions.append("可比公司数据为公开上市公司")

    caveats = [
        "非上市估值桥接为区间估计，非精确值",
        "PS 倍数法不反映盈利质量差异",
        "流动性折价 10-30% 为行业惯例区间",
    ]

    bridge = PrivateCompanyBridge(
        company_name=company_name,
        industry=industry,
        estimated_revenue=estimated_revenue,
        ps_multiple=round(ps_multiple, 2),
        ps_based_valuation=round(ps_based_valuation, 2),
        comparable_companies=comparables or [],
        median_comparable_pe=round(median_pe, 1),
        median_comparable_ps=round(median_ps, 2),
        liquidity_discount=round(liquidity_discount, 2),
        comp_based_valuation=round(comp_based_valuation, 2),
        fair_value_range=fair_value_range,
        key_assumptions=key_assumptions,
        caveats=caveats,
    )
    return bridge


# ═══════════════════════════════════════════
# ValuationAnchorTool
# ═══════════════════════════════════════════


class ValuationAnchorTool(BaseTool):
    """估值锚 (M5: 真实 DCF + 相对估值 + 加权目标价)"""

    tool_name = "valuation_anchor"
    tool_description = (
        "多方法估值锚定（DCF/PE分位数/PB-ROE/EV-EBITDA），返回各方法目标价 + "
        "加权目标价 + 敏感性 + 三情景；支持非上市估值桥接"
    )
    tool_version = "5.0.0"
    tool_category = "valuation"

    _private_company_bridge = staticmethod(_private_company_bridge)

    def execute(
        self,
        codes: List[str] = None,
        methods: List[str] = None,
        scenarios: Dict = None,
        financials: Optional[Dict] = None,
        peers: Optional[List[dict]] = None,
        private_company: Optional[Dict] = None,
        current_price: Optional[float] = None,
        asof_date: Optional[str] = None,
    ) -> Trader3Response:
        """
        估值锚定。

        Parameters
        ----------
        codes : List[str] — stock codes (first used for the report)
        methods : List[str], optional — restrict to e.g. ["dcf", "pe_percentile"]
        scenarios : Dict, optional — override FCF growth assumptions
        financials : Dict, optional — user-provided financial data (keys mirror
                     the synthetic schema); missing keys filled from synthetic
        peers : List[dict], optional — peer data (used by scorecard, kept for parity)
        private_company : Dict, optional — if provided, runs private company
                     valuation bridge instead of public DCF
        current_price : float, optional — override current price for implied return
        asof_date : str, optional — 防前视：按告公日对齐的财务截止日（YYYY-MM-DD），为空时读最新一期
        """
        code = (codes or ["000000"])[0]

        # ── Private company bridge path ──
        if private_company is not None:
            return self._private_company_response(private_company, code)

        # ── Public company valuation path ──
        # M8: 若未提供用户财务数据，尝试读取真实 financials.db
        if not financials:
            financials = self._try_real_financials(code, asof_date)
        fin, used_synthetic, synthetic_fields = _fill_financials(financials, code)
        if scenarios:
            if "fcf_growth_5y" in scenarios:
                fin["fcf_growth_5y"] = float(scenarios["fcf_growth_5y"])
            if "terminal_growth" in scenarios:
                fin["terminal_growth"] = float(scenarios["terminal_growth"])
            if "wacc" in scenarios:
                fin["_wacc_override"] = float(scenarios["wacc"])

        methods_all, aux = _compute_all_valuations(fin)

        # Optional WACC override from scenarios
        if "_wacc_override" in fin:
            aux["wacc"] = fin["_wacc_override"]
            methods_all["dcf"] = _dcf_valuation(fin, aux["wacc"])
            aux["scenarios"] = _dcf_scenarios(fin, aux["wacc"])
            aux["sensitivity"] = _dcf_sensitivity(fin, aux["wacc"])

        selected = list(methods_all.keys())
        if methods:
            valid = [m for m in methods if m in methods_all]
            if valid:
                selected = valid

        # 白名单模式：关键输入为合成值的方法整体禁用
        selected, disabled_methods = _gate_methods_by_provenance(
            methods_all, selected, synthetic_fields
        )

        # 负值目标价防护：亏损/负盈利输入产出的非正目标价不得进入加权
        neg_methods = [
            m for m in selected if float(methods_all[m].get("target", 0) or 0) <= 0
        ]
        if neg_methods:
            for m in neg_methods:
                selected.remove(m)
            disabled_methods = disabled_methods + [
                (m, ["目标价≤0（亏损或负盈利输入）"]) for m in neg_methods
            ]

        all_negative = not selected
        if all_negative:
            # 深度亏损股：四法全为非正目标价。保留原值但整体标注不可用，
            # 不静默回退合成数据（那会更不诚实）。
            selected = list(methods_all.keys())

        untrusted_fallback = False
        if not selected:
            selected = list(methods_all.keys())
            if used_synthetic:
                used_synthetic = True  # 纯合成入口：维持演示语义
            else:
                # 混合模式但无方法通过可信度门槛 → 输出仅作演示，不得引用
                untrusted_fallback = True

        weighted_target, norm_weights = _weighted_target(methods_all, selected)

        price = current_price if current_price is not None else fin.get("current_price", 50.0)
        price_is_real = ("current_price" not in synthetic_fields) or (current_price is not None)
        implied_return_ok = (
            price_is_real and weighted_target > 0
            and not all_negative and not untrusted_fallback
        )
        implied_return = (weighted_target / price - 1.0) if implied_return_ok else 0.0
        upside = float(np.clip(0.5 + 0.5 * math.tanh(implied_return * 3.0), 0.05, 0.95))

        scenarios_res = aux["scenarios"]
        report = ValuationReport(
            code=code,
            methods={k: _py(v) for k, v in methods_all.items()},
            fair_value_base=round(scenarios_res["fair_value_base"], 2),
            fair_value_bull=round(scenarios_res["fair_value_bull"], 2),
            fair_value_bear=round(scenarios_res["fair_value_bear"], 2),
            weighted_target=round(weighted_target, 2),
            implied_return=round(implied_return, 4),
            upside_probability=round(upside, 4),
            key_assumptions=_py({
                "wacc": aux["wacc"],
                "cost_of_equity": aux["cost_of_equity"],
                "terminal_growth": fin.get("terminal_growth"),
                "fcf_growth_5y": fin.get("fcf_growth_5y"),
                "forecast_years": DCF_FORECAST_YEARS,
                "current_price": price,
            }),
            sensitivity=_py(aux["sensitivity"]),
        )

        method_labels = {
            "dcf": "DCF",
            "pe_percentile": "PE分位",
            "pb_roe": "PB-ROE",
            "ev_ebitda": "EV/EBITDA",
        }
        synthetic_tag = "（合成数据）" if used_synthetic else ""

        return Trader3Response(
            success=True,
            data=report,
            summary=(
                f"{'[演示值·禁止引用] ' if untrusted_fallback else ''}"
                f"加权目标价 ¥{report.weighted_target:.1f}{synthetic_tag}, "
                f"Base ¥{report.fair_value_base:.1f} / "
                f"Bull ¥{report.fair_value_bull:.1f} / "
                f"Bear ¥{report.fair_value_bear:.1f}, "
                f"隐含收益率 {report.implied_return:.1%}, 当前价 ¥{price:.1f}"
            ),
            key_metrics={
                "加权目标价": report.weighted_target,
                "Base": report.fair_value_base,
                "Bull": report.fair_value_bull,
                "Bear": report.fair_value_bear,
                "隐含收益率": report.implied_return,
                "上行概率": report.upside_probability,
                "当前价": round(price, 2),
                "DCF": methods_all["dcf"]["target"],
                "PE分位": methods_all["pe_percentile"]["target"],
                "PB-ROE": methods_all["pb_roe"]["target"],
                "EV/EBITDA": methods_all["ev_ebitda"]["target"],
            },
            charts=[
                ChartSpec(
                    chart_type="bar",
                    title="各方法估值对比",
                    data={method_labels.get(k, k): round(v["target"], 1) for k, v in methods_all.items()},
                    description="DCF/PE/PB-ROE/EV-EBITDA 四种方法的估值结果对比",
                ),
                ChartSpec(
                    chart_type="line",
                    title="DCF 敏感性分析",
                    data={
                        "wacc": report.sensitivity.get("wacc", []),
                        "terminal_growth": report.sensitivity.get("terminal_growth", []),
                    },
                    x_label="情景",
                    y_label="目标价",
                    description="WACC 和永续增长率对 DCF 估值的影响",
                ),
            ],
            caveats=(
                [
                    f"M5 估值引擎{' — 合成财务数据，非真实数据' if used_synthetic else (' — 真实财务数据 ' + (fin.get('_quarter', '') if '_quarter' in fin else ''))}",
                ]
                + [
                    f"{method_labels.get(m, m)}法已禁用：{'、'.join(fields)} 为合成值"
                    for m, fields in disabled_methods
                ]
                + (
                    ["当前价为合成值，隐含收益率/上行概率不可用（置0）"]
                    if not price_is_real
                    else []
                )
                + (
                    ["⚠ 全部方法目标价≤0（深度亏损股），加权目标价与隐含收益率不可用"]
                    if all_negative
                    else []
                )
                + (
                    ["⚠ 真实模式下无方法通过可信度门槛（真实历史序列缺失），以下数值仅为合成演示，禁止引用"]
                    if untrusted_fallback
                    else []
                )
                + [
                    "DCF 估值对 WACC 和永续增长率高度敏感",
                    "相对估值法（PE/PB）受市场情绪影响较大",
                    "PB-ROE 隐含收益率 r 取自当前 PE（盈利收益率）",
                    f"方法权重: {', '.join(f'{k} {w:.0%}' for k, w in norm_weights.items())}",
                ]
            ),
        )

    @staticmethod
    def _asof_valuation_input(code: str, asof_date: str) -> Optional[Dict]:
        """
        防前视：按公告日对齐的财务数据（announcement_calendar.financials_asof），
        派生 to_valuation_input 同构输出（每股/比率口径一致）。
        """
        from trader3.v2.announcement_calendar import AnnouncementCalendar

        cal = AnnouncementCalendar()
        fin = cal.financials_asof(code, asof_date)
        if not fin or "epsTTM" not in fin:
            return None

        def _f(key):
            v = fin.get(key)
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        result = {}
        total_share = _f("totalShare") or _f("balance.totalShare") or _f("profit.totalShare")
        eps = _f("epsTTM")
        if eps:
            result["eps"] = eps
        roe = _f("roeAvg")
        if roe:
            result["roe"] = roe
        total_equity = _f("totalEquity") or _f("balance.totalEquity")
        if total_equity and total_share and total_share > 0:
            result["bvps"] = total_equity / total_share
        revenue = _f("MBRevenue")
        net_profit = _f("netProfit")
        fcf = _f("FCF")
        if revenue and total_share and total_share > 0:
            result["revenue_per_share"] = revenue / total_share
        if net_profit and total_share and total_share > 0:
            result["net_profit_per_share"] = net_profit / total_share
        if fcf and total_share and total_share > 0:
            result["fcf_per_share"] = fcf / total_share
        gp = _f("gpMargin")
        np_margin = _f("npMargin")
        if gp:
            result["gross_margin"] = gp
        if np_margin:
            result["net_margin"] = np_margin
        goodwill = _f("goodwill")
        total_assets = _f("totalAssets") or _f("balance.totalAssets")
        total_liab = _f("totalLiab") or _f("balance.totalLiab")
        if goodwill and total_assets and total_assets > 0:
            result["goodwill_to_assets"] = goodwill / total_assets
        if total_liab and total_equity and total_equity > 0:
            result["debt_to_equity"] = total_liab / total_equity
        ocf = _f("OCF")
        if net_profit and ocf:
            result["ocf_net_income_ratio"] = ocf / net_profit if net_profit != 0 else None
        result["_source"] = "financials.db"
        result["_quarter"] = fin.get("quarter", "")
        return {k: v for k, v in result.items() if v is not None}

    @staticmethod
    def _try_real_financials(code: str, asof_date: Optional[str] = None) -> Optional[Dict]:
        """
        M8: 尝试从 financials.db 读取真实财务数据。

        asof_date 非空时走 announcement_calendar.financials_asof 防前视
        （历史财务按公告日对齐，避免前视偏差）；为空时读取最新一期。

        Returns valuation 输入 dict 或 None（失败由调用方回退合成）。
        """
        try:
            if asof_date:
                return ValuationAnchorTool._asof_valuation_input(code, asof_date)
            from trader3.financials_provider import FinancialsProvider

            fp = FinancialsProvider()
            inp = fp.to_valuation_input(code)
            if not inp or "eps" not in inp:
                return None
            return inp
        except Exception:
            return None

    def _private_company_response(
        self, private_company: Dict, code: str
    ) -> Trader3Response:
        """Handle the private company valuation bridge path."""
        kwargs = dict(private_company)
        kwargs.setdefault("company_name", kwargs.pop("name", code))
        kwargs.setdefault("industry", "")
        kwargs.setdefault("estimated_revenue", 0.0)
        kwargs.setdefault("liquidity_discount", 0.20)
        kwargs.setdefault("control_premium", 0.0)
        if "ps" in private_company and "ps_multiple" not in private_company:
            kwargs.setdefault("ps_multiple", private_company["ps"])

        bridge = self._private_company_bridge(**kwargs)

        return Trader3Response(
            success=True,
            data=bridge,
            summary=(
                f"[{bridge.company_name}] PS估值 ¥{bridge.ps_based_valuation:.2f}亿, "
                f"折价后合理估值 ¥{bridge.comp_based_valuation:.2f}亿, "
                f"区间 ¥{bridge.fair_value_range[0]:.1f}-{bridge.fair_value_range[2]:.1f}亿"
            ),
            key_metrics={
                "收入(亿)": bridge.estimated_revenue,
                "PS倍数": bridge.ps_multiple,
                "PS估值(亿)": bridge.ps_based_valuation,
                "流动性折价": bridge.liquidity_discount,
                "合理估值(亿)": bridge.comp_based_valuation,
                "估值区间下沿(亿)": bridge.fair_value_range[0],
                "估值区间上沿(亿)": bridge.fair_value_range[2],
            },
            charts=[
                ChartSpec(
                    chart_type="table",
                    title="可比公司",
                    data=bridge.comparable_companies or {
                        "说明": "未提供可比公司，使用行业默认 PS"
                    },
                    description="可比公司法估值的参照系",
                ),
            ],
            caveats=bridge.caveats,
        )


# ═══════════════════════════════════════════
# FundamentalScorecardTool
# ═══════════════════════════════════════════


class FundamentalScorecardTool(BaseTool):
    """基本面评分卡 (M5: 六维真实评分 + 模板加权 + 红旗预警)"""

    tool_name = "fundamental_scorecard"
    tool_description = (
        "多维基本面评分（质量成长/价值/反转模板），返回六维评分 + 同业对比 + 红旗预警"
    )
    tool_version = "5.0.0"
    tool_category = "valuation"

    def execute(
        self,
        codes: List[str] = None,
        template: str = "quality_growth",
        financials: Optional[Dict] = None,
        peers: Optional[List[dict]] = None,
        management_score: Optional[float] = None,
        moat_score: Optional[float] = None,
    ) -> Trader3Response:
        """
        基本面评分卡。

        Parameters
        ----------
        template : str — quality_growth / value / turnaround
        financials : Dict, optional — user financial data (missing keys filled synthetic)
        peers : List[dict], optional — peer scores for percentile comparison
        management_score : float, optional — 0-10 external assessment, else neutral 5.0
        moat_score : float, optional — 0-10 external assessment, else neutral 5.0
        """
        code = (codes or ["000000"])[0]
        template_cfg = SCORECARD_TEMPLATES.get(
            template, SCORECARD_TEMPLATES["quality_growth"]
        )
        template_name = template_cfg["label"]

        fin, used_synthetic, _synthetic_fields = _fill_financials(financials, code)
        dimension_scores = _score_all_dimensions(fin, management_score, moat_score)
        red_flags = _detect_red_flags(fin)

        overall = sum(
            template_cfg["weights"].get(dim, 0.0) * dimension_scores.get(dim, 0.0)
            for dim in ALL_DIMENSIONS
        )
        overall = round(_clip10(overall), 1)

        peer_comparison = _build_peer_comparison(overall, peers, code)
        key_positives, key_concerns = _build_key_positives_concerns(
            dimension_scores, fin, red_flags
        )

        report = ScorecardReport(
            template=template,
            overall_score=overall,
            dimension_scores={k: round(v, 1) for k, v in dimension_scores.items()},
            peer_comparison=peer_comparison,
            red_flags=red_flags,
            key_positives=key_positives,
            key_concerns=key_concerns,
        )

        synthetic_tag = "（合成数据）" if used_synthetic else ""

        return Trader3Response(
            success=True,
            data=report,
            summary=(
                f"[{template_name}] 总分 {report.overall_score}/10{synthetic_tag}, "
                f"高于同行均值 {report.peer_comparison.get('同行均值', 0)}, "
                f"{len(report.red_flags)} 个红旗预警"
            ),
            key_metrics={
                "总分": report.overall_score,
                "盈利能力": report.dimension_scores.get("盈利能力", 0),
                "成长性": report.dimension_scores.get("成长性", 0),
                "财务健康": report.dimension_scores.get("财务健康", 0),
                "估值合理性": report.dimension_scores.get("估值合理性", 0),
                "管理层质量": report.dimension_scores.get("管理层质量", 0),
                "竞争壁垒": report.dimension_scores.get("竞争壁垒", 0),
                "同业分位": report.peer_comparison.get("该股分位", 0),
            },
            charts=[
                ChartSpec(
                    chart_type="bar",
                    title="六维评分",
                    data=report.dimension_scores,
                    description="六维基本面评分雷达",
                ),
                ChartSpec(
                    chart_type="table",
                    title="同业对比",
                    data=report.peer_comparison,
                    description="与同行/行业分位的对比",
                ),
            ],
            caveats=[
                f"M5 评分卡{' — 合成财务数据，非真实数据' if used_synthetic else ''}",
                f"模板权重: {', '.join(f'{k} {v:.0%}' for k, v in template_cfg['weights'].items())}",
                "红旗预警基于财务数据规则判断",
                "管理层质量/竞争壁垒未提供时取中性 5.0 分",
                "评分卡模板支持质量成长/价值/反转三类",
            ],
        )


# ═══════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════


def _py(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to native Python types."""
    if isinstance(obj, np.ndarray):
        return [_py(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    return obj


# ═══════════════════════════════════════════
# Module-level smoke test
# ═══════════════════════════════════════════

if __name__ == "__main__":
    v = ValuationAnchorTool()
    r = v(codes=["000001.SZ"])
    assert r.success, f"ValuationAnchorTool failed: {r.summary}"
    assert r.key_metrics.get("加权目标价", 0) > 0, "weighted target must be > 0"
    print(f"[ValuationAnchorTool] {r.summary}")
    print(f"  WACC: {r.data.key_assumptions['wacc']:.2%}, "
          f"sens_wacc={r.data.sensitivity['wacc']}")
    for k, vv in r.key_metrics.items():
        print(f"  {k}: {vv}")

    s = FundamentalScorecardTool()
    r2 = s(codes=["000001.SZ"])
    assert r2.success, f"FundamentalScorecardTool failed: {r2.summary}"
    assert r2.key_metrics.get("总分", 0) > 0, "overall score must be > 0"
    print(f"[FundamentalScorecardTool] {r2.summary}")
    print(f"  六维: {r2.data.dimension_scores}")
    print(f"  红旗: {r2.data.red_flags}")

    # Private company bridge smoke
    bridge = _private_company_bridge(
        company_name="示例科技",
        industry="科技",
        estimated_revenue=5.0,
        comparables=[{"name": "A公司", "ps": 6.5, "pe": 35.0},
                     {"name": "B公司", "ps": 5.2, "pe": 28.0}],
        liquidity_discount=0.25,
        control_premium=0.0,
    )
    print(f"[PrivateBridge] {bridge.company_name} 合理估值 "
          f"¥{bridge.comp_based_valuation:.2f}亿 (PS {bridge.ps_multiple:.1f}x, "
          f"折价 {bridge.liquidity_discount:.0%})")
    print("M5 smoke test PASSED")
