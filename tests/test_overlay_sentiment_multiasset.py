#!/usr/bin/env python3
"""risk_overlay / sentiment / multi_asset 模块测试"""

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, r"D:\Claude\projects\3号交易员")

from trader3.v2.multi_asset import (
    AssetType,
    get_spec,
    inverse_vol_weights,
    portfolio_margin,
    position_sizing,
)
from trader3.v2.risk_overlay import (
    OverlayConfig,
    apply_to_trade_pct,
    compute_risk_overlay,
    summarize,
)
from trader3.v2.sentiment import (
    SentimentStore,
    aggregate_daily,
    pipeline_from_sources,
    score_text,
    zscore_cross_section,
)


def make_returns(n=400, seed=7, vol=0.01):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0003, vol, n),
                     index=pd.bdate_range("2025-01-02", periods=n))


# ================= risk_overlay =================

class TestRiskOverlay:
    def test_all_pass_normal_market(self):
        r = make_returns(vol=0.008)
        res = compute_risk_overlay(r, regime_label="calm", regime_confidence=0.9,
                                   worst_scenario_pnl=-0.05)
        assert 0.8 <= res.size_multiplier <= 1.0
        assert not res.halt_new_buys

    def test_turbulent_regime_cuts_size(self):
        r = make_returns()
        base = compute_risk_overlay(r, regime_label="calm", regime_confidence=0.9,
                                    worst_scenario_pnl=-0.05)
        turb = compute_risk_overlay(r, regime_label="turbulent",
                                    regime_confidence=0.9, worst_scenario_pnl=-0.05)
        assert turb.size_multiplier < base.size_multiplier
        assert abs(turb.size_multiplier - 0.5 * base.size_multiplier) < 1e-9

    def test_low_confidence_ignored(self):
        r = make_returns()
        res = compute_risk_overlay(r, regime_label="turbulent", regime_confidence=0.3,
                                   worst_scenario_pnl=-0.05)
        assert res.size_multiplier > 0.9, "低置信度不应触发减仓"

    def test_high_vol_cuts_to_floor(self):
        r = make_returns(vol=0.06)   # 年化 ~95% 远超目标20%
        res = compute_risk_overlay(r, regime_label="calm", regime_confidence=0.9,
                                   worst_scenario_pnl=-0.05)
        assert res.size_multiplier <= OverlayConfig().max_vol_multiplier_cut + 1e-9

    def test_var_breach(self):
        r = make_returns(vol=0.03)   # 日VaR95 ≈ 4.9% > 3%
        OverlayConfig()
        res = compute_risk_overlay(r, regime_label="calm", regime_confidence=0.9,
                                   worst_scenario_pnl=-0.05)
        assert "var" in res.gates and ">" in res.gates["var"]

    def test_worst_scenario_triggers_stress_gate(self):
        r = make_returns()
        res = compute_risk_overlay(r, regime_label="calm", regime_confidence=0.9,
                                   worst_scenario_pnl=-0.45)
        assert res.size_multiplier <= 0.6 * 1.001
        assert "stress" in res.gates and "<" in res.gates["stress"]

    def test_circuit_breaker_halt(self):
        r = make_returns(vol=0.08)
        res = compute_risk_overlay(
            r, regime_label="turbulent", regime_confidence=0.95,
            worst_scenario_pnl=-0.50,
            config=OverlayConfig(stress_pnl_floor=-0.10))
        if res.size_multiplier <= 0.25:
            assert res.halt_new_buys

    def test_fail_open_on_empty_returns(self):
        res = compute_risk_overlay(pd.Series(dtype=float),
                                   regime_label=None, regime_confidence=None,
                                   worst_scenario_pnl=None)
        assert res.size_multiplier == 1.0
        assert not res.halt_new_buys

    def test_disabled(self):
        res = compute_risk_overlay(make_returns(), config=OverlayConfig(enabled=False))
        assert res.size_multiplier == 1.0 and res.gates.get("overlay") == "disabled"

    def test_apply_and_summarize(self):
        ov = compute_risk_overlay(make_returns(), regime_label="turbulent",
                                  regime_confidence=0.9, worst_scenario_pnl=-0.45)
        pct = apply_to_trade_pct(0.10, ov)
        assert pct <= 0.10 * ov.size_multiplier + 1e-12
        s = summarize(ov)
        assert "仓位乘数" in s and len(s.splitlines()) >= 2

    def test_live_regime_path(self):
        # 不传 label/conf → 现场跑 HMM（>=100 样本）
        r = make_returns(300)
        res = compute_risk_overlay(r)
        assert "regime" in res.gates


# ================= sentiment =================

class TestSentiment:
    def test_positive_negative_neutral(self):
        assert score_text("公司业绩超预期，订单大增").score > 0.5
        assert score_text("公司被立案调查，股价暴跌").score < -0.5
        assert abs(score_text("今天天气不错，公司发布公告").score) < 0.6

    def test_negation_flips(self):
        pos = score_text("业绩增长").score
        neg = score_text("业绩不增长").score
        assert pos > 0 > neg

    def test_intensifier_amplifies(self):
        a = score_text("利润增长").score
        b = score_text("利润大幅增长").score
        assert abs(b) > abs(a)

    def test_score_clamped(self):
        s = score_text("涨停 涨停 利好 利好 超预期 中标 获批")
        assert -3.0 <= s.score <= 3.0

    def test_aggregate_half_life_weighting(self):
        now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        old = (pd.Timestamp.now() - pd.Timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        out = aggregate_daily({"600519": [(old, 3.0), (now, -1.0)]},
                              half_life_hours=24)
        assert out["600519"] < 0, "新近负面应主导旧正面"

    def test_store_roundtrip_and_window(self, tmp_path):
        st = SentimentStore(state_dir=str(tmp_path))
        st.write({"600519": 1.2, "000858": -0.5}, date="20260826")
        got = st.read("20260826")
        assert got == {"600519": 1.2, "000858": -0.5}
        frame = st.window_frame(days=5)
        assert "600519" in frame.columns or frame.empty is False

    def test_pipeline_from_sources(self, tmp_path):
        class Item:
            def __init__(self, t, ts):
                self.title, self.content, self.ts = t, "", ts
        items = {"600519": [Item("重大利好：中标大单", "2026-08-26 09:00:00"),
                            Item("回购增持", "2026-08-26 10:00:00")],
                 "000858": [Item("立案调查 利空", "2026-08-26 11:00:00")]}
        scores = pipeline_from_sources(items, store=SentimentStore(str(tmp_path)))
        assert scores["600519"] > scores["000858"]

    def test_zscore_cross_section(self):
        s = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 10.0})
        z = zscore_cross_section(s)
        assert abs(z.mean()) < 1e-9 and z.abs().max() <= 2.5 + 1e-9


# ================= multi_asset =================

class TestMultiAsset:
    def test_stock_spec_defaults(self):
        spec = get_spec("600519")
        assert spec.asset_type == AssetType.STOCK
        assert spec.lot_size == 100 and spec.t_plus == 1

    def test_futures_specs(self):
        for sym, mult in [("IF", 300), ("IC", 200), ("IM", 200)]:
            spec = get_spec(sym)
            assert spec.multiplier == mult
            assert spec.margin_rate < 1.0
            assert spec.notional(4000.0, 2) == mult * 4000 * 2

    def test_contract_month_symbol(self):
        spec = get_spec("IF2609")
        assert spec.multiplier == 300, "IF2609 应继承 IF 规格"

    def test_position_sizing_basic(self):
        spec = get_spec("600519")
        plan = position_sizing(spec, target_notional=200_000, price=1000.0,
                               available_cash=500_000)
        assert plan.lots == 2                      # 200000 / (1000×100) = 2手
        assert plan.margin == plan.notional        # 现货保证金=名义

    def test_position_sizing_cash_cap(self):
        spec = get_spec("600519")
        plan = position_sizing(spec, target_notional=500_000, price=1000.0,
                               available_cash=80_000)
        assert plan.lots == 0 or "cash-capped" in plan.reason or plan.lots * 100 * 1000 <= 80_000 + 1e-6

    def test_position_sizing_insufficient(self):
        spec = get_spec("600519")
        plan = position_sizing(spec, target_notional=200_000, price=1000.0,
                               available_cash=5_000)
        assert plan.lots == 0 and plan.reason

    def test_futures_margin_check(self):
        spec = get_spec("IF")
        plan = position_sizing(spec, target_notional=1_800_000, price=4000.0,
                               available_cash=1_000_000)
        # IF 一手名义 = 4000×300 = 120万，保证金 12% = 14.4万
        assert plan.lots == 1
        assert plan.margin == pytest.approx(144_000)

    def test_portfolio_margin(self):
        pm = portfolio_margin([("600519", 1500.0, 2), ("IF", 4000.0, 1)], equity=2_000_000)
        assert pm.total_margin == pytest.approx(300_000 + 144_000)
        assert 0 < pm.gross_exposure_ratio < 2
        assert set(pm.by_symbol) == {"600519", "IF"}

    def test_inverse_vol_weights(self):
        vols = pd.Series({"stock": 0.20, "bond": 0.04, "gold": 0.12})
        w = inverse_vol_weights(vols)
        assert w.sum() == pytest.approx(1.0)
        assert w["bond"] > w["stock"], "低波动资产权重应更高"
        assert all(w >= 0.05 - 1e-9)

    def test_inverse_vol_bad_inputs(self):
        w = inverse_vol_weights(pd.Series({"a": 0.0, "b": np.nan}))
        assert w.sum() == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
