#!/usr/bin/env python3
"""risk_model / regime / ensemble / stress 模块测试"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader3.tools.stress import (
    compute_tail_metrics,
    factor_stress,
    historical_var,
    monte_carlo_var,
    parametric_var,
    run_stress_suite,
)
from trader3.v2.ensemble import build_ensemble, rank_ic_series
from trader3.v2.regime import GaussianHMM, current_regime, detect_regimes
from trader3.v2.risk_model import (
    FactorRiskModel,
    RiskModelConfig,
)


def make_returns(n=750, seed=7):
    rng = np.random.default_rng(seed)
    # 两个 regime：低波动牛市 + 高波动熊市
    bull = rng.normal(0.0008, 0.008, n // 2)
    bear = rng.normal(-0.0015, 0.022, n - n // 2)
    return pd.Series(np.concatenate([bear, bull]),
                     index=pd.bdate_range("2024-01-02", periods=n))


def make_panel(n_days=300, n_assets=30, seed=11):
    rng = np.random.default_rng(seed)
    assets = [f"S{i:03d}" for i in range(n_assets)]
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    mkt = rng.normal(0.0004, 0.01, n_days)
    beta = rng.uniform(0.6, 1.5, n_assets)
    idio = rng.normal(0, 0.008, (n_days, n_assets))
    rets = mkt[:, None] * beta[None, :] + idio + rng.normal(0.0003, 0.002)
    close = 100 * np.exp(np.cumsum(rets, axis=0))
    vol = rng.lognormal(mean=14, sigma=0.5, size=(n_days, n_assets))
    panel = pd.concat(
        {"close": pd.DataFrame(close, index=dates, columns=assets),
         "volume": pd.DataFrame(vol, index=dates, columns=assets)},
        axis=1,
    )
    return panel


# ================= risk_model =================

class TestRiskModel:
    def test_style_exposures_shape(self):
        panel = make_panel()
        exp = FactorRiskModel.build_style_exposures(panel)
        assert exp.shape[0] == 30
        for col in ["size", "value", "momentum", "volatility", "beta", "liquidity"]:
            assert col in exp.columns

    def test_fit_factor_returns_and_cov(self):
        panel = make_panel()
        close = panel.xs("close", axis=1, level=0)
        rets = np.log(close).diff().dropna()
        exp = FactorRiskModel.build_style_exposures(panel)
        model = FactorRiskModel(RiskModelConfig())
        fr = model.fit_factor_returns(rets.tail(200), exp)
        assert fr.shape[1] == 6 and len(fr) > 100
        cov = model.fit_factor_covariance()
        assert cov.shape == (6, 6)
        vals = np.linalg.eigvalsh(cov.values)
        assert vals.min() > -1e-10, "协方差矩阵应近似半正定"

    def test_portfolio_decomposition(self):
        panel = make_panel()
        close = panel.xs("close", axis=1, level=0)
        rets = np.log(close).diff().dropna()
        exp = FactorRiskModel.build_style_exposures(panel)
        model = FactorRiskModel()
        model.fit_factor_returns(rets.tail(200), exp)
        model.fit_factor_covariance()
        holdings = pd.Series(np.full(30, 1 / 30), index=exp.index)
        dec = model.decompose_portfolio(holdings)
        assert dec.total_risk >= 0
        assert abs(dec.systematic_risk**2 + dec.specific_risk**2 - dec.total_risk**2) < 1e-6
        assert set(dec.factor_contributions.keys()) >= {"size", "momentum"}


# ================= regime =================

class TestRegime:
    def test_detect_two_states(self):
        r = make_returns(600)
        res = detect_regimes(r.values, n_states=2, random_state=1)
        assert len(res.states) == 600
        assert res.state_probs.shape == (600, 2)
        assert np.allclose(res.transition.sum(axis=1), 1, atol=1e-6)

    def test_higher_vol_state_detected(self):
        r = make_returns(800, seed=3)
        res = detect_regimes(r.values, n_states=2, random_state=2)
        hi = max(res.labels_by_vol, key=lambda k: res.vols[k])
        lo = min(res.labels_by_vol, key=lambda k: res.vols[k])
        assert res.labels_by_vol[hi] == "turbulent"
        assert res.labels_by_vol[lo] == "calm"

    def test_current_regime_confidence(self):
        r = make_returns(500)
        res = detect_regimes(r.values, n_states=2, random_state=5)
        label, conf = current_regime(res)
        assert isinstance(label, str) and 0.0 <= conf <= 1.0

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            GaussianHMM(n_states=3).fit(np.zeros(15))

    def test_stationary_distribution(self):
        r = make_returns(400)
        res = detect_regimes(r.values, n_states=2, random_state=9)
        assert abs(res.stationary.sum() - 1) < 1e-6


# ================= ensemble =================

class TestEnsemble:
    def _make_features(self, n_days=420, n_assets=25, seed=21):
        rng = np.random.default_rng(seed)
        assets = [f"A{i:02d}" for i in range(n_assets)]
        dates = pd.bdate_range("2025-01-02", periods=n_days)
        f1 = pd.DataFrame(rng.normal(size=(n_days, n_assets)), index=dates, columns=assets)
        f2 = f1 * 0.6 + pd.DataFrame(rng.normal(size=(n_days, n_assets)), index=dates, columns=assets) * 0.8
        fwd = (0.01 * f1.shift(-1) + 0.004 * f2.shift(-1)).fillna(
            pd.DataFrame(rng.normal(0, 0.01, (n_days, n_assets)), index=dates, columns=assets))
        return {"f1": f1, "f2": f2}, fwd

    def test_build_ensemble_basic(self):
        feats, fwd = self._make_features()
        cfg_cfg = None
        try:
            from trader3.v2.ensemble import EnsembleConfig
            cfg_cfg = EnsembleConfig(min_train=200, retrain_every=90)
        except ImportError:
            pass
        res = build_ensemble(feats, fwd, config=cfg_cfg)
        assert not res.scores.empty
        assert res.model_used != ""
        assert 0 <= len(res.feature_importance) <= 2 or not res.used_ml

    def test_rank_ic_series(self):
        rng = np.random.default_rng(5)
        s = pd.DataFrame(rng.normal(size=(60, 20)))
        fwd = s * 0.5 + pd.DataFrame(rng.normal(size=(60, 20)))
        ic = rank_ic_series(s, fwd)
        assert len(ic) == 60
        assert ic.mean() > 0.3, "强信号 RankIC 应显著为正"

    def test_available_models(self):
        from trader3.v2.ensemble import available_models
        models = available_models()
        assert "ridge" in models and len(models) >= 2


# ================= stress =================

class TestStress:
    def test_tail_metrics(self):
        r = make_returns(500)
        tm = compute_tail_metrics(r)
        assert 0 < tm.var_95 < tm.var_99
        assert 0 < tm.cvar_95 < tm.cvar_99 or tm.cvar_95 > 0
        assert -1 < tm.max_drawdown < 0
        assert tm.worst_day < 0 < tm.best_day

    def test_historical_vs_parametric(self):
        # 熊市序列：均值<0，参数法 VaR 应为正
        r = pd.Series(np.random.default_rng(4).normal(-0.002, 0.02, 600))
        hv99 = historical_var(r, 0.99)
        pv99, pes99 = parametric_var(r, 0.99, method="cornish_fisher")
        nv99, nes99 = parametric_var(r, 0.99, method="normal")
        assert hv99 > 0
        assert all(v > 0 for v in [pv99, nv99])
        assert pes99 > nes99 - 1e-9

    def test_monte_carlo_var(self):
        r = make_returns(500)
        mv, mc = monte_carlo_var(r, confidence=0.99, n_sims=20000)
        assert mv > 0 and mc >= mv

    def test_stress_suite_ordering(self):
        r = make_returns(400)
        results = run_stress_suite(r)
        assert len(results) >= 9
        pnls = [x.portfolio_pnl_pct for x in results]
        assert pnls == sorted(pnls), "结果应按损失从大到小排序"
        assert all(x.portfolio_pnl_pct < 0 for x in results)

    def test_factor_stress_quantiles(self):
        cov = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.09]], index=["f1", "f2"], columns=["f1", "f2"]
        )
        expo = pd.Series({"f1": -0.8, "f2": -0.3})  # 负暴露 + 正冲击 → 亏损
        out = factor_stress(expo, cov, factor_shocks_z={"f1": 2.0}, n_mc=10000)
        assert out["p99_loss"] >= out["p95_loss"]
        assert out["p99_loss"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
