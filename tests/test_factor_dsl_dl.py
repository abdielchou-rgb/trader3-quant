"""Factor DSL 与深度学习基线测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader3.v2.ensemble import EnsembleConfig, available_models, build_ensemble
from trader3.v2.factor_dsl import (
    FactorDSL,
    list_available_factors,
    validate_expression,
)


def _make_panel(n_dates=60, n_assets=5, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    assets = [f"A{i}" for i in range(n_assets)]
    fields = ["open", "high", "low", "close", "volume", "vwap", "amount"]
    cols = pd.MultiIndex.from_product([assets, fields])
    data = rng.normal(size=(n_dates, len(cols)))
    df = pd.DataFrame(data, index=dates, columns=cols)
    # 让 close 单调偏随机游走
    for a in assets:
        walk = np.cumsum(rng.normal(0, 1, n_dates))
        df[(a, "close")] = walk + 100
        df[(a, "open")] = df[(a, "close")] - rng.normal(0, 0.5, n_dates)
        df[(a, "high")] = df[(a, "close")].max() + abs(rng.normal(0, 1, n_dates))
        df[(a, "low")] = df[(a, "close")].min() - abs(rng.normal(0, 1, n_dates))
        df[(a, "vwap")] = df[(a, "close")]
    return df


def test_dsl_compile_gp_expr():
    dsl = FactorDSL()
    cf = dsl.compile("add(log(close), delay(close, 5))")
    assert cf.is_gp
    panel = _make_panel()
    scores = cf.func(panel)
    assert isinstance(scores, pd.Series)
    assert len(scores) == 5


def test_dsl_validate_expressions():
    ok, err = validate_expression("mul(close, volume)")
    assert ok and err is None
    ok2, err2 = validate_expression("not_a_func(close)")
    assert not ok2 and err2 is not None


def test_dsl_list_available():
    info = list_available_factors()
    assert "close" in info["fields"]
    assert "ridge" in available_models() or "lgbm" in available_models()


def test_dsl_compile_cache():
    dsl = FactorDSL()
    a = dsl.compile("rank(close)")
    b = dsl.compile("rank(close)")
    assert a is b  # 同一对象（LRU缓存）


def test_ensemble_dl_baseline_lstm():
    panel = _make_panel(n_dates=120)
    fwd = panel.xs("close", axis=1, level=1).pct_change(5).shift(-5)
    feats = {
        "mom": panel.xs("close", axis=1, level=1).pct_change(10),
        "vol": panel.xs("volume", axis=1, level=1).pct_change(5),
    }
    feats = {k: v.div(v.std().replace(0, np.nan)).fillna(0) for k, v in feats.items()}
    cfg = EnsembleConfig(model="lstm", min_train=60, retrain_every=30)
    res = build_ensemble(feats, fwd, config=cfg, seed=1)
    assert hasattr(res, "scores")
    assert res.model_used in ("lstm", "equal_weight_baseline")


def test_ensemble_dl_baseline_transformer():
    panel = _make_panel(n_dates=120)
    fwd = panel.xs("close", axis=1, level=1).pct_change(5).shift(-5)
    feats = {
        "mom": panel.xs("close", axis=1, level=1).pct_change(10),
        "vol": panel.xs("volume", axis=1, level=1).pct_change(5),
    }
    feats = {k: v.div(v.std().replace(0, np.nan)).fillna(0) for k, v in feats.items()}
    cfg = EnsembleConfig(model="transformer", min_train=60)
    res = build_ensemble(feats, fwd, config=cfg, seed=2)
    assert res is not None


def test_ctp_broker_lifecycle():
    from trader3.v2.live.ctp_broker import ConnState, CTPBroker
    b = CTPBroker({"ctp_user_id": "test", "ctp_investor_id": "test"})
    import asyncio
    assert asyncio.run(b.connect())
    assert b.state == ConnState.LOGGED_IN
    acct = asyncio.run(b.get_account())
    assert acct is not None
    assert asyncio.run(b.disconnect())
    assert b.state == ConnState.DISCONNECTED
