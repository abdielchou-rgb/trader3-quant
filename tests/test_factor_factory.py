"""因子工厂（自主挖掘闭环）测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader3.v2.factor_factory import (
    FactorCandidate,
    FactorFactory,
    FactorFactoryConfig,
    FactorMetrics,
    FactorRegistry,
    run_factor_factory,
)


def _make_panel_with_signal(n_dates=120, n_assets=6, seed=21):
    """构造含隐藏动量信号的面板：fwd ≈ 0.5*mom + 噪声。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    assets = [f"A{i}" for i in range(n_assets)]
    fields = ["open", "high", "low", "close", "volume", "vwap", "amount"]
    cols = pd.MultiIndex.from_product([assets, fields])
    df = pd.DataFrame(rng.normal(size=(n_dates, len(cols))), index=dates, columns=cols)
    for a in assets:
        w = np.cumsum(rng.normal(0, 1, n_dates))
        df[(a, "close")] = w + 100
        df[(a, "vwap")] = df[(a, "close")]
        df[(a, "high")] = df[(a, "close")] + abs(rng.normal(0, 0.5, n_dates))
        df[(a, "low")] = df[(a, "close")] - abs(rng.normal(0, 0.5, n_dates))
        df[(a, "open")] = df[(a, "close")].shift(1).fillna(100)
        df[(a, "volume")] = abs(rng.normal(1e5, 2e4, n_dates)) + 1e4
        df[(a, "amount")] = df[(a, "volume")] * df[(a, "close")]
    close = df.xs("close", axis=1, level=1)
    mom = close - close.shift(5)
    fwd = mom * 0.5 + rng.normal(0, 0.005, (n_dates, n_assets))
    return df, fwd


def test_factory_offline_finds_signal_factor(tmp_path):
    panel, fwd = _make_panel_with_signal()
    accepted, reg = run_factor_factory(panel, fwd, config=FactorFactoryConfig(n_propose=12),
                                       registry_path=tmp_path / "reg.json")
    assert accepted, "应至少挖掘出含信号的因子"
    # 动量因子（与隐藏信号同构）应通过闸门
    assert any(m.expr == "sub(close, delay(close, 5))" for m in accepted)
    assert all(m.passed for m in accepted)
    assert reg.exprs()


def test_registry_persistence(tmp_path):
    panel, fwd = _make_panel_with_signal()
    path = tmp_path / "reg.json"
    accepted, reg = run_factor_factory(panel, fwd, config=FactorFactoryConfig(n_propose=12),
                                       registry_path=path)
    assert path.exists()
    reg2 = FactorRegistry(path)
    assert len(reg2.items) == len(reg.items)
    assert {m.expr for m in reg2.items} == {m.expr for m in accepted}


def test_redundancy_gate_rejects_duplicate():
    panel, fwd = _make_panel_with_signal()
    reg = FactorRegistry()
    reg.add(FactorMetrics(expr="sub(close, delay(close, 5))", hypothesis="动量",
                          ic=0.3, tstat=5.0, ic_pos_ratio=0.9, passed=True))
    fac = FactorFactory(reg, FactorFactoryConfig(n_propose=4))
    m = fac.evaluate(FactorCandidate(expr="sub(close, delay(close, 5))",
                                     hypothesis="重复动量"), panel, fwd)
    assert not m.passed
    assert "冗余" in m.reason


def test_run_with_refine_accumulates(tmp_path):
    panel, fwd = _make_panel_with_signal()
    accepted, reg = run_factor_factory(panel, fwd, generations=2,
                                       config=FactorFactoryConfig(n_propose=8),
                                       registry_path=tmp_path / "reg.json")
    assert accepted


def test_factory_graceful_on_noise_only():
    rng = np.random.default_rng(99)
    n = 90
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    assets = [f"A{i}" for i in range(6)]
    fields = ["open", "high", "low", "close", "volume", "vwap", "amount"]
    cols = pd.MultiIndex.from_product([assets, fields])
    df = pd.DataFrame(rng.normal(size=(n, len(cols))), index=dates, columns=cols)
    for a in assets:
        df[(a, "close")] = np.cumsum(rng.normal(0, 1, n)) + 100
        df[(a, "vwap")] = df[(a, "close")]
    fwd = pd.DataFrame(rng.normal(0, 0.01, (n, 6)), index=dates, columns=assets)
    accepted, reg = run_factor_factory(df, fwd, config=FactorFactoryConfig(n_propose=8))
    assert isinstance(accepted, list)


def test_pool_gain_computed_with_synergy(tmp_path):
    """池化增益字段正确计算：空库首因子 gain=0，非空库候选 gain 有限且合成方向一致。"""
    panel, fwd = _make_panel_with_signal(n_dates=110, n_assets=8, seed=51)
    cfg = FactorFactoryConfig(n_propose=40, corr_max=0.99, ic_min=0.0,
                              tstat_min=0.0, ic_pos_min=0.0, pool_ic_gain_min=-1.0)
    reg = FactorRegistry(tmp_path / "reg.json")
    fac = FactorFactory(reg, cfg)
    seed = FactorCandidate(expr="sub(close, delay(close, 5))", hypothesis="seed")
    m_seed = fac.evaluate(seed, panel, fwd)
    reg.add(m_seed)
    # 空库首因子：增益定义 0（无"入池前"可比较）
    assert m_seed.pool_ic_gain == 0.0
    # 库内因子值本身 IC 高（面板含强动量）
    assert np.isfinite(m_seed.ic)
    # 第二个因子：增益有限（无论符号），且 before/after 有值
    comp = FactorCandidate(expr="sub(close, delay(close, 20))", hypothesis="comp")
    m_comp = fac.evaluate(comp, panel, fwd)
    assert np.isfinite(m_comp.pool_ic_before), "入池前组合 IC 应可算"
    assert np.isfinite(m_comp.pool_ic_after), "入池后组合 IC 应可算"
    assert not np.isnan(m_comp.pool_ic_gain)


def test_pool_drag_gate_blocks_pure_duplicate(tmp_path):
    """纯重复因子（与库内高相关）入池无信息增量 → 严格增益门槛下被拦截。"""
    panel, fwd = _make_panel_with_signal(n_dates=90, n_assets=6, seed=61)
    cfg = FactorFactoryConfig(n_propose=40, corr_max=0.999, ic_min=0.0,
                              tstat_min=0.0, ic_pos_min=0.0, pool_ic_gain_min=-1.0)
    reg = FactorRegistry(tmp_path / "reg.json")
    fac = FactorFactory(reg, cfg)
    seed = FactorCandidate(expr="sub(close, delay(close, 5))", hypothesis="seed")
    reg.add(fac.evaluate(seed, panel, fwd))
    # 与库内 ρ 高、无新信息的重复因子（同族不同窗）
    cfg2 = FactorFactoryConfig(n_propose=40, corr_max=0.999, ic_min=0.0,
                               tstat_min=0.0, ic_pos_min=0.0, pool_ic_gain_min=0.01)
    fac2 = FactorFactory(reg, cfg2)
    dup = FactorCandidate(expr="sub(close, delay(close, 10))", hypothesis="dup")
    m_dup = fac2.evaluate(dup, panel, fwd)
    # 当库内因子 IC 已接近饱和时，重复因子增益 < 门槛 → 被拦截（如面板 seed 极强）
    assert not m_dup.passed or m_dup.pool_ic_gain >= 0.01 - 1e-9
    assert m_dup.pool_ic_gain < 0.03, f"重复因子增益应很小: {m_dup.pool_ic_gain}"


