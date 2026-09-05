"""P1 研究闭环缺口测试：因子动物园 / Barra 风险约束 / 子单调度 / LLM 研究代理。"""

from __future__ import annotations

import asyncio
import random

import numpy as np
import pandas as pd

from trader3.v2.child_orders import ChildOrderSchedulerConfig, execute_child_orders, schedule_children
from trader3.v2.factor_dsl import get_dsl
from trader3.v2.factor_factory import (
    _TEMPLATE_SPECS,
    FactorFactoryConfig,
    run_research_agent,
)
from trader3.v2.live.broker_base import Order, OrderSide
from trader3.v2.portfolio import PortfolioConfig, PortfolioConstruction
from trader3.v2.risk_model import FactorRiskModel, cap_industry, neutralize_beta


def _make_panel_with_signal(n_dates=160, n_assets=6, seed=21):
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
    # 信号有噪声，使逐期 RankIC 存在方差（否则 IC≈1 且方差≈0 会被 t 闸门误杀）
    fwd = mom * 0.2 + rng.normal(0, 0.04, (n_dates, n_assets))
    return df, fwd


# ── P1-3 因子动物园：所有模板可被 DSL 编译（含 min/max 新算子）──
def test_factor_zoo_all_templates_compile():
    dsl = get_dsl()
    for _hyp, tmpl, grid in _TEMPLATE_SPECS:
        for L in grid or [0]:
            expr = tmpl.replace("{L}", str(L))
            try:
                dsl.compile(expr)
            except Exception as e:  # noqa: BLE001
                raise AssertionError(f"模板编译失败 {expr!r}: {e}") from e
    assert len(_TEMPLATE_SPECS) >= 40  # 因子动物园已显著扩展


# ── P1-2 Barra 风险：VaR / CVaR / 跟踪误差 ──
def test_risk_var_cvar_te():
    panel, _ = _make_panel_with_signal()
    close = panel.xs("close", axis=1, level=1)
    returns = close.pct_change().dropna()
    assets = list(returns.columns)
    h = pd.Series(1.0 / len(assets), index=assets)
    model = FactorRiskModel()
    var, cvar = model.var_cvar(h, returns, alpha=0.95, method="historical")
    assert 0 <= var <= 1 and 0 <= cvar <= 1 and cvar >= var
    # 跟踪误差：基准=权重本身 → 0
    te = model.tracking_error(h, h, returns=returns)
    assert abs(te) < 1e-9
    # 参数法不报错
    vp, cp = model.var_cvar(h, returns, alpha=0.99, method="parametric")
    assert vp >= 0 and cp >= 0


# ── P1-2 β 中性投影使组合 beta≈0 ──
def test_beta_neutral_zeroes_exposure():
    panel, _ = _make_panel_with_signal()
    exposures = FactorRiskModel.build_style_exposures(panel)
    assets = list(exposures.index)
    h = pd.Series(1.0 / len(assets), index=assets)
    hn = neutralize_beta(h, exposures, target=0.0)
    port_beta = float(hn.reindex(exposures.index).fillna(0).values @ exposures["beta"].values)
    assert abs(port_beta) < 1e-6


def test_portfolio_beta_neutral_flag():
    panel, _ = _make_panel_with_signal()
    exposures = FactorRiskModel.build_style_exposures(panel)
    scores = pd.Series(np.arange(len(exposures.index)) + 1.0, index=exposures.index)
    cfg = PortfolioConfig(method="ic_weighted", beta_neutral=True)
    w = PortfolioConstruction(cfg).construct(scores, exposures=exposures)
    assert abs(float(w.reindex(exposures.index).fillna(0).values @ exposures["beta"].values)) < 0.05


# ── P1-2 行业权重上限 ──
def test_industry_cap():
    assets = [f"A{i}" for i in range(4)]
    industries = {a: ("tech" if i < 2 else "fin") for i, a in enumerate(assets)}
    w = pd.Series([0.4, 0.4, 0.1, 0.1], index=assets)
    out = cap_industry(w, industries, max_w=0.5)
    assert out[["A0", "A1"]].sum() <= 0.5 + 1e-9


# ── P1-6 子单调度：切片量守恒 + 异步提交 ──
def _fake_order(symbol, qty, side=OrderSide.BUY):
    return Order(symbol=symbol, side=side, quantity=qty, price=10.0,
                 client_order_id=symbol)


def test_child_schedule_quantity_conserved():
    parents = [_fake_order("A1", 100.0), _fake_order("A2", 200.0, OrderSide.SELL)]
    cfg = ChildOrderSchedulerConfig(method="twap", n_slices=5, horizon=10.0, jitter=0.0)
    kids = schedule_children(parents, cfg, rng=random.Random(1))
    by_sym = pd.Series([k.quantity for k in kids]).sum()
    assert abs(by_sym - 300.0) < 1e-6
    # 时间单调递增
    ts = [k.t_offset for k in kids]
    assert ts == sorted(ts)


def test_child_execute_submits_all():
    parents = [_fake_order("A1", 100.0)]
    cfg = ChildOrderSchedulerConfig(method="twap", n_slices=4, horizon=4.0, jitter=0.0)
    sent = []

    async def send(o):
        sent.append(o)
        return o

    placed = asyncio.run(execute_child_orders(parents, None, cfg, submit=send, sleep=lambda t: asyncio.sleep(0)))
    assert len(placed) == 4
    assert abs(sum(o.quantity for o in placed) - 100.0) < 1e-9


# ── P1-5 LLM 研究代理（离线）──
def test_research_agent_offline(tmp_path):
    panel, fwd = _make_panel_with_signal()
    accepted, reg = run_research_agent(panel, fwd, generations=2,
                                       config=FactorFactoryConfig(n_propose=10),
                                       registry_path=tmp_path / "ra.json")
    assert isinstance(accepted, list)
    # 离线多代应至少挖掘到信号因子
    assert any(m.expr == "sub(close, delay(close, 5))" for m in accepted)
