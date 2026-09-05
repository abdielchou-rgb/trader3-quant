"""行业分类基建 + cvxpy 求解器路径 — 零联网单测。

覆盖:
- trader3.v2.industry: akshare 列名解析/代码归一、缓存往返与过期刷新、get_industry
- trader3.tools.optimize: SOLVER_BACKEND 探测、caveats 后端文案、cvxpy 失败回退 SLSQP
"""
from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import trader3.tools.optimize as opt
import trader3.v2.industry as industry

# ═══════════════════════════════════════════
# fixtures
# ═══════════════════════════════════════════

@pytest.fixture(autouse=True)
def _isolated_industry_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADER3_INDUSTRY_MAP", str(tmp_path / "industry_map.json"))


@pytest.fixture(autouse=True)
def _restore_solver_backend():
    yield
    opt._refresh_solver_backend()


def _seed_cache(mapping: dict[str, str], age_days: float = 0.0) -> None:
    updated = datetime.now() - timedelta(days=age_days)
    industry._write_cache(industry.default_cache_path(), mapping, updated_at=updated)


class _Expr:
    """cvxpy 表达式哑对象: 吸收全部运算符, 不参与真实计算。"""

    __array_ufunc__ = None

    def __neg__(self):
        return self

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __matmul__(self, other):
        return self

    def __rmatmul__(self, other):
        return self

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __rsub__(self, other):
        return self

    def __le__(self, other):
        return self

    def __ge__(self, other):
        return self


class _FakeVariable(_Expr):
    value = None


def _make_fake_cvxpy(n_assets: int = 8):
    """可 import 的假 cvxpy: solve 时把变量设为等权（可行解）。"""
    holder: dict[str, _FakeVariable] = {}

    mod = types.ModuleType("cvxpy")

    class _Problem:
        def __init__(self, objective, constraints=None):
            self.objective = objective
            self.constraints = constraints or []

        def solve(self, **kwargs):
            var = holder["var"]
            var.value = np.full(n_assets, 1.0 / n_assets)

    mod.Variable = lambda n: holder.setdefault("var", _FakeVariable())
    mod.psd_wrap = lambda M: M
    mod.quad_form = lambda w, P: _Expr()
    mod.sum = lambda x: _Expr()
    mod.multiply = lambda a, b: _Expr()
    mod.log = lambda x: _Expr()
    mod.Minimize = lambda expr: expr
    mod.Problem = _Problem
    return mod


# ═══════════════════════════════════════════
# 任务A: 行业分类基建
# ═══════════════════════════════════════════

def test_fetch_parses_columns(monkeypatch):
    pytest.importorskip("akshare", reason="akshare 未安装（行业接口测试需要）")
    boards = pd.DataFrame(
        {"板块名称": ["半导体", "白酒"], "板块代码": ["BK1036", "BK0477"]}
    )
    cons_by_symbol = {
        "BK1036": pd.DataFrame({"代码": ["688981.SH", "002049.SZ"]}),
        "BK0477": pd.DataFrame({"代码": ["600519", "000858"]}),
    }
    monkeypatch.setattr("akshare.stock_board_industry_name_em", lambda: boards)
    monkeypatch.setattr(
        "akshare.stock_board_industry_cons_em",
        lambda symbol: cons_by_symbol[symbol],
    )

    result = industry.fetch_sw_industry_map()

    assert result["688981"] == "半导体"
    assert result["002049"] == "半导体"
    assert result["600519"] == "白酒"
    assert result["000858"] == "白酒"
    assert all(len(k) == 6 and k.isdigit() for k in result)


def test_fetch_failure_raises_runtimeerror(monkeypatch):
    akshare = pytest.importorskip("akshare", reason="akshare 未安装（行业接口测试需要）")

    def _boom():
        raise ConnectionError("remote disconnected")

    monkeypatch.setattr(akshare, "stock_board_industry_name_em", _boom)
    with pytest.raises(RuntimeError):
        industry.fetch_sw_industry_map()


def test_cache_roundtrip_and_expiry(monkeypatch):
    _seed_cache({"600519": "白酒"})

    def _must_not_fetch():
        raise AssertionError("缓存新鲜时不应触发 fetch")

    monkeypatch.setattr(industry, "fetch_sw_industry_map", _must_not_fetch)
    loaded = industry.load_industry_map(max_age_days=30)
    assert loaded == {"600519": "白酒"}

    # 过期缓存 → 触发 fetch 并原子重写
    _seed_cache({"600519": "旧数据"}, age_days=31.0)

    def _fake_fetch():
        return {"000001": "银行"}

    monkeypatch.setattr(industry, "fetch_sw_industry_map", _fake_fetch)
    refreshed = industry.load_industry_map(max_age_days=30)
    assert refreshed == {"000001": "银行"}

    saved = json.loads(industry.default_cache_path().read_text(encoding="utf-8"))
    assert saved["map"] == {"000001": "银行"}
    updated_at = datetime.fromisoformat(saved["updated_at"])
    assert updated_at > datetime.now() - timedelta(minutes=1)


def test_get_industry_missing_code():
    _seed_cache({"600519": "白酒"})
    assert industry.get_industry("600519") == "白酒"
    assert industry.get_industry("600519.SH") == "白酒"
    assert industry.get_industry("999999") is None
    assert industry.get_industry("") is None


# ═══════════════════════════════════════════
# 任务B: cvxpy 真优化器路径
# ═══════════════════════════════════════════

_SIGNALS = {f"60000{i}": float(i) - 4.0 for i in range(8)}


def _execute(method="risk_budget"):
    return opt.OptimizePortfolioTool().execute(signals=dict(_SIGNALS), method=method)


def test_solver_backend_flag_scipy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cvxpy", None)
    assert opt._refresh_solver_backend() == "scipy"

    resp = _execute("risk_budget")
    joined = "\n".join(resp.caveats)
    assert "求解后端: scipy-SLSQP" in joined
    assert all("cvxpy失败回退SLSQP" not in v for v in resp.data.constraint_violations)


def test_solver_backend_flag_cvxpy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cvxpy", _make_fake_cvxpy(len(_SIGNALS)))
    assert opt._refresh_solver_backend() == "cvxpy"

    resp = _execute("risk_budget")
    assert "求解后端: cvxpy" in "\n".join(resp.caveats)
    assert resp.success

    # 假解为等权 → 可行; 不应出现 cvxpy 回退标注
    assert all("cvxpy失败回退SLSQP" not in v for v in resp.data.constraint_violations)


def test_cvxpy_solve_failure_falls_back_to_slsqp(monkeypatch):
    broken = types.ModuleType("cvxpy")
    broken.Variable = lambda n: (_ for _ in ()).throw(RuntimeError("ECOS crashed"))
    monkeypatch.setitem(sys.modules, "cvxpy", broken)
    assert opt._refresh_solver_backend() == "cvxpy"

    resp = _execute("mean_variance")
    joined = "\n".join(resp.caveats)
    assert "求解后端: scipy-SLSQP" in joined
    assert any("cvxpy失败回退SLSQP" in v for v in resp.data.constraint_violations)
    assert resp.success


def test_mean_var_cvxpysmoke(monkeypatch):
    monkeypatch.setitem(sys.modules, "cvxpy", _make_fake_cvxpy(len(_SIGNALS)))
    opt._refresh_solver_backend()

    resp = _execute("mean_variance")
    assert resp.success
    weights = resp.data.target_weights
    assert weights
    total = sum(weights.values())
    assert total == pytest.approx(1.0, abs=1e-3)
    cap = resp.data.max_single_weight_cap
    assert all(w <= cap + 1e-6 for w in weights.values())
    assert all(w >= -1e-9 for w in weights.values())


@pytest.mark.skipif(opt.SOLVER_BACKEND != "cvxpy", reason="需要真实 cvxpy 环境")
def test_risk_budget_cvxpy_matches_scipy_quality():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(12, 252))
    np.cov(a) * 252.0
    signals = {f"60010{i}": float(i % 5) - 2.0 for i in range(12)}

    opt._refresh_solver_backend()
    assert opt.SOLVER_BACKEND == "cvxpy"
    resp_c = opt.OptimizePortfolioTool().execute(
        signals=signals, method="risk_budget"
    )
    vol_c = resp_c.data.expected_risk
    assert resp_c.success

    saved_cp, saved_flag = opt.cp, opt.SOLVER_BACKEND
    try:
        opt.cp, opt.SOLVER_BACKEND = None, "scipy"
        resp_s = opt.OptimizePortfolioTool().execute(
            signals=signals, method="risk_budget"
        )
    finally:
        opt.cp, opt.SOLVER_BACKEND = saved_cp, saved_flag
    vol_s = resp_s.data.expected_risk
    assert resp_s.success

    assert abs(vol_c - vol_s) / max(vol_s, 1e-12) < 0.10
