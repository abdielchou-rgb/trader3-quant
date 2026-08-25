"""行业数据消费落地 — Brinson 归因恢复 + 优化器行业中性（零联网单测）。

覆盖:
- trader3.tools.backtest: 真实路径 Brinson(BHB 简化) 行业归因（key_metrics/caveat 挂接、
  覆盖率<50% 跳过、get_industry 抛错不影响主回测）
- trader3.tools.optimize: industry_neutral 后处理投影（两工具共用、低覆盖 noop）

行业映射经 TRADER3_INDUSTRY_MAP 注入临时缓存；qlib 数据用临时目录构造，
全程不触发网络请求。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import trader3.data_provider as dp_mod
import trader3.tools.backtest as btmod
import trader3.tools.optimize as opt
import trader3.v2.industry as industry
from trader3.models import PortfolioConstraints

# ═══════════════════════════════════════════
# fixtures / 构造工具
# ═══════════════════════════════════════════


@pytest.fixture(autouse=True)
def _isolated_industry_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADER3_INDUSTRY_MAP", str(tmp_path / "industry_map.json"))


def _seed_cache(mapping: dict[str, str]) -> None:
    industry._write_cache(
        industry.default_cache_path(), mapping, updated_at=datetime.now()
    )


def _write_cal(tmp_path, dates):
    d = tmp_path / "calendars"
    d.mkdir(parents=True, exist_ok=True)
    (d / "day.txt").write_text("\n".join(dates) + "\n", encoding="utf-8")


def _write_bin(tmp_path, inst_dir, field, values):
    d = tmp_path / "features" / inst_dir.lower()
    d.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype="<f4").tofile(str(d / f"{field}.day.bin"))


def _build_constant_growth_qlib(tmp_path, daily_rets, n_days=40):
    """构造恒定日收益的最小 qlib 目录。

    daily_rets: {code: 日收益}，价格 = 10·(1+r)^t → 区间复合收益精确可控。
    返回 dates 列表；首只股票兼作基准。
    """
    start = __import__("datetime").date(2024, 1, 1)
    dates = []
    d = start
    while len(dates) < n_days:
        dates.append(d.isoformat())
        d += timedelta(days=1)

    rows = []
    for code, r in daily_rets.items():
        prices = 10.0 * np.cumprod(np.full(n_days, 1.0 + r))
        assert np.all(prices > 0) and prices[-1] < 10000
        _write_bin(tmp_path, code, "close", prices)
        rows.append(f"{code}\t{dates[0]}\t2099-12-31")
    _write_cal(tmp_path, dates)
    inst_dir = tmp_path / "instruments"
    inst_dir.mkdir(exist_ok=True)
    (inst_dir / "all.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return dates


def _force_tmp_qlib(monkeypatch, tmp_path):
    """RunBacktestTool 的真实路径强制使用临时 qlib 目录。"""
    real_cls = dp_mod.QlibDataProvider
    monkeypatch.setattr(
        dp_mod, "QlibDataProvider", lambda: real_cls(data_dir=str(tmp_path))
    )


def _stub_simulation(monkeypatch, final_weights):
    """替换月频调仓模拟：固定返回期末持仓权重，隔离权重对归因的影响。"""

    def _fake_sim(close_matrix, returns_matrix, valid_flags, *, n_hold,
                  max_single_w, commission=None, codes=None, score_matrix=None):
        t = returns_matrix.shape[0]
        stats = {
            "limit_up_blocked": 0,
            "limit_down_blocked": 0,
            "final_cash_weight": 0.0,
            "final_weights": np.asarray(final_weights, dtype=np.float64),
        }
        return np.ones(t), np.zeros(t), 0.0, 0, stats

    monkeypatch.setattr(btmod, "_run_momentum_backtest", _fake_sim)


# ── 共享场景：12 只股票、3 行业各 4 只，行业内收益异质（选股效应非零）──

_UNIVERSE = [f"SH60{i:04d}" for i in range(0, 12)]
_DAILY_RETS = (
    [0.004, 0.002, 0.001, 0.001]      # 白酒 SH600000..SH600003
    + [0.0005, 0.0005, 0.0005, 0.0005]  # 银行 SH600004..SH600007
    + [-0.001, 0.0, 0.0, 0.0]          # 半导体 SH600008..SH600011
)
_FULL_MAP = {
    **{c: "白酒" for c in _UNIVERSE[0:4]},
    **{c: "银行" for c in _UNIVERSE[4:8]},
    **{c: "半导体" for c in _UNIVERSE[8:12]},
}
_FINAL_WEIGHTS = [
    0.25, 0.15, 0.05, 0.05,   # 白酒 0.50（超配）
    0.08, 0.06, 0.04, 0.02,   # 银行 0.20（低配）
    0.08, 0.05, 0.04, 0.03,   # 半导体 0.20（低配）
]  # 合计 0.90（其余为现金）


def _expected_brinson():
    """独立手算口径（区间复合收益 + BHB 简化公式），与实现代码无共享。"""
    span = 39  # n_days - 1 次复合
    rets = np.array([(1.0 + r) ** span - 1.0 for r in _DAILY_RETS])
    w = np.array(_FINAL_WEIGHTS)
    idx = {"白酒": slice(0, 4), "银行": slice(4, 8), "半导体": slice(8, 12)}
    w_b = 4.0 / 12.0
    r_b = {g: float(np.mean(rets[s])) for g, s in idx.items()}
    w_p = {g: float(np.sum(w[s])) for g, s in idx.items()}
    r_p = {
        g: (float(np.dot(w[s], rets[s]) / w_p[g]) if w_p[g] > 1e-12 else r_b[g])
        for g, s in idx.items()
    }
    big_r = sum(w_b * v for v in r_b.values())
    allocation = sum((w_p[g] - w_b) * (r_b[g] - big_r) for g in idx)
    selection = sum(w_b * (r_p[g] - r_b[g]) for g in idx) + sum(
        (w_p[g] - w_b) * (r_p[g] - r_b[g]) for g in idx
    )
    return allocation, selection


def _run_real_backtest(monkeypatch, tmp_path, industry_map):
    _build_constant_growth_qlib(tmp_path, dict(zip(_UNIVERSE, _DAILY_RETS, strict=False)))
    _force_tmp_qlib(monkeypatch, tmp_path)
    _stub_simulation(monkeypatch, _FINAL_WEIGHTS)
    if industry_map is not None:
        _seed_cache(industry_map)
    tool = btmod.RunBacktestTool()
    tool._cache_dir = str(tmp_path / "bt_cache")  # 隔离共享缓存，防跨用例命中
    os.makedirs(tool._cache_dir, exist_ok=True)
    return tool.execute(
        universe=list(_UNIVERSE),
        start_date="2024-01-01",
        end_date="2024-02-20",
        benchmark="sh600000",
    )


# ═══════════════════════════════════════════
# 任务A: Brinson 行业归因
# ═══════════════════════════════════════════


def test_brinson_effects_computed(monkeypatch, tmp_path):
    """3 行业已知权重/收益场景：key_metrics 配置/选择效应与手算一致（容差 1e-6）。"""
    r = _run_real_backtest(monkeypatch, tmp_path, _FULL_MAP)
    assert r.success, r.summary

    alloc_exp, sel_exp = _expected_brinson()
    assert r.key_metrics["配置效应"] == pytest.approx(alloc_exp, abs=1e-6)
    assert r.key_metrics["选择效应"] == pytest.approx(sel_exp, abs=1e-6)

    joined = "\n".join(r.caveats)
    assert "Brinson(BHB简化): 配置" in joined and "选择" in joined
    assert "行业来源: industry_map" in joined
    assert "覆盖率 100%" in joined


def test_brinson_low_coverage_skips(monkeypatch, tmp_path):
    """覆盖率 <50%（2/12 有标签）→ 数字缺席，仅输出覆盖率不足 caveat。"""
    r = _run_real_backtest(
        monkeypatch, tmp_path, {"600000": "白酒", "600004": "银行"}
    )
    assert r.success, r.summary

    assert "配置效应" not in r.key_metrics
    assert "选择效应" not in r.key_metrics
    joined = "\n".join(r.caveats)
    assert "行业覆盖率不足，跳过归因" in joined
    assert "Brinson(BHB简化): 配置" not in joined


def test_brinson_exception_safe(monkeypatch, tmp_path):
    """get_industry 抛错 → 主响应仍 success=True，仅记跳过 caveat。"""

    def _boom(code):
        raise RuntimeError("industry lookup exploded")

    monkeypatch.setattr(industry, "get_industry", _boom)
    r = _run_real_backtest(monkeypatch, tmp_path, _FULL_MAP)

    assert r.success, r.summary
    assert "配置效应" not in r.key_metrics
    assert any("Brinson 归因跳过" in c for c in r.caveats)


# ═══════════════════════════════════════════
# 任务B: 优化器行业中性
# ═══════════════════════════════════════════

_NEU_SIGNALS = {f"60000{i}": float(i % 4) - 1.5 for i in range(1, 9)}
_NEU_INDUSTRIES = {
    **{f"60000{i}": "白酒" for i in range(1, 5)},
    **{f"60000{i}": "银行" for i in range(5, 9)},
}


def test_industry_neutral_projection():
    """超配单行业组合投影后：行业权重回到基准占比、sum=1、max≤cap。"""
    tickers = ["600001", "600002", "600003", "600004"]
    labels = {"600001": "白酒", "600002": "白酒", "600003": "银行", "600004": "银行"}
    w0 = np.array([0.70, 0.10, 0.10, 0.10])

    new_w, k = opt._industry_neutral_project(w0, tickers, labels, max_single=0.45)

    assert k == 2
    assert new_w == pytest.approx([0.4375, 0.0625, 0.25, 0.25], abs=1e-9)
    assert float(np.sum(new_w)) == pytest.approx(1.0, abs=1e-9)
    assert float(np.max(new_w)) <= 0.45 + 1e-12
    white = float(new_w[0] + new_w[1])
    bank = float(new_w[2] + new_w[3])
    assert white == pytest.approx(0.5, abs=1e-9)
    assert bank == pytest.approx(0.5, abs=1e-9)

    # 接线验证：execute 全链路（显式 industries，覆盖率 100%）→ caveat + 中性行业占比
    cons = PortfolioConstraints(max_single_weight=0.30)
    resp = opt.OptimizePortfolioTool().execute(
        signals=dict(_NEU_SIGNALS),
        method="risk_budget",
        constraints=cons,
        industry_neutral=True,
        industries=dict(_NEU_INDUSTRIES),
    )
    assert resp.success
    assert any("行业中性已启用(行业数=2)" in c for c in resp.caveats)
    tw = resp.data.target_weights
    assert sum(tw.values()) == pytest.approx(1.0, abs=1e-5)
    white_total = sum(v for key, v in tw.items() if _NEU_INDUSTRIES[key] == "白酒")
    bank_total = sum(v for key, v in tw.items() if _NEU_INDUSTRIES[key] == "银行")
    assert white_total == pytest.approx(0.5, abs=1e-5)
    assert bank_total == pytest.approx(0.5, abs=1e-5)
    assert all(v <= 0.30 + 1e-6 for v in tw.values())
    assert resp.data.constraints_satisfied

    # RegimeAware 同参行为一致
    regime_weights = {
        "bull": {t: (3.0 if _NEU_INDUSTRIES[t] == "白酒" else 1.0) for t in _NEU_SIGNALS}
    }
    r_resp = opt.RegimeAwareAllocationTool().execute(
        signals=dict(_NEU_SIGNALS),
        regime_probs={"bull": 1.0},
        regime_weights=regime_weights,
        constraints=PortfolioConstraints(max_single_weight=0.30),
        industry_neutral=True,
        industries=dict(_NEU_INDUSTRIES),
    )
    assert r_resp.success
    assert any("行业中性已启用(行业数=2)" in c for c in r_resp.caveats)
    rw = r_resp.data.target_weights
    r_white = sum(v for key, v in rw.items() if _NEU_INDUSTRIES[key] == "白酒")
    r_bank = sum(v for key, v in rw.items() if _NEU_INDUSTRIES[key] == "银行")
    assert r_white == pytest.approx(0.5, abs=1e-5)
    assert r_bank == pytest.approx(0.5, abs=1e-5)


def test_neutral_low_coverage_noop(monkeypatch):
    """自动补全覆盖率不足（1/8）→ 权重与未启用时完全一致 + caveat 明示。"""
    _seed_cache({"600001": "白酒"})  # 仅 1 只可识别 → 覆盖率 12.5% < 50%

    base = opt.OptimizePortfolioTool().execute(
        signals=dict(_NEU_SIGNALS), method="risk_budget"
    )
    neu = opt.OptimizePortfolioTool().execute(
        signals=dict(_NEU_SIGNALS),
        method="risk_budget",
        industry_neutral=True,
        industries=None,
    )
    assert base.success and neu.success
    assert neu.data.target_weights == base.data.target_weights
    assert any("行业覆盖率不足，未启用" in c for c in neu.caveats)
    assert not any("行业中性已启用" in c for c in neu.caveats)
