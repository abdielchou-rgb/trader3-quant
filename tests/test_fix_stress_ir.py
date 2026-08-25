"""
极端情景压力测试窗口 + 有效 IR 折算（post-audit-8）验证：

1. test_effective_ir_monthly_vs_daily    — rebalance_days=21 的有效IR < daily_ir*sqrt(252)
2. test_effective_ir_known_value         — daily_ir=0.03, rebalance=21 → eff ≈ 0.03*sqrt(12) ≈ 0.104
3. test_stress_windows_overlap_detection — 覆盖 2015-06 的区间 → 至少返回 2015_crisis 一个窗口；无交集返回空
4. test_stress_worst_window_identified   — 多窗口中最大回撤最大的被标记为 worst_window
5. test_stress_test_flag_off_by_default  — stress_test=False 时 key_metrics 无 stress_* 键
6. test_stress_test_flag_on              — stress_test=True 时有 stress_* 键 + 压测 caveat + 有效IR接入
"""
import datetime as dt
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

# ── 小型 qlib 目录构造工具（与 test_fix_signal_source 同款契约）──


def _write_cal(tmp_path, dates):
    d = tmp_path / "calendars"
    d.mkdir(parents=True, exist_ok=True)
    (d / "day.txt").write_text("\n".join(dates) + "\n", encoding="utf-8")


def _write_instruments(tmp_path, universe, rows):
    d = tmp_path / "instruments"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{universe}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_bin(tmp_path, inst_dir, field, values):
    d = tmp_path / "features" / inst_dir.lower()
    d.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype="<f4").tofile(str(d / f"{field}.day.bin"))


def _build_tmp_qlib(tmp_path, start: dt.date, n_days=520, n_stocks=12, seed=11):
    """构造最小 qlib 目录：几何随机游走价格 + 递增成交量。"""
    dates = []
    d = start
    while len(dates) < n_days:
        dates.append(d.isoformat())
        d += dt.timedelta(days=1)

    codes = [f"SH6000{i:02d}" for i in range(n_stocks)]
    rows = []
    for i, code in enumerate(codes):
        rng = np.random.default_rng(seed * 100 + i)
        rets = rng.normal(0.0003, 0.015, n_days)
        prices = 40.0 * np.exp(np.cumsum(rets))
        assert np.all(prices > 0)
        _write_bin(tmp_path, code, "close", prices)
        vol = (1e6 * (1.0 + i)) * np.exp(rng.normal(0.0, 0.2, n_days))
        _write_bin(tmp_path, code, "volume", vol)
        rows.append(f"{code}\t{dates[0]}\t2099-12-31")
    _write_cal(tmp_path, dates)
    _write_instruments(tmp_path, "all", rows)
    return dates, codes


def _fresh_tool(tmp_path):
    from trader3.tools.backtest import RunBacktestTool

    tool = RunBacktestTool()
    cache = tmp_path / f"cache_{abs(hash(str(tmp_path))) % 10000}"
    cache.mkdir(parents=True, exist_ok=True)
    tool._cache_dir = str(cache)
    return tool


# ═══════════════════════════════════════════
# 1-2. 有效 IR 折算（纯函数）
# ═══════════════════════════════════════════


def test_effective_ir_monthly_vs_daily():
    """月频调仓（21 日一次独立赌注）的折算 IR 必须低于传统日频年化口径。"""
    from trader3.tools.backtest import effective_ir

    d_ir = 0.05
    eff = effective_ir(d_ir, rebalance_days=21)
    trad = d_ir * math.sqrt(252)
    assert eff == pytest.approx(d_ir * math.sqrt(252 / 21), rel=1e-9)
    assert eff < trad, f"effective_ir({eff:.4f}) 应小于传统年化 {trad:.4f}"


def test_effective_ir_known_value():
    from trader3.tools.backtest import effective_ir

    eff = effective_ir(0.03, rebalance_days=21)
    assert eff == pytest.approx(0.03 * math.sqrt(12), rel=1e-9)
    assert eff == pytest.approx(0.104, abs=5e-4)

    # 非法调仓间隔必须显式报错，不得静默返回
    with pytest.raises(ValueError):
        effective_ir(0.03, rebalance_days=0)


# ═══════════════════════════════════════════
# 3. 极端情景窗口交集识别
# ═══════════════════════════════════════════


def test_stress_windows_overlap_detection(monkeypatch, tmp_path):
    """[start,end] 覆盖 2015-06 → 至少返回一个 stress 结果（2015_crisis）；
    与所有窗口无交集的区间 → 返回空列表。"""
    import trader3.data_provider as dp_mod
    from trader3.tools.backtest import STRESS_PERIODS, run_stress_test

    dates, codes = _build_tmp_qlib(tmp_path, dt.date(2014, 11, 1), n_days=520)
    monkeypatch.setattr(dp_mod, "DEFAULT_QLIB_DATA_DIR", str(tmp_path))
    tool = _fresh_tool(tmp_path)

    rows = run_stress_test(
        "2015-06-01", "2015-09-30", codes, _backtest_tool=tool
    )
    windows = [r for r in rows if r.get("period_name") != "_summary"]
    assert len(windows) >= 1, f"覆盖 2015-06 的区间应至少命中一个压测窗口: {rows}"
    names = {r["period_name"] for r in windows}
    expected = {
        p for p, (s, e) in STRESS_PERIODS.items()
        if s <= "2015-09-30" and e >= "2015-06-01"
    }
    assert names == expected, f"交集窗口应为 {expected}，实际 {names}"
    for r in windows:
        assert set(r) >= {"period_name", "ann_return", "sharpe", "max_drawdown", "excess"}
    # 汇总行
    summaries = [r for r in rows if r.get("period_name") == "_summary"]
    assert len(summaries) == 1, f"应恰好追加一行汇总: {rows}"
    assert "avg_stress_ann" in summaries[0] and "worst_window" in summaries[0]

    # 无任何交集 → 空结果
    empty = run_stress_test("2017-01-01", "2017-12-31", codes, _backtest_tool=tool)
    assert empty == [], f"无交集区间不应产生压测结果: {empty}"


# ═══════════════════════════════════════════
# 4. worst 窗口识别（受控桩工具）
# ═══════════════════════════════════════════


class _FakeBacktestTool:
    """按窗口起点返回预设指标的桩工具；记录全部调用供断言。"""

    DD_BY_START = {"2015-06-15": -0.05, "2016-01-04": -0.40}

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def execute(self, universe=None, start_date="", end_date="", signal_expr="", **kw):
        self.calls.append((start_date, end_date))
        from trader3.base_tool import Trader3Response
        from trader3.models import BacktestReport

        dd = self.DD_BY_START[start_date]
        rep = BacktestReport(
            start_date=start_date,
            end_date=end_date,
            annual_return=-0.20,
            sharpe_ratio=-1.0,
            max_drawdown=dd,
            excess_return=-0.15,
        )
        return Trader3Response(success=True, data=rep)


def test_stress_worst_window_identified():
    """多窗口中最大回撤最深者被标记为 worst_window；汇总均值为各窗口均值。"""
    from trader3.tools.backtest import run_stress_test

    fake = _FakeBacktestTool()
    rows = run_stress_test("2014-01-01", "2016-12-31", None, _backtest_tool=fake)

    starts = {s for s, _ in fake.calls}
    assert starts == {"2015-06-15", "2016-01-04"}, (
        f"两个相交窗口都应被回测: {fake.calls}"
    )
    summary = [r for r in rows if r.get("period_name") == "_summary"][0]
    assert summary["worst_window"] == "2016_circuit_breaker"
    assert summary["avg_stress_ann"] == pytest.approx(-0.20, abs=1e-9)


# ═══════════════════════════════════════════
# 5-6. execute 开关 + 有效IR主路径接入
# ═══════════════════════════════════════════


@pytest.fixture()
def env_2020(monkeypatch, tmp_path):
    """tmp qlib 目录覆盖 2020_covid 压测窗口；返回 (dates, codes)。"""
    import trader3.data_provider as dp_mod

    dates, codes = _build_tmp_qlib(tmp_path, dt.date(2019, 12, 1), n_days=220, seed=23)
    monkeypatch.setattr(dp_mod, "DEFAULT_QLIB_DATA_DIR", str(tmp_path))
    return dates, codes


def test_stress_test_flag_off_by_default(env_2020, tmp_path):
    """缺省 stress_test=False → key_metrics 无 stress_* 键、无压测 caveat。"""
    dates, codes = env_2020
    tool = _fresh_tool(tmp_path)

    r = tool.execute(universe=codes, start_date="2019-12-01", end_date="2020-07-08")
    assert r.success, r.summary
    stress_keys = [k for k in r.key_metrics if k.startswith("stress_")]
    assert stress_keys == [], f"缺省不应输出压测键: {stress_keys}"
    assert not any("极端情景压测" in c for c in r.caveats)


def test_stress_test_flag_on(env_2020, tmp_path):
    """stress_test=True → stress_* 键 + 压测/有效IR caveat；
    且缓存命中的无压测调用不受污染（缓存存未压测版本）。"""
    dates, codes = env_2020
    s, e = "2019-12-01", "2020-07-08"

    r_off = _fresh_tool(tmp_path).execute(universe=codes, start_date=s, end_date=e)
    assert r_off.success

    tool_on = _fresh_tool(tmp_path)
    r_on = tool_on.execute(universe=codes, start_date=s, end_date=e, stress_test=True)
    assert r_on.success, r_on.summary
    assert any(k.startswith("stress_") for k in r_on.key_metrics), (
        f"开启压测后应有 stress_* 键: {sorted(r_on.key_metrics)}"
    )
    assert any("极端情景压测" in c and "窗口已评估" in c for c in r_on.caveats), (
        f"缺少压测评估 caveat: {r_on.caveats}"
    )
    # 有效 IR 接入：双口径键 + 折算 caveat
    assert "信息比率" in r_on.key_metrics
    assert "有效IR(bet级)" in r_on.key_metrics
    assert any("有效IR按调仓频率折算" in c for c in r_on.caveats), (
        f"缺少有效IR折算 caveat: {r_on.caveats}"
    )
    # bet 级折算口径幅度不超过传统年化口径（同号时）
    ir_trad = r_on.key_metrics["信息比率"]
    ir_eff = r_on.key_metrics["有效IR(bet级)"]
    if ir_trad > 0:
        assert 0 < ir_eff < ir_trad

    # 缓存存的是无压测版本：同一指纹再次普通调用不得带回 stress_* 键
    r_cached = tool_on.execute(universe=codes, start_date=s, end_date=e)
    assert r_cached.success
    assert [k for k in r_cached.key_metrics if k.startswith("stress_")] == [], (
        "缓存不应携带压测键（指纹不含 stress 标志）"
    )
