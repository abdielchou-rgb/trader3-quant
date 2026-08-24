# -*- coding: utf-8 -*-
"""
自定义因子表达式接入回测（signal_expr / factor_from_selected）+ WFA 成本与涨跌停修复验证：

1. test_signal_expr_changes_result        — 同一面板下 rank(volume) 与默认动量的净值路径（持仓序列）不同
2. test_signal_expr_invalid_returns_error — 非法表达式 → success=False 且 summary 带明确原因
3. test_factor_from_selected_loads_expr   — factor_from_selected=N 读 selected.json 第 N 名 expr；文件缺失 → error
4. test_wfa_includes_costs                — WFA 测试段首日扣换手成本：含成本 OOS < 零成本对照；caveat 明示
5. test_cache_distinguishes_signal_expr   — 缓存指纹含 signal_expr 的 sha1，不同表达式互不串缓存
"""
import sys
import os
import json
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


def _free_commission():
    from trader3.v2.costs import CommissionInfo

    return CommissionInfo(commission_bp=0.0, stamp_tax_bp=0.0, slippage_bp=0.0)


# ── 小型 qlib 目录构造工具（close + volume 双字段）──


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


def _build_tmp_qlib(tmp_path, n_days=140, n_stocks=16, seed=11):
    """构造满足 load_stock 契约的最小 qlib 目录。

    反向设计保证两套信号排名显著不同：
    - 价格漂移随股票序号 i 递减（动量偏好低 i）
    - 成交量水平随序号 i 递增（rank(volume) 偏好高 i）
    """
    start = dt.date(2020, 1, 1)
    dates = []
    d = start
    while len(dates) < n_days:
        dates.append(d.isoformat())
        d += dt.timedelta(days=1)

    codes = [f"SH6000{i:02d}" for i in range(n_stocks)]
    rows = []
    for i, code in enumerate(codes):
        rng = np.random.default_rng(seed * 100 + i)
        drift = (0.30 - 0.025 * i) / 252.0
        rets = rng.normal(drift, 0.01, n_days)
        prices = 40.0 * np.exp(np.cumsum(rets))
        assert prices[0] < 10000 and np.all(prices > 0)
        _write_bin(tmp_path, code, "close", prices)
        vol = (1e6 * (1.0 + i)) * np.exp(rng.normal(0.0, 0.2, n_days))
        _write_bin(tmp_path, code, "volume", vol)
        rows.append(f"{code}\t{dates[0]}\t2099-12-31")
    _write_cal(tmp_path, dates)
    _write_instruments(tmp_path, "all", rows)
    return dates, codes


@pytest.fixture()
def real_env(monkeypatch, tmp_path):
    """tmp qlib 目录 + 数据目录重定向（真实面板路径）；返回 (dates, codes, tmp_path)。"""
    import trader3.data_provider as dp_mod

    dates, codes = _build_tmp_qlib(tmp_path)
    monkeypatch.setattr(dp_mod, "DEFAULT_QLIB_DATA_DIR", str(tmp_path))
    return dates, codes, tmp_path


def _fresh_tool(tmp_path):
    from trader3.tools.backtest import RunBacktestTool

    tool = RunBacktestTool()
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tool._cache_dir = str(cache)  # 缓存重定向，避免污染 shared_state
    return tool


# ═══════════════════════════════════════════
# 1. signal_expr 改变选股结果
# ═══════════════════════════════════════════


def test_signal_expr_changes_result(real_env):
    """同一面板下 signal_expr='rank(volume)' 与默认动量的持仓序列必须不同；
    表达式模式须带'信号源' caveat，默认模式不得误标。"""
    dates, codes, tmp_path = real_env
    tool = _fresh_tool(tmp_path)
    s, e = dates[0], dates[-1]

    r_mom = tool.execute(universe=codes, start_date=s, end_date=e)
    assert r_mom.success, r_mom.summary
    assert not any("信号源" in c for c in r_mom.caveats), (
        f"默认动量不应标注表达式信号源: {r_mom.caveats}"
    )

    r_vol = tool.execute(universe=codes, start_date=s, end_date=e,
                         signal_expr="rank(volume)")
    assert r_vol.success, r_vol.summary
    assert any(
        "信号源: 自定义表达式 rank(volume)" in c for c in r_vol.caveats
    ), f"缺少表达式信号源 caveat: {r_vol.caveats}"

    em = np.asarray(r_mom.data.equity_curve, dtype=np.float64)
    ev = np.asarray(r_vol.data.equity_curve, dtype=np.float64)
    assert em.shape == ev.shape and em.size > 0
    assert not np.allclose(em, ev), "rank(volume) 与默认动量的净值路径不应完全相同"


# ═══════════════════════════════════════════
# 2. 非法表达式 → 明确报错
# ═══════════════════════════════════════════


def test_signal_expr_invalid_returns_error(real_env):
    """语法非法的表达式必须返回 success=False，且 summary 说明原因。"""
    dates, codes, tmp_path = real_env
    tool = _fresh_tool(tmp_path)

    r = tool.execute(universe=codes, start_date=dates[0], end_date=dates[-1],
                     signal_expr="delay(close,-1))(")
    assert not r.success, f"非法表达式不应成功: {r.summary}"
    assert ("表达式" in r.summary) or ("解析" in r.summary), (
        f"summary 未说明原因: {r.summary}"
    )


# ═══════════════════════════════════════════
# 3. factor_from_selected 加载 selected.json
# ═══════════════════════════════════════════


def test_factor_from_selected_loads_expr(real_env, monkeypatch):
    """factor_from_selected=1 读取项目根下 evolve/strategies/selected.json 第 1 名的
    expr 并生效（caveat 可见）；文件缺失 → error。"""
    import trader3.tools.backtest as btmod

    dates, codes, tmp_path = real_env
    sel_dir = tmp_path / "evolve" / "strategies"
    sel_dir.mkdir(parents=True)
    expr = "rank(ts_mean(close, 10))"
    (sel_dir / "selected.json").write_text(
        json.dumps([{"expr": expr, "score": 1.23}], ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(btmod, "_PROJECT_ROOT", str(tmp_path))

    tool = _fresh_tool(tmp_path)
    r = tool.execute(universe=codes, start_date=dates[0], end_date=dates[-1],
                     factor_from_selected=1)
    assert r.success, r.summary
    assert any(
        "自定义表达式" in c and expr in c for c in r.caveats
    ), f"selected.json 的 expr 未生效: {r.caveats}"

    # 文件缺失 → 明确报错
    monkeypatch.setattr(btmod, "_PROJECT_ROOT", str(tmp_path / "missing"))
    r2 = tool.execute(universe=codes, start_date=dates[0], end_date=dates[-1],
                      factor_from_selected=1)
    assert not r2.success
    assert "selected.json" in r2.summary, f"报错未指明 selected.json: {r2.summary}"


# ═══════════════════════════════════════════
# 4. WFA 计入换手成本与首日涨跌停
# ═══════════════════════════════════════════


def test_wfa_includes_costs():
    """已知换手场景（一窗满仓建仓、无涨跌停拦截）：
    - 首日成本差 = 换手 1.0 × DEFAULT_COSTS 总费率（17bp），逐项可对账；
    - 含成本的 OOS 年化严格小于零成本对照。"""
    from trader3.tools.backtest import _run_wfa_rolling, _annualized_return
    from trader3.v2.costs import DEFAULT_COSTS

    rng = np.random.default_rng(7)
    # T=49 保证恰好一窗（s=0: 30+10≤49；s=10: 10+30+10=50>49）
    T, N, train, test_w, step = 49, 12, 30, 10, 10
    rets = rng.normal(0.0008, 0.01, (T, N))          # 日波动 1%，不会触及 ±9.8%
    fs = np.tile(np.arange(N, dtype=np.float64), (T, 1))  # 恒定偏好 → top_k 满仓
    codes = [f"SH6000{i:02d}" for i in range(N)]

    out_def = _run_wfa_rolling(rets, fs, train, test_w, step, codes=codes)
    out_free = _run_wfa_rolling(rets, fs, train, test_w, step, codes=codes,
                                commission=_free_commission())
    oos_def, oos_free = out_def[-1], out_free[-1]

    assert oos_free.size == test_w
    assert float(np.abs(oos_free).sum()) > 0, "零成本对照应有非零收益"

    # 首日成本精确对账：top_k = max(N//5, 10) = 10 → 等权 0.1×10 = 满仓换手 1.0
    expected_day0_cost = 1.0 * DEFAULT_COSTS.total_bp() / 10000.0
    day0_gap = float(oos_free[0] - oos_def[0])
    assert day0_gap == pytest.approx(expected_day0_cost, rel=1e-9), (
        f"首日成本差 {day0_gap} != 满仓换手×总费率 {expected_day0_cost}"
    )

    assert _annualized_return(oos_def) < _annualized_return(oos_free), (
        "含成本的 OOS 年化应严格低于零成本对照"
    )


def test_wfa_caveat_declares_costs_and_limits(monkeypatch):
    """汇总 caveats 必须声明 'WFA 含单边成本与首日涨跌停约束'，
    且不再出现旧的'未计入交易成本'声明。"""
    import trader3.tools.backtest as btmod

    def _raise(*a, **k):
        raise FileNotFoundError("no qlib in test")

    monkeypatch.setattr(btmod, "_open_qlib_dp", _raise)
    tool = btmod.WalkForwardAnalysisTool()
    r = tool.execute(train_window=60, test_window=20)
    assert r.success, r.summary

    joined = "\n".join(r.caveats)
    assert "WFA 含单边成本与首日涨跌停约束" in joined, f"caveats 未更新: {joined}"
    assert "未计入交易成本" not in joined


# ═══════════════════════════════════════════
# 5. 缓存指纹区分 signal_expr
# ═══════════════════════════════════════════


def test_cache_distinguishes_signal_expr(real_env):
    """不同 expr 指纹不同、同 expr 指纹稳定；端到端两次不同表达式各落盘一份缓存，
    重复请求命中缓存不再新增文件。"""
    dates, codes, tmp_path = real_env
    tool = _fresh_tool(tmp_path)
    args = (None, codes, dates[0], dates[-1], None, "000300.SH")

    fa = tool._fingerprint(*args, signal_expr="rank(volume)")
    fb = tool._fingerprint(*args, signal_expr="rank(close)")
    fc = tool._fingerprint(*args, signal_expr="rank(volume)")
    fd = tool._fingerprint(*args)  # 默认动量
    assert fa != fb, "不同表达式的指纹不应相同"
    assert fa == fc, "同一表达式的指纹应稳定"
    assert fd not in {fa, fb}, "默认动量指纹不应与任何表达式混同"

    s, e = dates[0], dates[-1]
    cache_dir = tool._cache_dir
    r1 = tool.execute(universe=codes, start_date=s, end_date=e, signal_expr="rank(volume)")
    r2 = tool.execute(universe=codes, start_date=s, end_date=e, signal_expr="rank(close)")
    assert r1.success and r2.success, f"{r1.summary} / {r2.summary}"
    assert len(os.listdir(cache_dir)) == 2, "两个不同表达式应各落盘一份缓存"
    tool.execute(universe=codes, start_date=s, end_date=e, signal_expr="rank(volume)")
    assert len(os.listdir(cache_dir)) == 2, "相同表达式重复请求应命中缓存而非新落盘"
