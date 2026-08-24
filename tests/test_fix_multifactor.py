"""
多因子等权合成（signal_exprs / top_k_combine）+ Deflated Sharpe Ratio 修复验证：

1. test_multi_factor_combination_runs — signal_exprs 两表达式在同一面板合成跑通，caveat 含 "K=2"
2. test_nan_factor_excluded           — 部分日期全 NaN 的因子被跳过，按剩余因子合成不崩；全无效日为 NaN
3. test_priority_order                — signal_expr 与 signal_exprs 同传 → 单表达式生效（caveat 可区分）
4. test_topk_combine_reads_selected   — top_k_combine 读 selected.json 前 N 名合成；不足 2 条 → error
5. test_dsr_known_values              — DSR 已知值：N=1 正态收益 → >0.95；N=1000 显著下降；非法输入 → 0
6. test_wfa_report_contains_dsr       — 真实小面板 WFA 结果 dict 含 dsr ∈ [0,1]，caveat 声明校正次数
"""
import datetime as dt
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

# ── 小型 qlib 目录构造工具（close + volume 双字段，与 test_fix_signal_source 同款）──


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
    """构造满足 load_stock 契约的最小 qlib 目录（全日历覆盖、价格恒正）。"""
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
        rets = rng.normal(0.0005, 0.01, n_days)
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


def _default_equity(tool, dates, codes):
    r = tool.execute(universe=codes, start_date=dates[0], end_date=dates[-1])
    assert r.success, r.summary
    return np.asarray(r.data.equity_curve, dtype=np.float64)


# ═══════════════════════════════════════════
# 1. 多因子等权合成跑通
# ═══════════════════════════════════════════


def test_multi_factor_combination_runs(real_env):
    """两个互补表达式在同一 aligned panel 上 z-score 等权合成：
    跑通且 caveat 含 'K=2'；净值路径与默认动量不同。"""
    dates, codes, tmp_path = real_env
    tool = _fresh_tool(tmp_path)

    em = _default_equity(tool, dates, codes)

    r = tool.execute(
        universe=codes, start_date=dates[0], end_date=dates[-1],
        signal_exprs=["rank(volume)", "rank(ts_mean(close, 10))"],
    )
    assert r.success, r.summary
    assert any(
        "多因子等权合成: K=2 个表达式" in c for c in r.caveats
    ), f"缺少 K=2 合成 caveat: {r.caveats}"

    ec = np.asarray(r.data.equity_curve, dtype=np.float64)
    assert ec.shape == em.shape and ec.size > 0
    assert not np.allclose(ec, em), "多因子合成的净值路径不应与默认动量完全相同"

    # 相同 exprs 列表重复请求应命中缓存而非新落盘
    cache_dir = tool._cache_dir
    n_before = len(os.listdir(cache_dir))
    tool.execute(
        universe=codes, start_date=dates[0], end_date=dates[-1],
        signal_exprs=["rank(volume)", "rank(ts_mean(close, 10))"],
    )
    assert len(os.listdir(cache_dir)) == n_before, "相同 exprs 重复请求应命中缓存"


# ═══════════════════════════════════════════
# 2. NaN 因子被排除
# ═══════════════════════════════════════════


def test_nan_factor_excluded(real_env):
    """其中一个因子在部分日期产生全 NaN（滚动窗口未成熟）：
    该日按剩余因子合成不崩；全部因子无效的日分数为 NaN。"""
    from trader3.tools.backtest import _combine_factor_scores

    # 纯函数层：NaN 跳过按可用因子数归一；全无效 → NaN
    stack = np.array(
        [[[1.0, 2.0], [np.nan, np.nan]],
         [[3.0, np.nan], [4.0, 6.0]]], dtype=np.float64,
    )  # (K=2, T=2, N=2)
    combined = _combine_factor_scores(stack)
    assert combined[0, 0] == pytest.approx(2.0)      # 两因子均值
    assert combined[0, 1] == pytest.approx(2.0)      # 仅剩第一个因子
    assert combined[1, 0] == pytest.approx(4.0)
    assert combined[1, 1] == pytest.approx(6.0)
    assert np.isnan(_combine_factor_scores(np.full((2, 1, 3), np.nan))).all()

    # 端到端：ts_mean(volume, 120) 在 T=140 面板前 ~119 天全 NaN
    dates, codes, tmp_path = real_env
    tool = _fresh_tool(tmp_path)
    r = tool.execute(
        universe=codes, start_date=dates[0], end_date=dates[-1],
        signal_exprs=["rank(volume)", "ts_mean(volume, 120)"],
    )
    assert r.success, r.summary
    assert any("K=2" in c for c in r.caveats), r.caveats
    assert np.all(np.isfinite(np.asarray(r.data.equity_curve, dtype=np.float64)))


# ═══════════════════════════════════════════
# 3. 信号源优先级：signal_expr > signal_exprs
# ═══════════════════════════════════════════


def test_priority_order(real_env):
    """同时传 signal_expr 与 signal_exprs → 单表达式生效；
    两者 caveat 措辞可区分（单表达式 '信号源: 自定义表达式 …' vs 合成 'K=…'）。"""
    dates, codes, tmp_path = real_env
    tool = _fresh_tool(tmp_path)
    s, e = dates[0], dates[-1]

    r_both = tool.execute(
        universe=codes, start_date=s, end_date=e,
        signal_expr="rank(close)",
        signal_exprs=["rank(volume)", "zscore(close)"],
    )
    assert r_both.success, r_both.summary
    assert any(
        "信号源: 自定义表达式 rank(close)" in c for c in r_both.caveats
    ), f"单表达式未生效: {r_both.caveats}"
    assert not any("多因子等权合成" in c for c in r_both.caveats), (
        f"不应触发合成模式: {r_both.caveats}"
    )

    r_single = tool.execute(universe=codes, start_date=s, end_date=e,
                            signal_expr="rank(close)")
    assert any("自定义表达式 rank(close)" in c for c in r_single.caveats)
    r_multi = tool.execute(universe=codes, start_date=s, end_date=e,
                           signal_exprs=["rank(volume)", "zscore(close)"])
    assert any("多因子等权合成" in c for c in r_multi.caveats)
    assert not any("信号源: 自定义表达式" in c for c in r_multi.caveats), (
        f"合成模式不应标注单表达式信号源: {r_multi.caveats}"
    )


# ═══════════════════════════════════════════
# 4. top_k_combine 读 selected.json 前 N 名
# ═══════════════════════════════════════════


def test_topk_combine_reads_selected(real_env, monkeypatch):
    """top_k_combine=True + factor_from_selected=2 → 读 selected.json 前 2 名做合成
    （caveat 'K=2'）；可用 expr 不足 2 条 / 文件缺失 → error。"""
    import trader3.tools.backtest as btmod

    dates, codes, tmp_path = real_env
    sel_dir = tmp_path / "evolve" / "strategies"
    sel_dir.mkdir(parents=True)
    exprs = ["rank(ts_mean(close, 10))", "rank(volume)", "zscore(close)"]
    (sel_dir / "selected.json").write_text(
        json.dumps(
            [{"expr": e, "score": float(i)} for i, e in enumerate(exprs)],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(btmod, "_PROJECT_ROOT", str(tmp_path))

    tool = _fresh_tool(tmp_path)
    s, e = dates[0], dates[-1]
    r = tool.execute(universe=codes, start_date=s, end_date=e,
                     top_k_combine=True, factor_from_selected=2)
    assert r.success, r.summary
    assert any(
        "多因子等权合成: K=2 个表达式" in c for c in r.caveats
    ), f"top_k_combine 未按前 2 名合成: {r.caveats}"

    # 仅 1 条可用 expr → error
    (sel_dir / "selected.json").write_text(
        json.dumps([{"expr": exprs[0]}], ensure_ascii=False),
        encoding="utf-8",
    )
    r2 = tool.execute(universe=codes, start_date=s, end_date=e,
                      top_k_combine=True, factor_from_selected=2)
    assert not r2.success, "不足 2 条 expr 不应成功"
    assert ("selected.json" in r2.summary) or ("2" in r2.summary), (
        f"报错未说明原因: {r2.summary}"
    )

    # 文件缺失 → error
    monkeypatch.setattr(btmod, "_PROJECT_ROOT", str(tmp_path / "missing"))
    r3 = tool.execute(universe=codes, start_date=s, end_date=e,
                      top_k_combine=True, factor_from_selected=2)
    assert not r3.success
    assert "selected.json" in r3.summary, f"报错未指明 selected.json: {r3.summary}"


# ═══════════════════════════════════════════
# 5. Deflated Sharpe Ratio 已知值
# ═══════════════════════════════════════════


def test_dsr_known_values():
    """构造年化 SR=2.0、T=252 的正态收益序列：
    N=1（无多重比较校正语境）→ DSR>0.95；
    同一观测但 N=1000 → DSR 显著下降；非法输入 → 0.0。"""
    from trader3.tools.backtest import deflated_sharpe_ratio

    rng = np.random.default_rng(2024)
    base = rng.standard_normal(252)
    daily = (base - base.mean()) / base.std(ddof=1) * 0.01 \
        + (2.0 / math.sqrt(252)) * 0.01          # 精确年化 SR=2.0
    sr_daily = float(np.mean(daily) / np.std(daily, ddof=1))

    d1 = deflated_sharpe_ratio(sr_daily, n_trials=1, tail_risk_adj=True, returns=daily)
    assert 0.0 < d1 <= 1.0
    assert d1 > 0.95, f"N=1 正态收益 PSR 应 >0.95，实际 {d1:.4f}"

    d1000 = deflated_sharpe_ratio(
        sr_daily, n_trials=1000, sr_variance=1.0, tail_risk_adj=True, returns=daily,
    )
    assert 0.0 <= d1000 < 0.5, f"N=1000 应显著下降，实际 {d1000:.4f}"
    assert d1000 < d1 - 0.5, f"DSR 未显著下降: N=1 {d1:.4f} vs N=1000 {d1000:.4f}"

    # 非法输入 → 0.0
    assert deflated_sharpe_ratio(sr_daily, n_trials=0) == 0.0
    assert deflated_sharpe_ratio(sr_daily, n_trials=-3) == 0.0
    assert deflated_sharpe_ratio(sr_daily, n_trials=5, n_periods=1) == 0.0
    assert deflated_sharpe_ratio(float("nan"), n_trials=5, n_periods=100) == 0.0
    assert deflated_sharpe_ratio(0.13, n_trials=5, returns=np.array([0.01])) == 0.0


# ═══════════════════════════════════════════
# 6. WFA 报告接入 DSR
# ═══════════════════════════════════════════


def test_wfa_report_contains_dsr(monkeypatch, tmp_path):
    """真实小面板 WFA：结果 dict 有 dsr 键且 ∈[0,1]；
    caveat 声明 'DSR(PSR)=X.XX（单策略口径）'（跨策略校正归基线报告）。"""
    import trader3.tools.backtest as btmod
    from trader3.data_provider import QlibDataProvider

    _build_tmp_qlib(tmp_path, n_days=140, n_stocks=12, seed=7)
    monkeypatch.setattr(
        btmod, "_open_qlib_dp", lambda: QlibDataProvider(data_dir=str(tmp_path))
    )
    tool = btmod.WalkForwardAnalysisTool()
    r = tool.execute(train_window=60, test_window=20)
    assert r.success, r.summary
    assert r.data.windows >= 2, f"窗口数异常: {r.data.windows}"

    assert "dsr" in r.key_metrics, f"WFA 结果 dict 缺少 dsr 键: {r.key_metrics}"
    dsr_val = float(r.key_metrics["dsr"])
    assert math.isfinite(dsr_val) and 0.0 <= dsr_val <= 1.0, f"dsr 越界: {dsr_val}"

    joined = "\n".join(r.caveats)
    assert re.search(r"DSR\(PSR\)=\d\.\d{2}", joined), f"caveat 缺少 DSR 数值: {joined}"
    assert "DSR(PSR)=" in joined and "单策略口径" in joined, (
        f"caveat 未声明校正次数: {joined}"
    )
