# -*- coding: utf-8 -*-
"""
审计修复验证（backtest 伪归因清除 + WFA 真实化）：

1. test_report_has_no_fake_attribution — 报告不再含伪造 Brinson/Barra 归因，caveat 明示不提供
2. test_wfa_uses_real_data            — qlib 可用时 WFA 走真实面板，OOS 窗口数 × test_window 校验
3. test_wfa_nonoverlap_default        — 默认 step==test_window（非重叠）；显式更小 step 触发重叠警告
4. test_oos_concat_metrics            — 拼接 OOS 日收益的年化/夏普公式（已知序列手工验证）
"""
import sys
import os
import math
import datetime as dt
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


# ── 小型 qlib 目录构造工具 ──

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


def _build_tmp_qlib(tmp_path, n_days=140, n_stocks=12, seed=7):
    """构造满足 load_stock 契约的最小 qlib 目录：全日历覆盖、价格恒正。

    返回 dates 列表；股票代码 SH600000..SH6000{n_stocks-1}。
    """
    start = dt.date(2020, 1, 1)
    dates = []
    d = start
    while len(dates) < n_days:
        dates.append(d.isoformat())
        d += dt.timedelta(days=1)

    rng = np.random.default_rng(seed)
    codes = [f"SH6000{i:02d}" for i in range(n_stocks)]
    rows = []
    for code in codes:
        rets = rng.normal(0.0004, 0.015, n_days)
        prices = 40.0 * np.exp(np.cumsum(rets))
        assert prices[0] < 10000 and np.all(prices > 0)
        _write_bin(tmp_path, code, "close", prices)
        rows.append(f"{code}\t{dates[0]}\t2099-12-31")
    _write_cal(tmp_path, dates)
    _write_instruments(tmp_path, "all", rows)
    return dates


# ═══════════════════════════════════════════
# 1. 伪归因清除
# ═══════════════════════════════════════════


def test_report_has_no_fake_attribution(monkeypatch, tmp_path):
    """最小回测的报告不得再冒充实测 Brinson/Barra 归因：
    - 模型字段为空（填充处已删除硬编码常数拆分）；
    - caveats 明示"行业归因(Brinson)与风险暴露(Barra)...当前版本不提供"。"""
    import trader3.data_provider as dp_mod
    import trader3.tools.backtest as btmod

    # 强制合成回退路径：探测与 QlibDataProvider 默认目录解析一并封死
    monkeypatch.setattr(dp_mod, "find_qlib_dir", lambda: None)
    monkeypatch.setattr(dp_mod, "DEFAULT_QLIB_DATA_DIR", str(tmp_path / "nonexistent"))

    from trader3.tools.backtest import RunBacktestTool

    tool = RunBacktestTool()
    r = tool.execute(start_date="2024-01-01", end_date="2024-06-30")
    assert r.success, r.summary

    report = r.data
    # 填充处已不再写入伪值：字段缺失或空 dict 均视为不含伪造归因
    assert not getattr(report, "brinson_allocation", None), (
        f"brinson_allocation 仍被填充: {report.brinson_allocation}"
    )
    assert not getattr(report, "barra_exposure", None), (
        f"barra_exposure 仍被填充: {report.barra_exposure}"
    )
    d = asdict(report)
    assert d.get("brinson_allocation", {}) == {}
    assert d.get("barra_exposure", {}) == {}

    joined = "\n".join(r.caveats)
    assert "Brinson" in joined and "不提供" in joined, f"caveats 缺少归因声明: {joined}"


# ═══════════════════════════════════════════
# 2. WFA 真实数据优先
# ═══════════════════════════════════════════


def test_wfa_uses_real_data(monkeypatch, tmp_path):
    """qlib 数据可用时，WFA 必须在真实面板上滚动而非种子合成数据。

    面板 T=140，train=60，test=20，默认 step=test_window=20：
      窗口数 = (140-60)//20 = 4，OOS 日收益总长 = 4×20 = 80。
    caveats 须标注真实数据与非重叠。"""
    import trader3.tools.backtest as btmod
    from trader3.data_provider import QlibDataProvider

    _build_tmp_qlib(tmp_path, n_days=140, n_stocks=12, seed=7)
    monkeypatch.setattr(
        btmod, "_open_qlib_dp", lambda: QlibDataProvider(data_dir=str(tmp_path))
    )

    tool = btmod.WalkForwardAnalysisTool()
    r = tool.execute(train_window=60, test_window=20)
    assert r.success, r.summary

    report = r.data
    assert report.windows == 4, f"窗口数异常: {report.windows}"
    assert report.step == 20 == report.test_window
    assert len(report.window_results) == report.windows

    # OOS 收益序列长度 ≈ 窗口数 × test_window（非重叠拼接）
    total_oos = sum(w.get("oos_days", 0) for w in report.window_results)
    assert total_oos == 4 * 20, f"OOS 天数异常: {total_oos}"
    assert r.key_metrics.get("OOS交易日") == 80

    # 窗口起点严格递进 step，OOS 区间互不重叠
    starts = [w["oos_start"] for w in report.window_results]
    assert starts == [60, 80, 100, 120]

    joined = "\n".join(r.caveats)
    assert "真实" in joined and "非重叠" in joined, f"caveats 未标注真实数据: {joined}"
    assert "合成数据" not in joined

    # 汇总指标可计算且有限
    for v in (report.is_mean_return, report.oos_mean_return,
              report.is_sharpe, report.oos_sharpe,
              report.parameter_stability, report.overfitting_probability):
        assert math.isfinite(v)


# ═══════════════════════════════════════════
# 3. 非重叠默认 / 重叠警告
# ═══════════════════════════════════════════


def _force_synthetic(monkeypatch):
    import trader3.tools.backtest as btmod

    def _raise(*a, **k):
        raise FileNotFoundError("no qlib in test")

    monkeypatch.setattr(btmod, "_open_qlib_dp", _raise)


def test_wfa_nonoverlap_default(monkeypatch):
    """/ 默认 step==test_window（无重叠警告）；显式传更小 step 时 caveat 含重叠警告；
    合成回退必须明示。"""
    _force_synthetic(monkeypatch)
    import trader3.tools.backtest as btmod

    tool = btmod.WalkForwardAnalysisTool()

    r_default = tool.execute(train_window=252, test_window=63)
    assert r_default.success, r_default.summary
    rep = r_default.data
    assert rep.step == rep.test_window == 63
    assert any("非重叠" in c for c in r_default.caveats)
    assert not any("高估显著性" in c for c in r_default.caveats), (
        f"非重叠不应警告: {r_default.caveats}"
    )
    assert any("WFA基于合成数据" in c for c in r_default.caveats), (
        f"合成回退未明示: {r_default.caveats}"
    )
    # 合成回退也按非重叠口径聚合
    assert r_default.key_metrics.get("OOS交易日", 0) > 0

    r_overlap = tool.execute(train_window=252, test_window=63, step=21)
    assert r_overlap.success
    assert r_overlap.data.step == 21
    assert any("重叠" in c and "高估显著性" in c for c in r_overlap.caveats), (
        f"缺少重叠警告: {r_overlap.caveats}"
    )


# ═══════════════════════════════════════════
# 4. 拼接 OOS 汇总公式
# ═══════════════════════════════════════════


def test_oos_concat_metrics():
    """汇总年化 = mean(日收益)×252；夏普 = mean/std(ddof=1)×√252；零波动/空序列安全守卫。"""
    from trader3.tools.backtest import _annualized_return, _annualized_sharpe

    # 已知序列 1：恒定日收益
    daily = np.full(252, 0.001)
    assert _annualized_return(daily) == pytest.approx(0.252, abs=1e-12)
    assert _annualized_sharpe(daily) == 0.0  # std=0 守卫

    # 已知序列 2：交替 ±，与 numpy 公式逐项对齐
    alt = np.tile([0.02, -0.01], 126)
    exp_ann = float(np.mean(alt)) * 252
    exp_sr = float(np.mean(alt)) / float(np.std(alt, ddof=1)) * math.sqrt(252)
    assert _annualized_return(alt) == pytest.approx(exp_ann, rel=1e-12)
    assert _annualized_sharpe(alt) == pytest.approx(exp_sr, rel=1e-12)

    # 负均值保留符号
    neg = -alt
    assert _annualized_return(neg) == pytest.approx(-exp_ann, rel=1e-12)
    assert _annualized_sharpe(neg) < 0

    # 边界守卫
    assert _annualized_return(np.array([])) == 0.0
    assert _annualized_sharpe(np.array([])) == 0.0
    assert _annualized_sharpe(np.array([0.01])) == 0.0
