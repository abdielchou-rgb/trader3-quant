"""
CPCV（López de Prado《AFML》ch.12 Combinatorial Purged CV）验证：

1. test_combinatorics              — C(6,2)=15 个测试块组合、各块覆盖次数相等、paths/oos_days_total 口径
2. test_purge_removes_leakage      — 因子=次日收益的完美泄漏信号：
                                     purge=0 时 prob_negative 显著偏离 0.5（泄漏被利用），
                                     purge>=K 时 ≈0.5（净化生效、泄漏被拦截）
3. test_deterministic              — 同输入两次运行结果完全一致
4. test_wfa_mode_default_unchanged — WFA execute mode 缺省时输出与旧版关键字段一致（缓存兼容）
5. test_cpcv_in_tool_response      — execute(mode="cpcv") 返回 cpcv_* 指标且 caveat 含 "CPCV"
"""
import math
import os
import sys
from collections import Counter
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from trader3.tools.cpcv import iter_test_combos, run_cpcv
from trader3.v2.costs import CommissionInfo

ZERO_COSTS = CommissionInfo(commission_bp=0.0, stamp_tax_bp=0.0, slippage_bp=0.0)


def _tiny_panel(seed=3, T=60, N=10):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.012, (T, N))
    fs = rng.normal(0.0, 1.0, (T, N))
    return rets, fs


# ═══════════════════════════════════════════
# 1. 组合数学
# ═══════════════════════════════════════════


def test_combinatorics():
    """n_blocks=6, test_blocks=2 → 15 个组合；每块被测次数相等=C(5,1)=5；
    paths=C(N,k)*k/N=5（AFML ch.12）；oos_days_total=15×2×10。"""
    rets, fs = _tiny_panel()
    res = run_cpcv(rets, fs, n_blocks=6, test_blocks=2, purge=5, top_k=3)

    combos = iter_test_combos(6, 2)
    assert res["n_combos"] == len(combos) == math.comb(6, 2) == 15
    cover = Counter(b for c in combos for b in c)
    assert set(cover.values()) == {math.comb(5, 1)}, f"块覆盖不均匀: {cover}"

    assert res["paths"] == 15 * 2 // 6 == 5
    assert len(res["sr_daily_list"]) == 15
    assert all(math.isfinite(x) for x in res["sr_daily_list"])
    assert res["oos_days_total"] == 15 * 20
    assert 0.0 <= res["prob_negative"] <= 1.0
    assert res["sr_ann_p05"] <= res["sr_ann_median"] <= res["sr_ann_p95"]


# ═══════════════════════════════════════════
# 2. purge 拦截泄漏
# ═══════════════════════════════════════════


def _leak_panel(seed, T=120, N=100, K=9, theta=4.0, sigma=0.003):
    """MA(K) 反转面板 + 完美泄漏因子（因子=次日收益）：因子对测试块的预测触达恰 K 日，
    purge>=K 后训练段不含任何测试段信息。"""
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, sigma, (T, N))
    rets = u.copy()
    for k in range(1, K + 1):
        rets[k:] -= theta * u[:-k]
    rets -= rets.mean(axis=1, keepdims=True)  # 日度市场中性，消共同模式运气
    fs = np.empty((T, N))
    fs[:-1] = rets[1:]
    fs[-1] = np.nan
    return rets, fs


def _mean_prob_negative(seed0, panels, purge):
    vals = [
        run_cpcv(*_leak_panel(sd), n_blocks=12, test_blocks=3, purge=purge,
                 top_k=3, commission=ZERO_COSTS)["prob_negative"]
        for sd in range(seed0, seed0 + panels)
    ]
    return float(np.mean(vals))


def test_purge_removes_leakage():
    """完美泄漏信号（因子=次日收益）：多面板平均下，
    purge=0 泄漏被利用 → prob_negative 显著 < 0.5；
    purge=9(>=触达K=9) 泄漏被拦截 → prob_negative≈0.5（无信息）。"""
    m0 = _mean_prob_negative(500, panels=24, purge=0)
    m9 = _mean_prob_negative(500, panels=24, purge=9)
    assert m0 <= 0.35, f"purge=0 未利用到泄漏: prob_negative={m0:.3f}"
    assert 0.40 <= m9 <= 0.60, f"purge=9 未净化至无信息: prob_negative={m9:.3f}"
    assert m9 - m0 >= 0.10, f"purge 生效性不足: gap={m9 - m0:.3f}"


# ═══════════════════════════════════════════
# 3. 确定性
# ═══════════════════════════════════════════


def test_deterministic():
    rets, fs = _tiny_panel(seed=11)
    r1 = run_cpcv(rets, fs, codes=[f"SH60000{i}" for i in range(10)],
                  n_blocks=6, test_blocks=2, purge=3, top_k=4)
    r2 = run_cpcv(rets, fs, codes=[f"SH60000{i}" for i in range(10)],
                  n_blocks=6, test_blocks=2, purge=3, top_k=4)
    assert r1 == r2


# ═══════════════════════════════════════════
# 4. 默认 wfa 行为不变
# ═══════════════════════════════════════════


def _force_synthetic(monkeypatch):
    import trader3.tools.backtest as btmod

    def _raise(*a, **k):
        raise FileNotFoundError("no qlib in test")

    monkeypatch.setattr(btmod, "_open_qlib_dp", _raise)


def test_wfa_mode_default_unchanged(monkeypatch):
    """mode 缺省与显式 'wfa' 输出完全一致，且不含任何 cpcv_* 字段/CPCV caveat。"""
    _force_synthetic(monkeypatch)
    from trader3.tools.backtest import WalkForwardAnalysisTool

    tool = WalkForwardAnalysisTool()
    r_default = tool.execute(train_window=252, test_window=63)
    r_explicit = tool.execute(train_window=252, test_window=63, mode="wfa")
    assert r_default.success and r_explicit.success

    old_keys = {
        "样本内收益", "样本外收益", "样本内夏普", "样本外夏普",
        "参数稳定性", "过拟合概率", "OOS交易日", "dsr",
    }
    for r in (r_default, r_explicit):
        assert set(r.key_metrics) == old_keys, f"key_metrics 集合漂移: {set(r.key_metrics)}"
        assert not any("CPCV" in c for c in r.caveats), f"缺省模式不应出现 CPCV caveat: {r.caveats}"

    assert r_default.key_metrics == r_explicit.key_metrics
    assert r_default.summary == r_explicit.summary
    assert r_default.caveats == r_explicit.caveats
    assert asdict(r_default.data) == asdict(r_explicit.data)


# ═══════════════════════════════════════════
# 5. 工具响应接入 CPCV
# ═══════════════════════════════════════════


def _write_cal(tmp_path, dates):
    d = tmp_path / "calendars"
    d.mkdir(parents=True, exist_ok=True)
    (d / "day.txt").write_text("\n".join(dates) + "\n", encoding="utf-8")


def _write_instruments(tmp_path, rows):
    d = tmp_path / "instruments"
    d.mkdir(parents=True, exist_ok=True)
    (d / "all.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_bin(tmp_path, code, field, values):
    d = tmp_path / "features" / code.lower()
    d.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype="<f4").tofile(str(d / f"{field}.day.bin"))


def _build_tmp_qlib(tmp_path, n_days=140, n_stocks=12, seed=7):
    """最小 qlib 目录（同 test_fix_wfa_real 契约）：全日历覆盖、价格恒正。"""
    import datetime as dt

    start = dt.date(2020, 1, 1)
    dates = []
    d = start
    while len(dates) < n_days:
        dates.append(d.isoformat())
        d += dt.timedelta(days=1)

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stocks):
        code = f"SH6000{i:02d}"
        rets = rng.normal(0.0004, 0.015, n_days)
        prices = 40.0 * np.exp(np.cumsum(rets))
        _write_bin(tmp_path, code, "close", prices)
        rows.append(f"{code}\t{dates[0]}\t2099-12-31")
    _write_cal(tmp_path, dates)
    _write_instruments(tmp_path, rows)


def test_cpcv_in_tool_response(monkeypatch, tmp_path):
    """/execute(mode='cpcv') 在真实面板上追加 cpcv_* 分布指标 + CPCV caveat；
    原 WFA 关键字段保持。"""
    import trader3.tools.backtest as btmod
    from trader3.data_provider import QlibDataProvider

    _build_tmp_qlib(tmp_path)
    monkeypatch.setattr(
        btmod, "_open_qlib_dp", lambda: QlibDataProvider(data_dir=str(tmp_path))
    )

    tool = btmod.WalkForwardAnalysisTool()
    r = tool.execute(train_window=60, test_window=20, mode="cpcv")
    assert r.success, r.summary

    km = r.key_metrics
    cpcv_keys = ["cpcv_median_sr", "cpcv_p05", "cpcv_p95", "cpcv_prob_negative"]
    for k in cpcv_keys:
        assert k in km, f"缺少 {k}: {km}"
        assert math.isfinite(km[k])
    assert 0.0 <= km["cpcv_prob_negative"] <= 1.0
    assert km["cpcv_p05"] <= km["cpcv_median_sr"] <= km["cpcv_p95"]

    # 原 WFA 字段仍在（叠加而非替换）
    assert "样本外夏普" in km and "OOS交易日" in km

    joined = "\n".join(r.caveats)
    assert "CPCV" in joined, f"caveat 缺少 CPCV 标注: {joined}"
    assert "purge=5" in joined, f"caveat 缺少 purge 参数: {joined}"

    # 未知 mode 报错而非静默回退
    r_bad = tool.execute(train_window=60, test_window=20, mode="xxx")
    assert not r_bad.success
