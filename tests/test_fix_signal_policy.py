"""
信号政策收口验证（基线 v2 裁定 2026-08-25）：

任务A — CPCV 主指标切换：
1. test_cpcv_headline_from_pooled — mode=cpcv 时 样本外收益/夏普 来自
   run_cpcv 的 oos_concat_pooled 组合路径聚合（≠ 同参 WFA 单点口径），
   且 *_wfa口径 对照键与 WFA 模式一致；caveat 注明聚合口径。
2. test_default_mode_untouched     — mode 缺省时 key_metrics 维持旧键集合，
   不出现任何 CPCV/口径键漂移。

任务B — selected.json OOS 否决机制：
3. test_oos_veto_skipped              — 带 oos_veto:true 的条目不进入合成候选，
   caveat 含 "OOS否决跳过 1 条"；可用不足 K 降级并明示。
4. test_veto_icir_weights_consistent  — weight_by=icir 时被否决条目权重为 0
   （不参与归一），净值与显式剔除该因子的加权结果一致。
5. test_all_vetoed_graceful           — 全部被否决 → error 且 summary 含 "无可入选因子"。
6. test_no_veto_field_backward_compat — 无 oos_veto 字段的旧格式行为完全不变。
7. test_veto_rank_mode_skips          — factor_from_selected 计数跳过被否决条目。
"""
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

ANNUAL_DAYS_POOLED = 244  # 与实现约定：CPCV pooled 年化基数


# ── 小型 qlib 目录构造工具（close + volume 双字段，与 test_fix_multifactor 同款）──


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


def _build_tmp_qlib_close_only(tmp_path, n_days=140, n_stocks=12, seed=7):
    """WFA/CPCV 用最小目录（仅 close，与 test_fix_cpcv 同款契约）。"""
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
    _write_instruments(tmp_path, "all", rows)


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


def _force_synthetic(monkeypatch):
    import trader3.tools.backtest as btmod

    def _raise(*a, **k):
        raise FileNotFoundError("no qlib in test")

    monkeypatch.setattr(btmod, "_open_qlib_dp", _raise)


# ═══════════════════════════════════════════
# 任务A — CPCV 主指标切换
# ═══════════════════════════════════════════


def test_cpcv_headline_from_pooled(monkeypatch, tmp_path):
    """mode=cpcv：样本外收益/夏普 = 由 oos_concat_pooled 手工计算值；
    与同参 WFA 模式的单点口径必然不同；*_wfa口径 键保留对照。"""
    import trader3.tools.backtest as btmod
    from trader3.data_provider import QlibDataProvider
    from trader3.tools.cpcv import run_cpcv

    _build_tmp_qlib_close_only(tmp_path)
    monkeypatch.setattr(
        btmod, "_open_qlib_dp", lambda: QlibDataProvider(data_dir=str(tmp_path))
    )

    # 复现 execute(mode=cpcv) 分支的输入，手工计算 pooled 年化期望值
    dp = btmod._open_qlib_dp()
    codes_list, close_matrix, returns_matrix, _valid_flags, _axis = (
        btmod._load_wfa_panel(dp)
    )
    factor_scores = btmod._momentum_scores(close_matrix)
    res = run_cpcv(
        returns_matrix, factor_scores,
        codes=codes_list, n_blocks=6, test_blocks=2, purge=5,
    )
    pooled = np.asarray(res["oos_concat_pooled"], dtype=np.float64)
    assert pooled.size == res["oos_days_total"]
    exp_ret = float(np.mean(pooled)) * ANNUAL_DAYS_POOLED
    p_std = float(np.std(pooled, ddof=1))
    exp_sharpe = float(np.mean(pooled)) / p_std * (ANNUAL_DAYS_POOLED ** 0.5)

    tool = btmod.WalkForwardAnalysisTool()
    r_cpcv = tool.execute(train_window=60, test_window=20, mode="cpcv")
    r_wfa = tool.execute(train_window=60, test_window=20, mode="wfa")
    assert r_cpcv.success and r_wfa.success

    km_cpcv, km_wfa = r_cpcv.key_metrics, r_wfa.key_metrics
    assert km_cpcv["样本外收益"] == pytest.approx(exp_ret)
    assert km_cpcv["样本外夏普"] == pytest.approx(exp_sharpe)

    # CPCV 聚合口径 ≠ WFA 单点拼接口径（确定性面板下差异显著且稳定）
    assert abs(km_cpcv["样本外收益"] - km_wfa["样本外收益"]) > 1e-6, (
        f"CPCV headline 应来自 pooled 聚合: {km_cpcv} vs {km_wfa}"
    )

    # WFA 口径对照键与 wfa 模式输出一致
    assert km_cpcv["样本外收益_wfa口径"] == pytest.approx(km_wfa["样本外收益"])
    assert km_cpcv["样本外夏普_wfa口径"] == pytest.approx(km_wfa["样本外夏普"])

    joined = "\n".join(r_cpcv.caveats)
    assert "headline 为 CPCV 组合聚合口径" in joined, f"缺少口径 caveat: {joined}"

    # wfa 模式不新增对照键、无口径 caveat
    assert "样本外收益_wfa口径" not in km_wfa
    assert "headline 为 CPCV 组合聚合口径" not in "\n".join(r_wfa.caveats)


def test_default_mode_untouched(monkeypatch):
    """mode 缺省回归：旧键集合不变、无 *_wfa口径 键、无 CPCV 聚合口径 caveat。"""
    _force_synthetic(monkeypatch)
    from trader3.tools.backtest import WalkForwardAnalysisTool

    tool = WalkForwardAnalysisTool()
    r = tool.execute(train_window=252, test_window=63)
    assert r.success

    old_keys = {
        "样本内收益", "样本外收益", "样本内夏普", "样本外夏普",
        "参数稳定性", "过拟合概率", "OOS交易日", "dsr",
    }
    assert set(r.key_metrics) == old_keys, f"key_metrics 集合漂移: {set(r.key_metrics)}"
    joined = "\n".join(r.caveats)
    assert "CPCV" not in joined
    assert "组合聚合口径" not in joined


# ═══════════════════════════════════════════
# 任务B — selected.json OOS 否决机制
# ═══════════════════════════════════════════

EXPR_A = "rank(ts_mean(close, 10))"
EXPR_B = "rank(volume)"
EXPR_C = "zscore(close)"


def _write_selected(tmp_path, entries):
    sel_dir = tmp_path / "evolve" / "strategies"
    sel_dir.mkdir(parents=True, exist_ok=True)
    (sel_dir / "selected.json").write_text(
        json.dumps(entries, ensure_ascii=False), encoding="utf-8"
    )


def test_oos_veto_skipped(real_env, monkeypatch):
    """三条 selected.json 中间一条带 oos_veto:true → 请求 K=3 实际只用 2 条合成
    （caveat 'K=2'），且 caveat 含 'OOS否决跳过 1 条' 与降级说明。"""
    import trader3.tools.backtest as btmod

    dates, codes, tmp_path = real_env
    _write_selected(tmp_path, [
        {"expr": EXPR_A, "score": 3.0},
        {"expr": EXPR_B, "score": 2.0, "oos_veto": True},
        {"expr": EXPR_C, "score": 1.0},
    ])
    monkeypatch.setattr(btmod, "_PROJECT_ROOT", str(tmp_path))

    tool = _fresh_tool(tmp_path)
    s, e = dates[0], dates[-1]
    r = tool.execute(universe=codes, start_date=s, end_date=e,
                     top_k_combine=True, factor_from_selected=3)
    assert r.success, r.summary
    assert any("多因子等权合成: K=2 个表达式" in c for c in r.caveats), (
        f"被否决条目仍参与合成: {r.caveats}"
    )
    assert any("OOS否决跳过 1 条" in c for c in r.caveats), (
        f"缺少否决统计 caveat: {r.caveats}"
    )
    assert any(("不足3" in c and "降级为2条" in c) for c in r.caveats), (
        f"缺少降级说明 caveat: {r.caveats}"
    )


def test_veto_icir_weights_consistent(real_env, monkeypatch):
    """weight_by=icir：被否决条目权重 0（不参与归一）——
    合成净值与显式 signal_weights=[2, 0]（只留未否决因子）完全一致。"""
    import trader3.tools.backtest as btmod

    dates, codes, tmp_path = real_env
    _write_selected(tmp_path, [
        {"expr": EXPR_A, "gates": {"icir": {"passed": True, "value": 2.0}}},
        {"expr": EXPR_B, "gates": {"icir": {"passed": True, "value": 5.0}},
         "oos_veto": True},
    ])
    monkeypatch.setattr(btmod, "_PROJECT_ROOT", str(tmp_path))

    # 纯函数层：被否决条目不出现在权重映射中
    stats = {}
    weights = btmod._load_icir_weights_from_selected([EXPR_A, EXPR_B], veto_stats=stats)
    assert weights == [2.0, 0.0], f"否决条目应得 0 权重: {weights}"
    assert stats.get("icir_skipped") == 1

    tool = _fresh_tool(tmp_path)
    s, e = dates[0], dates[-1]
    r_icir = tool.execute(universe=codes, start_date=s, end_date=e,
                          signal_exprs=[EXPR_A, EXPR_B], weight_by="icir")
    assert r_icir.success, r_icir.summary
    assert any("OOS否决跳过 1 条" in c for c in r_icir.caveats), r_icir.caveats
    # 归一化后权重 [2/(2+0), 0] = [1.0000, 0.0000]，被否决者不摊薄他人权重
    assert any(
        "weights=[1.0000, 0.0000]" in c for c in r_icir.caveats
    ), f"否决条目参与了权重归一: {r_icir.caveats}"

    r_explicit = tool.execute(universe=codes, start_date=s, end_date=e,
                              signal_exprs=[EXPR_A, EXPR_B],
                              signal_weights=[2.0, 0.0])
    assert r_explicit.success, r_explicit.summary
    ec_a = np.asarray(r_icir.data.equity_curve, dtype=np.float64)
    ec_b = np.asarray(r_explicit.data.equity_curve, dtype=np.float64)
    assert np.allclose(ec_a, ec_b), "icir 否决语义应等价于显式剔除该因子"


def test_all_vetoed_graceful(real_env, monkeypatch):
    """全部条目被否决 → error，summary 含 '无可入选因子'（而非崩溃/静默空合成）。"""
    import trader3.tools.backtest as btmod

    dates, codes, tmp_path = real_env
    _write_selected(tmp_path, [
        {"expr": EXPR_A, "oos_veto": True},
        {"expr": EXPR_B, "oos_veto": True},
    ])
    monkeypatch.setattr(btmod, "_PROJECT_ROOT", str(tmp_path))

    tool = _fresh_tool(tmp_path)
    r = tool.execute(universe=codes, start_date=dates[0], end_date=dates[-1],
                     top_k_combine=True, factor_from_selected=3)
    assert not r.success
    assert "无可入选因子" in r.summary, f"summary 未说明原因: {r.summary}"


def test_no_veto_field_backward_compat(real_env, monkeypatch):
    """旧格式（无 oos_veto 字段）：topk 取前 K 名、rank 取第 N 名、icir 全量加权，
    行为与措辞均不变、无否决 caveat。"""
    import trader3.tools.backtest as btmod

    dates, codes, tmp_path = real_env
    _write_selected(tmp_path, [
        {"expr": EXPR_A, "score": 3.0},
        {"expr": EXPR_B, "score": 2.0},
        {"expr": EXPR_C, "score": 1.0},
    ])
    monkeypatch.setattr(btmod, "_PROJECT_ROOT", str(tmp_path))

    # 纯函数层
    assert btmod._load_expr_from_selected(2) == EXPR_B
    assert btmod._load_topk_exprs_from_selected(3) == [EXPR_A, EXPR_B, EXPR_C]
    try:
        btmod._load_icir_weights_from_selected([EXPR_A])
        raised = False
    except ValueError as ex:
        raised = True
        assert "缺少" in str(ex)
    assert raised, "缺 icir 门禁的旧报错契约不应改变"

    tool = _fresh_tool(tmp_path)
    s, e = dates[0], dates[-1]
    r = tool.execute(universe=codes, start_date=s, end_date=e,
                     top_k_combine=True, factor_from_selected=3)
    assert r.success, r.summary
    assert any("多因子等权合成: K=3 个表达式" in c for c in r.caveats)
    assert not any("OOS否决" in c for c in r.caveats), (
        f"旧格式不应出现否决 caveat: {r.caveats}"
    )


def test_veto_rank_mode_skips(monkeypatch, tmp_path):
    """factor_from_selected 单因子模式：rank 计数跳过被否决条目；
    可用条目不足 rank 时报错并给出可用数。"""
    import trader3.tools.backtest as btmod

    monkeypatch.setattr(btmod, "_PROJECT_ROOT", str(tmp_path))
    _write_selected(tmp_path, [
        {"expr": EXPR_A, "score": 3.0},
        {"expr": EXPR_B, "score": 2.0, "oos_veto": True},
        {"expr": EXPR_C, "score": 1.0},
    ])

    stats = {}
    assert btmod._load_expr_from_selected(2, veto_stats=stats) == EXPR_C, (
        "rank=2 应落到第 2 个未被否决的条目"
    )
    assert stats.get("expr_skipped") == 1

    with pytest.raises(IndexError) as ei:
        btmod._load_expr_from_selected(3)
    assert "可用" in str(ei.value), f"报错应提示否决后的可用数: {ei.value}"
