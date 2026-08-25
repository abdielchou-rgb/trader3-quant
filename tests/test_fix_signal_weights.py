"""
ICIR 加权多因子合成（signal_weights / weight_by）修复验证：

1. test_icir_weights_change_result       — 同两表达式等权 vs icir 权重 → 净值序列不同且 caveat 含 weights
2. test_weight_length_mismatch_error     — weights 长度≠exprs → error；非多因子合成模式传 weights → error
3. test_weight_by_icir_reads_selected    — tmp selected.json 带 icir 门禁 → 生效（caveat 断言）；缺条目 → error
4. test_topk_combine_icir_path           — top_k_combine+weight_by 组合跑通；weight_by 无有效模式 → error
5. test_fingerprint_distinguishes_weights — 不同 weights 缓存文件不同；相同权重命中缓存
"""
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

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


def _write_selected(tmp_path, entries):
    sel_dir = tmp_path / "evolve" / "strategies"
    sel_dir.mkdir(parents=True, exist_ok=True)
    (sel_dir / "selected.json").write_text(
        json.dumps(entries, ensure_ascii=False), encoding="utf-8"
    )


def _sel_entry(expr, icir=None):
    entry = {"expr": expr, "score": 0.5}
    if icir is not None:
        entry["gates"] = {"icir": {"passed": True, "value": icir}}
    return entry


# ═══════════════════════════════════════════
# 1. 等权 vs icir 权重：净值不同 + caveat 标注
# ═══════════════════════════════════════════


def test_icir_weights_change_result(real_env):
    """同一对表达式：显式权重 [3,1]（归一化 [0.75,0.25]）的合成结果
    应与等权净值路径不同，且 caveat 变为 '多因子加权合成' 并含 weights。"""
    dates, codes, tmp_path = real_env
    tool = _fresh_tool(tmp_path)
    s, e = dates[0], dates[-1]
    exprs = ["rank(volume)", "rank(ts_mean(close, 10))"]

    r_eq = tool.execute(universe=codes, start_date=s, end_date=e, signal_exprs=exprs)
    assert r_eq.success, r_eq.summary
    assert any("多因子等权合成: K=2 个表达式" in c for c in r_eq.caveats), r_eq.caveats

    r_w = tool.execute(
        universe=codes, start_date=s, end_date=e,
        signal_exprs=exprs, signal_weights=[3.0, 1.0],
    )
    assert r_w.success, r_w.summary
    wc = [c for c in r_w.caveats if "多因子加权合成" in c]
    assert wc, f"缺少加权合成 caveat: {r_w.caveats}"
    assert "K=2" in wc[0] and "weights=" in wc[0], f"caveat 未含 K 与 weights: {wc[0]}"
    assert "0.75" in wc[0] and "0.25" in wc[0], f"caveat 未含归一化权重: {wc[0]}"

    ec_eq = np.asarray(r_eq.data.equity_curve, dtype=np.float64)
    ec_w = np.asarray(r_w.data.equity_curve, dtype=np.float64)
    assert ec_eq.shape == ec_w.shape and ec_eq.size > 0
    assert not np.allclose(ec_eq, ec_w), "加权合成的净值路径不应与等权完全相同"

    # 纯函数层：_combine_factor_scores 等权退化语义不变
    from trader3.tools.backtest import _combine_factor_scores

    stack = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])  # (K=2, T=1, N=2)
    assert np.allclose(_combine_factor_scores(stack), [[2.0, 3.0]])
    assert np.allclose(
        _combine_factor_scores(stack, weights=[3.0, 1.0]), [[1.5, 2.5]]
    )


# ═══════════════════════════════════════════
# 2. 长度不匹配 / 非法模式 → error
# ═══════════════════════════════════════════


def test_weight_length_mismatch_error(real_env):
    """weights 长度 ≠ exprs 数量 → error；
    signal_weights 出现在非 signal_exprs 模式（默认动量 / 单表达式）→ error。"""
    dates, codes, tmp_path = real_env
    tool = _fresh_tool(tmp_path)
    s, e = dates[0], dates[-1]

    r = tool.execute(
        universe=codes, start_date=s, end_date=e,
        signal_exprs=["rank(volume)", "zscore(close)"],
        signal_weights=[1.0],
    )
    assert not r.success, "长度不匹配不应成功"
    assert ("长度" in r.summary) and ("2" in r.summary), f"报错未说明原因: {r.summary}"

    # 默认动量模式 + weights → error（仅在 signal_exprs 模式下生效）
    r2 = tool.execute(universe=codes, start_date=s, end_date=e, signal_weights=[1.0, 2.0])
    assert not r2.success

    # 单表达式模式（signal_expr 优先）+ weights → error
    r3 = tool.execute(
        universe=codes, start_date=s, end_date=e,
        signal_expr="rank(close)", signal_weights=[1.0, 2.0],
    )
    assert not r3.success


# ═══════════════════════════════════════════
# 3. weight_by='icir' 从 selected.json 读门禁值
# ═══════════════════════════════════════════


def test_weight_by_icir_reads_selected(real_env, monkeypatch):
    """selected.json 各条目带 gates.icir.value → 按 |icir| 加权生效（caveat 含归一化权重）；
    任一表达式缺对应 icir 条目 → error 且指明缺失的表达式。"""
    import trader3.tools.backtest as btmod

    dates, codes, tmp_path = real_env
    expr_a = "rank(ts_mean(close, 10))"
    expr_b = "rank(volume)"
    _write_selected(tmp_path, [_sel_entry(expr_a, 0.6), _sel_entry(expr_b, 0.2)])
    monkeypatch.setattr(btmod, "_PROJECT_ROOT", str(tmp_path))

    tool = _fresh_tool(tmp_path)
    s, e = dates[0], dates[-1]
    r = tool.execute(
        universe=codes, start_date=s, end_date=e,
        signal_exprs=[expr_a, expr_b], weight_by="icir",
    )
    assert r.success, r.summary
    wc = [c for c in r.caveats if "多因子加权合成" in c]
    assert wc, f"icir 权重未生效: {r.caveats}"
    # |0.6| : |0.2| = 0.75 : 0.25
    assert "weights=" in wc[0] and "0.75" in wc[0] and "0.25" in wc[0], wc[0]

    # 缺少 expr_b 的条目 → error
    _write_selected(tmp_path, [_sel_entry(expr_a, 0.6)])
    r2 = tool.execute(
        universe=codes, start_date=s, end_date=e,
        signal_exprs=[expr_a, expr_b], weight_by="icir",
    )
    assert not r2.success, "缺失 icir 条目不应成功"
    assert f"selected.json 缺少 {expr_b} 的 icir" in r2.summary, (
        f"报错未指明缺失条目: {r2.summary}"
    )


# ═══════════════════════════════════════════
# 4. top_k_combine + weight_by 组合（主用例）
# ═══════════════════════════════════════════


def test_topk_combine_icir_path(real_env, monkeypatch):
    """top_k_combine=True + weight_by='icir' → 对前 K 名自动做 icir 加权合成跑通；
    weight_by 设置但既无 signal_exprs 也非该组合 → error。"""
    import trader3.tools.backtest as btmod

    dates, codes, tmp_path = real_env
    exprs = ["rank(ts_mean(close, 10))", "rank(volume)", "zscore(close)"]
    _write_selected(
        tmp_path,
        [_sel_entry(exprs[0], 0.6), _sel_entry(exprs[1], 0.2), _sel_entry(exprs[2], 0.4)],
    )
    monkeypatch.setattr(btmod, "_PROJECT_ROOT", str(tmp_path))

    tool = _fresh_tool(tmp_path)
    s, e = dates[0], dates[-1]
    r = tool.execute(
        universe=codes, start_date=s, end_date=e,
        top_k_combine=True, factor_from_selected=2, weight_by="icir",
    )
    assert r.success, r.summary
    wc = [c for c in r.caveats if "多因子加权合成" in c]
    assert wc and "K=2" in wc[0] and "weights=" in wc[0], (
        f"topk+icir 未做前 2 名加权合成: {r.caveats}"
    )
    assert "0.75" in wc[0] and "0.25" in wc[0], wc[0]

    # 互斥校验：weight_by 但无 signal_exprs / 非 topk 组合 → error
    r2 = tool.execute(universe=codes, start_date=s, end_date=e, weight_by="icir")
    assert not r2.success, "默认动量模式下 weight_by 应报错"

    # factor_from_selected 单因子模式 + weight_by → error
    r3 = tool.execute(
        universe=codes, start_date=s, end_date=e,
        factor_from_selected=1, weight_by="icir",
    )
    assert not r3.success

    # 不支持的 weight_by 取值 → error
    r4 = tool.execute(
        universe=codes, start_date=s, end_date=e,
        signal_exprs=exprs[:2], weight_by="sharpe",
    )
    assert not r4.success


# ═══════════════════════════════════════════
# 5. 指纹区分权重：不同 weights → 不同缓存文件
# ═══════════════════════════════════════════


def test_fingerprint_distinguishes_weights(real_env):
    """exprs+weights 联合指纹：不同 weights 落不同缓存文件；
    相同 weights 重复请求命中缓存不新增；无权重等权模式指纹亦不同。"""
    dates, codes, tmp_path = real_env
    tool = _fresh_tool(tmp_path)
    s, e = dates[0], dates[-1]
    exprs = ["rank(volume)", "rank(ts_mean(close, 10))"]
    base = {"universe": codes, "start_date": s, "end_date": e, "signal_exprs": exprs}

    tool.execute(**base, signal_weights=[0.7, 0.3])
    assert len(os.listdir(tool._cache_dir)) == 1

    tool.execute(**base, signal_weights=[0.7, 0.3])  # 同权重 → 命中缓存
    assert len(os.listdir(tool._cache_dir)) == 1, "相同 weights 应命中缓存"

    tool.execute(**base, signal_weights=[0.9, 0.1])  # 不同 weights → 新指纹
    assert len(os.listdir(tool._cache_dir)) == 2, "不同 weights 应产生不同缓存文件"

    tool.execute(**base)  # 等权模式 → 又一个指纹
    assert len(os.listdir(tool._cache_dir)) == 3
