"""
审计修复验证（data_provider + backtest）：

1. test_no_same_day_lookahead        — 真实/合成路径同日前视修复（pending_weights）
2. test_instruments_asof_filter      — instruments 成分股时段过滤（幸存者偏差）
3. test_load_stock_strips_leading_zeros — bin 首尾占位剥离 + 契约校验
4. test_benchmark_mapping            — 基准代码显式映射（000905 不再被换成沪深300）
5. test_cache_fingerprint_distinguishes_constraints — 缓存指纹含 constraints/engine_tag
"""
import json
import os
import sys

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


def _make_dp(tmp_path):
    from trader3.data_provider import QlibDataProvider

    return QlibDataProvider(data_dir=str(tmp_path))


# ═══════════════════════════════════════════
# 1. 同日前视修复
# ═══════════════════════════════════════════


def test_no_same_day_lookahead():
    """调仓日 t 收盘产生的信号不得吃 t 当日收益；最早 t+1 生效。

    场景：T=60, N=5。初始持仓股票 1/2。
    股票 0 在 t=20（恰为调仓日）暴涨 30%，信号随即切换到股票 0，
    此后每日 +1%。零费用下：
      - port_returns[20] 必须为 0（旧实现会拿到 0.30*0.5=0.15 的前视收益）
      - port_returns[21] == 0.01*0.5（生效日才开始吃到动量延续）
    """
    from trader3.tools.backtest import _run_portfolio_simulation
    from trader3.v2.costs import CommissionInfo

    free = CommissionInfo(commission_bp=0.0, stamp_tax_bp=0.0, slippage_bp=0.0)

    T, N = 60, 5
    stock_returns = np.zeros((T, N))
    factor_scores = np.zeros((T, N))

    # 初始信号：股票 1(5.0)、股票 2(4.0) 入选
    factor_scores[:, 1] = 5.0
    factor_scores[:, 2] = 4.0
    # t=20 暴涨（恰逢 (t+1)%21==0 调仓日），此后动量延续 +1%/日，信号切换至股票 0
    stock_returns[20, 0] = 0.30
    stock_returns[21:41, 0] = 0.01
    factor_scores[20:, 0] = 10.0

    _, port_returns, _, _ = _run_portfolio_simulation(
        stock_returns, factor_scores, N,
        n_hold=2, max_single_w=0.5, long_only=True, commission=free,
    )

    # 调仓日当天：组合不得包含暴涨收益（防同日前视核心断言）
    assert abs(port_returns[20]) < 1e-12, (
        f"调仓日当天吃到当日暴涨，前视! port_returns[20]={port_returns[20]:.6f}"
    )
    # 生效日起才吃到新持仓的收益：50% 权重 × 1% 日收益
    assert port_returns[21] == pytest.approx(0.005, abs=1e-9), (
        f"生效日收益异常: {port_returns[21]:.6f}"
    )
    # 暴涨当月之前组合无任何收益（对照基线）
    assert float(np.abs(port_returns[:20]).max()) < 1e-12


def test_synthetic_returns_use_lagged_alpha():
    """合成数据生成器：stock_returns[t] 由 alphas[t-1] 驱动（alpha 次日兑现）。

    注：alpha 为高持续 AR(1)（phi=0.95），同日 IC 会因信号持久性略低于次日 IC，
    但次日 IC 必须严格更高（若回退为当日兑现，IC(t,t) 将显著反超）。
    用长样本压低 IC 估计噪声。
    """
    from trader3.tools.backtest import _generate_market_data

    rng = np.random.default_rng(42)
    _, stock_returns, factor_scores, _ = _generate_market_data(rng, 1500, 20, 1)

    # t=0 无滞后 alpha 可用：个股收益只含市场 beta + 特异项（量级小）
    assert float(np.abs(stock_returns[0]).max()) < 0.05

    # 因子得分（基于 alphas[t]）应预测次日而非当日收益
    def _ic(a, b):
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        num = (a * b).sum(axis=1)
        den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1)) + 1e-12
        return num / den

    ic_same = float(_ic(factor_scores[:-1], stock_returns[:-1]).mean())
    ic_next = float(_ic(factor_scores[:-1], stock_returns[1:]).mean())
    assert ic_next > ic_same, (
        f"alpha 未滞后兑现: IC(t,t)={ic_same:.4f}, IC(t,t+1)={ic_next:.4f}"
    )


# ═══════════════════════════════════════════
# 2. instruments 时段过滤（幸存者偏差）
# ═══════════════════════════════════════════


def test_instruments_asof_filter(tmp_path):
    """asof_date 给定时仅返回该日仍在成分内的代码；None 保持并集旧行为。"""
    _write_cal(tmp_path, ["2015-01-05", "2025-01-06"])
    rows = [
        "SHA00001\t2000-01-01\t2020-12-31",   # A: 2000~2020
        "SHB00002\t2010-01-01\t2099-12-31",   # B: 2010~至今
        "SHC00003\t2000-01-01\t2005-06-30",   # C 第一段
        "SHC00003\t2006-07-01\t2010-12-31",   # C 第二段（多段覆盖）
    ]
    _write_instruments(tmp_path, "myuni", rows)
    dp = _make_dp(tmp_path)

    # asof=2015 → A∪B（C 已退出）
    assert set(dp.instruments("myuni", asof_date="2015-06-30")) == {"SHA00001", "SHB00002"}
    # asof=2025 → 仅 B
    assert set(dp.instruments("myuni", asof_date="2025-01-01")) == {"SHB00002"}
    # 边界：asof 恰在起止日期上（闭区间）
    assert set(dp.instruments("myuni", asof_date="2020-12-31")) == {"SHA00001", "SHB00002"}
    # None → 全历史并集（兼容旧调用）
    assert set(dp.instruments("myuni")) == {"SHA00001", "SHB00002", "SHC00003"}


# ═══════════════════════════════════════════
# 3. bin 加载防护（占位剥离 + 对齐 + 契约校验）
# ═══════════════════════════════════════════


def test_load_stock_strips_leading_zeros(tmp_path):
    """首尾 price<=0 占位剥离后，日期必须按 cal[start_idx+n_lead:] 映射。"""
    cal = [f"2024-01-{d:02d}" for d in range(1, 11)]  # 10 个日历条目
    _write_cal(tmp_path, cal)
    _write_instruments(tmp_path, "all", ["SH600000\t2024-01-01\t2099-12-31"])
    dp = _make_dp(tmp_path)

    raw = [0.0, 0.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    _write_bin(tmp_path, "sh600000", "close", raw)

    vals, dates = dp.load_stock("SH600000", "close")
    assert len(vals) == 6 and vals[0] == 10.0 and vals[-1] == 15.0
    # n_lead=2：首个有效价对应 cal[2]，而非错误地贴 cal[0]
    assert dates[0] == cal[2]
    assert dates[-1] == cal[7]
    assert len(dates) == len(vals)

    # 尾部 0 占位同样剥离，映射不变
    _write_bin(tmp_path, "sh600000", "close", raw + [0.0])
    vals2, dates2 = dp.load_stock("SH600000", "close")
    assert list(vals2) == list(vals) and dates2 == dates

    # 契约校验：剥离后长度与上市区间交易日数偏差 > 5 → ValueError
    _write_bin(tmp_path, "sh600000", "close", [10.0])
    with pytest.raises(ValueError):
        dp.load_stock("SH600000", "close")

    # 契约校验：close 场景剥离后首值明显非价格 (>10000) → ValueError
    _write_bin(tmp_path, "sh600000", "close",
               [99999.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    with pytest.raises(ValueError):
        dp.load_stock("SH600000", "close")

    # 全 0 数据 → 空结果而非崩溃
    _write_bin(tmp_path, "sh600000", "close", [0.0, 0.0])
    vals3, dates3 = dp.load_stock("SH600000", "close")
    assert len(vals3) == 0 and len(dates3) == 0


# ═══════════════════════════════════════════
# 4. 基准映射
# ═══════════════════════════════════════════


def test_benchmark_mapping():
    from trader3.tools.backtest import normalize_benchmark

    # 核心 bug 回归：000905（中证500）不得被静默替换为沪深300
    code, name = normalize_benchmark("000905")
    assert code == "SH000905"
    assert code != "SH000300"
    assert name is not None and "中证500" in name

    assert normalize_benchmark("000300") == ("SH000300", "沪深300")
    assert normalize_benchmark("000300.SH")[0] == "SH000300"
    assert normalize_benchmark("sh000300")[0] == "SH000300"
    assert normalize_benchmark("000906")[1] == "中证800"
    assert normalize_benchmark("000852")[1] == "中证1000"
    assert normalize_benchmark("000001")[0] == "SH000001"

    # 未知代码原样透传且标记未识别
    unk_code, unk_name = normalize_benchmark("999999")
    assert unk_code == "999999"
    assert unk_name is None

    # 空值回退沪深300
    assert normalize_benchmark("")[0] == "SH000300"


# ═══════════════════════════════════════════
# 5. 缓存指纹加固
# ═══════════════════════════════════════════


def test_cache_fingerprint_distinguishes_constraints(tmp_path):
    """不同 constraints / engine_tag 不共享缓存；指纹不匹配的缓存文件必须拒绝。"""
    from trader3.base_tool import Trader3Response
    from trader3.models import PortfolioConstraints
    from trader3.tools.backtest import RunBacktestTool

    tool = RunBacktestTool()
    tool._cache_dir = str(tmp_path / "cache")
    os.makedirs(tool._cache_dir, exist_ok=True)

    kw = dict(
        strategy_config=None,
        universe=["SH600000"],
        start_date="2020-01-01",
        end_date="2020-12-31",
        benchmark="000300",
        commission=None,
    )
    fp_a = tool._fingerprint(constraints=PortfolioConstraints(max_positions=10), **kw)
    fp_b = tool._fingerprint(constraints=PortfolioConstraints(max_positions=20), **kw)
    assert fp_a != fp_b, "不同 constraints 产生相同指纹 → 会发生张冠李戴缓存命中"

    # 引擎标识（真实/合成）参与指纹
    fp_r = tool._fingerprint(
        constraints=PortfolioConstraints(max_positions=10), engine_tag="real", **kw)
    fp_s = tool._fingerprint(
        constraints=PortfolioConstraints(max_positions=10), engine_tag="synthetic", **kw)
    assert fp_r != fp_s

    # 正常写入后可命中
    resp = Trader3Response(success=True, summary="ok")
    tool._save_cache(fp_a, resp)
    assert tool._load_cache(fp_a) is not None

    # 内容合法但存储指纹不匹配 → 拒绝（防跨参数串缓存）
    payload = json.loads(open(tool._cache_path(fp_a), encoding="utf-8").read())
    mismatched = tool._cache_path(fp_b)
    with open(mismatched, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    assert tool._load_cache(fp_b) is None
