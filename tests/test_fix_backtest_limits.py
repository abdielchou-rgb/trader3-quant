"""
A股交易约束修复验证（backtest 涨跌停可成交性 + 现金跟踪）：

1. test_price_limit_ratio_by_prefix   — 板块涨跌停幅度映射（30x/68x→19.5%、4x/8x/92x→29%、主板→9.8%）
2. test_limit_up_blocks_buy           — 执行日涨停 → 买入取消、资金回流现金、组合不吃当日涨幅
3. test_limit_down_blocks_sell        — 执行日跌停 → 卖出失败、保留旧权重承担后续收益（T+1 延迟退出）
4. test_prefix_ratio_wired_through_sim — 板块幅度经 codes 参数接入模拟（30x 股 -11% 不构成跌停）
5. test_limit_caveat_formatting       — 报告 caveats 文案与现金比例标注规则
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


def _approx(x):
    return pytest.approx(x, abs=1e-9)


def _free_commission():
    from trader3.v2.costs import CommissionInfo

    return CommissionInfo(commission_bp=0.0, stamp_tax_bp=0.0, slippage_bp=0.0)


# ═══════════════════════════════════════════
# 1. 板块幅度映射
# ═══════════════════════════════════════════


def test_price_limit_ratio_by_prefix():
    from trader3.tools.backtest import _price_limit_ratio

    r = _price_limit_ratio
    # 创业板 30x / 科创板 68x → 19.5%
    assert r("300750") == _approx(0.195)
    assert r("301236") == _approx(0.195)
    assert r("688981") == _approx(0.195)
    assert r("689009") == _approx(0.195)
    # 主板 → 9.8%
    assert r("600000") == _approx(0.098)
    assert r("000001") == _approx(0.098)
    # 北交所 4x / 8x / 92x → 29%
    assert r("430047") == _approx(0.29)
    assert r("833171") == _approx(0.29)
    assert r("880000") == _approx(0.29)
    assert r("920098") == _approx(0.29)
    # 带交易所前缀 / 大小写混排同样识别
    assert r("SH600000") == _approx(0.098)
    assert r("sz300750") == _approx(0.195)
    assert r("BJ920098") == _approx(0.29)


# ═══════════════════════════════════════════
# 2. 涨停拦截买入
# ═══════════════════════════════════════════


def _sim(returns, scores, N, stats, codes=None):
    from trader3.tools.backtest import _run_portfolio_simulation

    return _run_portfolio_simulation(
        returns, scores, N,
        n_hold=2, max_single_w=0.6, long_only=True,
        commission=_free_commission(),
        codes=codes, stats=stats,
    )


def test_limit_up_blocks_buy():
    """执行日股票 0 暴涨 12%（>9.8% 主板涨停）→ 买单被取消。

    断言：
    - 组合当日及全程均不包含其 +12% 的涨幅贡献；
    - stats 记录 2 笔涨停拦截（t=21 与 t=42 两个执行日各一次）；
    - 被拦截资金回流现金：期末 cash_weight = 0.5。
    """
    T, N = 45, 4
    returns = np.zeros((T, N))
    scores = np.zeros((T, N))
    # 初始信号：持有股票 1(5.0)、2(4.0)，各 50%
    scores[:, 1] = 5.0
    scores[:, 2] = 4.0
    scores[:, 3] = -9.0
    # t=20 收盘信号切换至 {0, 1}；此后股票 0 每日 +12%（持续涨停，买入始终无法成交）
    scores[20:, 0] = 10.0
    scores[20:, 2] = -10.0
    returns[21:, 0] = 0.12

    stats = {}
    equity, port_returns, _, _ = _sim(returns, scores, N, stats)

    # 执行日 t=21 组合收益为 0（不含 0.5 * 12% 的前视/越权成交收益）
    assert port_returns[21] == _approx(0.0), (
        f"涨停日买入仍成交! port_returns[21]={port_returns[21]:.6f}"
    )
    # 全程净值恒为 1（+12% 从未进入组合）
    assert np.allclose(equity, 1.0), f"净值异常: {equity[[20, 21, 42]]}"
    assert stats["limit_up_blocked"] == 2
    assert stats["limit_down_blocked"] == 0
    assert stats["final_cash_weight"] == _approx(0.5)


# ═══════════════════════════════════════════
# 3. 跌停滞留卖出
# ═══════════════════════════════════════════


def test_limit_down_blocks_sell():
    """执行日持仓股票 1 暴跌 -11%（< -9.8% 主板跌停）→ 卖单失败、保留旧权重。

    断言：
    - 当日组合承担跌停损失 0.5 * (-11%)（滞留仓位无法止损）；
    - 次日反弹 +2% 时仍以旧权重 0.5 参与收益（证明权重未被减掉）；
    - stats 记录 1 笔跌停拦截；资金守恒（期末无杠杆现金）。
    """
    T, N = 45, 4
    returns = np.zeros((T, N))
    scores = np.zeros((T, N))
    # 初始信号：持有股票 0(5.0)、1(4.0)，各 50%
    scores[:, 0] = 5.0
    scores[:, 1] = 4.0
    scores[:, 2:] = -9.0
    # t=20 收盘信号全部切走（目标 {2,3}）→ t=21 需清仓 0 和 1
    scores[20:, 0] = -10.0
    scores[20:, 1] = -10.0
    scores[20:, 2] = 10.0
    scores[20:, 3] = 9.0
    returns[21, 1] = -0.11   # 执行日股票 1 跌停，卖单失败
    returns[22, 1] = 0.02    # 次日反弹，验证旧权重仍在

    stats = {}
    _, port_returns, _, _ = _sim(returns, scores, N, stats)

    assert stats["limit_down_blocked"] == 1
    assert stats["limit_up_blocked"] == 0
    # 跌停日：滞留 50% 权重硬吃 -11%
    assert port_returns[21] == _approx(0.5 * -0.11)
    # 次日：仍以旧权重 0.5 参与反弹收益
    assert port_returns[22] == _approx(0.5 * 0.02)
    # 无隐含杠杆：跌停卖单筹资不足 → 买单缩量，期末现金不为负
    assert stats["final_cash_weight"] == _approx(0.0)


def test_prefix_ratio_wired_through_sim():
    """板块幅度接入模拟：同样的 -11% 发生在创业板股（30x，幅度 19.5%）不构成跌停，
    卖单照常成交、次日不再有该股敞口。"""
    T, N = 45, 4
    returns = np.zeros((T, N))
    scores = np.zeros((T, N))
    scores[:, 0] = 5.0
    scores[:, 1] = 4.0
    scores[:, 2:] = -9.0
    scores[20:, 0] = -10.0
    scores[20:, 1] = -10.0
    scores[20:, 2] = 10.0
    scores[20:, 3] = 9.0
    returns[21, 1] = -0.11
    returns[22, 1] = 0.02

    # 股票 1 用创业板代码 → -11% 未触及 -19.5% 跌停线
    codes = ["600000", "300001", "600002", "600003"]
    stats = {}
    _, port_returns, _, _ = _sim(returns, scores, N, stats, codes=codes)

    assert stats["limit_down_blocked"] == 0
    assert port_returns[21] == _approx(0.0)   # 卖出成交，且当日其余收益为 0
    assert port_returns[22] == _approx(0.0)   # 反弹时已无该股敞口


# ═══════════════════════════════════════════
# 4. 报告 caveats
# ═══════════════════════════════════════════


def test_limit_caveat_formatting():
    from trader3.tools.backtest import _format_limit_caveats

    # 无拦截、无滞留现金 → 不追加任何 caveat
    assert _format_limit_caveats(0, 0, 0.0) == []

    c1 = _format_limit_caveats(3, 0, 0.0)
    assert any("3 笔买入" in s and "涨停" in s for s in c1)
    assert any("0 笔卖出" in s for s in c1)   # 单行合并格式，未拦截数显式为 0
    assert all("现金比例" not in s for s in c1)

    c2 = _format_limit_caveats(0, 2, 0.05)
    assert any("2 笔卖出" in s and "跌停" in s for s in c2)
    assert any("现金" in s for s in c2)  # 5% > 1% 需注明

    c3 = _format_limit_caveats(1, 1, 0.005)
    assert any("1 笔买入" in s and "1 笔卖出" in s for s in c3)
    assert not any("现金比例" in s for s in c3)  # 0.5% ≤ 1% 不注明
