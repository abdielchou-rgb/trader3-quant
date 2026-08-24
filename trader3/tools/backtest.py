"""
3号交易员 — 回测验证 Tools (M1: 真实回测引擎)

M1 upgrades:
1. Vectorized backtest engine (Qlib fallback) — generates synthetic data, runs portfolio
   simulation, computes all real metrics
2. Real Walk-Forward Analysis with rolling IS/OOS windows
3. Backtest result caching (sha256 key, shared_state/backtest_cache/)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from trader3.base_tool import BaseTool, ChartSpec, Trader3Response
from trader3.v2.costs import CommissionInfo, DEFAULT_COSTS, build_commission
from trader3.models import (
    BacktestReport,
    FactorConfig,
    PortfolioConstraints,
    StrategyConfig,
    WFAReport,
)


# ═══════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════

TRADING_DAYS_PER_YEAR = 252
STAMP_TAX_BP = 10          # 印花税（卖出单边）
COMMISSION_BP = 2           # 佣金（双边）
SLIPPAGE_BP = 5             # 冲击成本（双边）
TOTAL_COST_BP = int(DEFAULT_COSTS.total_bp())  # 17bp / 每笔交易（默认，可插拔见 CommissionInfo）
DEFAULT_N_STOCKS = 50       # 默认股票数
REBALANCE_FREQ = 21         # 月频调仓
MAX_UNIVERSE = 150          # 默认股票池上限（超出则确定性抽样）
DEFAULT_PRICE_LIMIT = 0.098  # 涨跌停幅度默认值（主板；合成数据路径统一使用）
CODE_VERSION = "post-audit-4"   # 回测代码版本号（参与缓存指纹，逻辑变更时递增）
# 归因诚实声明：Brinson 分解需要行业分类、Barra 暴露需要多因子库，
# 数据缺位时不以常数拆分冒充实测（post-audit-4 移除伪归因）。
ATTRIBUTION_CAVEAT = "行业归因(Brinson)与风险暴露(Barra)需要行业分类与多因子库，当前版本不提供"


# ═══════════════════════════════════════════
# Module-level helpers (shared by both Tools)
# ═══════════════════════════════════════════


# 基准指数显式映射表：纯数字代码 → (qlib 目录代码, 显示名)
_BENCHMARK_MAP = {
    "000300": ("SH000300", "沪深300"),
    "000905": ("SH000905", "中证500"),
    "000906": ("SH000906", "中证800"),
    "000852": ("SH000852", "中证1000"),
    "000001": ("SH000001", "上证指数"),
    "000016": ("SH000016", "上证50"),
    "399001": ("SZ399001", "深证成指"),
    "399006": ("SZ399006", "创业板指"),
}


def normalize_benchmark(benchmark: str) -> Tuple[str, Optional[str]]:
    """
    归一化基准代码 → (qlib 目录代码, 显示名或 None)。

    支持 '000300'、'000300.SH'、'sh000300'、'SH000300' 等形式。
    显式带 SH/SZ/BJ 前缀的代码直接透传（按数字段查显示名）；
    纯数字/带后缀形式的未知代码原样透传且返回 None（调用方应标注"基准未识别" caveat）。
    """
    if not benchmark:
        return "SH000300", "沪深300"
    raw = str(benchmark).strip()
    code_part = raw.split(".")[0].strip().upper()
    if len(code_part) > 2 and code_part[:2] in ("SH", "SZ", "BJ"):
        digits = code_part[2:]
        mapped = _BENCHMARK_MAP.get(digits)
        return code_part, (mapped[1] if mapped else None)
    mapped = _BENCHMARK_MAP.get(code_part)
    if mapped:
        return mapped
    # 未识别：原样透传，由调用方标注 caveat
    return raw.upper(), None


def _estimate_trading_days(start_date: str, end_date: str) -> int:
    """日历区间 → 估算交易日数"""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days
    return max(int(total_days * TRADING_DAYS_PER_YEAR / 365), 20)


def _factor_count(strategy_config: Optional[StrategyConfig]) -> int:
    """提取因子数量"""
    if strategy_config and strategy_config.factors:
        return len(strategy_config.factors)
    return 0


def _parse_constraints(
    constraints: Optional[PortfolioConstraints], N: int
) -> Dict[str, Any]:
    """解析组合约束"""
    if constraints is None:
        return {"n_hold": max(N // 5, 10), "max_single_w": 0.05, "long_only": True}
    max_pos = constraints.max_positions or 80
    min_pos = constraints.min_positions or 20
    return {
        "n_hold": min(max_pos, max(min_pos, max(N // 5, 10))),
        "max_single_w": constraints.max_single_weight or 1.0,
        "long_only": True if constraints.long_only is None else constraints.long_only,
    }


def _generate_market_data(
    rng: np.random.Generator, T: int, N: int, n_factors: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成合成市场数据。

    个股收益包含缓慢演化的持久 alpha（AR(1) 过程）。
    防前视设计：alpha 次日兑现 —— stock_returns[t] 由 alphas[t-1] 驱动
    （t=0 无滞后 alpha，仅市场 beta + 特异项），而因子得分 factor_scores[t]
    基于 alphas[t]（t 日收盘可见），因此信号只能预测次日收益。

    Parameters
    ----------
    rng : np.random.Generator
    T : int — 交易日数
    N : int — 股票数
    n_factors : int — 因子数量

    Returns
    -------
    market_returns : (T,) — 基准日收益
    stock_returns  : (T, N) — 个股日收益
    factor_scores  : (T, N) — 因子得分（Z-score 标准化）
    benchmark_prices : (T,) — 基准价格序列
    """
    # ── 基准收益：年化 8% 漂移 + 20% 波动 ──
    mu_market = 0.08 / TRADING_DAYS_PER_YEAR
    sigma_market = 0.20 / math.sqrt(TRADING_DAYS_PER_YEAR)
    market_returns = rng.normal(mu_market, sigma_market, T).astype(np.float64)
    benchmark_prices = 100.0 * np.exp(np.cumsum(market_returns))

    # ── 个股收益：beta * 市场 + 持久 alpha + 特异波动 ──
    betas = rng.normal(1.0, 0.15, N).astype(np.float64)

    # Alpha 为 AR(1) 过程：年化 4% 标准差，高持续性
    alpha_annual_std = 0.04
    alpha_daily_std = alpha_annual_std / math.sqrt(TRADING_DAYS_PER_YEAR)
    phi_alpha = 0.95  # 半衰期 ~ 13 个交易日
    alpha_innov_std = alpha_daily_std * math.sqrt(1.0 - phi_alpha ** 2)

    alphas = np.zeros((T, N), dtype=np.float64)
    alphas[0] = rng.normal(0.0, alpha_daily_std, N)
    for t in range(1, T):
        alphas[t] = phi_alpha * alphas[t - 1] + rng.normal(0.0, alpha_innov_std, N)

    specific_vol = 0.25 / math.sqrt(TRADING_DAYS_PER_YEAR)
    epsilons = rng.normal(0.0, specific_vol, (T, N)).astype(np.float64)

    # alpha 次日兑现：t 日收益由 t-1 日已实现的 alpha 驱动（防前视）
    alpha_realized = np.zeros((T, N), dtype=np.float64)
    if T > 1:
        alpha_realized[1:] = alphas[:-1]

    stock_returns = (
        np.outer(market_returns, betas) + alpha_realized + epsilons
    )

    # ── 因子得分：alpha + 噪声（IC ≈ 0.32，基于 alphas[t]，仅可预测 t+1 收益）──
    # noise_std = 3 * alpha_daily_std → IC = 1/sqrt(1+9) ≈ 0.316
    signal_noise_std = 3.0 * alpha_daily_std
    factor_scores = alphas + rng.normal(0.0, signal_noise_std, (T, N)).astype(np.float64)

    # Z-score 逐日标准化
    mean_t = np.mean(factor_scores, axis=1, keepdims=True)
    std_t = np.std(factor_scores, axis=1, keepdims=True)
    factor_scores = (factor_scores - mean_t) / (std_t + 1e-10)

    return market_returns, stock_returns, factor_scores, benchmark_prices


def _equal_weight_targets(
    scores_row: np.ndarray,
    valid_mask: np.ndarray,
    n_hold: int,
    long_only: bool,
    max_single_w: float,
) -> np.ndarray:
    """
    单日信号行 → 等权目标权重向量（合成/真实两条路径共用）。

    仅 finite 且 valid_mask 为真的股票参与排名，取前 n_hold 只等权配置；
    long_only 时按 max_single_w 截断并重新归一。无有效股票时返回全零向量
    （全零目标在执行层不产生任何订单）。
    """
    N = scores_row.shape[0]
    scores = np.where(valid_mask & np.isfinite(scores_row), scores_row, -np.inf)
    ranked = np.argsort(scores)[::-1]
    selected = [c for c in ranked if np.isfinite(scores[c])][:n_hold]

    target = np.zeros(N, dtype=np.float64)
    if selected:
        target[selected] = 1.0 / len(selected)
        if long_only:
            target = np.clip(target, 0.0, max_single_w)
            total = float(np.sum(target))
            if total > 0:
                target /= total  # 重新归一
    return target


def _annualized_return(daily_returns: np.ndarray) -> float:
    """日收益序列 → 简单年化（mean × 252）；空序列返回 0。"""
    if daily_returns is None or len(daily_returns) == 0:
        return 0.0
    return float(np.mean(daily_returns)) * TRADING_DAYS_PER_YEAR


def _annualized_sharpe(daily_returns: np.ndarray) -> float:
    """日收益序列 → 年化夏普（mean/std(ddof=1)×√252）；样本<2 或零波动返回 0。"""
    if daily_returns is None or len(daily_returns) < 2:
        return 0.0
    std = float(np.std(daily_returns, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(daily_returns)) / std * math.sqrt(TRADING_DAYS_PER_YEAR)


def _price_limit_ratio(code: str) -> float:
    """
    按代码前缀返回板块涨跌停幅度：
    - 30x（创业板）/ 68x（科创板）→ 0.195（±20%）
    - 4x / 8x / 92x 开头（北交所）→ 0.29（±30%）
    - 其余主板 → 0.098（±10%）
    ST 无法从代码判断，统一按所属板块幅度处理。
    兼容 'SH600000'/'sz300750' 等带前缀形式（仅取数字段判断）。
    """
    digits = "".join(ch for ch in str(code).upper() if ch.isdigit())
    if digits.startswith(("30", "68")):
        return 0.195
    if digits[:1] in ("4", "8") or digits.startswith("92"):
        return 0.29
    return DEFAULT_PRICE_LIMIT


def _apply_execution_constraints(
    old_weights: np.ndarray,
    target_weights: np.ndarray,
    day_returns: np.ndarray,
    limit_ratios: np.ndarray,
    cash_weight: float,
) -> Tuple[np.ndarray, float, int, int]:
    """
    执行日撮合：按 A股涨跌停约束过滤调仓订单。

    - 买入订单（目标权重 > 旧持仓）：当日涨幅 >= 板块幅度（涨停）→ 取消，
      对应目标权重回流现金（现金收益按 0 计）；
    - 卖出订单（目标权重 < 旧持仓）：当日跌幅 <= -板块幅度（跌停）→ 卖出失败，
      该股票保持旧权重。

    T+1 语义说明：本引擎只在调仓执行日交易（每 REBALANCE_FREQ 个交易日一次），
    同一执行日内"同日买、同日卖"的回转交易在结构上不可能发生，天然满足 T+1；
    因跌停滞留的仓位无法当日止损，只能等到下一个执行日（约 21 个交易日后）
    再次尝试卖出 —— 等效于被强制的 T+1 延迟退出，期间继续承担该股票涨跌。

    资金守恒：跌停卖单筹资不足时，买单按比例缩量成交
    （对应 A股资金不足时买单部分成交的实际规则）；未配置余额视为可部署现金，
    保证现金不为负、无隐含杠杆；被取消买入的资金全额滞留现金。
    恒等式 sum(effective_weights) + new_cash_weight == 1 恒成立。

    Returns
    -------
    effective_weights : (N,) — 实际生效持仓权重
    new_cash_weight : float — 更新后的现金权重（收益按 0 计）
    n_buy_blocked : int — 涨停拦截的买入笔数
    n_sell_blocked : int — 跌停拦截的卖出笔数
    """
    eps = 1e-12
    delta = target_weights - old_weights
    buy_orders = delta > eps
    sell_orders = delta < -eps

    blocked_buy = buy_orders & (day_returns >= limit_ratios)
    blocked_sell = sell_orders & (day_returns <= -limit_ratios)

    sell_executed = sell_orders & ~blocked_sell
    buy_executed = buy_orders & ~blocked_buy

    sell_proceeds = float(np.sum(-delta[sell_executed]))
    # 可部署资金 = 跟踪现金 + 未配置余额（账户总值 1 中未被持仓/现金记账的部分；
    # 引擎允许从空账直接建仓——如真实路径首信号日动量尚未成熟、旧持仓为空）+ 卖出回款。
    # 恒等式 sum(effective) + new_cash == 1 由此保持，无隐含杠杆。
    unallocated = max(0.0, 1.0 - float(np.sum(old_weights)) - cash_weight)
    buy_budget = cash_weight + unallocated + sell_proceeds
    buy_demand = float(np.sum(delta[buy_executed]))
    fill_scale = min(1.0, buy_budget / buy_demand) if buy_demand > eps else 1.0

    effective = np.where(blocked_sell, old_weights, target_weights)
    effective = np.where(blocked_buy, old_weights, effective)
    effective[buy_executed] = old_weights[buy_executed] + delta[buy_executed] * fill_scale

    executed_buy_amount = float(np.sum(effective[buy_executed] - old_weights[buy_executed]))
    # 被取消的买入不产生现金流动（钱从未离开现金账户）；未配置余额动用后并入现金记账
    new_cash = cash_weight + unallocated + sell_proceeds - executed_buy_amount

    return (
        effective,
        new_cash,
        int(np.count_nonzero(blocked_buy)),
        int(np.count_nonzero(blocked_sell)),
    )


def _format_limit_caveats(
    limit_up_blocked: int, limit_down_blocked: int, final_cash_weight: float
) -> List[str]:
    """涨跌停拦截统计 → 报告 caveats 行（无拦截且现金比例 ≤1% 时返回空）。"""
    caveats: List[str] = []
    if limit_up_blocked or limit_down_blocked:
        caveats.append(
            f"涨跌停约束: {limit_up_blocked} 笔买入因涨停无法成交、"
            f"{limit_down_blocked} 笔卖出因跌停无法成交"
        )
    if final_cash_weight > 0.01:
        caveats.append(
            f"期末现金比例约 {final_cash_weight:.1%}"
            "（部分买入被涨停拦截，资金滞留现金、收益按 0 计）"
        )
    return caveats


def _run_portfolio_simulation(
    stock_returns: np.ndarray,
    factor_scores: np.ndarray,
    N: int,
    n_hold: int,
    max_single_w: float,
    long_only: bool,
    commission: Optional[CommissionInfo] = None,
    codes: Optional[List[str]] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    月频调仓等权组合模拟。

    防前视（pending_weights 模式）：
    - t 日收盘产生的信号只计算目标权重（pending），t+1 日才生效；
    - 换手成本计入生效日；首日建仓视为立即生效（空仓建仓不构成前视收益）；
    - 末一日（t == T-1）的调仓信号跳过（无次日可生效）。

    A股交易约束（执行日撮合）：
    - 涨停（当日涨幅 >= 板块幅度）→ 买入取消、资金回流现金；
    - 跌停（当日跌幅 <= -板块幅度）→ 卖出失败、保留旧权重（T+1 延迟退出，
      见 _apply_execution_constraints 注释）；
    - codes 提供时按板块幅度逐股判定，否则统一按主板 DEFAULT_PRICE_LIMIT
      （合成数据无真实代码，固定 0.098）；
    - 现金权重显式跟踪：买入被拦截后组合部分持现金，现金收益为 0。

    Returns
    -------
    equity : (T,) — 净值曲线
    port_returns : (T,) — 每日组合收益
    turnover_total : float — 双边总换手（sum |delta|，仅计实际成交部分）
    positive_days : int — 正收益天数

    Other Parameters
    ----------------
    stats : dict, optional — 传入时回填
        ``limit_up_blocked`` / ``limit_down_blocked`` / ``final_cash_weight``。
    """
    T = len(stock_returns)
    weights = np.zeros(N, dtype=np.float64)
    old_weights = np.zeros(N, dtype=np.float64)
    equity = np.ones(T, dtype=np.float64)
    port_returns = np.zeros(T, dtype=np.float64)
    turnover_total = 0.0
    positive_days = 0
    cm = commission if commission is not None else DEFAULT_COSTS
    pending = None
    cash_weight = 0.0
    limit_up_blocked = 0
    limit_down_blocked = 0

    if codes is not None:
        limit_ratios = np.array(
            [_price_limit_ratio(c) for c in codes], dtype=np.float64
        )
    else:
        limit_ratios = np.full(N, DEFAULT_PRICE_LIMIT, dtype=np.float64)

    for t in range(T):
        # 1) 生效昨日产生的调仓信号（成本计入生效日；涨跌停订单拦截）
        day_cost = 0.0
        if pending is not None:
            effective, cash_weight, n_up, n_down = _apply_execution_constraints(
                old_weights, pending, stock_returns[t], limit_ratios, cash_weight
            )
            limit_up_blocked += n_up
            limit_down_blocked += n_down
            executed_delta = float(np.sum(np.abs(effective - old_weights)))
            day_cost = cm.turnover_cost(executed_delta)
            turnover_total += executed_delta
            weights = effective.copy()
            old_weights = effective.copy()
            pending = None

        # 2) t 日收盘产生新信号 → 只挂起不生效（末日跳过）
        if ((t == 0) or ((t + 1) % REBALANCE_FREQ == 0)) and (t < T - 1):
            pending = _equal_weight_targets(
                factor_scores[t], np.ones(N, dtype=bool), n_hold, long_only, max_single_w
            )
            if t == 0:
                # 首日空仓建仓，立即生效（不受涨跌停约束）
                weights = pending.copy()
                old_weights = pending.copy()
                pending = None

        # 3) 用当前（已生效）权重结算当日收益；现金权重收益按 0 计
        port_return = float(np.dot(weights, stock_returns[t])) + cash_weight * 0.0 - day_cost
        port_returns[t] = port_return
        equity[t] = equity[t - 1] * (1 + port_return) if t > 0 else 1.0 + port_return
        if port_return > 0:
            positive_days += 1

    if stats is not None:
        stats["limit_up_blocked"] = limit_up_blocked
        stats["limit_down_blocked"] = limit_down_blocked
        stats["final_cash_weight"] = float(cash_weight)

    return equity, port_returns, turnover_total, positive_days


def _classify_period_returns(
    port_returns: np.ndarray,
    benchmark_prices: np.ndarray,
) -> Dict[str, float]:
    """按基准趋势分类的区间年化收益"""
    T = len(port_returns)
    ma_short, ma_long = 20, 60
    regime_rets: Dict[str, list] = {"trending_up": [], "ranging": [], "bearish": []}

    for t in range(max(ma_long, 5), T):
        short_ma = np.mean(benchmark_prices[max(0, t - ma_short) : t])
        long_ma = np.mean(benchmark_prices[max(0, t - ma_long) : t])
        cur = benchmark_prices[t]
        if cur > short_ma * 1.01 and short_ma > long_ma:
            regime = "trending_up"
        elif cur < short_ma * 0.99:
            regime = "bearish"
        else:
            regime = "ranging"
        regime_rets[regime].append(port_returns[t])

    result: Dict[str, float] = {}
    for regime, rets in regime_rets.items():
        result[regime] = (
            float(np.mean(rets) * TRADING_DAYS_PER_YEAR) if len(rets) >= 5 else 0.0
        )
    return result


def _compute_metrics(
    equity: np.ndarray,
    port_returns: np.ndarray,
    market_returns: np.ndarray,
    benchmark_prices: np.ndarray,
    T: int,
    years: float,
    turnover_total: float,
    positive_days: int,
) -> dict:
    """计算全套回测指标，返回 dict"""
    # 年化收益
    annual_return = (equity[-1] / equity[0]) ** (1.0 / years) - 1.0
    benchmark_return = (
        (benchmark_prices[-1] / benchmark_prices[0]) ** (1.0 / years) - 1.0
    )
    excess_return = annual_return - benchmark_return

    # 波动率
    daily_vol = float(np.std(port_returns, ddof=1))
    volatility = daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR)

    # 夏普
    sharpe_ratio = annual_return / volatility if volatility > 1e-10 else 0.0

    # 最大回撤
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = float(np.min(drawdown))

    # Calmar
    calmar_ratio = annual_return / abs(max_drawdown) if abs(max_drawdown) > 1e-10 else 0.0

    # 胜率
    win_rate = positive_days / T if T > 0 else 0.0

    # 年化换手（单边）
    annual_turnover = (turnover_total / 2.0) / years if years > 0 else 0.0

    # 平均持有期
    avg_holding_period = (
        TRADING_DAYS_PER_YEAR / annual_turnover if annual_turnover > 1e-10 else float(TRADING_DAYS_PER_YEAR)
    )

    # 信息比 & 统计检验
    excess_daily = port_returns - market_returns
    tracking_error = float(np.std(excess_daily, ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)
    information_ratio = excess_return / tracking_error if tracking_error > 1e-10 else 0.0

    mean_excess = float(np.mean(excess_daily))
    se_excess = float(np.std(excess_daily, ddof=1)) / math.sqrt(T)
    t_statistic = mean_excess / se_excess if se_excess > 1e-10 else 0.0
    p_value = math.erfc(abs(t_statistic) / math.sqrt(2.0))

    return {
        "annual_return": annual_return,
        "benchmark_return": benchmark_return,
        "excess_return": excess_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "annual_turnover": annual_turnover,
        "avg_holding_period": avg_holding_period,
        "information_ratio": information_ratio,
        "t_statistic": t_statistic,
        "p_value": p_value,
        # 伪归因（brinson_allocation/barra_exposure 常数拆分）已于 post-audit-4 移除：
        # 行业归因与多因子暴露需相应数据库，缺位时以 ATTRIBUTION_CAVEAT 明示而非伪造。
        "period_returns": _classify_period_returns(port_returns, benchmark_prices),
    }


# ═══════════════════════════════════════════
# RunBacktestTool
# ═══════════════════════════════════════════


def get_financials_asof(code: str, asof_date: str) -> Optional[dict]:
    """
    防前视财务查询辅助：按公告日对齐返回 code 在 asof_date 当日可见的最新一期财务快照。
    供回测信号因子（估值锚、财务分位等）在历史时点安全引用财务数据，
    避免使用最新一期财务造成前视偏差。
    返回 dict（含 quarter/epsTTM 等）或 None。
    """
    try:
        from trader3.v2.announcement_calendar import AnnouncementCalendar

        return AnnouncementCalendar().financials_asof(code, asof_date)
    except Exception:
        return None


# ── 真实面板共享组件（run_backtest 与 WFA 共用）──


def _momentum_scores(close_matrix: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    对齐收盘矩阵 → 20 日对数动量分数矩阵（真实路径信号定义，两处共用）。

    价格缺失/非正处为 -inf（自然落选）；前 lookback 日为 -inf（动量未成熟）。
    """
    T, N = close_matrix.shape
    momentum = np.full((T, N), -np.inf, dtype=np.float64)
    if T > lookback:
        prev_all = close_matrix[:-lookback]
        cur_all = close_matrix[lookback:]
        mask = (
            (prev_all > 0) & (cur_all > 0)
            & np.isfinite(prev_all) & np.isfinite(cur_all)
        )
        rows, cols = np.where(mask)
        momentum[rows + lookback, cols] = np.log(cur_all[rows, cols] / prev_all[rows, cols])
    return momentum


def _build_aligned_panel(dp: Any, codes: List[str], start_date: str, end_date: str) -> Tuple[
    List[str], List[str], np.ndarray, np.ndarray, np.ndarray, int
]:
    """
    读取并按交易日历对齐面板（逐股容错：契约校验失败的股票跳过）。

    Returns
    -------
    time_axis : List[str] — 交易日时间轴
    codes_list : List[str] — 实际入池代码
    close_matrix / returns_matrix : (T, M) float64（缺失为 0；收益跨缺口 ≤5 天不计算）
    valid_flags : (T, M) bool
    n_skipped : int — 因契约校验失败被跳过的股票数

    Raises
    ------
    RuntimeError — 可用股票 <10 或区间交易日 <30。
    """
    stock_closes: Dict[str, np.ndarray] = {}
    stock_dates: Dict[str, List[str]] = {}
    n_skipped = 0
    for code in codes:
        try:
            close, dates = dp.load_stock(code.lower(), "close", start_date, end_date)
        except Exception:
            n_skipped += 1
            continue
        if len(close) >= 30:  # 至少 30 个交易日
            stock_closes[code] = close
            stock_dates[code] = dates

    if len(stock_closes) < 10:
        raise RuntimeError(f"真实数据可用股票数不足 ({len(stock_closes)} < 10)")

    cal = dp.calendar()
    start_idx = dp._lower_bound(cal, start_date)
    end_idx = dp._upper_bound(cal, end_date)
    time_axis = cal[start_idx:end_idx]
    T = len(time_axis)
    if T < 30:
        raise RuntimeError("回测区间交易日不足 30 天")

    M = len(stock_closes)
    close_matrix = np.zeros((T, M), dtype=np.float64)
    valid_flags = np.zeros((T, M), dtype=bool)

    date_index = {d: i for i, d in enumerate(time_axis)}  # O(1) 日期定位
    codes_list = list(stock_closes.keys())
    for j, code in enumerate(codes_list):
        close = stock_closes[code]
        dates = stock_dates[code]
        for i, d in enumerate(dates):
            pos = date_index.get(d)
            if pos is not None and close[i] > 0 and np.isfinite(close[i]):
                close_matrix[pos, j] = close[i]
                valid_flags[pos, j] = True

    # 收益率（逐股计算日收益，缺口不超过5天，避免跨停牌期）
    returns_matrix = np.zeros_like(close_matrix)
    for j in range(M):
        col = close_matrix[:, j]
        valid_pos = np.where(col > 0)[0]
        for idx in range(1, len(valid_pos)):
            i_prev = valid_pos[idx - 1]
            i_curr = valid_pos[idx]
            if i_curr - i_prev <= 5:
                returns_matrix[i_curr, j] = (col[i_curr] - col[i_prev]) / col[i_prev]

    return time_axis, codes_list, close_matrix, returns_matrix, valid_flags, n_skipped


def _run_momentum_backtest(
    close_matrix: np.ndarray,
    returns_matrix: np.ndarray,
    valid_flags: np.ndarray,
    *,
    n_hold: int,
    max_single_w: float,
    commission: Optional[CommissionInfo] = None,
    codes: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, int, Dict[str, Any]]:
    """
    真实面板动量组合回测（月频调仓；与合成引擎共用执行语义）：

    - 防前视 pending：t 日收盘动量信号只挂起，t+1 生效（首日建仓立即生效、末日跳过）；
    - 执行日涨跌停拦截 + 显式现金跟踪（见 _apply_execution_constraints）；
    - 供 run_backtest 真实路径调用。

    Returns
    -------
    equity, port_returns, turnover_total, positive_days, stats
    （stats 含 limit_up_blocked / limit_down_blocked / final_cash_weight）
    """
    momentum = _momentum_scores(close_matrix)
    T, N = returns_matrix.shape
    weights = np.zeros(N, dtype=np.float64)
    old_weights = np.zeros(N, dtype=np.float64)
    equity = np.ones(T, dtype=np.float64)
    port_returns = np.zeros(T, dtype=np.float64)
    turnover_total = 0.0
    positive_days = 0
    cm = commission if commission is not None else DEFAULT_COSTS
    pending = None
    cash_weight = 0.0
    limit_up_blocked = 0
    limit_down_blocked = 0

    if codes is not None:
        limit_ratios = np.array(
            [_price_limit_ratio(c) for c in codes], dtype=np.float64
        )
    else:
        limit_ratios = np.full(N, DEFAULT_PRICE_LIMIT, dtype=np.float64)

    for t in range(T):
        # 1) 生效昨日收盘产生的调仓信号（成本计入生效日；涨跌停订单拦截）
        day_cost = 0.0
        if pending is not None:
            effective, cash_weight, n_up, n_down = _apply_execution_constraints(
                old_weights, pending, returns_matrix[t], limit_ratios, cash_weight
            )
            limit_up_blocked += n_up
            limit_down_blocked += n_down
            executed_delta = float(np.sum(np.abs(effective - old_weights)))
            day_cost = cm.turnover_cost(executed_delta)
            turnover_total += executed_delta
            weights = effective.copy()
            old_weights = effective.copy()
            pending = None

        # 2) t 日收盘信号 → 只计算目标权重挂起，t+1 生效（末日跳过）
        if ((t == 0) or ((t + 1) % REBALANCE_FREQ == 0)) and (t < T - 1):
            pending = _equal_weight_targets(
                momentum[t], valid_flags[t], n_hold, True, max_single_w
            )
            if t == 0:
                # 首日空仓建仓，立即生效（不受涨跌停约束）
                weights = pending.copy()
                old_weights = pending.copy()
                pending = None

        # 3) 用当前（已生效）权重结算当日收益；现金权重收益按 0 计
        port_return = (
            float(np.dot(weights, returns_matrix[t])) + cash_weight * 0.0 - day_cost
        )
        port_returns[t] = port_return
        equity[t] = equity[t - 1] * (1 + port_return) if t > 0 else 1.0 + port_return
        if port_return > 0:
            positive_days += 1

    stats: Dict[str, Any] = {
        "limit_up_blocked": limit_up_blocked,
        "limit_down_blocked": limit_down_blocked,
        "final_cash_weight": float(cash_weight),
    }
    return equity, port_returns, turnover_total, positive_days, stats


def _open_qlib_dp() -> Any:
    """打开默认 qlib 数据源（不可用时抛异常；测试可替换本函数注入临时数据源）。"""
    from trader3.data_provider import QlibDataProvider

    return QlibDataProvider()


def _load_wfa_panel(dp: Any) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    WFA 全历史面板：成分池取 csi300（按日历起点 asof 过滤，防幸存者偏差），
    缺失时回退 all；超上限按固定种子确定性抽样。

    Returns
    -------
    (codes_list, close_matrix, returns_matrix, valid_flags)
    """
    cal = dp.calendar()
    if not cal:
        raise RuntimeError("qlib 日历为空")
    start_date, end_date = cal[0], cal[-1]

    codes: List[str] = []
    for universe in ("csi300", "all"):
        try:
            codes = dp.instruments(universe, asof_date=start_date) or []
        except Exception:
            codes = []
        if codes:
            break
    if not codes:
        raise RuntimeError("无可用成分股")

    if len(codes) > MAX_UNIVERSE:
        rng = np.random.default_rng(42)
        pick = rng.choice(len(codes), size=MAX_UNIVERSE, replace=False)
        codes = sorted(codes[i] for i in pick)

    _, codes_list, close_matrix, returns_matrix, valid_flags, _ = _build_aligned_panel(
        dp, codes, start_date, end_date
    )
    return codes_list, close_matrix, returns_matrix, valid_flags


def _run_wfa_rolling(
    stock_returns: np.ndarray,
    factor_scores: np.ndarray,
    train_window: int,
    test_window: int,
    step: int,
) -> Tuple[List[dict], List[float], List[float], List[float], List[float],
           List[np.ndarray], np.ndarray]:
    """
    滚动 IS/OOS（真实/合成面板共用）。

    每窗以训练段因子均值确定等权 top-k 持仓，固定应用于紧随其后的测试段 ——
    训练段收盘信息最早于测试段首日生效，与回测引擎的 pending 次日生效语义一致；
    窗口间前进 step 天。OOS 日收益逐窗拼接（step ≥ test_window 时天然非重叠），
    供汇总指标在非重叠样本上计算。

    Returns
    -------
    windows, is_ann_list, oos_ann_list, is_sr_list, oos_sr_list,
    param_weights, oos_concat（拼接 OOS 日收益 np.ndarray）
    """
    T, N = stock_returns.shape
    top_k = max(N // 5, 10)

    windows: List[dict] = []
    is_ann_list: List[float] = []
    oos_ann_list: List[float] = []
    is_sr_list: List[float] = []
    oos_sr_list: List[float] = []
    param_weights: List[np.ndarray] = []
    oos_chunks: List[np.ndarray] = []

    w = 0
    s = 0
    while s + train_window + test_window <= T:
        is_slice = slice(s, s + train_window)
        oos_slice = slice(s + train_window, s + train_window + test_window)

        # IS 因子均值 → 等权持仓（忽略 -inf/无效分数：仅按有效观测平均，无观测者落选）
        fs_block = factor_scores[is_slice]
        finite_mask = np.isfinite(fs_block)
        obs_counts = finite_mask.sum(axis=0)
        obs_sums = np.where(finite_mask, fs_block, 0.0).sum(axis=0)
        is_signal = np.full(N, -np.inf, dtype=np.float64)
        np.divide(obs_sums, obs_counts, out=is_signal, where=obs_counts > 0)

        ranked = np.argsort(is_signal)[::-1]
        selected = ranked[:top_k]
        weights = np.zeros(N, dtype=np.float64)
        weights[selected] = 1.0 / top_k

        # IS 表现
        is_rets = stock_returns[is_slice] @ weights
        is_ann = float(np.mean(is_rets)) * TRADING_DAYS_PER_YEAR
        is_std = float(np.std(is_rets, ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)
        is_sr = is_ann / is_std if is_std > 1e-10 else 0.0

        # OOS 表现（固定 IS 权重 → 测试段日收益序列，拼接进非重叠 OOS 样本）
        oos_rets = stock_returns[oos_slice] @ weights
        oos_ann = float(np.mean(oos_rets)) * TRADING_DAYS_PER_YEAR
        oos_std = float(np.std(oos_rets, ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)
        oos_sr = oos_ann / oos_std if oos_std > 1e-10 else 0.0

        windows.append({
            "window": w,
            "is_return": is_ann,
            "oos_return": oos_ann,
            "is_sharpe": is_sr,
            "oos_sharpe": oos_sr,
            "oos_start": s + train_window,
            "oos_days": int(test_window),
        })
        is_ann_list.append(is_ann)
        oos_ann_list.append(oos_ann)
        is_sr_list.append(is_sr)
        oos_sr_list.append(oos_sr)
        param_weights.append(weights.copy())
        oos_chunks.append(oos_rets)

        s += step
        w += 1

    oos_concat = (
        np.concatenate(oos_chunks) if oos_chunks else np.array([], dtype=np.float64)
    )
    return (
        windows, is_ann_list, oos_ann_list, is_sr_list, oos_sr_list,
        param_weights, oos_concat,
    )


class RunBacktestTool(BaseTool):
    """运行回测 (M1: 真实回测引擎)"""

    tool_name = "run_backtest"
    tool_description = "运行策略回测，返回标准化回测报告（收益/风险/归因/换手/分时段表现/统计检验）"
    tool_version = "1.0.0"
    tool_category = "backtest"

    def __init__(self):
        super().__init__()
        self._cache_dir = self._resolve_cache_dir()

    # ── 缓存 ──

    def _resolve_cache_dir(self) -> str:
        """shared_state/backtest_cache/"""
        base = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "shared_state")
        )
        cache_dir = os.path.join(base, "backtest_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _fingerprint(
        self,
        strategy_config: Optional[StrategyConfig],
        universe: Optional[List[str]],
        start_date: str,
        end_date: str,
        constraints: Optional[PortfolioConstraints],
        benchmark: str,
        commission: Optional[CommissionInfo] = None,
        *,
        engine_tag: str = "real",
        data_end: str = "",
    ) -> str:
        """sha256 策略指纹（含 constraints/引擎标识/数据末端/代码版本，防张冠李戴命中）"""
        factors = []
        if strategy_config and strategy_config.factors:
            factors = [
                {"name": f.name, "weight": f.weight, "direction": f.direction}
                for f in strategy_config.factors
            ]
        data = {
            "strategy": strategy_config.name if strategy_config else "default",
            "factors": factors,
            "universe": sorted(universe) if universe else [],
            "start": start_date,
            "end": end_date,
            "benchmark": benchmark,
            "costs": commission.as_dict() if commission else None,
            "constraints": constraints.to_dict() if constraints is not None else None,
            "engine_tag": engine_tag,          # 'real' / 'synthetic'
            "data_end": data_end,              # qlib 日历末日
            "code_version": CODE_VERSION,
        }
        raw = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _probe_engine(self) -> Tuple[str, str]:
        """探测可用引擎与数据末端（不执行回测）：('real'|'synthetic', data_end)"""
        try:
            from trader3.data_provider import find_qlib_dir

            qdir = find_qlib_dir()
            if qdir:
                cal_path = os.path.join(qdir, "calendars", "day.txt")
                with open(cal_path, "r") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                if lines:
                    return "real", lines[-1]
        except Exception:
            pass
        return "synthetic", ""

    def _cache_path(self, fp: str) -> str:
        return os.path.join(self._cache_dir, f"{fp}.json")

    def _load_cache(self, fp: str) -> Optional[Trader3Response]:
        path = self._cache_path(fp)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # 指纹完整匹配校验：存储指纹不一致（旧格式/串参数）一律视为未命中
            if saved.pop("_fingerprint", None) != fp:
                return None
            return self._deserialize_response(saved)
        except Exception:
            return None

    def _save_cache(self, fp: str, response: Trader3Response) -> None:
        path = self._cache_path(fp)
        try:
            saved = self._serialize_response(response)
            saved["_fingerprint"] = fp
            with open(path, "w", encoding="utf-8") as f:
                json.dump(saved, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # cache is best-effort

    def _serialize_response(self, response: Trader3Response) -> dict:
        data = response.to_dict()
        if isinstance(response.data, BacktestReport):
            data["_data_type"] = "BacktestReport"
            data["data"] = asdict(response.data)
        else:
            data["_data_type"] = "unknown"
        return data

    def _deserialize_response(self, data: dict) -> Trader3Response:
        data_type = data.pop("_data_type", None)
        if data_type == "BacktestReport":
            data["data"] = BacktestReport(**data["data"])
        charts = [ChartSpec(**c) for c in data.get("charts", [])]
        data["charts"] = charts
        return Trader3Response(**data)

    # ── execute ──

    def execute(
        self,
        strategy_config: Optional[StrategyConfig] = None,
        universe: Optional[List[str]] = None,
        start_date: str = "2020-01-01",
        end_date: str = "2025-12-31",
        constraints: Optional[PortfolioConstraints] = None,
        benchmark: str = "000300.SH",
        commission: Optional[CommissionInfo] = None,
    ) -> Trader3Response:
        """执行回测（M8: 真实 qlib 数据优先 → 向量化合成回退 + 缓存）"""
        # 先探测引擎与数据末端，指纹含 engine_tag，避免真实/合成结果串缓存
        engine_tag, data_end = self._probe_engine()
        fp = self._fingerprint(
            strategy_config, universe, start_date, end_date,
            constraints, benchmark, commission,
            engine_tag=engine_tag, data_end=data_end,
        )

        cached = self._load_cache(fp)
        if cached is not None:
            return cached

        # M8: 优先真实数据
        try:
            result = self._real_data_backtest(
                strategy_config, universe, start_date, end_date, constraints, benchmark, commission
            )
        except Exception as e:
            # 真实数据不可用 → 回退合成数据
            result = self._vectorized_backtest(
                strategy_config, universe, start_date, end_date, constraints, benchmark, commission
            )
            result.caveats.append(f"真实数据不可用，回退合成数据: {e}")
            if engine_tag != "synthetic":
                # 实际落到的引擎与探测不符 → 以合成引擎指纹保存
                fp = self._fingerprint(
                    strategy_config, universe, start_date, end_date,
                    constraints, benchmark, commission,
                    engine_tag="synthetic", data_end=data_end,
                )

        self._save_cache(fp, result)
        return result

    # ── M8: 真实 qlib 数据回测 ──

    def _real_data_backtest(
        self,
        strategy_config: Optional[StrategyConfig],
        universe: Optional[List[str]],
        start_date: str,
        end_date: str,
        constraints: Optional[PortfolioConstraints],
        benchmark: str,
        commission: Optional[CommissionInfo] = None,
    ) -> Trader3Response:
        """
        基于真实 qlib 数据的回测。

        策略：动量因子（20日收益率）选股，月频调仓，等权持有。
        基准：CSI300 指数（或用户指定）。
        """
        from trader3.data_provider import QlibDataProvider

        dp = QlibDataProvider()
        sampled_note = ""

        # ── 确定股票池 ──
        if not universe:
            # 默认用 CSI300 成分股；按窗口起点过滤成分（防幸存者偏差）
            codes = dp.instruments("csi300", asof_date=start_date)
            # 限制数量：确定性抽样（固定种子），不用字典序截断
            if len(codes) > MAX_UNIVERSE:
                rng = np.random.default_rng(42)
                pick = rng.choice(len(codes), size=MAX_UNIVERSE, replace=False)
                codes = sorted(codes[i] for i in pick)
                sampled_note = (
                    f"窗口起点成分股超过 {MAX_UNIVERSE} 只，"
                    f"已用固定种子(42)确定性抽样至 {MAX_UNIVERSE} 只"
                )
        else:
            codes = [c.upper() for c in universe]

        # 规范化代码: SH600000 -> sh600000 (目录名)
        def _norm(c: str) -> str:
            return c.lower()

        # ── 面板构建（逐股容错 + 日历对齐，与 WFA 共享 _build_aligned_panel）──
        time_axis, codes_list, close_matrix, returns_matrix, valid_flags, n_skipped = (
            _build_aligned_panel(dp, codes, start_date, end_date)
        )
        T = len(time_axis)

        # ── 基准收益（显式映射，不再 startswith("000") 一刀切） ──
        benchmark_code, benchmark_name = normalize_benchmark(benchmark)
        bench_caveat = (
            f"基准未识别: {benchmark}，按原始代码 {benchmark_code} 计算"
            if benchmark_name is None else ""
        )

        bench_close, bench_dates = dp.load_stock(_norm(benchmark_code), "close", start_date, end_date)
        bench_returns = np.zeros(T, dtype=np.float64)
        if len(bench_close) > 1:
            # 对齐基准到时间轴
            bench_map = {d: close for d, close in zip(bench_dates, bench_close)}
            bench_series = np.zeros(T, dtype=np.float64)
            for i, d in enumerate(time_axis):
                if d in bench_map and bench_map[d] > 0:
                    bench_series[i] = bench_map[d]
            bench_pos = np.where(bench_series > 0)[0]
            for idx in range(1, len(bench_pos)):
                i_prev = bench_pos[idx - 1]
                i_curr = bench_pos[idx]
                if i_curr - i_prev <= 5:
                    bench_returns[i_curr] = (bench_series[i_curr] - bench_series[i_prev]) / bench_series[i_prev]
        benchmark_prices = 100.0 * np.cumprod(1.0 + bench_returns)

        # ── 组合模拟（月频调仓，动量前 n_hold；与合成引擎共用执行语义）──
        n_hold = min(max(len(codes_list) // 5, 10), 50)
        equity, port_returns, turnover_total, positive_days, sim_stats = (
            _run_momentum_backtest(
                close_matrix, returns_matrix, valid_flags,
                n_hold=n_hold, max_single_w=0.05,
                commission=commission, codes=codes_list,
            )
        )

        # ── 指标 ──
        years = T / TRADING_DAYS_PER_YEAR
        metrics = _compute_metrics(
            equity, port_returns, bench_returns, benchmark_prices,
            T, years, turnover_total, positive_days,
        )

        report = BacktestReport(
            start_date=start_date,
            end_date=end_date,
            benchmark=benchmark_name or benchmark_code,
            **metrics,
            equity_curve=list(equity),
        )

        name = strategy_config.name if strategy_config else "动量策略"
        pool_desc = (
            f"真实 qlib 数据回测（{len(codes_list)} 只股票，"
            f"取 {start_date} 窗口起点成分池{'，固定种子确定性抽样' if sampled_note else ''}）"
        )
        caveats = [
            pool_desc,
            "策略：20日动量因子，月频调仓，等权持有（信号次日生效，无同日前视）",
            "已扣除印花税/佣金/冲击成本",
            "未处理停牌；涨跌停按板块幅度拦截（ST 无法从代码判断，统一按板块幅度处理）",
            ATTRIBUTION_CAVEAT,
        ]
        caveats.extend(
            _format_limit_caveats(
                sim_stats["limit_up_blocked"],
                sim_stats["limit_down_blocked"],
                sim_stats["final_cash_weight"],
            )
        )
        if sampled_note:
            caveats.append(sampled_note)
        if bench_caveat:
            caveats.append(bench_caveat)
        if n_skipped:
            caveats.append(f"{n_skipped} 只股票因数据契约校验失败被跳过")
        return Trader3Response(
            success=True,
            data=report,
            summary=(
                f"[{name}·真实数据] {start_date}~{end_date} 年化 {report.annual_return:.1%}, "
                f"夏普 {report.sharpe_ratio:.2f}, 超额 {report.excess_return:.1%}, "
                f"最大回撤 {report.max_drawdown:.1%}"
            ),
            key_metrics={
                "年化收益": report.annual_return,
                "超额收益": report.excess_return,
                "夏普比": report.sharpe_ratio,
                "最大回撤": report.max_drawdown,
                "信息比": report.information_ratio,
                "年化换手": report.annual_turnover,
                "t统计量": report.t_statistic,
            },
            charts=[
                ChartSpec(
                    chart_type="line",
                    title=f"净值曲线 vs {benchmark_name or benchmark_code} 基准",
                    data={
                        "equity_curve": list(equity),
                        "benchmark": list(benchmark_prices),
                    },
                    description="真实数据动量策略净值 vs 基准",
                ),
            ],
            caveats=caveats,
        )

    def _qlib_backtest(
        self,
        strategy_config: Optional[StrategyConfig],
        universe: Optional[List[str]],
        start_date: str,
        end_date: str,
        constraints: Optional[PortfolioConstraints],
        benchmark: str,
    ) -> Trader3Response:
        """Qlib 回测（完整 qlib 安装时使用）"""
        raise NotImplementedError("完整 qlib 回测接线规划在后续版本")

    # ── 向量化回测 ──

    def _vectorized_backtest(
        self,
        strategy_config: Optional[StrategyConfig],
        universe: Optional[List[str]],
        start_date: str,
        end_date: str,
        constraints: Optional[PortfolioConstraints],
        benchmark: str,
        commission: Optional[CommissionInfo] = None,
    ) -> Trader3Response:
        """
        向量化回测（无 Qlib 时使用）。
        生成合成市场数据，运行策略模拟，计算全部指标。
        """
        # ── 参数 ──
        N = len(universe) if universe else DEFAULT_N_STOCKS
        if N < 5:
            N = DEFAULT_N_STOCKS

        T = _estimate_trading_days(start_date, end_date)
        years = T / TRADING_DAYS_PER_YEAR

        n_factors = _factor_count(strategy_config)
        cons = _parse_constraints(constraints, N)
        n_hold, max_single_w, long_only = cons["n_hold"], cons["max_single_w"], cons["long_only"]

        # ── 数据生成 ──
        rng = np.random.default_rng(42)
        market_returns, stock_returns, factor_scores, benchmark_prices = _generate_market_data(
            rng, T, N, n_factors
        )

        # ── 组合模拟 ──
        sim_stats: Dict[str, Any] = {}
        equity, port_returns, turnover_total, positive_days = _run_portfolio_simulation(
            stock_returns, factor_scores, N, n_hold, max_single_w, long_only,
            commission, stats=sim_stats,
        )

        # ── 指标 ──
        metrics = _compute_metrics(
            equity, port_returns, market_returns, benchmark_prices,
            T, years, turnover_total, positive_days,
        )

        report = BacktestReport(
            start_date=start_date,
            end_date=end_date,
            benchmark=benchmark,
            **metrics,
            equity_curve=list(equity),
        )

        name = strategy_config.name if strategy_config else "未命名策略"
        return Trader3Response(
            success=True,
            data=report,
            summary=(
                f"[{name}] {start_date}~{end_date} 年化 {report.annual_return:.1%}, "
                f"夏普 {report.sharpe_ratio:.2f}, 超额 {report.excess_return:.1%}, "
                f"最大回撤 {report.max_drawdown:.1%}"
            ),
            key_metrics={
                "年化收益": report.annual_return,
                "超额收益": report.excess_return,
                "夏普比": report.sharpe_ratio,
                "最大回撤": report.max_drawdown,
                "信息比": report.information_ratio,
                "年化换手": report.annual_turnover,
                "t统计量": report.t_statistic,
            },
            charts=[
                ChartSpec(
                    chart_type="line",
                    title="净值曲线 vs 基准",
                    data={
                        "equity_curve": list(equity),
                        "benchmark": list(benchmark_prices),
                    },
                    description="策略净值 vs 基准指数",
                ),
            ],
            caveats=[
                "合成数据回测（无 Qlib），实际表现可能差异显著",
                ATTRIBUTION_CAVEAT,
                "已扣除千分之一印花税 + 万二佣金 + 千分之五冲击成本",
                "组合按月频调仓，等权持有",
                "信号为模拟生成，非真实因子数据",
                "涨跌停约束已接入（合成路径统一主板幅度 9.8%）",
            ]
            + _format_limit_caveats(
                sim_stats.get("limit_up_blocked", 0),
                sim_stats.get("limit_down_blocked", 0),
                sim_stats.get("final_cash_weight", 0.0),
            ),
        )


# ═══════════════════════════════════════════
# WalkForwardAnalysisTool
# ═══════════════════════════════════════════


class WalkForwardAnalysisTool(BaseTool):
    """Walk-Forward Analysis (M1: 真实滚动验证)"""

    tool_name = "walk_forward_analysis"
    tool_description = "滚动 WFA 验证，检测过拟合，返回 IS/OOS 对比 + 参数稳定性 + 过拟合概率"
    tool_version = "1.0.0"
    tool_category = "backtest"

    def execute(
        self,
        strategy_config: Optional[StrategyConfig] = None,
        train_window: int = 252,
        test_window: int = 63,
        step: Optional[int] = None,
    ) -> Trader3Response:
        """
        滚动 WFA（真实 qlib 数据优先）。

        - qlib 可用：在真实面板上滚动 —— 每窗以训练段动量排名确定等权持仓，
          固定应用于测试段（训练段收盘信息最早测试段首日生效，与回测引擎
          pending 语义一致）；OOS 汇总指标基于拼接的非重叠 OOS 日收益计算。
        - step 缺省等于 test_window（OOS 窗口非重叠，显著性不被共享样本抬高）；
          显式传入更小的 step 时 caveats 警告窗口重叠会高估显著性。
        - qlib 不可用：回退种子 123 合成数据，并在 caveats 明示"WFA基于合成数据"。
        """
        eff_step = int(step) if step is not None else int(test_window)

        panel = None
        panel_err: Optional[Exception] = None
        try:
            panel = _load_wfa_panel(_open_qlib_dp())
        except Exception as e:  # 数据缺失/损坏 → 合成回退
            panel_err = e

        if panel is not None:
            codes_list, close_matrix, returns_matrix, _valid_flags = panel
            stock_returns = returns_matrix
            factor_scores = _momentum_scores(close_matrix)
            engine_caveat = "WFA 基于真实 qlib 数据"
            if eff_step >= test_window:
                engine_caveat += "，OOS 窗口非重叠"
            engine_caveat += (
                f"（面板 {stock_returns.shape[0]} 日 × {len(codes_list)} 股，"
                "成分按起点 asof 过滤）"
            )
        else:
            T = max(252 * 6, train_window + test_window + eff_step * 5 + 10)
            N = DEFAULT_N_STOCKS
            rng = np.random.default_rng(123)  # WFA 合成回退独立种子
            _, stock_returns, factor_scores, _ = _generate_market_data(
                rng, T, N, _factor_count(strategy_config)
            )
            engine_caveat = (
                f"WFA基于合成数据（真实 qlib 数据不可用: {panel_err}），"
                "结果仅验证方法学、不代表实际表现"
            )

        (windows, is_returns, oos_returns, is_sharpes, oos_sharpes,
         param_weights, oos_concat) = _run_wfa_rolling(
            stock_returns, factor_scores, train_window, test_window, eff_step
        )

        # ── 参数稳定性 = 相邻窗口权重相关系数均值 ──
        if len(param_weights) >= 2:
            corrs = []
            for i in range(1, len(param_weights)):
                c = np.corrcoef(param_weights[i - 1], param_weights[i])[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
            parameter_stability = float(np.mean(corrs)) if corrs else 0.0
        else:
            parameter_stability = 1.0

        # ── 汇总指标：OOS 基于拼接的非重叠 OOS 日收益 ──
        mean_is_sr = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        oos_mean_return = _annualized_return(oos_concat)
        oos_sharpe = _annualized_sharpe(oos_concat)

        if mean_is_sr > 1e-10:
            overfitting_probability = min(
                max(0.0, 1.0 - oos_sharpe / mean_is_sr), 1.0
            )
        else:
            overfitting_probability = 0.5

        report = WFAReport(
            train_window=train_window,
            test_window=test_window,
            step=eff_step,
            windows=len(windows),
            is_mean_return=float(np.mean(is_returns)) if is_returns else 0.0,
            oos_mean_return=oos_mean_return,
            is_sharpe=mean_is_sr,
            oos_sharpe=oos_sharpe,
            parameter_stability=parameter_stability,
            overfitting_probability=overfitting_probability,
            window_results=windows,
        )

        overlap_note = "非重叠" if eff_step >= test_window else "重叠"
        caveats = [
            engine_caveat,
            f"OOS 汇总基于拼接的{overlap_note} OOS 日收益（共 {int(oos_concat.size)} 个交易日）",
            "每窗以训练段信号确定等权持仓并固定应用于测试段（训练段信息不泄漏至段内收益）",
            "未计入交易成本与涨跌停约束",
            "过拟合概率基于 IS/OOS 夏普比衰减: max(0, 1 - OOS_Sharpe / IS_Sharpe)",
        ]
        if eff_step < test_window:
            caveats.insert(
                1,
                f"step={eff_step} < test_window={test_window}: "
                "OOS 窗口重叠，相邻窗口共享样本会高估显著性",
            )

        return Trader3Response(
            success=True,
            data=report,
            summary=(
                f"WFA 样本外年化 {report.oos_mean_return:.1%}, "
                f"夏普 {report.oos_sharpe:.2f}, "
                f"过拟合概率 {report.overfitting_probability:.0%}, "
                f"参数稳定性 {report.parameter_stability:.0%}"
            ),
            key_metrics={
                "样本内收益": report.is_mean_return,
                "样本外收益": report.oos_mean_return,
                "样本内夏普": report.is_sharpe,
                "样本外夏普": report.oos_sharpe,
                "参数稳定性": report.parameter_stability,
                "过拟合概率": report.overfitting_probability,
                "OOS交易日": int(oos_concat.size),
            },
            caveats=caveats,
        )