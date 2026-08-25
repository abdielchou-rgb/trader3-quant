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
import warnings
from dataclasses import asdict
from datetime import datetime
from typing import Any

import numpy as np

from trader3.base_tool import BaseTool, ChartSpec, Trader3Response
from trader3.models import (
    BacktestReport,
    PortfolioConstraints,
    StrategyConfig,
    WFAReport,
)
from trader3.v2.costs import DEFAULT_COSTS, CommissionInfo

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
CODE_VERSION = "post-audit-8"   # 回测代码版本号（参与缓存指纹，逻辑变更时递增；
# post-audit-8: 极端情景压测(stress_test)与有效IR折算接入，key_metrics 输出结构变更，
#               旧缓存作废；
# post-audit-7: 真实路径接入 Brinson(BHB简化) 行业归因，输出内容变更，旧缓存作废；
# post-audit-6: 数据内容纪元——治愈尾部拼接修复后旧缓存全部作废）

# 极端情景压力测试窗口（A股历史危机样本；run_stress_test 按 [start,end] 交集筛选）
STRESS_PERIODS: dict[str, tuple[str, str]] = {
    "2015_crisis": ("2015-06-15", "2015-09-30"),
    "2016_circuit_breaker": ("2016-01-04", "2016-02-29"),
    "2020_covid": ("2020-01-20", "2020-03-31"),
    "2022_bear": ("2022-01-04", "2022-12-30"),
    "2024_small_cap_crash": ("2024-01-01", "2024-02-29"),
}
STRESS_CAVEAT_FMT = "极端情景压测: {n} 窗口已评估"
EFFECTIVE_IR_CAVEAT_FMT = "有效IR按调仓频率折算(每{n}天一次独立赌注)"
STRESS_SUMMARY_NAME = "_summary"


def _data_version_stamp() -> str:
    """读取增量管线盖的章（qlib_bin 数据内容纪元）；缺失返回空串。"""
    try:
        from trader3.shared_state import SharedState
        v = SharedState().read_json("data_version") or {}
        return str((v.get("versions") or {}).get("qlib_bin", ""))
    except Exception:
        return ""
# post-audit-5: 自定义信号表达式接入回测（signal_expr/factor_from_selected）；
# WFA 测试段首日计入换手成本并套用涨跌停约束。
# 归因诚实声明：Brinson 分解需要行业分类、Barra 暴露需要多因子库，
# 数据缺位时不以常数拆分冒充实测（post-audit-4 移除伪归因）。
ATTRIBUTION_CAVEAT = "行业归因(Brinson)与风险暴露(Barra)需要行业分类与多因子库，当前版本不提供"

# 项目根目录（selected.json 等共享资源基于此解析；测试可 monkeypatch 重定向）
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SELECTED_JSON_REL = os.path.join("evolve", "strategies", "selected.json")
SIGNAL_SOURCE_CAVEAT_FMT = "信号源: 自定义表达式 {expr}（evolve GP 语法）"


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


def normalize_benchmark(benchmark: str) -> tuple[str, str | None]:
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


def _factor_count(strategy_config: StrategyConfig | None) -> int:
    """提取因子数量"""
    if strategy_config and strategy_config.factors:
        return len(strategy_config.factors)
    return 0


def _parse_constraints(
    constraints: PortfolioConstraints | None, N: int
) -> dict[str, Any]:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


# ═══════════════════════════════════════════
# Deflated Sharpe Ratio（Bailey & López de Prado 多重比较校正）
# ═══════════════════════════════════════════

_EULER_GAMMA = 0.5772156649015329  # 欧拉-马歇罗尼常数


def _norm_cdf(x: float) -> float:
    """标准正态分布 CDF（stdlib 实现，避免引入 scipy）。"""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_ppf(p: float) -> float:
    """标准正态分布分位数函数；p 被夹入开区间 (1e-15, 1-1e-15) 防溢出。"""
    from statistics import NormalDist

    return NormalDist().inv_cdf(min(max(float(p), 1e-15), 1.0 - 1e-15))


def _sample_skew_kurt(returns: np.ndarray) -> tuple[float, float]:
    """
    收益序列的标准化三/四阶矩（总体口径）：γ3=偏度、γ4=峰度（正态 → (0, 3)）。
    样本退化（<2 个有限观测或零方差）→ 返回 (0.0, 3.0) 正态默认值。
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0, 3.0
    mu = float(np.mean(r))
    m2 = float(np.mean((r - mu) ** 2))
    if m2 <= 1e-300:
        return 0.0, 3.0
    m3 = float(np.mean((r - mu) ** 3))
    m4 = float(np.mean((r - mu) ** 4))
    return m3 / m2**1.5, m4 / m2**2


def deflated_sharpe_ratio(
    sharpe_observed: float,
    n_trials: int,
    sr_variance: float | None = None,
    tail_risk_adj: bool = True,
    *,
    n_periods: int = 0,
    returns: np.ndarray | None = None,
) -> float:
    """
    Deflated Sharpe Ratio：对"从 n_trials 次试验中挑出的最优策略"的夏普做多重比较校正。

    DSR = Φ( ((SR_obs − SR_0) · √(T−1)) /
             √(1 − γ3·SR + ((γ4−1)/4)·SR²) )

    - SR_0 = √(V[SR across trials]) · ((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)))，
      γ=0.5772（欧拉常数）、N=n_trials、V[SR]=各试验 Sharpe 的样本方差；
    - γ3/γ4 为收益序列的偏度/峰度；tail_risk_adj=True 且提供 returns 时按序列
      三/四阶矩估计，否则设 γ3=0、γ4=3（正态默认）；
    - sr_variance=None 时以单序列估计量方差近似 V[SR] ≈ (1 − γ3·SR + ((γ4−1)/4)·SR²)/(T−1)
      （多重比较校正的保守下界）；
    - n_trials=1 无多重比较语境，退化为 PSR(SR*=0)；

    Parameters
    ----------
    sharpe_observed : 观测 Sharpe，须与 returns 同频率的单期（如日频）单位；
                      年化 Sharpe 需先除以 √252 再传入。
    n_trials : 试验次数（策略配置数/参数网格数/WFA 滚动窗口数等诚实下界）。
    sr_variance : 各试验 Sharpe 的样本方差；None 时用估计量方差近似。
    tail_risk_adj : 是否用收益序列高阶矩校正偏度/峰度。
    n_periods : 单期观测数 T（returns 缺失时使用）。
    returns : 收益序列（提供时 T=len(returns)，并用于高阶矩估计）。

    Returns
    -------
    float — DSR ∈ [0, 1]；输入不合法（n_trials<1、T<2、非有限 SR 等）返回 0.0。
    """
    sr_obs = float(sharpe_observed)
    trials = int(n_trials)
    rets = (
        np.asarray(returns, dtype=np.float64).ravel()
        if returns is not None and np.size(returns)
        else None
    )
    t_obs = int(rets.size) if rets is not None else int(n_periods)

    if trials < 1 or t_obs < 2 or not math.isfinite(sr_obs):
        return 0.0

    if tail_risk_adj and rets is not None:
        gamma3, gamma4 = _sample_skew_kurt(rets)
    else:
        gamma3, gamma4 = 0.0, 3.0

    denom_sq = 1.0 - gamma3 * sr_obs + (gamma4 - 1.0) / 4.0 * sr_obs**2
    if not math.isfinite(denom_sq) or denom_sq <= 1e-12:
        return 0.0

    if trials == 1:
        sr_0 = 0.0
    else:
        var_sr = (
            float(sr_variance)
            if sr_variance is not None
            else denom_sq / (t_obs - 1)
        )
        if not math.isfinite(var_sr) or var_sr < 0:
            return 0.0
        expected_max = (
            (1.0 - _EULER_GAMMA) * _norm_ppf(1.0 - 1.0 / trials)
            + _EULER_GAMMA * _norm_ppf(1.0 - 1.0 / (trials * math.e))
        )
        sr_0 = math.sqrt(var_sr) * expected_max

    z_stat = (sr_obs - sr_0) * math.sqrt(t_obs - 1) / math.sqrt(denom_sq)
    return float(min(max(_norm_cdf(z_stat), 0.0), 1.0))


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
) -> tuple[np.ndarray, float, int, int]:
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
) -> list[str]:
    """涨跌停拦截统计 → 报告 caveats 行（无拦截且现金比例 ≤1% 时返回空）。"""
    caveats: list[str] = []
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


# ═══════════════════════════════════════════
# Brinson(BHB 简化) 行业归因 — 最小诚实版（真实路径）
# ═══════════════════════════════════════════

BRINSON_COVERAGE_MIN = 0.5   # 行业覆盖率低于此值 → 跳过归因（只出 caveat 不出数字）


def _stock_interval_returns(returns_matrix: np.ndarray) -> np.ndarray:
    """
    (T, N) 日收益矩阵 → (N,) 全区间复合收益。

    缺失/停牌日在 returns_matrix 中记 0，复合时为恒等因子，不产生虚假贡献。
    """
    growth = np.prod(1.0 + np.asarray(returns_matrix, dtype=np.float64), axis=0)
    return growth - 1.0


def _brinson_attribution(
    codes_list: list[str],
    final_weights: np.ndarray,
    returns_matrix: np.ndarray,
) -> dict[str, Any] | None:
    """
    单期（整个回测区间）Brinson-Hood-Beebower 简化归因。

    诚实口径：
    - 行业标签来自 v2.industry.get_industry（只读本地缓存，不触发联网）；
      无行业数据的股票记 "Unknown"，覆盖率 = 有标签股票 / 全池；
    - 行业基准收益 r_b,i = 该行业成分股等权平均全区间复合收益；
    - 基准行业权重 w_b,i = 全池（asof 成分）等权占比 n_i/N（市值无关）；
    - 组合行业权重 w_p,i = 回测期末持仓权重按行业聚合；
    - 组合行业收益 r_p,i = 行业内按期末持仓权重加权（区别于基准等权 → 选股效应来源；
      无持仓的行业回退 r_b,i，选股项自然归零）；
    - allocation_effect = Σ(w_p,i − w_b,i)(r_b,i − R_b)，R_b = Σ w_b,i·r_b,i；
    - selection_effect = Σ w_b,i(r_p,i − r_b,i) + Σ(w_p,i − w_b,i)(r_p,i − r_b,i)
      （交互项并入 selection，即合并后 Σ w_p,i(r_p,i − r_b,i)）。

    Returns
    -------
    dict(allocation, selection, coverage, n_industries)；覆盖率 < 50% 或输入退化返回 None
    （调用方据此仅输出"覆盖率不足"caveat，不产出数字）。查找抛错由调用方 try/except 兜底。
    """
    from trader3.v2.industry import get_industry

    N = len(codes_list)
    if N == 0 or final_weights is None or len(final_weights) != N:
        return None
    raw_labels = [get_industry(c) for c in codes_list]
    coverage = sum(1 for x in raw_labels if x) / N
    if coverage < BRINSON_COVERAGE_MIN:
        return None

    labels = ["Unknown" if x is None else str(x) for x in raw_labels]
    stock_rets = _stock_interval_returns(returns_matrix)
    weights = np.asarray(final_weights, dtype=np.float64)

    members: dict[str, list[int]] = {}
    for j, g in enumerate(labels):
        members.setdefault(g, []).append(j)

    industry_w_b: dict[str, float] = {}
    industry_r_b: dict[str, float] = {}
    industry_w_p: dict[str, float] = {}
    industry_r_p: dict[str, float] = {}
    for g, idx in members.items():
        rows = np.asarray(idx, dtype=int)
        industry_w_b[g] = len(idx) / N
        industry_r_b[g] = float(np.mean(stock_rets[rows]))
        w_sum = float(np.sum(weights[rows]))
        industry_w_p[g] = w_sum
        if w_sum > 1e-12:
            industry_r_p[g] = float(np.dot(weights[rows], stock_rets[rows]) / w_sum)
        else:
            industry_r_p[g] = industry_r_b[g]

    R_b = float(sum(industry_w_b[g] * industry_r_b[g] for g in members))
    allocation_effect = float(sum(
        (industry_w_p[g] - industry_w_b[g]) * (industry_r_b[g] - R_b)
        for g in members
    ))
    selection_effect = float(sum(
        industry_w_b[g] * (industry_r_p[g] - industry_r_b[g])
        + (industry_w_p[g] - industry_w_b[g]) * (industry_r_p[g] - industry_r_b[g])
        for g in members
    ))
    return {
        "allocation": allocation_effect,
        "selection": selection_effect,
        "coverage": float(coverage),
        "n_industries": len(members),
    }


def _run_portfolio_simulation(
    stock_returns: np.ndarray,
    factor_scores: np.ndarray,
    N: int,
    n_hold: int,
    max_single_w: float,
    long_only: bool,
    commission: CommissionInfo | None = None,
    codes: list[str] | None = None,
    stats: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int]:
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
) -> dict[str, float]:
    """按基准趋势分类的区间年化收益"""
    T = len(port_returns)
    ma_short, ma_long = 20, 60
    regime_rets: dict[str, list] = {"trending_up": [], "ranging": [], "bearish": []}

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

    result: dict[str, float] = {}
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


# ── 有效 IR 折算（post-audit-8）──


def annualized_ir(daily_ir: float, trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """传统年化口径 IR = daily_ir * sqrt(trading_days)。"""
    return float(daily_ir) * math.sqrt(trading_days)


def effective_ir(
    daily_ir: float,
    rebalance_days: int,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    赌注级（bet-level）有效 IR 折算：

    - 独立赌注数 independent_bets = trading_days / rebalance_days；
    - effective_ir = daily_ir * sqrt(independent_bets)。

    调仓间隔内信号不更新 → 同一持仓期的日度超额收益高度自相关，
    直接按 sqrt(252) 年化会高估信息比率；按真实独立赌注数折算更保守。
    rebalance_days 非正时显式报错（不静默吞掉配置错误）。
    """
    if int(rebalance_days) <= 0:
        raise ValueError(f"rebalance_days 必须为正整数，收到 {rebalance_days!r}")
    independent_bets = float(trading_days) / int(rebalance_days)
    return float(daily_ir) * math.sqrt(independent_bets)


def _daily_information_ratio(
    port_returns: np.ndarray, market_returns: np.ndarray
) -> float:
    """日频 IR = mean(日超额) / std(日超额)；样本不足或零波动返回 0。"""
    ex = np.asarray(port_returns, dtype=np.float64) - np.asarray(
        market_returns, dtype=np.float64
    )
    if ex.size < 2:
        return 0.0
    sd = float(np.std(ex, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(ex)) / sd


def _attach_effective_ir(
    key_metrics: dict,
    caveats: list[str],
    port_returns: np.ndarray,
    market_returns: np.ndarray,
    rebalance_days: int,
) -> None:
    """
    回测主路径接入双口径 IR（原地更新 key_metrics / caveats）：

    - "信息比率" ← 传统年化口径 daily_ir*sqrt(252)（键名向后兼容）；
    - "有效IR(bet级)" ← 按调仓频率折算口径；
    调仓间隔从 is_rebalance 条件推导：两个回测引擎均为
    (t == 0) or ((t + 1) % REBALANCE_FREQ == 0)，即每 REBALANCE_FREQ 个
    交易日产生一次独立调仓赌注。
    """
    d_ir = _daily_information_ratio(port_returns, market_returns)
    key_metrics["信息比率"] = annualized_ir(d_ir)
    # 向后兼容别名：历史消费者（gates/既有测试）按旧键名 "信息比" 精确取值
    key_metrics["信息比"] = key_metrics["信息比率"]
    key_metrics["有效IR(bet级)"] = effective_ir(d_ir, rebalance_days)
    caveats.append(EFFECTIVE_IR_CAVEAT_FMT.format(n=int(rebalance_days)))


# ═══════════════════════════════════════════
# RunBacktestTool
# ═══════════════════════════════════════════


def get_financials_asof(code: str, asof_date: str) -> dict | None:
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


class _SignalExprError(ValueError):
    """自定义信号表达式解析/求值失败（execute 层转为 error 响应，不走合成回退）。"""


def _prepare_expr(expr: str) -> Any:
    """
    解析并规范化因子表达式（evolve GP 语法）。

    Returns
    -------
    normalize 后的 Node 表达式树。

    Raises
    ------
    _SignalExprError — 语法非法（含原因）。
    """
    try:
        from evolve.core.gp import normalize
        from evolve.core.parser import parse_expr

        node = parse_expr(str(expr))
        return normalize(node)
    except _SignalExprError:
        raise
    except Exception as e:
        raise _SignalExprError(f"表达式无法解析: {e}") from e


def _collect_expr_fields(node: Any) -> set:
    """收集表达式树引用的行情字段集合（open/high/low/close/volume/vwap/amount）。"""
    from evolve.core.gp import FIELDS

    if node.op == "const":
        return set()
    out = {node.op} if node.op in FIELDS else set()
    for child in node.children:
        out |= _collect_expr_fields(child)
    return out


def _load_expression_panels(
    dp: Any,
    codes_list: list[str],
    time_axis: list[str],
    fields: set,
) -> dict[str, np.ndarray]:
    """
    按 codes_list × time_axis 对齐加载表达式所需字段面板 {field: (T,M)}。

    缺失/停牌处为 NaN；整列字段加载失败（bin 不存在等）则不进入返回 dict，
    由调用方做字段缺失校验。
    """
    date_index = {d: i for i, d in enumerate(time_axis)}
    T, M = len(time_axis), len(codes_list)
    panels: dict[str, np.ndarray] = {}
    for fld in sorted(fields):
        mat = np.full((T, M), np.nan, dtype=np.float64)
        loaded_any = False
        for j, code in enumerate(codes_list):
            try:
                vals, dates = dp.load_stock(code.lower(), fld, time_axis[0], time_axis[-1])
            except Exception:
                continue
            for i, d in enumerate(dates):
                pos = date_index.get(d)
                if pos is not None and np.isfinite(vals[i]) and vals[i] > 0:
                    mat[pos, j] = vals[i]
                    loaded_any = True
        if loaded_any:
            panels[fld] = mat
    return panels


def _expr_zscores(
    node: Any,
    panels: dict[str, np.ndarray],
    valid_flags: np.ndarray,
) -> np.ndarray:
    """
    在已对齐面板上求值表达式 → 逐日横截面 z-score 的 (T,N) 矩阵（无效处为 NaN）。

    - 引用字段缺面板 → 报错（gp.evaluate 对缺失字段默认补零，静默补零会伪装信号）；
    - 求值抛错 / 结果全 NaN → 报错；
    - 无效（NaN 或 valid_flags=False）处置 NaN，不参与当日截面统计，
      供多因子合成做"NaN 跳过、按可用因子数归一"。

    Raises
    ------
    _SignalExprError — 字段缺失 / 求值失败 / 全 NaN / 形状不符。
    """
    need = _collect_expr_fields(node)
    missing = sorted(f for f in need if f not in panels)
    if missing:
        raise _SignalExprError(f"表达式引用字段数据缺失: {', '.join(missing)}")

    try:
        from evolve.core.gp import evaluate

        with np.errstate(all="ignore"):
            raw = np.asarray(evaluate(node, panels), dtype=np.float64)
    except _SignalExprError:
        raise
    except Exception as e:
        raise _SignalExprError(f"表达式求值失败: {e}") from e

    if raw.ndim != 2 or raw.shape != valid_flags.shape:
        raise _SignalExprError(
            f"表达式结果形状异常: {getattr(raw, 'shape', None)} != {(valid_flags.shape)}"
        )
    if not np.isfinite(raw).any():
        raise _SignalExprError("表达式计算结果全为 NaN，无法作为选股信号")

    vals = np.where(np.isfinite(raw) & valid_flags, raw, np.nan)
    # 整行无效时 nanmean/nanstd 产生 "Mean of empty slice" 警告属预期路径（结果即 NaN）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu = np.nanmean(vals, axis=1, keepdims=True)
        sd = np.nanstd(vals, axis=1, keepdims=True)
    z = (vals - mu) / (sd + 1e-12)
    return np.where(np.isfinite(z), z, np.nan)


def _combine_factor_scores(
    score_stack: np.ndarray,
    weights: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    """
    K 个因子分数堆栈 (K,T,N) → 合成 (T,N)。

    - weights=None：等权合成 —— NaN 跳过后按当日可用因子数归一；
    - weights 给定：先归一化 w/Σw，再按各层权重加权合成；
      NaN 跳过、按当日可用因子的权重和归一（等权语义的自然推广，
      全因子可用时退化为加权平均）；
    - 任一日全部因子无效 → 该日分数为 NaN（调用方选股时视为 -inf 排除）。

    Raises
    ------
    ValueError — weights 长度与因子数不符 / 之和为 0 或非有限。
    """
    k = score_stack.shape[0]
    if weights is None:
        w = np.full(k, 1.0 / max(k, 1), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape[0] != k:
            raise ValueError(f"weights 长度 ({w.shape[0]}) 与因子数 ({k}) 不一致")
        total = float(np.sum(w))
        if not math.isfinite(total) or abs(total) < 1e-12:
            raise ValueError("weights 之和为 0 或非有限，无法归一化")
        w = w / total
    finite = np.isfinite(score_stack)
    denom = np.where(finite, w[:, None, None], 0.0).sum(axis=0)
    numer = np.where(finite, score_stack * w[:, None, None], 0.0).sum(axis=0)
    combined = np.full(denom.shape, np.nan, dtype=np.float64)
    np.divide(numer, denom, out=combined, where=denom > 0)
    return combined


def _expr_scores(
    node: Any,
    panels: dict[str, np.ndarray],
    valid_flags: np.ndarray,
) -> np.ndarray:
    """
    单表达式分数矩阵（选股口径）：_expr_zscores 结果中无效处置 -inf 落选。
    """
    z = _expr_zscores(node, panels, valid_flags)
    return np.where(np.isfinite(z), z, -np.inf)


def _read_selected_entries() -> list:
    """
    读取 <项目根>/evolve/strategies/selected.json 原始条目数组（共享解析与校验）。

    Raises
    ------
    FileNotFoundError / ValueError — 均带明确原因，由 execute 层转 error 响应。
    """
    path = os.path.join(_PROJECT_ROOT, SELECTED_JSON_REL)
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 selected.json: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
    except Exception as e:
        raise ValueError(f"selected.json 解析失败 ({path}): {e}") from e
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"selected.json 为空或格式非法（应为非空数组）: {path}")
    return entries


def _entry_expr(entry: Any) -> str:
    """selected.json 条目 → 非空 expr 字符串（缺失返回空串）。"""
    return str(entry.get("expr", "")).strip() if isinstance(entry, dict) else ""


def _entry_vetoed(entry: Any) -> bool:
    """selected.json 条目的 OOS 否决标记：oos_veto 为真值即被样本外检验否决。"""
    return bool(entry.get("oos_veto")) if isinstance(entry, dict) else False


def _kept_entries(entries: list) -> tuple[list[Any], int]:
    """剔除 oos_veto 为真的条目；返回 (保留条目, 否决数)。"""
    kept = [en for en in entries if not _entry_vetoed(en)]
    return kept, len(entries) - len(kept)


def _load_expr_from_selected(rank: int, veto_stats: dict | None = None) -> str:
    """
    从 selected.json 读取第 rank 名因子的 expr。

    文件按分数降序排列（run_evolution.py 写出），rank=1 即第一名；
    oos_veto 为真的条目不参与 rank 计数（基线 v2 裁定 2026-08-25）。

    veto_stats : 可选输出 dict — 写入 "expr_skipped"（被否决跳过的条目数），
                 供 execute 层写 caveat。

    Raises
    ------
    FileNotFoundError / ValueError / IndexError — 均带明确原因，由 execute 层转 error 响应。
    """
    entries = _read_selected_entries()
    kept, n_veto = _kept_entries(entries)
    if veto_stats is not None:
        veto_stats["expr_skipped"] = n_veto
    if not 1 <= int(rank) <= len(kept):
        msg = f"factor_from_selected={rank} 超出范围（selected.json 共 {len(entries)} 名"
        msg += (
            f"，OOS 否决 {n_veto} 名后可用 {len(kept)} 名）" if n_veto else "）"
        )
        raise IndexError(msg)
    expr = _entry_expr(kept[int(rank) - 1])
    if not expr:
        raise ValueError(f"selected.json 第 {rank} 名缺少 expr 字段")
    return expr


def _load_topk_exprs_from_selected(k: int, veto_stats: dict | None = None) -> list[str]:
    """
    top_k_combine 合成模式：读取 selected.json 前 k 名的 expr 列表。

    oos_veto 为真的条目先被剔除再取前 k 名（不进入候选）；剔除后可用条目
    不足 k 时降级使用全部可用条目（veto_stats["topk_degraded_k"]=k 供 execute
    层写降级 caveat）。可用 expr 少于 2 个（文件缺失/为空/条目缺 expr）
    → ValueError，单因子无合成意义，由 execute 层转 error 响应。
    """
    entries = _read_selected_entries()
    kept, n_veto = _kept_entries(entries)
    if veto_stats is not None:
        veto_stats["topk_skipped"] = n_veto
    k_eff = max(int(k), 0)
    usable = [e for e in (_entry_expr(en) for en in kept) if e]
    # 过滤后可用条目不足 K → 降级用全部可用条目（execute 层写降级 caveat）
    if veto_stats is not None and k_eff > 0 and len(usable) < k_eff:
        veto_stats["topk_degraded_k"] = k_eff
    exprs = usable[:k_eff]
    if not exprs and n_veto:
        raise ValueError(
            f"selected.json 共 {len(entries)} 名因子全部被 OOS 否决，无可入选因子"
        )
    if len(exprs) < 1:
        raise ValueError(
            f"selected.json 前 {k} 名可用 expr 为空（文件缺失或全部无效）"
        )
    return exprs


def _load_icir_weights_from_selected(
    exprs: list[str], veto_stats: dict | None = None
) -> list[float]:
    """
    ICIR 加权模式：按表达式精确匹配读取 selected.json 条目的 gates.icir.value，
    返回 |icir| 权重列表（顺序与入参 exprs 一致）。

    oos_veto 为真的条目不进入权重映射（其表达式权重记 0，等价于退出加权
    归一，_combine_factor_scores 按 Σw 归一时自然排除）。

    veto_stats : 可选输出 dict — 累加写入 "icir_skipped"（被否决跳过的条目数）。

    Raises
    ------
    ValueError — selected.json 缺失/非法，或任一表达式缺少对应 icir 门禁值
                 （报错文案: "selected.json 缺少 <expr> 的 icir"，execute 层转 error 响应）。
    """
    entries = _read_selected_entries()
    icir_map: dict[str, float] = {}
    vetoed_exprs: set[str] = set()
    n_veto = 0
    for en in entries:
        ex = _entry_expr(en)
        if not ex:
            continue
        if _entry_vetoed(en):
            n_veto += 1
            vetoed_exprs.add(ex)
            continue
        gates = en.get("gates") if isinstance(en, dict) else None
        g = gates.get("icir") if isinstance(gates, dict) else None
        val = g.get("value") if isinstance(g, dict) else None
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            icir_map[ex] = float(val)
    if veto_stats is not None:
        veto_stats["icir_skipped"] = veto_stats.get("icir_skipped", 0) + n_veto
    weights: list[float] = []
    for ex in exprs:
        if ex in icir_map:
            weights.append(abs(icir_map[ex]))
        elif ex in vetoed_exprs:
            weights.append(0.0)  # 被否决：不参与加权归一（Σw 归一自然排除）
        else:
            raise ValueError(f"selected.json 缺少 {ex} 的 icir")
    return weights


def _build_aligned_panel(dp: Any, codes: list[str], start_date: str, end_date: str) -> tuple[
    list[str], list[str], np.ndarray, np.ndarray, np.ndarray, int
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
    stock_closes: dict[str, np.ndarray] = {}
    stock_dates: dict[str, list[str]] = {}
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
    commission: CommissionInfo | None = None,
    codes: list[str] | None = None,
    score_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int, dict[str, Any]]:
    """
    真实面板动量组合回测（月频调仓；与合成引擎共用执行语义）：

    - 防前视 pending：t 日收盘信号只挂起，t+1 生效（首日建仓立即生效、末日跳过）；
    - 执行日涨跌停拦截 + 显式现金跟踪（见 _apply_execution_constraints）；
    - 打分函数可插拔：score_matrix 为 None 时内部计算 20 日对数动量；
      传入 (T,N) 矩阵（如自定义因子表达式归一化结果）时替代动量进入同一
      pending 权重模拟（选股=分数 top-n_hold）；
    - 供 run_backtest 真实路径调用。

    Returns
    -------
    equity, port_returns, turnover_total, positive_days, stats
    （stats 含 limit_up_blocked / limit_down_blocked / final_cash_weight /
     final_weights —— 期末实际生效持仓权重，供 Brinson 归因聚合行业敞口）
    """
    scores = (
        _momentum_scores(close_matrix) if score_matrix is None else score_matrix
    )
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
                scores[t], valid_flags[t], n_hold, True, max_single_w
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

    stats: dict[str, Any] = {
        "limit_up_blocked": limit_up_blocked,
        "limit_down_blocked": limit_down_blocked,
        "final_cash_weight": float(cash_weight),
        "final_weights": weights.copy(),
    }
    return equity, port_returns, turnover_total, positive_days, stats


def _open_qlib_dp() -> Any:
    """打开默认 qlib 数据源（不可用时抛异常；测试可替换本函数注入临时数据源）。"""
    from trader3.data_provider import QlibDataProvider

    return QlibDataProvider()


def _load_wfa_panel(dp: Any) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    WFA 全历史面板：成分池取 csi300（按日历起点 asof 过滤，防幸存者偏差），
    缺失时回退 all；超上限按固定种子确定性抽样。

    Returns
    -------
    (codes_list, close_matrix, returns_matrix, valid_flags, time_axis)
    """
    cal = dp.calendar()
    if not cal:
        raise RuntimeError("qlib 日历为空")
    start_date, end_date = cal[0], cal[-1]

    codes: list[str] = []
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

    time_axis, codes_list, close_matrix, returns_matrix, valid_flags, _n_skip = (
        _build_aligned_panel(dp, codes, start_date, end_date)
    )
    return codes_list, close_matrix, returns_matrix, valid_flags, time_axis


def _run_wfa_rolling(
    stock_returns: np.ndarray,
    factor_scores: np.ndarray,
    train_window: int,
    test_window: int,
    step: int,
    codes: list[str] | None = None,
    commission: CommissionInfo | None = None,
) -> tuple[list[dict], list[float], list[float], list[float], list[float],
           list[np.ndarray], np.ndarray]:
    """
    滚动 IS/OOS（真实/合成面板共用）。

    每窗以训练段因子均值确定等权 top-k 持仓，固定应用于紧随其后的测试段 ——
    训练段收盘信息最早于测试段首日生效，与回测引擎的 pending 次日生效语义一致；
    窗口间前进 step 天。OOS 日收益逐窗拼接（step ≥ test_window 时天然非重叠），
    供汇总指标在非重叠样本上计算。

    交易现实性（post-audit-5）：
    - 测试段首日视为调仓执行日：复用主循环同款规则
      （_apply_execution_constraints：涨停拦买、跌停滞卖；空仓建仓场景仅买单）
      并按 DEFAULT_COSTS（或显式传入的 commission）对实际成交换手扣单边成本；
    - 段内不调仓，持仓权重保持首日实际成交结果至窗口结束；
    - 被涨停拦截未建仓的资金按现金处理（收益 0），不参与段内收益。

    Parameters
    ----------
    codes : 可选，与 stock_returns 列一一对应；提供时涨跌停幅度逐股按板块判定，
            否则统一主板 DEFAULT_PRICE_LIMIT。
    commission : 可选费用模型；None 用 DEFAULT_COSTS。

    Returns
    -------
    windows, is_ann_list, oos_ann_list, is_sr_list, oos_sr_list,
    param_weights, oos_concat（拼接 OOS 日收益 np.ndarray，已扣首日建仓成本）
    """
    T, N = stock_returns.shape
    top_k = max(N // 5, 10)
    cm = commission if commission is not None else DEFAULT_COSTS

    if codes is not None:
        limit_ratios = np.array(
            [_price_limit_ratio(c) for c in codes], dtype=np.float64
        )
    else:
        limit_ratios = np.full(N, DEFAULT_PRICE_LIMIT, dtype=np.float64)

    windows: list[dict] = []
    is_ann_list: list[float] = []
    oos_ann_list: list[float] = []
    is_sr_list: list[float] = []
    oos_sr_list: list[float] = []
    param_weights: list[np.ndarray] = []
    oos_chunks: list[np.ndarray] = []

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

        # IS 表现（参数拟合口径，毛收益）
        is_rets = stock_returns[is_slice] @ weights
        is_ann = float(np.mean(is_rets)) * TRADING_DAYS_PER_YEAR
        is_std = float(np.std(is_rets, ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)
        is_sr = is_ann / is_std if is_std > 1e-10 else 0.0

        # 测试段首日=调仓执行日：主循环同款约束撮合 + 实际成交换手计成本
        oos_start = s + train_window
        effective, cash_weight, _n_up, _n_down = _apply_execution_constraints(
            np.zeros(N, dtype=np.float64),
            weights,
            stock_returns[oos_start],
            limit_ratios,
            0.0,
        )
        executed_delta = float(np.sum(np.abs(effective)))  # 空仓建仓：|eff - 0|
        day0_cost = cm.turnover_cost(executed_delta)

        # OOS 表现（首日实际成交权重持有全段；首日扣建仓成本，现金收益按 0 计）
        oos_rets = stock_returns[oos_slice] @ effective
        oos_rets[0] = float(effective @ stock_returns[oos_start]) + cash_weight * 0.0 - day0_cost
        oos_ann = float(np.mean(oos_rets)) * TRADING_DAYS_PER_YEAR
        oos_std = float(np.std(oos_rets, ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)
        oos_sr = oos_ann / oos_std if oos_std > 1e-10 else 0.0

        windows.append({
            "window": w,
            "is_return": is_ann,
            "oos_return": oos_ann,
            "is_sharpe": is_sr,
            "oos_sharpe": oos_sr,
            "oos_start": oos_start,
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
        strategy_config: StrategyConfig | None,
        universe: list[str] | None,
        start_date: str,
        end_date: str,
        constraints: PortfolioConstraints | None,
        benchmark: str,
        commission: CommissionInfo | None = None,
        *,
        engine_tag: str = "real",
        data_end: str = "",
        signal_expr: str = "",
        signal_exprs: list[str] | None = None,
        signal_weights: list[float] | None = None,
    ) -> str:
        """sha256 策略指纹（含 constraints/引擎标识/数据末端/代码版本/信号表达式，防张冠李戴命中）"""
        factors = []
        if strategy_config and strategy_config.factors:
            factors = [
                {"name": f.name, "weight": f.weight, "direction": f.direction}
                for f in strategy_config.factors
            ]
        exprs = [str(e) for e in (signal_exprs or [])]
        weights = [float(w) for w in (signal_weights or [])]
        # 多因子合成指纹：加权时对 (exprs+weights) 联合 json 取 sha1（不同权重互不共享缓存）；
        # 等权模式保持 exprs 列表整体 sha1 不变
        if exprs and weights:
            joint_payload = json.dumps(
                {"exprs": exprs, "weights": weights}, ensure_ascii=False
            )
            exprs_sha1 = hashlib.sha1(joint_payload.encode("utf-8")).hexdigest()
        else:
            exprs_sha1 = (
                hashlib.sha1(
                    json.dumps(exprs, ensure_ascii=False).encode("utf-8")
                ).hexdigest()
                if exprs else ""
            )
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
            # 数据内容纪元：增量管线盖章的 qlib_bin 版本（治愈/重刷后自动失效旧缓存）
            "data_version": _data_version_stamp(),
            # 信号源指纹：自定义表达式取 sha1（空串=默认动量），不同表达式互不共享缓存
            "signal_sha1": (
                hashlib.sha1(signal_expr.encode("utf-8")).hexdigest()
                if signal_expr else ""
            ),
            # 多因子合成指纹：exprs 列表（有序）整体 sha1（加权模式见上方联合指纹）
            "exprs_sha1": exprs_sha1,
        }
        raw = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _probe_engine(self) -> tuple[str, str]:
        """探测可用引擎与数据末端（不执行回测）：('real'|'synthetic', data_end)"""
        try:
            from trader3.data_provider import find_qlib_dir

            qdir = find_qlib_dir()
            if qdir:
                cal_path = os.path.join(qdir, "calendars", "day.txt")
                with open(cal_path) as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                if lines:
                    return "real", lines[-1]
        except Exception:
            pass
        return "synthetic", ""

    def _cache_path(self, fp: str) -> str:
        return os.path.join(self._cache_dir, f"{fp}.json")

    def _load_cache(self, fp: str) -> Trader3Response | None:
        path = self._cache_path(fp)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
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
        strategy_config: StrategyConfig | None = None,
        universe: list[str] | None = None,
        start_date: str = "2020-01-01",
        end_date: str = "2025-12-31",
        constraints: PortfolioConstraints | None = None,
        benchmark: str = "000300.SH",
        commission: CommissionInfo | None = None,
        signal_expr: str = "",
        factor_from_selected: int = 0,
        signal_exprs: list[str] | None = None,
        top_k_combine: bool = False,
        signal_weights: list[float] | None = None,
        weight_by: str = "",
        stress_test: bool = False,
    ) -> Trader3Response:
        """
        执行回测（M8: 真实 qlib 数据优先 → 向量化合成回退 + 缓存）。

        信号源优先级（高 → 低，post-audit-6）：
        1. signal_expr 非空：单表达式模式 —— evolve GP 语法表达式在真实对齐面板上
           求值，归一化后替代动量进入同一 pending 权重模拟（仅真实数据路径支持；
           表达式解析/求值失败或全 NaN → error 响应）。此时忽略
           signal_exprs / signal_weights / weight_by / factor_from_selected /
           top_k_combine。
        2. 否则 signal_exprs 非空：多因子合成模式 —— 每个表达式在同一面板上求值并
           逐日横截面 z-score，NaN 跳过按可用因子（权重）归一合成：
           - 无权重参数 → 等权合成，caveat "多因子等权合成: K=<N> 个表达式"；
           - signal_weights 显式给定（长度须等于 exprs）→ 加权合成，
             caveat "多因子加权合成: K=<N>, weights=[...]"；
           - weight_by="icir" → 从 selected.json 按 expr 精确匹配读取
             gates.icir.value，以 |icir| 为权重做加权合成。
        3. 否则 top_k_combine=True：读 selected.json 前 N 名做合成
           （N=factor_from_selected>0 时取之，否则默认 3；文件缺失或可用 expr
           不足 2 个 → error 响应）。weight_by="icir" 时对前 N 名自动做 icir
           加权合成（主用例）；否则等权。
        4. 否则 factor_from_selected>0：读取 selected.json 第 N 名因子的 expr 作为
           单表达式（文件缺失/为空/越界 → error 响应）。
        5. 均缺省：走默认 20 日动量逻辑（完全兼容旧行为）。

        权重校验（先于指纹与缓存）：signal_weights 与 weight_by 互斥；
        长度不符/非有限值/和为 0、weight_by 设置但既无 signal_exprs 也非
        top_k_combine 组合、selected.json 缺 icir 条目 → 一律 error 响应。

        缓存：指纹含 signal_expr 的 sha1 与 signal_exprs 列表整体的 sha1；
        加权模式下 exprs_sha1 为 (exprs+weights) 联合 json 的 sha1，
        不同信号源/不同权重互不命中。

        stress_test=True（post-audit-8）：额外对 STRESS_PERIODS 中与
        [start_date, end_date] 有交集的极端窗口逐一回测，结果以 stress_* 前缀
        键合并进 key_metrics 并追加压测 caveat。压测不参与缓存指纹——
        缓存始终保存未压测版本，压测层在缓存读取/落盘之后叠加。
        """
        # ── 信号源解析（先于指纹与缓存；selected.json 加载失败直接报错）──
        signal_expr = str(signal_expr or "").strip()
        if isinstance(signal_exprs, str):
            signal_exprs = [signal_exprs]
        raw_exprs = [
            str(e).strip() for e in (signal_exprs or []) if str(e or "").strip()
        ]
        expr_list: list[str] = []
        # selected.json OOS 否决统计（基线 v2 裁定 2026-08-25）：
        # 各加载器写入各自键；top_k_combine+icir 组合读同一文件，取 max 防重复计数
        veto_stats: dict[str, int] = {}

        if signal_expr:
            try:
                _prepare_expr(signal_expr)  # 语法预检（快速失败，不做 IO）
            except _SignalExprError as e:
                return Trader3Response.error(f"信号表达式错误: {e}")
        elif raw_exprs:
            expr_list = raw_exprs
            for expr in expr_list:
                try:
                    _prepare_expr(expr)
                except _SignalExprError as e:
                    return Trader3Response.error(f"信号表达式错误: {e}")
        elif top_k_combine:
            k_top = int(factor_from_selected) if int(factor_from_selected) > 0 else 3
            try:
                expr_list = _load_topk_exprs_from_selected(k_top, veto_stats=veto_stats)
            except Exception as e:
                return Trader3Response.error(f"selected.json 多因子加载失败: {e}")
        elif int(factor_from_selected) > 0:
            try:
                signal_expr = _load_expr_from_selected(
                    int(factor_from_selected), veto_stats=veto_stats
                )
            except Exception as e:
                return Trader3Response.error(f"selected.json 因子加载失败: {e}")

        # ── 合成权重解析（先于指纹与缓存；显式 signal_weights 与派生 weight_by 互斥）──
        weights_list: list[float] | None = None
        weight_by_norm = str(weight_by or "").strip().lower()
        if weight_by_norm and weight_by_norm != "icir":
            return Trader3Response.error(
                f"不支持的 weight_by={weight_by}（当前仅支持 'icir'）"
            )
        if signal_weights is not None:
            if weight_by_norm:
                return Trader3Response.error(
                    "signal_weights 与 weight_by 互斥，请只指定其一"
                )
            if not expr_list:
                return Trader3Response.error(
                    "signal_weights 仅在 signal_exprs 多因子合成模式下生效"
                )
            try:
                cand = [float(w) for w in signal_weights]
            except (TypeError, ValueError):
                return Trader3Response.error("signal_weights 含非数值元素")
            if len(cand) != len(expr_list):
                return Trader3Response.error(
                    f"signal_weights 长度 ({len(cand)}) 与 "
                    f"signal_exprs 数量 ({len(expr_list)}) 不一致"
                )
            if any(not math.isfinite(w) for w in cand):
                return Trader3Response.error("signal_weights 含非有限值 (NaN/inf)")
            if abs(sum(cand)) < 1e-12:
                return Trader3Response.error("signal_weights 之和为 0，无法归一化")
            weights_list = cand
        elif weight_by_norm == "icir":
            # 仅适用于 signal_exprs 多因子模式或 top_k_combine 前 K 名组合（主用例）
            if not expr_list:
                return Trader3Response.error(
                    "weight_by='icir' 仅适用于 signal_exprs 或 "
                    "top_k_combine 组合模式"
                )
            try:
                weights_list = _load_icir_weights_from_selected(
                    expr_list, veto_stats=veto_stats
                )
            except ValueError as e:
                return Trader3Response.error(str(e))
            except Exception as e:
                return Trader3Response.error(f"selected.json icir 权重加载失败: {e}")

        # 先探测引擎与数据末端，指纹含 engine_tag，避免真实/合成结果串缓存
        engine_tag, data_end = self._probe_engine()
        fp = self._fingerprint(
            strategy_config, universe, start_date, end_date,
            constraints, benchmark, commission,
            engine_tag=engine_tag, data_end=data_end,
            signal_expr=signal_expr, signal_exprs=expr_list,
            signal_weights=weights_list,
        )

        cached = self._load_cache(fp)
        if cached is not None:
            if stress_test:
                self._apply_stress_results(
                    cached, universe, start_date, end_date, signal_expr
                )
            return cached

        # M8: 优先真实数据
        try:
            result = self._real_data_backtest(
                strategy_config, universe, start_date, end_date, constraints, benchmark, commission,
                signal_expr=signal_expr, signal_exprs=expr_list,
                signal_weights=weights_list,
            )
        except _SignalExprError as e:
            # 表达式字段缺失/求值失败/全 NaN —— 明确报错，不静默换信号源
            return Trader3Response.error(f"信号表达式错误: {e}")
        except Exception as e:
            if signal_expr or expr_list:
                # 自定义表达式依赖真实面板（close/volume 等），合成路径无对应数据，
                # 静默回退会悄悄丢弃用户的信号定义 → 显式报错
                return Trader3Response.error(
                    f"自定义表达式回测需要真实数据面板，真实数据不可用: {e}"
                )
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

        # ── selected.json OOS 否决 caveat（先于缓存落盘，命中缓存同样携带）──
        n_veto_skipped = max(
            veto_stats.get("expr_skipped", 0),
            veto_stats.get("topk_skipped", 0),
            veto_stats.get("icir_skipped", 0),
        )
        if n_veto_skipped > 0:
            result.caveats.append(f"OOS否决跳过 {n_veto_skipped} 条")
        degraded_k = veto_stats.get("topk_degraded_k", 0)
        if degraded_k > 0:
            result.caveats.append(
                f"selected.json 可用因子不足{degraded_k}，降级为{len(expr_list)}条"
            )

        # 缓存先落盘（指纹不含 stress 标志）→ 缓存保存未压测版本，
        # 压测层只在本次响应上叠加，避免普通调用经缓存带回 stress_* 键
        self._save_cache(fp, result)
        if stress_test:
            self._apply_stress_results(
                result, universe, start_date, end_date, signal_expr
            )
        return result

    def _apply_stress_results(
        self,
        result: Trader3Response,
        universe: list[str] | None,
        start_date: str,
        end_date: str,
        signal_expr: str = "",
    ) -> None:
        """
        将 run_stress_test 结果合并进响应（原地修改，post-audit-8）：

        - 每个窗口 → key_metrics["stress_{窗口名}_{ann_return|sharpe|max_drawdown|excess}"]；
        - 汇总行 → "stress_avg_ann" / "stress_worst_window"；
        - caveats 追加 "极端情景压测: N 窗口已评估"。
        压测自身异常只记 caveat，不影响主回测结果。
        """
        try:
            rows = run_stress_test(
                start_date, end_date, universe,
                signal_expr=signal_expr, _backtest_tool=self,
            )
        except Exception as exc:
            result.caveats.append(f"极端情景压测失败: {type(exc).__name__}: {exc}")
            return
        n_windows = 0
        for row in rows:
            nm = str(row.get("period_name", ""))
            if nm == STRESS_SUMMARY_NAME:
                result.key_metrics["stress_avg_ann"] = float(row["avg_stress_ann"])
                result.key_metrics["stress_worst_window"] = str(row["worst_window"])
                continue
            n_windows += 1
            result.key_metrics[f"stress_{nm}_ann_return"] = float(row["ann_return"])
            result.key_metrics[f"stress_{nm}_sharpe"] = float(row["sharpe"])
            result.key_metrics[f"stress_{nm}_max_drawdown"] = float(row["max_drawdown"])
            result.key_metrics[f"stress_{nm}_excess"] = float(row["excess"])
        result.caveats.append(STRESS_CAVEAT_FMT.format(n=n_windows))

    # ── M8: 真实 qlib 数据回测 ──

    def _real_data_backtest(
        self,
        strategy_config: StrategyConfig | None,
        universe: list[str] | None,
        start_date: str,
        end_date: str,
        constraints: PortfolioConstraints | None,
        benchmark: str,
        commission: CommissionInfo | None = None,
        signal_expr: str = "",
        signal_exprs: list[str] | None = None,
        signal_weights: list[float] | None = None,
    ) -> Trader3Response:
        """
        基于真实 qlib 数据的回测。

        策略：动量因子（20日收益率）选股，月频调仓，等权持有；
        signal_expr 非空时以单表达式因子替代动量打分；
        signal_exprs 非空时逐表达式求值并逐日横截面 z-score 合成
        （无效处置 NaN 跳过、按可用因子权重归一；全无效日分数 NaN → 选股 -inf 排除；
        signal_weights 给定时按 w/Σw 加权替代等权）；
        其余模拟语义不变。基准：CSI300 指数（或用户指定）。
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
            bench_map = {d: close for d, close in zip(bench_dates, bench_close, strict=False)}
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

        # ── 组合模拟（月频调仓；与合成引擎共用执行语义）──
        # 信号源：默认 20 日动量；signal_expr 单表达式 / signal_exprs 多因子等权合成
        n_hold = min(max(len(codes_list) // 5, 10), 50)
        score_matrix = None
        multi_caveat = ""
        if signal_expr:
            node = _prepare_expr(signal_expr)
            expr_fields = _collect_expr_fields(node)
            close_panel = np.where(close_matrix > 0, close_matrix, np.nan)
            extra_fields = {f for f in expr_fields if f != "close"}
            expr_panels: dict[str, np.ndarray] = {"close": close_panel}
            if extra_fields:
                expr_panels.update(
                    _load_expression_panels(dp, codes_list, time_axis, extra_fields)
                )
            score_matrix = _expr_scores(node, expr_panels, valid_flags)
        elif signal_exprs:
            nodes = [_prepare_expr(e) for e in signal_exprs]
            union_fields: set[str] = set()
            for node in nodes:
                union_fields |= _collect_expr_fields(node)
            close_panel = np.where(close_matrix > 0, close_matrix, np.nan)
            extra_fields = {f for f in union_fields if f != "close"}
            expr_panels = {"close": close_panel}
            if extra_fields:
                expr_panels.update(
                    _load_expression_panels(dp, codes_list, time_axis, extra_fields)
                )
            z_stack = np.stack(
                [_expr_zscores(node, expr_panels, valid_flags) for node in nodes]
            )
            combined = _combine_factor_scores(z_stack, weights=signal_weights)
            # 全因子无效日分数为 NaN → 选股时视为 -inf 排除
            score_matrix = np.where(np.isfinite(combined), combined, -np.inf)
            if signal_weights:
                w_arr = np.asarray(signal_weights, dtype=np.float64)
                w_arr = w_arr / float(np.sum(w_arr))
                multi_caveat = (
                    f"多因子加权合成: K={len(nodes)}, "
                    f"weights=[{', '.join(f'{w:.4f}' for w in w_arr)}]"
                )
            else:
                multi_caveat = f"多因子等权合成: K={len(nodes)} 个表达式"

        equity, port_returns, turnover_total, positive_days, sim_stats = (
            _run_momentum_backtest(
                close_matrix, returns_matrix, valid_flags,
                n_hold=n_hold, max_single_w=0.05,
                commission=commission, codes=codes_list,
                score_matrix=score_matrix,
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
            (
                SIGNAL_SOURCE_CAVEAT_FMT.format(expr=signal_expr)
                if signal_expr else
                (
                    multi_caveat
                    if multi_caveat else
                    "策略：20日动量因子，月频调仓，等权持有（信号次日生效，无同日前视）"
                )
            ),
            "已扣除印花税/佣金/冲击成本",
            "未处理停牌；涨跌停按板块幅度拦截（ST 无法从代码判断，统一按板块幅度处理）",
        ]
        caveats.extend(
            _format_limit_caveats(
                sim_stats["limit_up_blocked"],
                sim_stats["limit_down_blocked"],
                sim_stats["final_cash_weight"],
            )
        )

        # ── Brinson(BHB 简化) 行业归因：任何异常只记 caveat，不影响主回测 ──
        brinson_ok = False
        br_allocation = br_selection = 0.0
        try:
            br = _brinson_attribution(
                codes_list, sim_stats.get("final_weights"), returns_matrix
            )
            if br is None:
                caveats.append("行业覆盖率不足，跳过归因")
            else:
                brinson_ok = True
                br_allocation = float(br["allocation"])
                br_selection = float(br["selection"])
                caveats.append(
                    f"Brinson(BHB简化): 配置 {br_allocation:.1%} 选择 {br_selection:.1%}"
                    f"（行业来源: industry_map, 覆盖率 {float(br['coverage']):.0%}）"
                )
        except Exception as exc:
            caveats.append(f"Brinson 归因跳过: {type(exc).__name__}: {exc}")
        # Barra 暴露仍无多因子库支撑；Brinson 成功时归因声明收窄为 Barra 单项
        caveats.append(
            "风险暴露(Barra)需要多因子库，当前版本不提供"
            if brinson_ok else ATTRIBUTION_CAVEAT
        )
        if sampled_note:
            caveats.append(sampled_note)
        if bench_caveat:
            caveats.append(bench_caveat)
        if n_skipped:
            caveats.append(f"{n_skipped} 只股票因数据契约校验失败被跳过")

        # ── 有效 IR 双口径（post-audit-8；调仓间隔 = REBALANCE_FREQ，见 _attach_effective_ir）──
        key_metrics = {
            "年化收益": report.annual_return,
            "超额收益": report.excess_return,
            "夏普比": report.sharpe_ratio,
            "最大回撤": report.max_drawdown,
            "信息比率": report.information_ratio,
            "年化换手": report.annual_turnover,
            "t统计量": report.t_statistic,
            **(
                {"配置效应": br_allocation, "选择效应": br_selection}
                if brinson_ok else {}
            ),
        }
        _attach_effective_ir(
            key_metrics, caveats, port_returns, bench_returns, REBALANCE_FREQ
        )
        return Trader3Response(
            success=True,
            data=report,
            summary=(
                f"[{name}·真实数据] {start_date}~{end_date} 年化 {report.annual_return:.1%}, "
                f"夏普 {report.sharpe_ratio:.2f}, 超额 {report.excess_return:.1%}, "
                f"最大回撤 {report.max_drawdown:.1%}"
            ),
            key_metrics=key_metrics,
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
        strategy_config: StrategyConfig | None,
        universe: list[str] | None,
        start_date: str,
        end_date: str,
        constraints: PortfolioConstraints | None,
        benchmark: str,
    ) -> Trader3Response:
        """Qlib 回测（完整 qlib 安装时使用）"""
        raise NotImplementedError("完整 qlib 回测接线规划在后续版本")

    # ── 向量化回测 ──

    def _vectorized_backtest(
        self,
        strategy_config: StrategyConfig | None,
        universe: list[str] | None,
        start_date: str,
        end_date: str,
        constraints: PortfolioConstraints | None,
        benchmark: str,
        commission: CommissionInfo | None = None,
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
        sim_stats: dict[str, Any] = {}
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
        key_metrics = {
            "年化收益": report.annual_return,
            "超额收益": report.excess_return,
            "夏普比": report.sharpe_ratio,
            "最大回撤": report.max_drawdown,
            "信息比率": report.information_ratio,
            "年化换手": report.annual_turnover,
            "t统计量": report.t_statistic,
        }
        caveats = [
            "合成数据回测（无 Qlib），实际表现可能差异显著",
            ATTRIBUTION_CAVEAT,
            "已扣除千分之一印花税 + 万二佣金 + 千分之五冲击成本",
            "组合按月频调仓，等权持有",
            "信号为模拟生成，非真实因子数据",
            "涨跌停约束已接入（合成路径统一主板幅度 9.8%）",
        ] + _format_limit_caveats(
            sim_stats.get("limit_up_blocked", 0),
            sim_stats.get("limit_down_blocked", 0),
            sim_stats.get("final_cash_weight", 0.0),
        )
        # 合成引擎与真实引擎同款调仓节奏：每 REBALANCE_FREQ 个交易日一次
        _attach_effective_ir(
            key_metrics, caveats, port_returns, market_returns, REBALANCE_FREQ
        )
        return Trader3Response(
            success=True,
            data=report,
            summary=(
                f"[{name}] {start_date}~{end_date} 年化 {report.annual_return:.1%}, "
                f"夏普 {report.sharpe_ratio:.2f}, 超额 {report.excess_return:.1%}, "
                f"最大回撤 {report.max_drawdown:.1%}"
            ),
            key_metrics=key_metrics,
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
            caveats=caveats,
        )


# ═══════════════════════════════════════════
# 极端情景压力测试（post-audit-8）
# ═══════════════════════════════════════════


def _stress_windows_overlap(
    a_start: str, a_end: str, b_start: str, b_end: str
) -> bool:
    """闭区间日期交集判断（ISO yyyy-mm-dd 字符串序即时间序）。"""
    return a_start <= b_end and b_start <= a_end


def _summarize_stress_rows(rows: list[dict]) -> list[dict]:
    """
    追加压测汇总行（纯函数）：

    - avg_stress_ann = 各窗口 ann_return 均值；
    - worst_window = max_drawdown 最深（最负）窗口名。
    空输入原样返回。
    """
    if not rows:
        return rows
    avg_ann = float(np.mean([float(r.get("ann_return", 0.0)) for r in rows]))
    worst = min(rows, key=lambda r: float(r.get("max_drawdown", 0.0)))
    return rows + [
        {
            "period_name": STRESS_SUMMARY_NAME,
            "avg_stress_ann": avg_ann,
            "worst_window": str(worst.get("period_name", "")),
        }
    ]


def run_stress_test(
    start_date: str,
    end_date: str,
    universe_codes: list[str] | None = None,
    signal_expr: str = "",
    **kwargs,
) -> list[dict]:
    """
    极端情景压力测试：对 STRESS_PERIODS 中与 [start_date, end_date] 有交集的
    每个历史危机窗口分别运行回测，返回逐窗结果 + 汇总行：

    - 逐窗 {period_name, ann_return, sharpe, max_drawdown, excess}；
    - 汇总行 {"period_name": "_summary", "avg_stress_ann": 各窗口年化均值,
      "worst_window": 最大回撤最深的窗口名}；
    与所有窗口无交集时返回空列表。

    kwargs 透传给 RunBacktestTool.execute；内部复用同一工具实例
    （测试可经 _backtest_tool 注入重定向缓存/桩工具）。
    """
    tool: Any = kwargs.pop("_backtest_tool", None)
    if tool is None:
        tool = RunBacktestTool()
    rows: list[dict] = []
    for period_name, (p_start, p_end) in STRESS_PERIODS.items():
        if not _stress_windows_overlap(start_date, end_date, p_start, p_end):
            continue
        resp = tool.execute(
            universe=universe_codes,
            start_date=p_start,
            end_date=p_end,
            signal_expr=signal_expr,
            **kwargs,
        )
        report = getattr(resp, "data", None)
        if not (getattr(resp, "success", False) and report is not None):
            continue
        rows.append(
            {
                "period_name": period_name,
                "ann_return": float(report.annual_return),
                "sharpe": float(report.sharpe_ratio),
                "max_drawdown": float(report.max_drawdown),
                "excess": float(report.excess_return),
            }
        )
    return _summarize_stress_rows(rows)


# ═══════════════════════════════════════════
# WalkForwardAnalysisTool
# ═══════════════════════════════════════════

# CPCV 组合聚合口径年化基数（基线 v2 裁定 2026-08-25）：pooled 序列按组合重复
# 计日、非日历年化，采用 A 股年均交易日 244，与 WFA 的 TRADING_DAYS_PER_YEAR=252 区分
_CPCV_POOLED_ANNUAL_DAYS = 244


class WalkForwardAnalysisTool(BaseTool):
    """Walk-Forward Analysis (M1: 真实滚动验证)"""

    tool_name = "walk_forward_analysis"
    tool_description = "滚动 WFA 验证，检测过拟合，返回 IS/OOS 对比 + 参数稳定性 + 过拟合概率"
    tool_version = "1.0.0"
    tool_category = "backtest"

    def execute(
        self,
        strategy_config: StrategyConfig | None = None,
        train_window: int = 252,
        test_window: int = 63,
        signal_expr: str = "",
        signal_exprs: list[str] | None = None,
        step: int | None = None,
        mode: str = "wfa",
    ) -> Trader3Response:
        """
        滚动 WFA（真实 qlib 数据优先）。

        - qlib 可用：在真实面板上滚动 —— 每窗以训练段动量排名确定等权持仓，
          固定应用于测试段（训练段收盘信息最早测试段首日生效，与回测引擎
          pending 语义一致）；OOS 汇总指标基于拼接的非重叠 OOS 日收益计算。
        - 测试段首日视为调仓执行日（post-audit-5）：按主循环同款规则扣
          DEFAULT_COSTS 单边换手成本并套用涨跌停拦截，段内不调仓。
        - step 缺省等于 test_window（OOS 窗口非重叠，显著性不被共享样本抬高）；
          显式传入更小的 step 时 caveats 警告窗口重叠会高估显著性。
        - qlib 不可用：回退种子 123 合成数据，并在 caveats 明示"WFA基于合成数据"。
        - mode="cpcv"：headline（样本外收益/样本外夏普）改由 CPCV 组合路径聚合
          （oos_concat_pooled，全部测试片段按组合序路径依赖拼接）计算
          （年化基数 244），并保留 WFA 单点口径对照键 样本外收益_wfa口径 /
          样本外夏普_wfa口径；另叠加运行 CPCV 分布指标（AFML ch.12，
          n_blocks=6/test_blocks=2/purge=5，key_metrics 追加 cpcv_*）；
          caveat 注明 "headline 为 CPCV 组合聚合口径"。缺省 "wfa" 完全不变。
        """
        eff_step = int(step) if step is not None else int(test_window)
        mode_eff = (mode or "wfa").strip().lower()
        if mode_eff not in ("wfa", "cpcv"):
            return Trader3Response.error(
                f"未知 mode={mode!r}（支持 'wfa' / 'cpcv'）"
            )

        panel = None
        panel_err: Exception | None = None
        dp = None
        try:
            dp = _open_qlib_dp()
            panel = _load_wfa_panel(dp)
        except Exception as e:  # 数据缺失/损坏 → 合成回退
            panel_err = e

        exec_codes: list[str] | None = None
        if panel is not None:
            codes_list, close_matrix, returns_matrix, _valid_flags, time_axis = panel
            stock_returns = returns_matrix
            if signal_expr:
                try:
                    node = _prepare_expr(signal_expr)
                except Exception as e:
                    return Trader3Response.error(f"signal_expr 非法: {e}")
                expr_fields = _collect_expr_fields(node)
                close_panel = np.where(close_matrix > 0, close_matrix, np.nan)
                extra_fields = {f for f in expr_fields if f != "close"}
                expr_panels: dict[str, np.ndarray] = {"close": close_panel}
                if extra_fields:
                    expr_panels.update(
                        _load_expression_panels(dp, codes_list, time_axis, extra_fields)
                    )
                missing = {f for f in expr_fields if f not in expr_panels}
                if missing - {"close"}:
                    return Trader3Response.error(
                        f"WFA 表达式字段缺失: {sorted(missing)}"
                    )
                factor_scores = _expr_scores(node, expr_panels, _valid_flags)
            elif signal_exprs:
                try:
                    nodes = [_prepare_expr(e) for e in signal_exprs]
                except Exception as e:
                    return Trader3Response.error(f"signal_exprs 非法: {e}")
                union_fields: set[str] = set()
                for node in nodes:
                    union_fields |= _collect_expr_fields(node)
                close_panel = np.where(close_matrix > 0, close_matrix, np.nan)
                extra_fields = {f for f in union_fields if f != "close"}
                expr_panels = {"close": close_panel}
                if extra_fields:
                    expr_panels.update(
                        _load_expression_panels(dp, codes_list, time_axis, extra_fields)
                    )
                missing = {f for f in union_fields if f not in expr_panels}
                if missing - {"close"}:
                    return Trader3Response.error(
                        f"WFA 表达式字段缺失: {sorted(missing)}"
                    )
                z_stack = np.stack(
                    [_expr_zscores(n, expr_panels, _valid_flags) for n in nodes]
                )
                combined = _combine_factor_scores(z_stack)
                factor_scores = np.where(np.isfinite(combined), combined, -np.inf)
            else:
                factor_scores = _momentum_scores(close_matrix)
            exec_codes = codes_list  # 涨跌停幅度逐股按板块判定
            engine_caveat = "WFA 基于真实 qlib 数据"
            if signal_expr or signal_exprs:
                src = signal_expr or " + ".join(signal_exprs or [])
                engine_caveat += f"；信号源: {src}"
            if eff_step >= test_window:
                engine_caveat += "，OOS 窗口非重叠"
            engine_caveat += (
                f"（面板 {stock_returns.shape[0]} 日 × {len(codes_list)} 股，"
                "成分按起点 asof 过滤）"
            )
        else:
            if signal_expr or signal_exprs:
                return Trader3Response.error(
                    "signal_expr(s) 模式需要真实 qlib 面板；当前不可用，"
                    "拒绝静默回退合成动量（会丢弃用户信号定义）"
                )
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
            stock_returns, factor_scores, train_window, test_window, eff_step,
            codes=exec_codes,
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

        # ── Deflated Sharpe（单策略口径 = PSR）：滚动窗口是"同策略的时间分段"，
        #    不构成独立试验数；跨策略的多重比较校正由基线跑批脚本在
        #    策略集合层面统一计算（n_trials=策略数）。此处 n_trials=1。──
        sqrt_ann = math.sqrt(TRADING_DAYS_PER_YEAR)
        dsr_value = deflated_sharpe_ratio(
            sharpe_observed=oos_sharpe / sqrt_ann,
            n_trials=1,
            sr_variance=None,
            tail_risk_adj=True,
            returns=oos_concat if oos_concat.size else None,
            n_periods=int(oos_concat.size),
        )
        dsr_caveat = f"DSR(PSR)={dsr_value:.2f}（单策略口径；跨策略校正见基线报告）"

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
            "WFA 含单边成本与首日涨跌停约束"
            "（测试段首日按主循环规则执行建仓并扣 DEFAULT_COSTS 换手成本，段内不调仓）",
            "过拟合概率基于 IS/OOS 夏普比衰减: max(0, 1 - OOS_Sharpe / IS_Sharpe)",
            dsr_caveat,
        ]
        if eff_step < test_window:
            caveats.insert(
                1,
                f"step={eff_step} < test_window={test_window}: "
                "OOS 窗口重叠，相邻窗口共享样本会高估显著性",
            )

        key_metrics = {
            "样本内收益": report.is_mean_return,
            "样本外收益": report.oos_mean_return,
            "样本内夏普": report.is_sharpe,
            "样本外夏普": report.oos_sharpe,
            "参数稳定性": report.parameter_stability,
            "过拟合概率": report.overfitting_probability,
            "OOS交易日": int(oos_concat.size),
            "dsr": dsr_value,
        }

        if mode_eff == "cpcv":
            # 延迟导入避免模块级环（cpcv 复用本模块的执行约束/涨跌停函数）
            from trader3.tools.cpcv import run_cpcv

            cpcv_res = run_cpcv(
                stock_returns, factor_scores,
                codes=exec_codes, n_blocks=6, test_blocks=2, purge=5,
            )
            # headline 切换为 CPCV 组合聚合口径（基线 v2 裁定 2026-08-25）：
            # 全部测试片段按组合序路径依赖拼接后计算年化收益/夏普
            pooled = np.asarray(cpcv_res["oos_concat_pooled"], dtype=np.float64)
            pooled_mean = float(np.mean(pooled)) if pooled.size else 0.0
            pooled_std = (
                float(np.std(pooled, ddof=1)) if pooled.size >= 2 else 0.0
            )
            ann_days = _CPCV_POOLED_ANNUAL_DAYS
            key_metrics["样本外收益"] = pooled_mean * ann_days
            key_metrics["样本外夏普"] = (
                pooled_mean / pooled_std * math.sqrt(ann_days)
                if pooled_std > 1e-12 else 0.0
            )
            # WFA 单点口径对照（report 字段仍由 _run_wfa_rolling 拼接口径计算）
            key_metrics["样本外收益_wfa口径"] = report.oos_mean_return
            key_metrics["样本外夏普_wfa口径"] = report.oos_sharpe
            key_metrics.update({
                "cpcv_median_sr": cpcv_res["sr_ann_median"],
                "cpcv_p05": cpcv_res["sr_ann_p05"],
                "cpcv_p95": cpcv_res["sr_ann_p95"],
                "cpcv_prob_negative": cpcv_res["prob_negative"],
            })
            caveats.append(
                f"CPCV: C(6,2) 组合净化交叉验证 (purge=5)："
                f"{cpcv_res['n_combos']} 条组合路径（{cpcv_res['paths']} 条独立路径）、"
                f"年化夏普 p05/p50/p95 = {cpcv_res['sr_ann_p05']:.2f}/"
                f"{cpcv_res['sr_ann_median']:.2f}/{cpcv_res['sr_ann_p95']:.2f}、"
                f"日SR<0 占比 {cpcv_res['prob_negative']:.0%}"
                " —— OOS 表现为分布而非单点"
            )
            caveats.append(
                "headline 为 CPCV 组合聚合口径"
                "（全部测试片段按组合序路径依赖拼接，同一测试日可重复计入；"
                f"年化基数 {ann_days}）；WFA 单点口径见 样本外*_wfa口径 键"
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
            key_metrics=key_metrics,
            caveats=caveats,
        )
