"""
3号交易员 — 数据模型
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════
# 策略/配置类
# ═══════════════════════════════════════════

@dataclass
class FactorConfig:
    name: str
    weight: float = 0.0
    direction: str = "long"          # long / short / overlay
    type: str = ""                   # 因子类型标记（如 overlay）
    neutralize: List[str] = field(default_factory=list)  # ["industry", "size", ...]
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """策略定义"""
    name: str = ""
    version: str = "0.1.0"
    factors: List[FactorConfig] = field(default_factory=list)
    universe: Dict[str, Any] = field(default_factory=dict)
    constraints_path: str = ""
    risk_model: Dict[str, Any] = field(default_factory=dict)
    optimizer: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioConstraints:
    """组合约束"""
    max_single_weight: float = 0.05         # 单票上限
    max_industry_deviation: float = 0.10    # 行业偏离上限 vs 基准
    max_style_deviation: float = 0.15       # 风格因子暴露上限
    max_turnover_annual: float = 3.0        # 年化换手上限
    min_liquidity_buffer: int = 5           # 单票 ≤ 日均成交额 %
    long_only: bool = True
    max_positions: int = 80
    min_positions: int = 20
    cash_buffer: float = 0.02
    max_sector: float = 0.25

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RiskModel:
    """风险模型配置"""
    type: str = "sample_covariance"   # sample_covariance / barra_cne6 / factor_covariance
    frequency: str = "daily"
    half_life: int = 252
    decay: float = 0.94


# ═══════════════════════════════════════════
# 输出模型
# ═══════════════════════════════════════════

@dataclass
class BacktestReport:
    """回测报告"""
    start_date: str = ""
    end_date: str = ""
    benchmark: str = "000300.SH"

    # 核心绩效
    annual_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0

    # 换手
    annual_turnover: float = 0.0
    avg_holding_period: float = 0.0

    # 归因
    brinson_allocation: Dict[str, float] = field(default_factory=dict)
    barra_exposure: Dict[str, float] = field(default_factory=dict)

    # 分段表现
    period_returns: Dict[str, float] = field(default_factory=dict)  # {"trending_up": 0.25, ...}

    # 统计检验
    t_statistic: float = 0.0
    p_value: float = 0.0
    information_ratio: float = 0.0

    # 序列
    equity_curve: List[float] = field(default_factory=list)


@dataclass
class WFAReport:
    """Walk-Forward Analysis 报告"""
    train_window: int = 252
    test_window: int = 63
    step: int = 21
    windows: int = 0

    is_mean_return: float = 0.0
    oos_mean_return: float = 0.0
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    parameter_stability: float = 0.0     # 参数稳定性（1 为完全稳定）
    overfitting_probability: float = 0.0 # 过拟合概率

    window_results: List[dict] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """组合优化结果"""
    target_weights: Dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    expected_risk: float = 0.0
    expected_sharpe: float = 0.0
    factor_exposure: Dict[str, float] = field(default_factory=dict)
    turnover_cost_bp: float = 0.0
    constraints_satisfied: bool = True
    constraint_violations: List[str] = field(default_factory=list)
    max_single_weight_cap: float = 0.0  # 单票权重上限（供门禁独立复检；0=未提供）


@dataclass
class TCAEstimate:
    """交易成本估算"""
    total_cost_bp: float = 0.0
    commission_bp: float = 0.0
    stamp_tax_bp: float = 0.0
    impact_bp: float = 0.0
    timing_risk_bp: float = 0.0
    opportunity_cost_bp: float = 0.0
    total_cost_cny: float = 0.0
    recommended_urgency: str = "normal"
    execution_suggestions: List[str] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """执行计划"""
    algorithm: str = "adaptive_vwap"
    urgency: str = "normal"

    slices: List[dict] = field(default_factory=list)  # [{time: "09:35", symbol: "...", quantity: ..., price_limit: ...}]
    expected_completion_rate: float = 0.0
    expected_total_cost_bp: float = 0.0

    risk_limits: Dict[str, float] = field(default_factory=dict)  # {"max_price_deviation": 0.02, ...}


@dataclass
class SignalValidationReport:
    """信号验证报告"""
    signal_name: str = ""

    ic_mean: float = 0.0
    ic_std: float = 0.0
    icir: float = 0.0
    ic_series: List[float] = field(default_factory=list)

    group_returns: Dict[str, float] = field(default_factory=dict)  # {"Q1": 0.05, "Q5": -0.03}
    monotonicity: float = 0.0

    half_life_periods: float = 0.0  # 半衰期（交易日），自相关衰减法
    half_life_months: float = 0.0   # 已弃用，兼容旧字段（= periods/21）
    crowding_index: float = 0.0     # 拥挤度代理：信号截面平均|成对相关|

    conditional_validity: Dict[str, float] = field(default_factory=dict)  # {"low_vol": 0.12, "high_vol": 0.01}

    long_short_return: float = 0.0
    long_only_return: float = 0.0


@dataclass
class RegimeDiagnosis:
    """市场状态诊断"""
    current_regime: str = ""         # trending_up / ranging / bearish / high_vol / liquidity_crisis
    regime_probabilities: Dict[str, float] = field(default_factory=dict)  # {regime: prob, ...}
    regime_entropy: float = 0.0      # 不确定性（越低越明确）

    key_indicators: Dict[str, float] = field(default_factory=dict)  # {indicator: value}
    historical_analog: str = ""       # 历史类比期

    strategy_suggestion: str = ""     # 策略建议（仓位/风格/信号权重倾向）
    suggested_position: float = 1.0   # 建议仓位 0.0~1.0


@dataclass
class ValuationReport:
    """估值报告"""
    code: str = ""
    methods: Dict[str, dict] = field(default_factory=dict)  # {"dcf": {target, base_assumptions}, ...}

    fair_value_base: float = 0.0
    fair_value_bull: float = 0.0
    fair_value_bear: float = 0.0
    weighted_target: float = 0.0

    implied_return: float = 0.0       # 隐含收益率
    upside_probability: float = 0.0   # 上行概率

    key_assumptions: Dict[str, Any] = field(default_factory=dict)
    sensitivity: Dict[str, List[float]] = field(default_factory=dict)  # 敏感性分析


@dataclass
class ScorecardReport:
    """评分卡报告"""
    template: str = "quality_growth"  # quality_growth / value / turnaround

    overall_score: float = 0.0
    dimension_scores: Dict[str, float] = field(default_factory=dict)  # {"quality": 7.5, ...}
    peer_comparison: Dict[str, float] = field(default_factory=dict)

    red_flags: List[str] = field(default_factory=list)
    key_positives: List[str] = field(default_factory=list)
    key_concerns: List[str] = field(default_factory=list)


@dataclass
class PrivateCompanyBridge:
    """非上市公司估值桥"""
    company_name: str = ""
    industry: str = ""
    estimated_revenue: float = 0.0           # 推算收入（亿元）
    ps_multiple: float = 0.0                 # 行业适用PS倍数
    ps_based_valuation: float = 0.0          # 收入×PS倍数
    comparable_companies: List[dict] = field(default_factory=list)  # [{name, pe, ps, ev_ebitda, liquidity_discount_adjusted}]
    median_comparable_pe: float = 0.0
    median_comparable_ps: float = 0.0
    liquidity_discount: float = 0.30         # 流动性折价（25-35%）
    comp_based_valuation: float = 0.0        # 可比公司法估值（含流动性折价）
    fair_value_range: List[float] = field(default_factory=list)  # [保守, 合理, 乐观]
    key_assumptions: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)