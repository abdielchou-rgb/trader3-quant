"""
3号交易员 — 交易成本/执行规划 (M3: 真实 Almgren-Chriss + 执行引擎)
"""

from __future__ import annotations

import math

from trader3.base_tool import BaseTool, ChartSpec, Trader3Response
from trader3.models import ExecutionPlan, TCAEstimate

# ── A股默认参数 ──
_DEFAULT_COMMISSION_BP = 2.0        # 佣金 2bp
_DEFAULT_STAMP_TAX_BP = 10.0        # 印花税 10bp (仅卖出)
_DEFAULT_SPREAD_BP = 15.0           # 买卖价差 15bp (A股典型)
_DEFAULT_VOLATILITY_ANNUAL = 0.30   # 年化波动率 30%
_DEFAULT_DAILY_VOLUME_CNY = 1e8     # 日均成交额 1亿
_DEFAULT_TRADING_HORIZON = 0.5      # 默认交易时长 0.5 天 (约4小时)

# Almgren-Chriss 模型系数
_K_PERMANENT = 0.1     # 永久冲击系数
_ETA_TEMPORARY = 1.0   # 临时冲击系数

# 执行计划参数
_SLICE_INTERVAL_MINUTES = 30        # 切片间隔


def _daily_vol(sigma_annual: float) -> float:
    """年化波动率 -> 日波动率"""
    return sigma_annual / math.sqrt(252)


def _almgren_chriss_impact(
    order_value: float,
    daily_volume: float,
    sigma_annual: float,
    horizon_days: float,
    side: str,
) -> tuple[float, float]:
    """
    Almgren-Chriss 冲击成本模型

    永久冲击:  k * sigma * sqrt(Q/V) * sign(Q)
    临时冲击:  eta * sigma * sqrt(Q/(V*T)) * sign(Q)

    Returns:
        (permanent_impact_bp, temporary_impact_bp)
    """
    sign = 1.0 if side == "buy" else -1.0
    participation = order_value / daily_volume if daily_volume > 0 else 0.0
    participation = min(participation, 1.0)

    sigma_d = _daily_vol(sigma_annual)

    perm_bp = _K_PERMANENT * sigma_d * math.sqrt(max(participation, 1e-10)) * sign
    temp_bp = _ETA_TEMPORARY * sigma_d * math.sqrt(
        max(participation / max(horizon_days, 0.01), 1e-10)
    ) * sign

    return abs(perm_bp) * 10000, abs(temp_bp) * 10000


def _lookup(key: str, symbol: str, market_data: dict, fallback: float) -> float:
    """从 market_data 中安全取值"""
    if sd := market_data.get(symbol):
        val = sd.get(key)
        if val is not None and val > 0:
            return val
    return fallback


class EstimateTransactionCostTool(BaseTool):
    """交易成本估算（M3: Almgren-Chriss 模型 + A股特有成本）"""

    tool_name = "estimate_transaction_cost"
    tool_description = "估算交易成本（滑点/印花税/冲击/时机风险/机会成本），返回总成本bp + 人民币金额 + 执行建议"
    tool_version = "3.0.0"
    tool_category = "execution"

    def execute(  # type: ignore[override]
        self,
        orders: list[dict] | None = None,
        method: str = "implementation_shortfall",
        market_data: dict | None = None,
    ) -> Trader3Response:
        """Almgren-Chriss 驱动交易成本估算"""
        if not orders:
            return Trader3Response.error("orders 参数不能为空")

        orders = orders or []
        market_data = market_data or {}

        # 逐订单估算
        total_value_cny = 0.0
        sum_stamp_bp = 0.0
        sum_impact_bp = 0.0
        sum_timing_bp = 0.0
        sum_opp_cost_bp = 0.0
        details = []

        for o in orders:
            symbol = o.get("symbol", "unknown")
            side = o.get("side", "buy")
            value_cny = float(o.get("value_cny", 0))
            horizon = float(o.get("trading_horizon_days", _DEFAULT_TRADING_HORIZON))

            total_value_cny += value_cny

            dv = _lookup("avg_daily_value_cny", symbol, market_data, _DEFAULT_DAILY_VOLUME_CNY)
            vol = _lookup("volatility_annual", symbol, market_data, _DEFAULT_VOLATILITY_ANNUAL)
            spread = _lookup("spread_bp", symbol, market_data, _DEFAULT_SPREAD_BP)

            perm_bp, temp_bp = _almgren_chriss_impact(value_cny, dv, vol, horizon, side)
            impact_bp = perm_bp + temp_bp

            stamp = _DEFAULT_STAMP_TAX_BP if side == "sell" else 0.0
            timing = spread * 0.2
            participation = value_cny / max(dv, 1)
            opp_cost = min(participation * 20, 10.0)

            sum_impact_bp += impact_bp
            sum_stamp_bp += stamp
            sum_timing_bp += timing
            sum_opp_cost_bp += opp_cost

            details.append({
                "symbol": symbol,
                "side": side,
                "value_cny": value_cny,
                "daily_volume_cny": dv,
                "participation_pct": round(participation * 100, 2),
                "impact_bp": round(impact_bp, 1),
                "stamp_bp": stamp,
                "timing_bp": round(timing, 1),
            })

        # 按订单金额加权平均（简单平均会放大/缩小大单的真实成本占比）
        w_total = sum(o.get("value_cny", 0) for o in orders) or 1.0
        avg_stamp = sum(
            (d["stamp_bp"] * d["value_cny"]) / w_total for d in details
        ) if details else 0.0
        avg_impact = sum(
            (d["impact_bp"] * d["value_cny"]) / w_total for d in details
        ) if details else 0.0
        avg_timing = sum(
            (d["timing_bp"] * d["value_cny"]) / w_total for d in details
        ) if details else 0.0
        avg_opp = sum(
            (min(d["value_cny"] / max(d["daily_volume_cny"], 1) * 20, 10.0) * d["value_cny"]) / w_total
            for d in details
        ) if details else 0.0

        total_bp = _DEFAULT_COMMISSION_BP + avg_stamp + avg_impact + avg_timing + avg_opp
        total_cny = total_value_cny * total_bp / 10000

        # 紧急度判断
        max_part = max(
            (o.get("value_cny", 0) / max(_lookup("avg_daily_value_cny", o.get("symbol", ""), market_data, _DEFAULT_DAILY_VOLUME_CNY), 1))
            for o in orders
        )
        urgency = "high" if max_part > 0.15 else "normal" if max_part > 0.05 else "low"

        estimate = TCAEstimate(
            total_cost_bp=round(total_bp, 1),
            commission_bp=_DEFAULT_COMMISSION_BP,
            stamp_tax_bp=round(avg_stamp, 1),
            impact_bp=round(avg_impact, 1),
            timing_risk_bp=round(avg_timing, 1),
            opportunity_cost_bp=round(avg_opp, 1),
            total_cost_cny=round(total_cny, 2),
            recommended_urgency=urgency,
            execution_suggestions=self._suggestions(orders, market_data, max_part),
        )

        return Trader3Response(
            success=True,
            data=estimate,
            summary=(
                f"预期总成本 {estimate.total_cost_bp:.0f}bp "
                f"(¥{estimate.total_cost_cny:,.0f}), "
                f"其中冲击成本 {estimate.impact_bp:.0f}bp 为主力贡献"
            ),
            key_metrics={
                "总成本(bp)": estimate.total_cost_bp,
                "佣金(bp)": estimate.commission_bp,
                "印花税(bp)": estimate.stamp_tax_bp,
                "冲击成本(bp)": estimate.impact_bp,
                "时机风险(bp)": estimate.timing_risk_bp,
                "机会成本(bp)": estimate.opportunity_cost_bp,
                "总成本(元)": estimate.total_cost_cny,
            },
            charts=[
                ChartSpec(
                    chart_type="bar",
                    title="交易成本分解",
                    data={
                        "佣金": estimate.commission_bp,
                        "印花税": estimate.stamp_tax_bp,
                        "冲击": estimate.impact_bp,
                        "时机风险": estimate.timing_risk_bp,
                        "机会成本": estimate.opportunity_cost_bp,
                    },
                    description="交易成本分项构成（Almgren-Chriss 模型）",
                ),
                ChartSpec(
                    chart_type="table",
                    title="逐订单成本明细",
                    data={"details": details},
                    description="每个订单的独立成本估算",
                ),
            ],
            caveats=[
                "冲击成本基于 Almgren-Chriss 模型估算",
                "日均成交额和波动率为估算值，实际可能大幅偏离",
                "A股特有：印花税仅卖出时收取（10bp）",
                "T+1 制度下当日买入不可卖出",
                "实际成本可能因市场条件大幅偏离",
            ],
        )

    @staticmethod
    def _suggestions(orders: list[dict], market_data: dict, max_part: float) -> list[str]:
        s = []
        if max_part > 0.15:
            s.append("交易量较大，建议分批执行，每批不超过日均成交量的5%")
            s.append("使用冰山订单隐藏大单意图")
        elif max_part > 0.05:
            s.append("建议使用TWAP/VWAP算法分散执行")
        else:
            s.append("订单规模适中，可择机直接执行")
        s.append("避开开盘前15分钟和收盘前30分钟")
        s.append("当前A股适用±10%涨跌停限制")
        if any(o.get("side") == "sell" for o in orders):
            s.append("卖出订单需考虑T+1可用余额")
        return s


class GenerateExecutionPlanTool(BaseTool):
    """生成执行计划（M3: TWAP/VWAP/IS/Adaptive 真实算法）"""

    tool_name = "generate_execution_plan"
    tool_description = "生成分时执行计划（TWAP/VWAP/IS/Adaptive），含分时切片 + 预期的完成度 + 风控熔断线"
    tool_version = "3.0.0"
    tool_category = "execution"

    def execute(  # type: ignore[override]
        self,
        target_weights: dict[str, float] | None = None,
        current_weights: dict[str, float] | None = None,
        algorithm: str = "adaptive_vwap",
        urgency: str = "normal",
        portfolio_value: float = 10_000_000.0,
        market_data: dict | None = None,
    ) -> Trader3Response:
        """生成执行计划"""
        target_weights = target_weights or {}
        current_weights = current_weights or {}
        market_data = market_data or {}

        if not target_weights:
            return Trader3Response.error("target_weights 参数不能为空")

        orders = self._compute_orders(target_weights, current_weights, portfolio_value)
        slices = self._generate_slices(orders, algorithm, urgency, market_data)

        plan = ExecutionPlan(
            algorithm=algorithm,
            urgency=urgency,
            slices=slices,
            expected_completion_rate=round(self._estimate_completion(algorithm, urgency), 4),
            expected_total_cost_bp=round(
                self._estimate_plan_cost(urgency, orders, market_data), 1
            ),
            risk_limits=self._compute_risk_limits(urgency),
        )

        return Trader3Response(
            success=True,
            data=plan,
            summary=(
                f"执行计划 ({algorithm}): {len(plan.slices)} 笔切片, "
                f"预期完成率 {plan.expected_completion_rate:.0%}, "
                f"预期成本 {plan.expected_total_cost_bp:.0f}bp"
            ),
            key_metrics={
                "切片数": len(plan.slices),
                "预期完成率": plan.expected_completion_rate,
                "预期成本(bp)": plan.expected_total_cost_bp,
            },
            charts=[
                ChartSpec(
                    chart_type="table",
                    title="执行时间表",
                    data={"slices": slices},
                    description="分时执行计划明细",
                ),
            ],
            caveats=[
                f"算法: {algorithm}",
                f"紧急度: {urgency}",
                "真实成交价可能偏离计划价格",
                "市场剧烈波动时风控熔断线将触发",
            ],
        )

    # ── 内部方法 ──

    @staticmethod
    def _compute_orders(
        target_weights: dict[str, float],
        current_weights: dict[str, float],
        portfolio_value: float,
    ) -> list[dict]:
        """目标权重 -> 待执行订单（遍历 target ∪ current 并集，清仓股生成卖单）"""
        orders = []
        for sym in set(target_weights) | set(current_weights):
            tw = target_weights.get(sym, 0.0)
            cw = current_weights.get(sym, 0.0)
            diff = tw - cw
            if abs(diff) < 0.001:
                continue
            orders.append({
                "symbol": sym,
                "side": "buy" if diff > 0 else "sell",
                "value_cny": round(abs(diff) * portfolio_value, 2),
                "weight_diff": diff,
            })
        return orders

    def _generate_slices(
        self,
        orders: list[dict],
        algorithm: str,
        urgency: str,
        market_data: dict,
    ) -> list[dict]:
        """根据算法路由到具体切片策略"""
        if not orders:
            return []
        if algorithm == "twap":
            return self._twap(orders, urgency)
        elif algorithm == "vwap":
            return self._vwap(orders, urgency)
        elif algorithm in ("implementation_shortfall", "is"):
            return self._is(orders, urgency)
        else:  # adaptive_vwap
            return self._adaptive(orders, urgency)

    @staticmethod
    def _time_grid() -> list[str]:
        """09:35 - 14:55 每30分钟，跳过 11:30-13:00 午休休市时段"""
        times = []
        h, m = 9, 35
        while h < 15 or (h == 15 and m <= 0):
            t = f"{h:02d}:{m:02d}"
            in_morning = t <= "11:30"
            in_afternoon = "13:05" <= t <= "15:00"
            if in_morning or in_afternoon:
                times.append(t)
            m += _SLICE_INTERVAL_MINUTES
            if m >= 60:
                h += 1
                m %= 60
        return [t for t in times if t <= "15:00"]

    @staticmethod
    def _u_shape_weights(n: int) -> list[float]:
        """A股 U 型成交量分布：开盘/收盘放量，午间缩量"""
        if n <= 1:
            return [1.0]
        # cos 形状：首尾高(1.0)、中间低(0.0)，与真实 U 型分布同相
        raw = [0.5 + 0.5 * math.cos(math.pi * (2.0 * i / (n - 1) - 1.0)) for i in range(n)]
        s = sum(raw)
        return [w / s for w in raw]

    @staticmethod
    def _front_load_factor(urgency: str) -> float:
        return {"high": 0.50, "normal": 0.35, "low": 0.20}.get(urgency, 0.35)

    @staticmethod
    def _slice_count(urgency: str) -> int:
        return {"high": 4, "normal": 6, "low": 10}.get(urgency, 6)

    # ── 四种算法 ──

    def _twap(self, orders: list[dict], urgency: str) -> list[dict]:
        """TWAP: 时间均匀切分"""
        times = self._time_grid()[:self._slice_count(urgency)]
        n = len(times)
        slices = []
        for t in times:
            for o in orders:
                slices.append({
                    "time": t,
                    "symbol": o["symbol"],
                    "side": o["side"],
                    "value_cny": round(o["value_cny"] / n, 2),
                    "weight": round(1.0 / n, 4),
                })
        return slices

    def _vwap(self, orders: list[dict], urgency: str) -> list[dict]:
        """VWAP: U型成交量加权分布"""
        times = self._time_grid()[:self._slice_count(urgency)]
        n = len(times)
        vw = self._u_shape_weights(n)
        slices = []
        for i, t in enumerate(times):
            for o in orders:
                slices.append({
                    "time": t,
                    "symbol": o["symbol"],
                    "side": o["side"],
                    "value_cny": round(o["value_cny"] * vw[i], 2),
                    "weight": round(vw[i], 4),
                })
        return slices

    def _is(self, orders: list[dict], urgency: str) -> list[dict]:
        """Implementation Shortfall: 前端加载"""
        times = self._time_grid()[:max(4, self._slice_count(urgency))]
        n = len(times)
        front_n = max(1, int(n * 0.3))
        front_w = self._front_load_factor(urgency)
        back_w = 1.0 - front_w
        slices = []
        for i, t in enumerate(times):
            w = (front_w / front_n) if i < front_n else (back_w / (n - front_n))
            for o in orders:
                slices.append({
                    "time": t,
                    "symbol": o["symbol"],
                    "side": o["side"],
                    "value_cny": round(o["value_cny"] * w, 2),
                    "weight": round(w, 4),
                })
        return slices

    def _adaptive(self, orders: list[dict], urgency: str) -> list[dict]:
        """Adaptive: high->IS, normal->VWAP, low->TWAP"""
        if urgency == "high":
            return self._is(orders, urgency)
        elif urgency == "low":
            return self._twap(orders, urgency)
        return self._vwap(orders, urgency)

    @staticmethod
    def _estimate_completion(algorithm: str, urgency: str) -> float:
        base = {"high": 0.92, "normal": 0.95, "low": 0.98}.get(urgency, 0.95)
        bonus = {"vwap": 0.01, "implementation_shortfall": -0.02}.get(algorithm, 0.0)
        return min(base + bonus, 0.99)

    @staticmethod
    def _estimate_plan_cost(
        urgency: str,
        orders: list[dict] | None = None,
        market_data: dict | None = None,
    ) -> float:
        """计划成本 = 佣金/印花税底仓 + 冲击成本（随参与率平方根增长）+ 紧急度罚项"""
        penalty = {"high": 10.0, "normal": 0.0, "low": -3.0}.get(urgency, 0.0)
        base = _DEFAULT_COMMISSION_BP + 1.0  # 佣金 + 印花税近似（卖出单边）
        orders = orders or []
        market_data = market_data or {}
        impact = 0.0
        for o in orders:
            dv = _DEFAULT_DAILY_VOLUME_CNY
            for k in ("avg_daily_value_cny", "daily_volume_cny"):
                v = market_data.get(o.get("symbol", ""), {})
                if isinstance(v, dict) and v.get(k):
                    dv = float(v[k])
                    break
            participation = o.get("value_cny", 0.0) / max(dv, 1.0)
            side_mult = 0.0 if o.get("side") == "buy" else 1.0  # 印花税只在卖出
            stamp = side_mult * 1.0
            impact += (base + stamp + min(math.sqrt(participation) * 45.0, 60.0)) * (
                o.get("value_cny", 0.0)
            )
        total_value = sum(o.get("value_cny", 0.0) for o in orders)
        if total_value <= 0:
            return 20.0 + penalty
        return (impact / total_value) + penalty

    @staticmethod
    def _compute_risk_limits(urgency: str) -> dict[str, float]:
        dev = {"high": 0.03, "normal": 0.02, "low": 0.01}.get(urgency, 0.02)
        part = {"high": 0.15, "normal": 0.10, "low": 0.05}.get(urgency, 0.10)
        return {
            "max_price_deviation": dev,
            "max_participation_rate": part,
            "min_slice_interval_seconds": 30,
            "circuit_breaker_deviation": dev * 2,
        }
