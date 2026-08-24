"""
3号交易员 — M1~M6 全量系统测试

覆盖所有里程碑：
- M1: 回测引擎 / WFA
- M2: 组合优化（风险平价/均值-方差/Black-Litterman）/ 情境路由
- M3: TCA（含 A股印花税规则）/ 执行计划（四种算法）
- M4: 信号验证 / HMM 市场状态
- M5: 估值锚 / 基本面评分卡
- M6: IronGate 门禁 / FastAPI 服务

所有断言基于确定性（固定随机种子），可重复运行。
"""

import math

import pytest

from trader3 import Trader3, Trader3Response


# ═══════════════════════════════════════════
# M1 回测引擎
# ═══════════════════════════════════════════


class TestBacktestReal:
    """M1: 真实回测引擎"""

    def setup_method(self):
        self.t3 = Trader3()

    def test_backtest_real(self):
        """run_backtest 返回真实净值曲线与合理指标"""
        r = self.t3.run_backtest(start_date="2020-01-01", end_date="2025-12-31")
        assert r.success, r.summary
        assert isinstance(r, Trader3Response)
        assert "年化" in r.summary

        report = r.data
        # 净值曲线：6 年日频约 1500 点
        assert len(report.equity_curve) > 500
        assert report.equity_curve[-1] > 0

        # 指标合理
        assert report.annual_return > -0.5
        assert report.volatility > 0
        assert report.sharpe_ratio != 0
        assert report.max_drawdown < 0
        assert report.calmar_ratio != 0
        assert 0 < report.win_rate <= 1
        assert report.t_statistic != 0
        assert report.p_value > 0

        # key_metrics 齐全
        for k in ("年化收益", "超额收益", "夏普比", "最大回撤", "信息比"):
            assert k in r.key_metrics, f"missing key_metrics: {k}"

        # 分时段表现保留；伪归因已移除（post-audit-4）：字段必须为空且报告明示不提供
        assert not report.brinson_allocation and not report.barra_exposure
        assert any("不提供" in c for c in r.caveats)
        assert "trending_up" in report.period_returns

        # 元信息
        assert r.metadata.get("tool") == "run_backtest"
        assert r.metadata.get("version") >= "1.0.0"

    def test_wfa_real(self):
        """walk_forward 返回 IS/OOS 对比与过拟合概率"""
        r = self.t3.walk_forward_analysis(train_window=252, test_window=63)
        assert r.success, r.summary
        assert "过拟合概率" in r.summary

        report = r.data
        assert report.windows >= 5
        assert report.is_mean_return != 0
        assert report.oos_mean_return != 0
        assert report.parameter_stability > 0
        assert 0 <= report.overfitting_probability <= 1.0
        assert len(report.window_results) == report.windows

        # 每个窗口都有 IS/OOS 记录
        for w in report.window_results:
            assert "is_return" in w and "oos_return" in w
            assert "is_sharpe" in w and "oos_sharpe" in w


# ═══════════════════════════════════════════
# M2 组合优化
# ═══════════════════════════════════════════

SIGNALS = {
    "000001.SZ": 80, "000002.SZ": 72, "000003.SZ": 65, "000004.SZ": 60,
    "000005.SZ": 55, "000006.SZ": 50, "000007.SZ": 45, "000008.SZ": 40,
}


class TestOptimizeReal:
    """M2: 真实组合优化器"""

    def setup_method(self):
        self.t3 = Trader3()

    def test_optimize_real(self):
        """risk_budget / mean_variance / black_litterman 三种方法均可用"""
        for method in ("risk_budget", "mean_variance", "black_litterman"):
            r = self.t3.optimize_portfolio(signals=SIGNALS, method=method)
            assert r.success, f"{method} failed: {r.summary}"
            assert "优化完成" in r.summary

            result = r.data
            assert result.target_weights, f"{method} 无目标权重"
            # 权重和为 1（归一化）
            total = sum(result.target_weights.values())
            assert total == pytest.approx(1.0, abs=0.02), f"{method} 权重和={total}"
            assert result.expected_risk > 0
            assert result.constraints_satisfied is True
            # 关键指标
            assert r.key_metrics["持仓数量"] >= 1
            assert r.key_metrics["换手成本(bp)"] >= 0

    def test_regime_allocate(self):
        """regime_aware_allocation 概率加权调整权重"""
        r = self.t3.regime_aware_allocation(
            signals={"000001.SZ": 80, "000002.SZ": 72, "000003.SZ": 65},
            regime_probs={"ranging": 0.6, "bearish": 0.3, "trending_up": 0.1},
            regime_weights={
                "ranging": {"000001.SZ": 0.4, "000002.SZ": 0.4, "000003.SZ": 0.2},
                "bearish": {"000001.SZ": 0.2, "000002.SZ": 0.3, "000003.SZ": 0.5},
                "trending_up": {"000001.SZ": 0.5, "000002.SZ": 0.3, "000003.SZ": 0.2},
            },
        )
        assert r.success, r.summary
        assert "情境路由" in r.summary
        assert "当前状态" in r.key_metrics
        assert r.key_metrics["主导概率"] > 0.5

        # 概率加权会改变权重：与等权不同，且建议仓位被风险概率折价
        result = r.data
        assert result.target_weights
        total = sum(result.target_weights.values())
        assert total == pytest.approx(1.0, abs=0.02)
        assert r.key_metrics["建议仓位"] <= 1.0

    def test_regime_allocate_errors(self):
        """缺失参数返回错误"""
        r = self.t3.regime_aware_allocation(signals={"A": 1})
        assert not r.success


# ═══════════════════════════════════════════
# M3 执行 / TCA
# ═══════════════════════════════════════════


class TestExecutionReal:
    """M3: 真实 TCA 与执行引擎"""

    def setup_method(self):
        self.t3 = Trader3()

    def test_tca_real(self):
        """成本模型产生合理 bp；卖出含印花税、买入不含"""
        buy = self.t3.estimate_transaction_cost(
            orders=[{"symbol": "301150.SZ", "side": "buy", "value_cny": 5000000}]
        )
        assert buy.success
        assert buy.data.total_cost_bp > 0
        assert buy.data.impact_bp > 0
        assert buy.data.total_cost_cny > 0
        # A股：买入不收印花税
        assert buy.data.stamp_tax_bp == 0.0
        assert buy.key_metrics["印花税(bp)"] == 0.0

        sell = self.t3.estimate_transaction_cost(
            orders=[{"symbol": "301150.SZ", "side": "sell", "value_cny": 5000000}]
        )
        assert sell.success
        # A股：卖出单边收取印花税 10bp
        assert sell.data.stamp_tax_bp > 0
        assert sell.data.stamp_tax_bp == pytest.approx(10.0, abs=0.1)
        # 卖出总成本高于买入（印花税差额）
        assert sell.data.total_cost_bp > buy.data.total_cost_bp

        # 建议与紧急度
        assert sell.data.recommended_urgency in ("low", "normal", "high")
        assert isinstance(sell.data.execution_suggestions, list)

    def test_execution_plan(self):
        """twap / vwap / is / adaptive 四种算法均产生切片"""
        target = {"000001.SZ": 0.3, "000002.SZ": 0.2, "000003.SZ": 0.1}
        for algo in ("twap", "vwap", "is", "adaptive_vwap", "implementation_shortfall"):
            r = self.t3.generate_execution_plan(
                target_weights=target, algorithm=algo, urgency="normal"
            )
            assert r.success, f"{algo} failed: {r.summary}"
            plan = r.data
            assert plan.slices, f"{algo} 无切片"
            assert len(plan.slices) > 0
            assert plan.expected_completion_rate > 0
            assert plan.expected_total_cost_bp > 0
            assert "max_price_deviation" in plan.risk_limits
            # 每片都有时间/标的/方向
            first = plan.slices[0]
            assert "time" in first and "symbol" in first and "side" in first
            assert first["side"] in ("buy", "sell")


# ═══════════════════════════════════════════
# M4 信号验证 / 市场状态
# ═══════════════════════════════════════════


class TestSignalValidation:
    """M4: 真实信号验证"""

    def setup_method(self):
        self.t3 = Trader3()

    def test_signal_validation(self):
        """ICIR 计算、分组收益单调性、半衰期与拥挤度"""
        r = self.t3.validate_signal(signal_name="动量因子")
        assert r.success, r.summary
        assert "ICIR" in r.summary

        report = r.data
        # IC 统计
        assert report.icir != 0
        assert len(report.ic_series) > 20
        assert report.ic_mean != 0
        # 分组
        assert set(report.group_returns) >= {"Q1 (多头)", "Q5 (空头)"}
        assert report.long_short_return != 0
        assert report.long_only_return != 0
        # 半衰期/拥挤度
        assert report.half_life_months > 0
        assert 0 <= report.crowding_index <= 1.0
        # 条件有效性
        assert "低波动" in report.conditional_validity

        # key_metrics 对齐
        assert "ICIR" in r.key_metrics

    def test_regime_hmm(self):
        """市场状态诊断：概率和为 1，输出结构完整"""
        r = self.t3.diagnose_market_regime()
        assert r.success, r.summary
        assert "当前状态" in r.summary

        diagnosis = r.data
        probs = diagnosis.regime_probabilities
        # 状态概率和为 1
        total = sum(probs.values())
        assert total == pytest.approx(1.0, abs=0.01), f"prob sum={total}"
        assert max(probs.values()) > 0.5
        # 完整状态空间
        assert set(probs) >= {"trending_up", "ranging", "bearish", "high_vol", "liquidity_crisis"}
        # 关键字段
        assert diagnosis.current_regime in probs
        assert diagnosis.regime_entropy >= 0
        assert 0 < diagnosis.suggested_position <= 1.0
        # 伪造指标已移除：市盈率/融资余额不得出现；真实可算指标存在
        assert "沪深300市盈率" not in diagnosis.key_indicators
        assert "融资余额(亿)" not in diagnosis.key_indicators
        assert "20日年化波动率(%)" in diagnosis.key_indicators
        assert any(c for c in r.caveats if "已移除" in c)
        assert diagnosis.strategy_suggestion


# ═══════════════════════════════════════════
# M5 估值 / 基本面
# ═══════════════════════════════════════════


class TestValuationReal:
    """M5: 估值锚与基本面评分卡"""

    def setup_method(self):
        self.t3 = Trader3()

    def test_valuation_real(self):
        """混合模式诚实性：无真实历史序列时方法被禁用，输出标记为不可引用演示值"""
        r = self.t3.valuation_anchor(codes=["301150.SZ"])
        assert r.success, r.summary
        assert "目标价" in r.summary

        report = r.data
        # 四种方法仍在 methods 里（结构完整）
        assert set(report.methods) >= {"dcf", "pe_percentile", "pb_roe", "ev_ebitda"}
        honesty_markers = ("已禁用", "禁止引用", "当前价为合成值", "目标价≤0")
        if report.implied_return == 0:
            # 隐含收益置0必须有诚实标注：方法禁用/演示值/合成价/负目标价之一
            assert any(any(m in c for m in honesty_markers) for c in r.caveats), (
                f"隐含收益为0但缺少诚实标注: {r.summary}"
            )
        else:
            assert report.weighted_target > 0
        # 敏感性网格
        assert "wacc" in report.sensitivity
        assert "terminal_growth" in report.sensitivity
        assert len(report.sensitivity["wacc"]) >= 3  # M5 真实引擎输出 5 点网格
        # key_metrics
        assert r.key_metrics["加权目标价"] >= 0
        # 隐含收益率在诚实化后可为 0（价格合成/演示值/负目标价），不再强制非零

    def test_scorecard(self):
        """六维评分 + 红旗预警"""
        r = self.t3.fundamental_scorecard(codes=["000001.SZ"], template="quality_growth")
        assert r.success, r.summary
        assert "总分" in r.summary

        report = r.data
        # 六维
        assert len(report.dimension_scores) == 6
        for dim in ("盈利能力", "成长性", "财务健康", "估值合理性", "管理层质量", "竞争壁垒"):
            assert dim in report.dimension_scores, f"missing dim: {dim}"
            assert 0 < report.dimension_scores[dim] <= 10
        # 红旗预警
        assert report.red_flags, "应检测到红旗预警"
        assert report.overall_score > 0
        # 同业对比
        assert "同行均值" in report.peer_comparison


# ═══════════════════════════════════════════
# M6 门禁 / API
# ═══════════════════════════════════════════


class TestGatesAndApi:
    """M6: IronGate 门禁 + FastAPI"""

    def test_gates_real(self):
        """门禁启用时 run_backtest 元数据含 gate_results"""
        t3 = Trader3(gates_enabled=True)
        assert t3.gates.enabled is True

        r = t3.run_backtest(start_date="2020-01-01", end_date="2025-12-31")
        assert r.success, r.summary
        # M6 接入后 metadata 必然含门禁键
        assert "gate_results" in r.metadata
        assert isinstance(r.metadata["gate_results"], list)
        assert "gates_passed" in r.metadata
        # 门禁摘要可查询
        summary = t3.gates.summary()
        assert summary["enabled"] is True

    def test_gates_default_on(self):
        """M6: 默认门禁启用 (完整 IronGate)"""
        t3 = Trader3()
        assert t3.gates.enabled is True

    def test_gates_off_explicit(self):
        """M6: 可显式关闭门禁 (M0 兼容)"""
        t3 = Trader3(gates_enabled=False)
        assert t3.gates.enabled is False

    def test_api_import(self):
        """FastAPI 应用可导入且有路由（M6 交付后生效，未交付则跳过）"""
        pytest.importorskip("trader3.api", reason="M6 API 模块未交付，跳过")
        pytest.importorskip("fastapi", reason="fastapi 未安装，跳过")

        from trader3.api import app  # noqa: F401

        assert app is not None
        routes = [r.path for r in app.routes]
        # 应包含核心 Tool 路由
        assert "/backtest" in routes or any("backtest" in p for p in routes)
