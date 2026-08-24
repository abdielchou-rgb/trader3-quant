"""3号交易员 — M0 骨架验证测试"""

from trader3 import Trader3, Trader3Response
from trader3.models import (
    BacktestReport, OptimizationResult, TCAEstimate,
    SignalValidationReport, RegimeDiagnosis, ValuationReport,
    ScorecardReport, WFAReport, PortfolioConstraints, StrategyConfig,
)


class TestM0Skeleton:
    """验证 M0 骨架的完整性和接口契约"""

    def setup_method(self):
        self.t3 = Trader3()

    def test_trader3_init(self):
        """验证初始化"""
        assert self.t3 is not None
        tools = self.t3.list_tools()
        assert len(tools) == 10  # 10 个 Tool

    def test_list_tools(self):
        """验证所有 Tool 注册正确"""
        tools = {t["name"] for t in self.t3.list_tools()}
        expected = {
            "run_backtest", "walk_forward_analysis",
            "optimize_portfolio", "regime_aware_allocation",
            "estimate_transaction_cost", "generate_execution_plan",
            "validate_signal", "diagnose_market_regime",
            "valuation_anchor", "fundamental_scorecard",
        }
        missing = expected - tools
        extra = tools - expected
        assert not missing, f"缺少 Tool: {missing}"
        print(f"  ✅ 10 个 Tool 全部注册")

    def test_run_backtest(self):
        """验证回测接口"""
        result = self.t3.run_backtest(start_date="2020-01-01", end_date="2025-12-31")
        assert result.success
        assert result.summary
        assert result.key_metrics.get("夏普比", 0) > 0
        assert result.key_metrics.get("年化收益", 0) > 0
        assert result.key_metrics.get("最大回撤", 0) < 0
        assert result.metadata.get("tool") == "run_backtest"
        print(f"  ✅ run_backtest: {result.summary}")

    def test_walk_forward_analysis(self):
        """验证 WFA 接口"""
        result = self.t3.walk_forward_analysis()
        assert result.success
        assert "过拟合概率" in result.summary
        print(f"  ✅ walk_forward_analysis: {result.summary}")

    def test_optimize_portfolio(self):
        """验证组合优化接口"""
        result = self.t3.optimize_portfolio(signals={"000001.SZ": 80})
        assert result.success
        assert "夏普" in result.summary
        print(f"  ✅ optimize_portfolio: {result.summary}")

    def test_estimate_transaction_cost(self):
        """验证交易成本接口"""
        result = self.t3.estimate_transaction_cost(
            orders=[{"symbol": "000001.SZ", "side": "buy", "value_cny": 5000000}]
        )
        assert result.success
        assert "总成本" in result.summary
        assert result.key_metrics.get("总成本(bp)", 0) > 0
        print(f"  ✅ estimate_transaction_cost: {result.summary}")

    def test_validate_signal(self):
        """验证信号验证接口"""
        result = self.t3.validate_signal(signal_name="动量因子")
        assert result.success
        assert "ICIR" in result.summary
        assert result.key_metrics.get("ICIR", 0) > 0
        print(f"  ✅ validate_signal: {result.summary}")

    def test_diagnose_market_regime(self):
        """验证市场状态诊断接口"""
        result = self.t3.diagnose_market_regime()
        assert result.success
        assert "当前状态" in result.summary
        assert result.key_metrics.get("建议仓位", 0) > 0
        print(f"  ✅ diagnose_market_regime: {result.summary}")

    def test_valuation_anchor(self):
        """验证估值锚接口"""
        result = self.t3.valuation_anchor(codes=["301150.SZ"])
        assert result.success
        assert "目标价" in result.summary
        assert result.key_metrics.get("加权目标价", 0) > 0
        print(f"  ✅ valuation_anchor: {result.summary}")

    def test_fundamental_scorecard(self):
        """验证基本面评分卡接口"""
        result = self.t3.fundamental_scorecard(codes=["000001.SZ"])
        assert result.success
        assert "总分" in result.summary
        assert result.key_metrics.get("总分", 0) > 0
        print(f"  ✅ fundamental_scorecard: {result.summary}")

    def test_trader3_response_uniformity(self):
        """验证所有 Tool 返回格式一致"""
        methods = [
            (self.t3.run_backtest, {}),
            (self.t3.walk_forward_analysis, {}),
            (self.t3.optimize_portfolio, {"signals": {"A": 80}}),
            (self.t3.estimate_transaction_cost, {"orders": [{"symbol": "A", "side": "buy", "value_cny": 100000}]}),
            (self.t3.validate_signal, {"signal_name": "test"}),
            (self.t3.diagnose_market_regime, {}),
            (self.t3.valuation_anchor, {"codes": ["000001.SZ"]}),
            (self.t3.fundamental_scorecard, {"codes": ["000001.SZ"]}),
            (self.t3.regime_aware_allocation, {}),
            (self.t3.generate_execution_plan, {}),
        ]

        for method, kwargs in methods:
            result = method(**kwargs)
            assert isinstance(result, Trader3Response)
            assert hasattr(result, "success")
            assert hasattr(result, "summary")
            assert hasattr(result, "key_metrics")
            assert hasattr(result, "charts")
            assert hasattr(result, "caveats")
            assert hasattr(result, "metadata")
            assert result.metadata.get("tool"), f"metadata missing tool: {method.__name__}"
            assert result.metadata.get("request_id"), f"metadata missing request_id: {method.__name__}"
            assert result.metadata.get("elapsed_seconds", 0) > 0, f"elapsed=0: {method.__name__}"

        print(f"  ✅ 全部 10 个 Tool 返回格式一致 (含 request_id/elapsed/tool)")

    def test_gates_default_on(self):
        """验证门禁默认启用 (M6: 完整 IronGate)"""
        assert self.t3.gates.enabled
        summary = self.t3.gates.summary()
        assert summary["enabled"] is True
        print(f"  ✅ 门禁默认启用 (M6 完整 IronGate 模式)")

    def test_gates_can_disable(self):
        """验证门禁可显式关闭 (M0 兼容)"""
        from trader3 import Trader3
        t3_disabled = Trader3(gates_enabled=False)
        assert not t3_disabled.gates.enabled
        summary = t3_disabled.gates.summary()
        assert summary["enabled"] is False
        print(f"  ✅ 门禁可显式关闭 (M0 存根模式)")

    def test_shared_state(self):
        """验证共享状态可用"""
        from trader3.shared_state import SharedState
        state = SharedState()

        # 写/读版本号
        state.set_data_version({"financials": "v20260809", "market": "v20260810"})
        version = state.get_data_version()
        assert version["versions"]["financials"] == "v20260809"
        print(f"  ✅ shared_state: 读写正常")

    def test_config_manager(self):
        """验证配置管理"""
        from trader3.config import ConfigManager
        cm = ConfigManager()
        config = cm.load("3号交易员_core")
        assert config is not None
        assert "factors" in config
        assert len(config["factors"]) > 0
        print(f"  ✅ config: 配置文件加载正常, {len(config['factors'])} 个因子定义")


if __name__ == "__main__":
    t = TestM0Skeleton()
    t.setup_method()

    print("\n" + "=" * 60)
    print("📊 3号交易员 M0 骨架验证")
    print("=" * 60)

    for name in dir(t):
        if name.startswith("test_"):
            try:
                getattr(t, name)()
            except Exception as e:
                print(f"  ❌ {name}: {e}")

    print("\n✅ M0 骨架验证完成")