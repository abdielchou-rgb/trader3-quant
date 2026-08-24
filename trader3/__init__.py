"""
3号交易员 — M0 骨架

包结构:
trader3/
├── __init__.py         # 包入口, 暴露 Trader3, Trader3Response, Trader3Gates
├── base_tool.py        # Tool 抽象基类 + Trader3Response 统一返回
├── models.py           # 数据模型定义
├── config.py           # 配置管理 (YAML 加载/合并/验证)
├── registry.py         # Tool 注册表
├── shared_state.py     # 共享状态管理
├── gates.py            # IronGate 门禁存根
├── tools/              # 具体 Tool 实现
│   ├── __init__.py
│   ├── backtest.py     # run_backtest, walk_forward_analysis
│   ├── optimize.py     # optimize_portfolio, regime_aware_allocation
│   ├── execution.py    # estimate_transaction_cost, generate_execution_plan
│   ├── signal.py       # validate_signal, diagnose_market_regime
│   └── valuation.py    # valuation_anchor, fundamental_scorecard
├── cli.py              # CLI 入口（命令行调用）
└── api/                # FastAPI 服务（M2+）

使用方式:
  # 作为包导入（2号分析师嵌入模式）
  from trader3 import Trader3
  t3 = Trader3()
  result = t3.run_backtest(start_date="2020-01-01", end_date="2025-12-31")
  print(result.summary)

  # 作为 CLI（独立运行模式）
  python -m trader3.cli backtest --start 2020-01-01 --end 2025-12-31
  python -m trader3.cli regime
  python -m trader3.cli list
"""

from __future__ import annotations

from trader3.base_tool import BaseTool, ChartSpec, Trader3Response
from trader3.gates import GateResult, Trader3Gates
from trader3.models import (
    BacktestReport,
    ExecutionPlan,
    OptimizationResult,
    PortfolioConstraints,
    RegimeDiagnosis,
    ScorecardReport,
    SignalValidationReport,
    StrategyConfig,
    TCAEstimate,
    ValuationReport,
    WFAReport,
)
from trader3.registry import ToolRegistry
from trader3.shared_state import SharedState

# 公开 API（嵌入模式：from trader3 import Trader3, ...）
__all__ = [
    "BacktestReport",
    "BaseTool",
    "ChartSpec",
    "ExecutionPlan",
    "GateResult",
    "OptimizationResult",
    "PortfolioConstraints",
    "RegimeDiagnosis",
    "ScorecardReport",
    "SharedState",
    "SignalValidationReport",
    "StrategyConfig",
    "TCAEstimate",
    "Trader3",
    "Trader3Gates",
    "Trader3Response",
    "ToolRegistry",
    "ValuationReport",
    "WFAReport",
]


class Trader3:
    """
    3号交易员 — 主入口。

    双模式运行：
    1. 嵌入模式：2号分析师直接导入并调用方法
    2. 独立模式：CLI / FastAPI

    所有方法返回统一的 Trader3Response，包含 summary / key_metrics / charts / caveats / metadata。
    """

    def __init__(self, gates_enabled: bool = True, config_dir: str = ""):
        # 注册所有 Tool
        self._registry = ToolRegistry()
        self._register_tools()

        # 共享状态
        self.state = SharedState()

        # 门禁（M6 默认启用完整 IronGate 检查，gates_enabled=False 回退 M0 存根）
        self.gates = Trader3Gates(enabled=gates_enabled, state=self.state)

    def _register_tools(self):
        """注册所有内置 Tool"""
        from trader3.tools.backtest import RunBacktestTool, WalkForwardAnalysisTool
        from trader3.tools.execution import EstimateTransactionCostTool, GenerateExecutionPlanTool
        from trader3.tools.optimize import OptimizePortfolioTool, RegimeAwareAllocationTool
        from trader3.tools.signal import DiagnoseMarketRegimeTool, ValidateSignalTool
        from trader3.tools.valuation import FundamentalScorecardTool, ValuationAnchorTool

        for tool_cls in [
            RunBacktestTool,
            WalkForwardAnalysisTool,
            OptimizePortfolioTool,
            RegimeAwareAllocationTool,
            EstimateTransactionCostTool,
            GenerateExecutionPlanTool,
            ValidateSignalTool,
            DiagnoseMarketRegimeTool,
            ValuationAnchorTool,
            FundamentalScorecardTool,
        ]:
            self._registry.register(tool_cls())

    # ── Tool 调用方法（直接映射） ──

    def run_backtest(self, **kwargs) -> Trader3Response:
        return self._call_tool("run_backtest", **kwargs)

    def walk_forward_analysis(self, **kwargs) -> Trader3Response:
        return self._call_tool("walk_forward_analysis", **kwargs)

    def optimize_portfolio(self, **kwargs) -> Trader3Response:
        return self._call_tool("optimize_portfolio", **kwargs)

    def regime_aware_allocation(self, **kwargs) -> Trader3Response:
        return self._call_tool("regime_aware_allocation", **kwargs)

    def estimate_transaction_cost(self, **kwargs) -> Trader3Response:
        return self._call_tool("estimate_transaction_cost", **kwargs)

    def generate_execution_plan(self, **kwargs) -> Trader3Response:
        return self._call_tool("generate_execution_plan", **kwargs)

    def validate_signal(self, **kwargs) -> Trader3Response:
        return self._call_tool("validate_signal", **kwargs)

    def diagnose_market_regime(self, **kwargs) -> Trader3Response:
        return self._call_tool("diagnose_market_regime", **kwargs)

    def push_regime_to_state(self, diagnosis=None) -> str:
        """
        将市场状态诊断写入共享状态 shared_state/regime_current.json。

        优先使用传入的 RegimeDiagnosis；否则重新运行 diagnose_market_regime 读取。
        返回写入的 JSON 路径。
        """
        if diagnosis is None:
            response = self.diagnose_market_regime()
            if not response.success:
                raise RuntimeError(f"无法诊断市场状态: {response.summary}")
            diagnosis = response.data

        regime_data = {
            "current_regime": getattr(diagnosis, "current_regime", ""),
            "regime_probabilities": getattr(diagnosis, "regime_probabilities", {}),
            "regime_entropy": getattr(diagnosis, "regime_entropy", 0.0),
            "key_indicators": getattr(diagnosis, "key_indicators", {}),
            "historical_analog": getattr(diagnosis, "historical_analog", ""),
            "strategy_suggestion": getattr(diagnosis, "strategy_suggestion", ""),
            "suggested_position": getattr(diagnosis, "suggested_position", 1.0),
        }
        return self.state.set_regime(regime_data)

    def valuation_anchor(self, **kwargs) -> Trader3Response:
        return self._call_tool("valuation_anchor", **kwargs)

    def fundamental_scorecard(self, **kwargs) -> Trader3Response:
        return self._call_tool("fundamental_scorecard", **kwargs)

    # ── 工具方法 ──

    def _call_tool(self, name: str, **kwargs) -> Trader3Response:
        """调用 Tool 并运行门禁检查"""
        result = self._registry.call(name, **kwargs)
        if not result.success:
            return result
        # 运行门禁（如果启用）。tool_category 由 Tool 定义；WFA 归入 backtest。
        tool_category = result.metadata.get("tool_category", "")
        if not tool_category:
            tool = self._registry.get(name)
            tool_category = tool.tool_category if tool else ""
        result.metadata["tool_category"] = tool_category

        gate_results = self.gates.run_all(type=tool_category, data=result.data)
        result.metadata["gate_results"] = [
            {"check_name": g.check_name, "passed": g.passed, "score": g.score,
             "message": g.message}
            for g in gate_results
        ]
        result.metadata["gates_passed"] = all(g.passed for g in gate_results)
        result.metadata["gates_summary"] = self.gates.summary()
        return result

    def list_tools(self) -> list:
        return self._registry.summary()

    def get_call_history(self) -> list:
        """获取调用历史（给 2号分析师的审计用）"""
        history = []
        for name, tool in self._registry._tools.items():
            for entry in tool.get_call_history():
                history.append({"tool": name, **entry})
        return sorted(history, key=lambda x: x.get("elapsed", 0), reverse=True)[:50]

    # ═══════════════════════════════════════════════════════
    # v2.0 扩展接口（自选股状态机 / 三因子触发 / 可比分析 / 每日管线）
    # ═══════════════════════════════════════════════════════

    # ── 自选股状态机 ──

    def watchlist_add(self, code: str, name: str = "", note: str = "") -> bool:
        """加入自选股跟踪（状态=观察中）"""
        from trader3.v2.watchlist import get_watchlist
        wl = get_watchlist()
        ok = wl.add(code, name, note)
        wl.close()
        return ok

    def watchlist_remove(self, code: str, reason: str = "用户移除") -> bool:
        from trader3.v2.watchlist import get_watchlist
        wl = get_watchlist()
        ok = wl.remove(code, reason)
        wl.close()
        return ok

    def watchlist_list(self, priority: bool = True) -> list:
        """列出自选股（默认按状态优先级：卖出触发/买入区间/预警在前）"""
        from trader3.v2.watchlist import get_watchlist
        wl = get_watchlist()
        items = wl.list_by_priority() if priority else wl.list()
        wl.close()
        return [dict(
            code=i.code, name=i.name, status=i.status, add_date=i.add_date,
            last_change=i.last_change, note=i.note, trigger_score=i.trigger_score,
            trigger_reason=i.trigger_reason, valuation_anchor=i.valuation_anchor,
            current_price=i.current_price,
        ) for i in items]

    def watchlist_transition(self, code: str, to_status: str, reason: str = "") -> bool:
        """手动迁移状态（白名单校验）"""
        from trader3.v2.watchlist import get_watchlist
        wl = get_watchlist()
        ok = wl.transition(code, to_status, reason)
        wl.close()
        return ok

    def watchlist_events(self, code: str | None = None, limit: int = 20) -> list:
        """状态迁移事件（审计用）"""
        from trader3.v2.watchlist import get_watchlist
        wl = get_watchlist()
        events = wl.events(code, limit)
        wl.close()
        return [dict(code=e.code, from_status=e.from_status, to_status=e.to_status,
                     reason=e.reason, event_time=e.event_time) for e in events]

    # ── 三因子触发 ──

    def trigger_scan(self, catalyst_scores: dict | None = None) -> list:
        """扫描自选股三因子（催化×预期差×技术），催化自动从事件库读取"""
        from trader3.v2.trigger import get_trigger_engine
        from trader3.v2.watchlist import get_watchlist
        engine = get_trigger_engine(self)
        wl = get_watchlist()
        items = wl.list()
        results = engine.scan(items, catalyst_scores)
        # 自动状态迁移（触发 → 买入区间）
        for r in results:
            if r.triggered:
                item = wl.get(r.code)
                if item:
                    from trader3.v2.watchlist import ATTENTION, BUY_ZONE
                    if item.status in (ATTENTION, "观察中"):
                        wl.transition(r.code, BUY_ZONE, "三因子触发", r.to_dict())
        wl.close()
        return [r.to_dict() for r in results]

    # ── 可比公司分析 ──

    def comps_analysis(self, code: str, industry: str = "",
                       peer_codes: list | None = None, n_peers: int = 8) -> dict:
        """可比公司分析：建可比池 → 倍数 + 四分位 → 异常标红 → 溢价/折价结论"""
        from trader3.v2.comps import get_comps_analyzer
        analyzer = get_comps_analyzer()
        table = analyzer.analyze(code, industry, peer_codes, n_peers)
        return table.to_dict()

    # ── 每日管线 ──

    def daily_run(self, codes: list | None = None, collect_only: bool = False) -> dict:
        """每日扫描：采集事件 → 三因子触发 → 状态迁移（对齐 daily_pipeline）"""
        from trader3.v2.daily_pipeline import run_daily
        return run_daily(codes=codes, collect_only=collect_only)

    def daily_alert_text(self, summary: dict | None = None) -> str:
        """把每日扫描结果格式化为提醒文本（写作规范三道关）"""
        from trader3.v2.daily_pipeline import format_alerts
        if summary is None:
            summary = self.daily_run()
        return format_alerts(summary)
