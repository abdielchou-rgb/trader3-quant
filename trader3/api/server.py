"""
3号交易员 — FastAPI 服务 (M6)

暴露全部 10 个 Tool 为 POST 端点，统一返回 Trader3Response.to_dict()，
并在 metadata 中包含门禁摘要 (gate_results / gates_passed / gates_summary)。

安全:
  - 默认仅绑定 127.0.0.1；需要局域网访问时必须设置 TRADER3_API_KEY
  - 设置 TRADER3_API_KEY 后所有 Tool 端点要求请求头 X-API-Key 匹配
  - payload 经 pydantic 白名单校验（extra=forbid，未知字段/越界数值 → 422），
    仅声明的字段以 model_dump(exclude_unset=True) 透传给 Tool（未传即用工具默认），
    不执行任何动态求值

端点:
  POST /backtest           → run_backtest
  POST /wfa                → walk_forward_analysis
  POST /optimize           → optimize_portfolio
  POST /regime_allocate    → regime_aware_allocation
  POST /estimate_cost      → estimate_transaction_cost
  POST /execution_plan     → generate_execution_plan
  POST /validate_signal    → validate_signal
  POST /regime             → diagnose_market_regime
  POST /valuation          → valuation_anchor
  POST /scorecard          → fundamental_scorecard
  GET  /health             健康检查
  GET  /tools              工具清单

启动:
  uvicorn trader3.api.server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field

from trader3 import Trader3

# 共享单例引擎（门禁默认启用）
_t3: Trader3 = None

# API Key：环境变量 TRADER3_API_KEY 设置后强制鉴权
_API_KEY: str | None = os.environ.get("TRADER3_API_KEY") or None


def get_trader3() -> Trader3:
    """懒加载共享 Trader3 实例"""
    global _t3
    if _t3 is None:
        _t3 = Trader3()
    return _t3


def _require_api_key(x_api_key: str | None) -> None:
    """配置了 TRADER3_API_KEY 时校验请求头 X-API-Key。"""
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


app = FastAPI(
    title="3号交易员 — IronGate API",
    description="3号交易员量化交易引擎（10 个 Tool + 完整 IronGate 门禁）",
    version="6.2.0",
)


# ── 可观测性：Prometheus 指标端点（E5，无鉴权——生产部署由内网隔离保护）──
_METRICS = None


def _get_metrics_registry():
    global _METRICS
    if _METRICS is None:
        from trader3.obs.metrics import build_default_registry
        _METRICS = build_default_registry()
    return _METRICS


@app.get("/metrics")
def metrics_endpoint() -> Any:
    """Prometheus 抓取端点（text/plain; version=0.0.4）。"""
    from fastapi.responses import PlainTextResponse

    reg = _get_metrics_registry()
    # 喂运行时快照（幂等：不存在则跳过）
    try:
        t3 = get_trader3()
        hist = t3.get_call_history()
        for h in hist:
            if h.get("tool"):
                reg.counter("tool_calls_total").inc(tool=str(h["tool"]))
    except Exception:  # noqa: BLE001 — 指标路径绝不影响主服务
        pass
    return PlainTextResponse(
        reg.render(), media_type="text/plain; version=0.0.4; charset=utf-8"
    )


# ═══════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════

def _run_tool(method_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """调用 Tool 并返回可 JSON 序列化的 Trader3Response.to_dict()"""
    t3 = get_trader3()
    method = getattr(t3, method_name)
    result = method(**(payload or {}))
    # jsonable_encoder 处理 dataclass / numpy 标量 → JSON 安全
    return jsonable_encoder(result.to_dict())


# ═══════════════════════════════════════════
# 请求白名单模型（字段与各 Tool execute 签名严格对齐）
# ═══════════════════════════════════════════


class BacktestPayload(BaseModel):
    """POST /backtest ← RunBacktestTool.execute"""

    model_config = ConfigDict(extra="forbid")

    strategy_config: dict[str, Any] | None = None
    universe: list[str] | None = None
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    constraints: dict[str, Any] | None = None
    benchmark: str = "000300.SH"
    commission: dict[str, Any] | None = None
    signal_expr: str = ""
    factor_from_selected: int = Field(default=0, ge=0)


class WfaPayload(BaseModel):
    """POST /wfa ← WalkForwardAnalysisTool.execute"""

    model_config = ConfigDict(extra="forbid")

    strategy_config: dict[str, Any] | None = None
    train_window: int = Field(default=252, gt=10)
    test_window: int = Field(default=63, gt=10)
    step: int | None = None


class OptimizePayload(BaseModel):
    """POST /optimize ← OptimizePortfolioTool.execute"""

    model_config = ConfigDict(extra="forbid")

    signals: dict[str, float]
    method: str = "risk_budget"
    constraints: dict[str, Any] | None = None
    risk_model: dict[str, Any] | None = None


class RegimeAllocatePayload(BaseModel):
    """POST /regime_allocate ← RegimeAwareAllocationTool.execute"""

    model_config = ConfigDict(extra="forbid")

    signals: dict[str, float]
    regime_probs: dict[str, float] | None = None
    regime_weights: dict[str, dict[str, float]] | None = None
    constraints: dict[str, Any] | None = None


class EstimateCostPayload(BaseModel):
    """POST /estimate_cost ← EstimateTransactionCostTool.execute"""

    model_config = ConfigDict(extra="forbid")

    orders: list[dict[str, Any]] | None = None
    method: str = "implementation_shortfall"
    market_data: dict[str, Any] | None = None


class ExecutionPlanPayload(BaseModel):
    """POST /execution_plan ← GenerateExecutionPlanTool.execute"""

    model_config = ConfigDict(extra="forbid")

    target_weights: dict[str, float] | None = None
    current_weights: dict[str, float] | None = None
    algorithm: str = "adaptive_vwap"
    urgency: str = "normal"
    portfolio_value: float = Field(default=10_000_000.0, gt=0)
    market_data: dict[str, Any] | None = None


class ValidateSignalPayload(BaseModel):
    """POST /validate_signal ← ValidateSignalTool.execute"""

    model_config = ConfigDict(extra="forbid")

    signal_name: str = ""
    signal_values: list[float] | None = None
    forward_returns: dict[int, list[float]] | None = None
    horizons: list[int] | None = None


class RegimePayload(BaseModel):
    """POST /regime ← DiagnoseMarketRegimeTool.execute（允许空请求体）"""

    model_config = ConfigDict(extra="forbid")

    lookback: int = 60
    prices: list[float] | None = None
    volumes: list[float] | None = None


class ValuationPayload(BaseModel):
    """POST /valuation ← ValuationAnchorTool.execute"""

    model_config = ConfigDict(extra="forbid")

    codes: list[str] | None = None
    methods: list[str] | None = None
    scenarios: dict[str, Any] | None = None
    financials: dict[str, Any] | None = None
    peers: list[dict[str, Any]] | None = None
    private_company: dict[str, Any] | None = None
    current_price: float | None = None
    asof_date: str | None = None


class ScorecardPayload(BaseModel):
    """POST /scorecard ← FundamentalScorecardTool.execute"""

    model_config = ConfigDict(extra="forbid")

    codes: list[str] | None = None
    template: str = "quality_growth"
    financials: dict[str, Any] | None = None
    peers: list[dict[str, Any]] | None = None
    management_score: float | None = None
    moat_score: float | None = None


# ═══════════════════════════════════════════
# 10 个 Tool 端点
# ═══════════════════════════════════════════


@app.post("/backtest")
def api_backtest(payload: BacktestPayload, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """运行策略回测"""
    _require_api_key(x_api_key)
    return _run_tool("run_backtest", payload.model_dump(exclude_unset=True))


@app.post("/wfa")
def api_wfa(payload: WfaPayload, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """滚动 Walk-Forward Analysis"""
    _require_api_key(x_api_key)
    return _run_tool("walk_forward_analysis", payload.model_dump(exclude_unset=True))


@app.post("/optimize")
def api_optimize(payload: OptimizePayload, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """组合优化"""
    _require_api_key(x_api_key)
    return _run_tool("optimize_portfolio", payload.model_dump(exclude_unset=True))


@app.post("/regime_allocate")
def api_regime_allocate(payload: RegimeAllocatePayload, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """市场状态路由分配"""
    _require_api_key(x_api_key)
    return _run_tool("regime_aware_allocation", payload.model_dump(exclude_unset=True))


@app.post("/estimate_cost")
def api_estimate_cost(payload: EstimateCostPayload, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """交易成本估算"""
    _require_api_key(x_api_key)
    return _run_tool("estimate_transaction_cost", payload.model_dump(exclude_unset=True))


@app.post("/execution_plan")
def api_execution_plan(payload: ExecutionPlanPayload, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """生成执行计划"""
    _require_api_key(x_api_key)
    return _run_tool("generate_execution_plan", payload.model_dump(exclude_unset=True))


@app.post("/validate_signal")
def api_validate_signal(payload: ValidateSignalPayload, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """信号有效性验证"""
    _require_api_key(x_api_key)
    return _run_tool("validate_signal", payload.model_dump(exclude_unset=True))


@app.post("/regime")
def api_regime(
    payload: RegimePayload = Body(default=RegimePayload()),
    x_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    """市场状态诊断（允许空请求体，等价 {}）"""
    _require_api_key(x_api_key)
    return _run_tool("diagnose_market_regime", payload.model_dump(exclude_unset=True))


@app.post("/valuation")
def api_valuation(payload: ValuationPayload, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """估值锚定"""
    _require_api_key(x_api_key)
    return _run_tool("valuation_anchor", payload.model_dump(exclude_unset=True))


@app.post("/scorecard")
def api_scorecard(payload: ScorecardPayload, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """基本面评分卡"""
    _require_api_key(x_api_key)
    return _run_tool("fundamental_scorecard", payload.model_dump(exclude_unset=True))


# ═══════════════════════════════════════════
# 元信息端点
# ═══════════════════════════════════════════


@app.get("/health")
def api_health() -> dict[str, Any]:
    """健康检查"""
    t3 = get_trader3()
    return {
        "status": "ok",
        "app": app.title,
        "version": "6.1.0",
        "tools_count": len(t3.list_tools()),
        "gates_enabled": t3.gates.enabled,
        "auth_required": bool(_API_KEY),
    }


@app.get("/tools")
def api_tools(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """工具清单"""
    _require_api_key(x_api_key)
    t3 = get_trader3()
    return {
        "tools": t3.list_tools(),
        "count": len(t3.list_tools()),
        "gates": t3.gates.summary(),
    }


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("TRADER3_API_HOST", "127.0.0.1")
    if host != "127.0.0.1" and not _API_KEY:
        raise SystemExit(
            "拒绝启动: 绑定非回环地址必须先设置 TRADER3_API_KEY（防止未鉴权暴露）"
        )
    uvicorn.run(app, host=host, port=int(os.environ.get("TRADER3_API_PORT", "8000")))
