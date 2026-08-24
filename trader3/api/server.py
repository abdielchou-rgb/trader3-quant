"""
3号交易员 — FastAPI 服务 (M6)

暴露全部 10 个 Tool 为 POST 端点，统一返回 Trader3Response.to_dict()，
并在 metadata 中包含门禁摘要 (gate_results / gates_passed / gates_summary)。

安全:
  - 默认仅绑定 127.0.0.1；需要局域网访问时必须设置 TRADER3_API_KEY
  - 设置 TRADER3_API_KEY 后所有 Tool 端点要求请求头 X-API-Key 匹配
  - payload 键透传给 Tool（未知键由 BaseTool 统一报错），不执行任何动态求值

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
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.encoders import jsonable_encoder

from trader3 import Trader3

# 共享单例引擎（门禁默认启用）
_t3: Trader3 = None

# API Key：环境变量 TRADER3_API_KEY 设置后强制鉴权
_API_KEY: Optional[str] = os.environ.get("TRADER3_API_KEY") or None


def get_trader3() -> Trader3:
    """懒加载共享 Trader3 实例"""
    global _t3
    if _t3 is None:
        _t3 = Trader3()
    return _t3


def _require_api_key(x_api_key: Optional[str]) -> None:
    """配置了 TRADER3_API_KEY 时校验请求头 X-API-Key。"""
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


app = FastAPI(
    title="3号交易员 — IronGate API",
    description="3号交易员量化交易引擎（10 个 Tool + 完整 IronGate 门禁）",
    version="6.1.0",
)


# ═══════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════

def _run_tool(method_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """调用 Tool 并返回可 JSON 序列化的 Trader3Response.to_dict()"""
    t3 = get_trader3()
    method = getattr(t3, method_name)
    result = method(**(payload or {}))
    # jsonable_encoder 处理 dataclass / numpy 标量 → JSON 安全
    return jsonable_encoder(result.to_dict())


# ═══════════════════════════════════════════
# 10 个 Tool 端点
# ═══════════════════════════════════════════


@app.post("/backtest")
def api_backtest(payload: Dict[str, Any] = Body(default={}), x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """运行策略回测"""
    _require_api_key(x_api_key)
    return _run_tool("run_backtest", payload)


@app.post("/wfa")
def api_wfa(payload: Dict[str, Any] = Body(default={}), x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """滚动 Walk-Forward Analysis"""
    _require_api_key(x_api_key)
    return _run_tool("walk_forward_analysis", payload)


@app.post("/optimize")
def api_optimize(payload: Dict[str, Any] = Body(default={}), x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """组合优化"""
    _require_api_key(x_api_key)
    return _run_tool("optimize_portfolio", payload)


@app.post("/regime_allocate")
def api_regime_allocate(payload: Dict[str, Any] = Body(default={}), x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """市场状态路由分配"""
    _require_api_key(x_api_key)
    return _run_tool("regime_aware_allocation", payload)


@app.post("/estimate_cost")
def api_estimate_cost(payload: Dict[str, Any] = Body(default={}), x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """交易成本估算"""
    _require_api_key(x_api_key)
    return _run_tool("estimate_transaction_cost", payload)


@app.post("/execution_plan")
def api_execution_plan(payload: Dict[str, Any] = Body(default={}), x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """生成执行计划"""
    _require_api_key(x_api_key)
    return _run_tool("generate_execution_plan", payload)


@app.post("/validate_signal")
def api_validate_signal(payload: Dict[str, Any] = Body(default={}), x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """信号有效性验证"""
    _require_api_key(x_api_key)
    return _run_tool("validate_signal", payload)


@app.post("/regime")
def api_regime(payload: Dict[str, Any] = Body(default={}), x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """市场状态诊断"""
    _require_api_key(x_api_key)
    return _run_tool("diagnose_market_regime", payload)


@app.post("/valuation")
def api_valuation(payload: Dict[str, Any] = Body(default={}), x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """估值锚定"""
    _require_api_key(x_api_key)
    return _run_tool("valuation_anchor", payload)


@app.post("/scorecard")
def api_scorecard(payload: Dict[str, Any] = Body(default={}), x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """基本面评分卡"""
    _require_api_key(x_api_key)
    return _run_tool("fundamental_scorecard", payload)


# ═══════════════════════════════════════════
# 元信息端点
# ═══════════════════════════════════════════


@app.get("/health")
def api_health() -> Dict[str, Any]:
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
def api_tools(x_api_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
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
