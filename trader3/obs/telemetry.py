"""
全局遥测桥（F3）—— 让既有链路（quant_pipeline/daily/OrderManager）把真实产出
喂进 Prometheus 注册表，/metrics 与本地 render 共享同一实例。

设计：模块级单例 registry（build_default_registry 的惰性实例），
各模块在自然事件点（下单/成交/风控/挂起/权益快照）调用 record_*，
不引入跨模块对象依赖，也无需改写既有调用签名。
"""

from __future__ import annotations

_registry = None


def registry():
    """默认 Prometheus 注册表单例（惰性）。"""
    global _registry
    if _registry is None:
        from trader3.obs.metrics import build_default_registry
        _registry = build_default_registry()
    return _registry


def record_order_allowed() -> None:
    registry().counter("risk_allowed_total").inc()


def record_order_denied(reason: str) -> None:
    registry().counter("risk_denied_total").inc(reason=str(reason))


def record_fill_latency_ms(lag_ms: float) -> None:
    registry().histogram("fill_latency_ms").observe(float(lag_ms))


def record_equity(equity: float) -> None:
    registry().gauge("account_equity").set(float(equity))


def record_cash(cash: float) -> None:
    registry().gauge("account_cash").set(float(cash))


def record_positions(n_open: int) -> None:
    registry().gauge("positions_open").set(float(n_open))


def record_kill_switch(tripped: bool) -> None:
    registry().gauge("kill_switch_tripped").set(1.0 if tripped else 0.0)


def record_drift_halt(halted: bool) -> None:
    registry().gauge("drift_halt_active").set(1.0 if halted else 0.0)


def render() -> str:
    """抓取当前全部指标（API /metrics 端点直接调用）。"""
    return registry().render()
