"""
E5：Prometheus 指标暴露（文本格式，零新依赖）— 回归测试
契约：
  1. Counter/Gauge 输出标准 exposition 文本格式（# HELP/# TYPE/名称 值）
  2. 指标集：风控拒绝计数（按 reason）、放行计数、成交延迟直方图（简化为
     sum/count/桶）、实时权益 gauge、事件总线排队延迟
  3. render() 幂等可拼接；多实例注册表聚合
  4. FastAPI /metrics 集成：内容类型 text/plain; version=0.0.4
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.obs.metrics import (  # noqa: E402
    MetricsRegistry,
)


def test_counter_render():
    reg = MetricsRegistry()
    c = reg.counter("risk_denied_total", "风控拒绝总数")
    c.inc(reason="PANIC")
    c.inc(reason="PANIC")
    c.inc(reason="OTR_TRIPPED")
    out = reg.render()
    assert "# TYPE risk_denied_total counter" in out
    assert 'risk_denied_total{reason="PANIC"} 2' in out
    assert 'risk_denied_total{reason="OTR_TRIPPED"} 1' in out


def test_gauge_render():
    reg = MetricsRegistry()
    g = reg.gauge("account_equity", "实时权益（元）")
    g.set(1_234_567.89)
    out = reg.render()
    assert "# TYPE account_equity gauge" in out
    assert "account_equity 1234567.89" in out


def test_histogram_buckets_and_quantiles():
    reg = MetricsRegistry()
    h = reg.histogram("fill_latency_ms", "成交回报延迟（毫秒）",
                      buckets=(1, 5, 10, 50, 100, 500))
    for v in (2, 3, 7, 12, 80):
        h.observe(v)
    out = reg.render()
    assert "# TYPE fill_latency_ms histogram" in out
    assert 'fill_latency_ms_bucket{le="5"}' in out
    assert 'fill_latency_ms_bucket{le="+Inf"} 5' in out
    assert "fill_latency_ms_count 5" in out
    assert "fill_latency_ms_sum 104" in out  # 2+3+7+12+80


def test_registry_idempotent_and_namespaced():
    reg = MetricsRegistry()
    c1 = reg.counter("x_total", "d")
    c2 = reg.counter("x_total", "d")  # 同名 → 同实例（幂等）
    assert c1 is c2
    c1.inc()
    c1.inc()
    assert 'x_total 2' in reg.render()


def test_full_observability_snapshot():
    """生产口径快照：风控 + 延迟 + 权益一次性可采集。"""
    from trader3.obs.metrics import build_default_registry

    reg = build_default_registry()
    reg.counter("risk_denied_total").inc(reason="CONCENTRATION")
    reg.gauge("account_equity").set(2_000_000.0)
    reg.histogram("fill_latency_ms").observe(42)
    out = reg.render()
    for name in ("risk_denied_total", "risk_allowed_total",
                 "account_equity", "fill_latency_ms",
                 "event_bus_lag_ms"):
        assert name in out
