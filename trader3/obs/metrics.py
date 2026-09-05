"""
Prometheus 指标暴露（E5）—— 纯文本 exposition 格式，零第三方依赖。

设计：
  1. Counter / Gauge / Histogram 三个原语，标签集（labels）支持
  2. render() 输出标准 `text/plain; version=0.0.4` 格式，可被
     Prometheus server 直接抓取
  3. MetricsRegistry 幂等注册（同名同实例），多模块共享
  4. FastAPI 接线：trader3.api.server /metrics 路由直接返回 render()

零依赖理由：本机/容器内不引 prometheus_client，保持最小充分 ——
文本格式协议极简（名称 值 + HELP/TYPE 注释），自实现 60 行内，
避免为一个 exposition 字符串引入新供应链。
"""

from __future__ import annotations

from collections.abc import Iterable

DEFAULT_BUCKETS_MS = (1, 5, 10, 50, 100, 250, 500, 1000, 5000)


def _fmt_labels(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + inner + "}"


class Counter:
    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help = help_text
        self._values: dict[tuple, float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        self._values[key] = self._values.get(key, 0.0) + amount

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}",
                 f"# TYPE {self.name} counter"]
        for key, val in sorted(self._values.items()):
            labels = dict(key)
            lines.append(f"{self.name}{_fmt_labels(labels)} {val}")
        return lines


class Gauge:
    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help = help_text
        self._values: dict[tuple, float] = {}

    def set(self, value: float, **labels: str) -> None:
        self._values[tuple(sorted(labels.items()))] = float(value)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}",
                 f"# TYPE {self.name} gauge"]
        for key, val in sorted(self._values.items()):
            lines.append(f"{self.name}{_fmt_labels(dict(key))} {val}")
        return lines


class Histogram:
    def __init__(self, name: str, help_text: str,
                 buckets: Iterable[float] = DEFAULT_BUCKETS_MS):
        self.name = name
        self.help = help_text
        self.buckets = tuple(sorted(set(float(b) for b in buckets)))
        self._counts: dict[tuple, list[int]] = {}
        self._sums: dict[tuple, float] = {}
        self._n: dict[tuple, int] = {}

    def observe(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        if key not in self._counts:
            self._counts[key] = [0] * len(self.buckets)
            self._sums[key] = 0.0
            self._n[key] = 0
        for i, b in enumerate(self.buckets):
            if value <= b:
                self._counts[key][i] += 1
        self._sums[key] += float(value)
        self._n[key] += 1

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}",
                 f"# TYPE {self.name} histogram"]
        for key in sorted(self._counts):
            labels = dict(key)
            cumulative = 0
            for i, b in enumerate(self.buckets):
                cumulative = max(cumulative, self._counts[key][i])
                b_str = str(int(b)) if float(b).is_integer() else str(b)
                lines.append(
                    f'{self.name}_bucket{_fmt_labels({**labels, "le": b_str})} {cumulative}'
                )
            lines.append(
                f'{self.name}_bucket{_fmt_labels({**labels, "le": "+Inf"})} {self._n[key]}'
            )
            lines.append(f"{self.name}_sum{_fmt_labels(labels)} {self._sums[key]}")
            lines.append(f"{self.name}_count{_fmt_labels(labels)} {self._n[key]}")
        return lines


class MetricsRegistry:
    """幂等注册表：同名同配置 → 同实例。"""

    def __init__(self) -> None:
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}

    def counter(self, name: str, help_text: str = "") -> Counter:
        if name not in self._metrics:
            self._metrics[name] = Counter(name, help_text)
        return self._metrics[name]  # type: ignore[return-value]

    def gauge(self, name: str, help_text: str = "") -> Gauge:
        if name not in self._metrics:
            self._metrics[name] = Gauge(name, help_text)
        return self._metrics[name]  # type: ignore[return-value]

    def histogram(self, name: str, help_text: str = "",
                  buckets: Iterable[float] = DEFAULT_BUCKETS_MS) -> Histogram:
        if name not in self._metrics:
            self._metrics[name] = Histogram(name, help_text, buckets)
        return self._metrics[name]  # type: ignore[return-value]

    def render(self) -> str:
        lines: list[str] = []
        for name in sorted(self._metrics):
            lines.extend(self._metrics[name].render())
        return "\n".join(lines) + "\n"


def build_default_registry() -> MetricsRegistry:
    """生产默认指标集（trader3 观测面）。"""
    reg = MetricsRegistry()
    reg.counter("risk_denied_total", "风控拒绝订单数（按原因）")
    reg.counter("risk_allowed_total", "风控放行订单数")
    reg.gauge("account_equity", "实时权益（元）")
    reg.gauge("account_cash", "可用现金（元）")
    reg.gauge("positions_open", "持仓标的数")
    reg.histogram("fill_latency_ms", "成交回报延迟（毫秒）")
    reg.histogram("event_bus_lag_ms", "事件总线排队延迟（毫秒）")
    reg.gauge("kill_switch_tripped", "回撤熔断状态（1=已熔断）")
    reg.gauge("drift_halt_active", "持仓漂移挂起状态（1=挂起中）")
    return reg
