"""
持仓漂移对账挂起引擎（E4，Reconciliation Engine）。

痛点：本地账与券商账漂移（网络丢回报/重复回报/人工干预）后，
若继续开新仓，敞口计算建立在错误持仓之上 —— 生产事故放大器。

机制（复用 v2.execution.reconcile_positions 的差异检测）：
  1. reconcile(internal, broker)：比对全量持仓
  2. 漂移标的数 ≥ halt_threshold 且差异超容差 → 引擎挂起（halted）
     并联动风控网关拒开新仓（平仓放行：降风险优先）
  3. 人工核对/重同步后 resolve() 解除

生产接线：定时任务（每分钟）拉券商持仓 vs shared_state 本地账调用本引擎。
"""

from __future__ import annotations

from typing import Any


class DriftHaltEngine:
    """持仓漂移 → 自动挂起开仓权限。"""

    def __init__(self, tolerance_qty: float = 0.0, halt_threshold: int = 1):
        """
        Parameters
        ----------
        tolerance_qty : 单票持仓差异数量容差（股）
        halt_threshold : 漂移标的数达到该阈值才挂起（防单票误报）
        """
        self.tolerance_qty = float(tolerance_qty)
        self.halt_threshold = int(halt_threshold)
        self.halted = False
        self.last_report: dict[str, Any] = {}
        self._gateway: Any = None  # PreTradeRiskGateway（可选绑定）

    def bind_gateway(self, gateway: Any) -> None:
        """绑定风控网关：挂起时置网关 drift_halt 标志。"""
        self._gateway = gateway

    def reconcile(
        self,
        internal: dict[str, float],
        broker: dict[str, float],
    ) -> dict[str, Any]:
        """比对持仓并按阈值决定挂起。返回差异报告。"""
        # 对齐键集（单侧缺失=差异）
        symbols = sorted(set(internal) | set(broker))
        diffs: dict[str, float] = {}
        for s in symbols:
            iq = float(internal.get(s, 0.0))
            bq = float(broker.get(s, 0.0))
            if abs(iq - bq) > self.tolerance_qty:
                diffs[s] = iq - bq

        self.last_report = {
            "drifted": len(diffs) >= self.halt_threshold and bool(diffs),
            "drift_symbols": diffs,
            "n_drift": len(diffs),
            "halting": False,
        }
        if self.last_report["drifted"]:
            self.halted = True
            self.last_report["halting"] = True
            if self._gateway is not None:
                self._gateway.drift_halt()
        return self.last_report

    def resolve(self) -> None:
        """人工核对/重同步完成后解除挂起。"""
        self.halted = False
        if self._gateway is not None:
            self._gateway.drift_resume()
