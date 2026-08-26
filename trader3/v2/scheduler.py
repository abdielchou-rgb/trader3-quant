"""
统一调度器（Unified Scheduler）—— 桥接触发式 daily_pipeline 与截面量化 quant_pipeline。

职责：
- 按市场时段（工作日 9:30–15:00）判断是否应运行；同一自然日不重复触发。
- 每步：可选刷新因子工厂（每 N 步）→ 跑 quant_pipeline（截面组合）→ 可选跑 legacy
  daily_pipeline（事件/三因子触发式）→ 持久化运行态（净值/权重/状态/熔断/计数）。
- 运行态落盘 shared_state/scheduler_state.json，支持跨进程恢复与调度去重。

quant_pipeline 全流程离线可跑；legacy daily_pipeline 为可选钩子（缺依赖时优雅跳过）。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("trader3.v2.scheduler")


@dataclass
class SchedulerConfig:
    state_path: str = "shared_state/scheduler_state.json"
    market_open: time = time(9, 30)
    market_close: time = time(15, 0)
    enable_factor_factory_refresh: bool = False
    factor_factory_every_n: int = 5
    factor_factory_path: str = "shared_state/factor_registry.json"
    legacy_daily: bool = False          # 是否同时跑触发式 daily_pipeline
    min_run_interval_hours: float = 6.0
    # 自主面板抓取
    auto_panel: bool = False             # 未显式传 panel 时自动构建
    universe: list[str] = field(default_factory=list)
    lookback_days: int = 250
    panel_source: Callable | None = None  # source(code, lookback, end) -> 日线 DF


@dataclass
class SchedulerRun:
    ts: str
    kind: str                           # "scheduled" | "trigger" | "manual"
    quant: dict[str, Any]
    legacy: dict[str, Any] | None
    state: dict[str, Any]
    factory_refreshed: bool = False


def _is_weekday(now: datetime) -> bool:
    return now.weekday() < 5


def _is_market_open(now: datetime, cfg: SchedulerConfig | None = None) -> bool:
    cfg = cfg or SchedulerConfig()
    t = time(now.hour, now.minute)
    return _is_weekday(now) and cfg.market_open <= t <= cfg.market_close


class QuantScheduler:
    """统一调度器：按调度运行量化主链路并维护运行态。"""

    def __init__(self, config: SchedulerConfig | None = None):
        self.config = config or SchedulerConfig()
        self.state: dict[str, Any] = {"run_count": 0, "last_run": None, "history": []}
        self.load_state()

    # ── 调度判定 ───────────────────────────────────────
    def should_run(self, now: datetime | None = None) -> bool:
        now = now or datetime.now()
        if not _is_market_open(now, self.config):
            return False
        last = self.state.get("last_run")
        if not last:
            return True
        last_dt = datetime.fromisoformat(last) if isinstance(last, str) else last
        if last_dt.date() == now.date():
            return False
        if now > last_dt and (now - last_dt).total_seconds() \
                < self.config.min_run_interval_hours * 3600:
            return False
        return True

    # ── 运行态持久化 ───────────────────────────────────
    def load_state(self) -> None:
        p = Path(self.config.state_path)
        if p.exists():
            try:
                self.state = json.loads(p.read_text("utf-8"))
            except Exception as e:  # noqa: BLE001
                logger.warning("[scheduler] 状态读取失败: %s", e)

    def save_state(self) -> None:
        p = Path(self.config.state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        hist = self.state.get("history", [])[-30:]
        self.state["history"] = hist
        p.write_text(json.dumps(self.state, ensure_ascii=False, indent=2, default=str),
                     "utf-8")

    # ── 主步 ──────────────────────────────────────────
    async def step(self, *, panel: pd.DataFrame | None = None,
                   forward_returns: pd.DataFrame | None = None,
                   broker=None, overlay=None, kill_switch=None,
                   kind: str = "scheduled",
                   quant_config=None,
                   factor_factory_config=None) -> SchedulerRun:
        from trader3.v2.quant_pipeline import QuantPipelineConfig, run_quant_pipeline

        now = datetime.now()
        run_count = int(self.state.get("run_count", 0)) + 1
        factory_refreshed = False

        # 0. 自主面板抓取（未显式传入 panel 时）
        if panel is None and self.config.auto_panel and self.config.universe:
            from trader3.v2.panel_builder import build_panel
            panel = build_panel(self.config.universe, self.config.lookback_days,
                                source=self.config.panel_source)

        # 1. 可选刷新因子工厂（每 N 步）
        if self.config.enable_factor_factory_refresh and panel is not None \
                and run_count % self.config.factor_factory_every_n == 0:
            try:
                from trader3.v2.factor_factory import run_factor_factory
                run_factor_factory(panel, forward_returns,
                                   config=factor_factory_config,
                                   registry_path=self.config.factor_factory_path)
                factory_refreshed = True
                logger.info("[scheduler] 因子工厂刷新完成 -> %s",
                            self.config.factor_factory_path)
            except Exception as e:  # noqa: BLE001
                logger.warning("[scheduler] 因子工厂刷新失败: %s", e)

        # 2. 截面量化主链路
        qcfg = quant_config or QuantPipelineConfig()
        if factory_refreshed and not qcfg.factor_registry_path:
            qcfg.factor_registry_path = self.config.factor_factory_path
        quant = await run_quant_pipeline(
            panel, forward_returns=forward_returns, broker=broker,
            overlay=overlay, config=qcfg, kill_switch=kill_switch)

        # 3. 可选 legacy 触发式 daily_pipeline
        legacy = None
        if self.config.legacy_daily:
            legacy = self._run_legacy_daily(overlay=overlay)

        # 4. 更新运行态
        self.state["run_count"] = run_count
        self.state["last_run"] = now.isoformat()
        self.state["last_equity"] = quant.get("equity")
        weights = quant.get("weights")
        self.state["last_weights"] = (
            {k: round(float(v), 4) for k, v in weights.items()}
            if hasattr(weights, "items") else {})
        meta = quant.get("meta", {})
        self.state["last_regime"] = (
            meta.get("regime", {}).get("label") if meta.get("regime") else None)
        self.state["last_drawdown"] = meta.get("drawdown")
        self.state["last_kill_switch"] = meta.get("kill_switch_tripped")
        self.state["history"].append({
            "ts": now.isoformat(), "kind": kind,
            "equity": quant.get("equity"),
            "n_orders": meta.get("n_orders", 0),
            "regime": self.state["last_regime"],
            "factory_refreshed": factory_refreshed,
        })
        self.save_state()

        return SchedulerRun(ts=now.isoformat(), kind=kind, quant=quant,
                            legacy=legacy, state=dict(self.state),
                            factory_refreshed=factory_refreshed)

    @staticmethod
    def _run_legacy_daily(overlay=None) -> dict[str, Any] | None:
        try:
            from trader3.v2.daily_pipeline import run_daily
            summary = run_daily(overlay=overlay)
            return {"mode": summary.get("mode"), "scanned": summary.get("scanned"),
                    "triggered": len(summary.get("triggered", []) or []),
                    "paper": summary.get("paper_trading") is not None}
        except Exception as e:  # noqa: BLE001
            logger.warning("[scheduler] legacy daily_pipeline 跳过: %s", e)
            return {"error": str(e)}


def run_scheduler_step(*, panel=None, forward_returns=None, broker=None,
                       overlay=None, kill_switch=None, kind="manual",
                       config: SchedulerConfig | None = None,
                       quant_config=None, factor_factory_config=None) -> SchedulerRun:
    """同步便利入口（内部用 asyncio 跑异步 step）。"""
    sched = QuantScheduler(config)
    return asyncio.run(sched.step(
        panel=panel, forward_returns=forward_returns, broker=broker,
        overlay=overlay, kill_switch=kill_switch, kind=kind,
        quant_config=quant_config, factor_factory_config=factor_factory_config))
