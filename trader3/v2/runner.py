"""
顶层编排器 / CLI（runner）—— 把整套量化系统串成一次可运行的主链路。

把 Settings（环境变量）→ 券商选择（真实 CTP / SIMULATED / 影子）→ 自主面板 →
统一调度器一步跑通：因子工厂刷新 + 截面集成（MoE/状态路由/组合优化/风险覆盖/嵌套执行）
+ 成本门禁 + 持仓对账 + Kill-Switch，并落盘运行态。

用法：
    python -m trader3.v2.runner                       # 按 Settings 运行（UNIVERSE 非空时自主抓面板）
    python -m trader3.v2.runner --universe 600519,000858 --shadow
    python -m trader3.v2.runner --once                # 单次（不依赖调度去重）

离线演示：未装 vnpy_ctp 时 CTPBroker 走 SIMULATED；未设 OPENROUTER_API_KEY 时因子工厂离线闭环。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

from trader3.v2.config import Settings
from trader3.v2.data_sources import make_panel_source
from trader3.v2.execution import KillSwitch
from trader3.v2.factor_factory import FactorFactoryConfig
from trader3.v2.live.broker_base import ShadowBroker
from trader3.v2.live.ctp_broker import CTPBroker
from trader3.v2.quant_pipeline import QuantPipelineConfig
from trader3.v2.scheduler import QuantScheduler, SchedulerConfig, SchedulerRun

logger = logging.getLogger("trader3.v2.runner")


def build_broker(settings: Settings, *, shadow: bool = False) -> CTPBroker | ShadowBroker:
    """券商选择：凭据齐全且非 paper 时走真实 CTP（否则 SIMULATED）；可选影子包装。"""
    brk: Any = CTPBroker(settings.ctp_config())
    if shadow or settings.shadow_mode:
        return ShadowBroker(brk)
    return brk


def build_pipeline_config(settings: Settings) -> QuantPipelineConfig:
    return QuantPipelineConfig(
        method=settings.quant_method,
        use_moe=True,
        moe_experts=["lgbm", "et", "ridge"],
        min_train=60,
        use_cost_gate=True,
        use_reconcile=True,
        kill_switch_dd=settings.kill_switch_dd,
        use_nested_execution=settings.nested_execution,
        edge_per_score_bps=50.0,
        factor_registry_path="shared_state/factor_registry.json",
    )


def build_scheduler_config(settings: Settings, universe: list[str]) -> SchedulerConfig:
    return SchedulerConfig(
        auto_panel=bool(universe),
        universe=universe,
        lookback_days=settings.lookback_days,
        enable_factor_factory_refresh=settings.factory_refresh,
        factor_factory_every_n=settings.factory_every_n,
        legacy_daily=settings.legacy_daily,
        panel_source=make_panel_source(settings.data_source, settings.qlib_uri),
    )


async def run_once(settings: Settings, *,
                  universe: list[str] | None = None,
                  shadow: bool = False,
                  panel=None,
                  kill_switch: KillSwitch | None = None,
                  kind: str = "manual") -> SchedulerRun:
    universe = universe if universe is not None else settings.universe_list()
    brk = build_broker(settings, shadow=shadow)
    await brk.connect()
    sched = QuantScheduler(build_scheduler_config(settings, universe))
    run = await sched.step(
        panel=panel, broker=brk, kill_switch=kill_switch,
        kind=kind,
        quant_config=build_pipeline_config(settings),
        factor_factory_config=FactorFactoryConfig(),
    )
    await brk.disconnect()
    return run


def summarize(run: SchedulerRun) -> str:
    w = run.quant.get("weights", {})
    meta = run.quant.get("meta", {})
    lines = ["# 3号交易员 · 量化主链路运行", ""]
    lines.append(f"- 模式: {run.kind}  时间: {run.ts}")
    lines.append(f"- 组合权重数: {len(w)}  下单数: {meta.get('n_orders', 0)}")
    lines.append(f"- 状态: {meta.get('regime', {}).get('label') if meta.get('regime') else 'n/a'}"
                 f"  回撤: {meta.get('drawdown')}  KillSwitch: {meta.get('kill_switch_tripped')}")
    if "execution_plan" in meta:
        ep = meta["execution_plan"]
        lines.append(f"- 嵌套执行: 预期成本 {ep['expected_cost_bps']}bp / "
                     f"缺口 {ep['expected_shortfall']:.2f} / 切片 {ep['n_slices']}")
    if run.factory_refreshed:
        lines.append("- 因子工厂：本轮已刷新并入库")
    if run.legacy is not None:
        lines.append(f"- legacy daily_pipeline: {run.legacy.get('triggered')} 触发")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="3号交易员 量化主链路编排")
    ap.add_argument("--universe", default="", help="逗号分隔标的，覆盖 UNIVERSE")
    ap.add_argument("--shadow", action="store_true", help="影子模式（真实券商仅空跑）")
    ap.add_argument("--once", action="store_true", help="强制单次运行（忽略调度去重）")
    ap.add_argument("--no-auto-panel", action="store_true", help="不自主抓面板（需另行喂数据）")
    args = ap.parse_args()

    settings = Settings.load()
    universe = [u.strip() for u in args.universe.split(",") if u.strip()] or settings.universe_list()
    if args.no_auto_panel:
        universe = []

    if args.once:
        run = asyncio.run(run_once(settings, universe=universe, shadow=args.shadow, kind="manual"))
    else:
        # 受调度去重约束：仅在应运行时执行
        sched = QuantScheduler(build_scheduler_config(settings, universe))
        if sched.should_run():
            run = asyncio.run(run_once(settings, universe=universe, shadow=args.shadow))
        else:
            logger.info("调度去重：当前不在运行窗口或今日已运行。")
            return
    print(summarize(run))


if __name__ == "__main__":
    main()
