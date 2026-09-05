"""
影子对账（shadow reconcile）— 回测→实盘并行验证的每日落地环节。

职责：
  1. 读取目标组合（targets.json：权重和=1）
  2. 用 ShadowBroker 包装 QMTBroker，把目标权重的调仓订单全部走影子通道
     （记录、不打发、不产生真实成交）
  3. 计算目标组合与影子持仓的缺口（gaps），落盘 shadow_run.json
  4. 每日积累 → 3-6 个月后可对照：影子执行 vs 回测预期 vs （未来）真实执行

诚实边界：simulated=True 时 QMTBroker 走离线 SIM 撮合（订单打标 QMT-SIM），
影子层仍然拦截在先——验证的是下单链路与权重计算，不是市场成交质量。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

from trader3.v2.live.broker_base import (
    Order,
    OrderSide,
    OrderType,
    ShadowBroker,
)

logger = logging.getLogger("trader3.v2.shadow_reconcile")

SHADOW_STATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "shared_state", "shadow", "shadow_run.json",
)


def _load_targets(targets_path: str) -> tuple[dict[str, float], float]:
    with open(targets_path, encoding="utf-8-sig") as f:
        data = json.load(f)
    weights = {str(k): float(v) for k, v in (data.get("weights") or {}).items()}
    if not weights:
        raise ValueError("targets.json 无 weights")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"权重和必须为 1.0，当前 {total:.6f}")
    cash_pct = float(data.get("cash_pct", 0.0))
    return weights, cash_pct


def run_shadow_reconcile(
    targets_path: str,
    qmt_path: str,
    account_id: str,
    simulated: bool = True,
    dry_run: bool = False,
    state_file: str | None = None,
    equity: float = 1_000_000.0,
) -> dict[str, Any]:
    """跑一次影子对账。返回并（非 dry_run 时）落盘执行记录。

    equity 用于把权重换算成股数（整手取整）；SIM 路径下不涉及真实资金。
    """
    from trader3.v2.live.qmt_broker import QMTBroker, QMTConfig

    weights, _cash_pct = _load_targets(targets_path)

    async def _run() -> tuple[list[dict[str, Any]], dict[str, float]]:
        inner = QMTBroker(QMTConfig(qmt_path=qmt_path, account_id=account_id),
                          simulated=simulated)
        await inner.connect()
        sh = ShadowBroker(inner)
        await sh.connect()

        orders_meta: list[dict[str, Any]] = []
        gaps: dict[str, float] = {}
        for sym, w in weights.items():
            notional = equity * w
            # 估值价：SIM 内部价目；真实路径由行情接口取（当前 SIM 联调）
            px = inner._sim_prices.get(sym, 10.0)
            qty = int(notional / px)
            qty -= qty % 100  # 整手
            if qty < 100:
                gaps[sym] = w
                continue
            o = Order(symbol=sym, side=OrderSide.BUY, quantity=qty,
                      order_type=OrderType.MARKET)
            res = await sh.place_order(o)
            orders_meta.append({
                "symbol": sym,
                "quantity": res.quantity,
                "shadow": bool(res.metadata.get("shadow")),
                "simulated": bool(res.metadata.get("simulated")),
                "status": res.status.value,
                "client_order_id": res.client_order_id,
                "broker_order_id": res.broker_order_id,
            })
            gaps[sym] = w - (res.quantity * px) / equity  # 目标权重 - 影子已下权重

        await sh.disconnect()
        return orders_meta, gaps

    orders, gaps = asyncio.run(_run())
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    if dry_run:
        return {
            "mode": "shadow-dry", "ts": ts, "n_orders": len(orders),
            "orders": orders, "gaps": gaps,
        }

    payload = {
        "mode": "shadow",
        "ts": ts,
        "datetime": datetime.now().isoformat(),
        "equity": equity,
        "n_orders": len(orders),
        "orders": orders,
        "gaps": gaps,
        "targets": weights,
    }
    out_path = state_file or SHADOW_STATE_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("影子对账落盘: %s（%d 订单）", out_path, len(orders))
    return payload
