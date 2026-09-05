"""可执行子单调度（P1-6）：把父单拆成 TWAP/VWAP 子单并按节奏提交给券商。

与执行层（nested_execution 给出交易轨迹 + LiquidityEngine 参与率约束）协同：
- 时间加权（TWAP）或量加权（VWAP）切片
- 切片时间加抖动以降辨识度（简易反博弈）
- 单切片参与率上限保护（与 LiquidityEngine.max_participation 对齐）
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

from trader3.v2.live.broker_base import Order, OrderSide


@dataclass
class ChildOrderSchedulerConfig:
    method: str = "twap"          # twap | vwap
    n_slices: int = 10            # 拆单数
    horizon: float = 60.0         # 执行总跨度（秒，回测可缩放）
    max_participation: float = 0.3   # 单切片参与率上限（仅用于告警/裁剪提示）
    jitter: float = 0.1           # 切片时间抖动比例（0~1）


@dataclass
class ChildOrder:
    symbol: str
    side: OrderSide
    quantity: float
    price: float | None
    t_offset: float               # 距起点的秒数
    slice_id: int
    parent_id: str = ""


def _slice_weights(method: str, n_slices: int,
                   volume_profile: list[float] | None) -> list[float]:
    if method == "vwap" and volume_profile:
        w = [max(x, 0.0) for x in volume_profile[:n_slices]]
        s = sum(w) or 1.0
        return [x / s for x in w]
    return [1.0 / n_slices] * n_slices


def schedule_children(parents: list[Order], cfg: ChildOrderSchedulerConfig,
                      volume_profile: list[float] | None = None,
                      rng: random.Random | None = None) -> list[ChildOrder]:
    """父单 → 子单列表（纯函数，便于测试回放）。"""
    rng = rng or random.Random(0)
    n = max(1, cfg.n_slices)
    weights = _slice_weights(cfg.method, n, volume_profile)
    out: list[ChildOrder] = []
    for p in parents:
        pid = getattr(p, "client_order_id", "") or p.symbol
        for i, w in enumerate(weights):
            base = (i + 0.5) / n * cfg.horizon
            jit = rng.uniform(-cfg.jitter, cfg.jitter) * (cfg.horizon / n)
            t = max(0.0, base + jit)
            qty = p.quantity * w
            if qty <= 0:
                continue
            out.append(ChildOrder(
                symbol=p.symbol, side=p.side, quantity=qty,
                price=getattr(p, "price", None), t_offset=t,
                slice_id=i, parent_id=pid,
            ))
    out.sort(key=lambda c: c.t_offset)
    return out


def _to_order(child: ChildOrder, seq: int) -> Order:
    return Order(
        symbol=child.symbol, side=child.side, quantity=child.quantity,
        price=child.price,
        client_order_id=f"{child.parent_id}_c{child.slice_id}_{seq}",
    )


async def execute_child_orders(
    parents: list[Order], broker, cfg: ChildOrderSchedulerConfig,
    *, volume_profile: list[float] | None = None,
    rng: random.Random | None = None,
    submit=None, sleep=None,
) -> list[Order]:
    """把父单拆子单并按 t_offset 节奏提交（异步）。

    submit / sleep 可注入以便测试（默认 broker.place_order / asyncio.sleep）。
    """
    children = schedule_children(parents, cfg, volume_profile=volume_profile, rng=rng)
    submit_fn = submit or (lambda o: broker.place_order(o))
    sleep_fn = sleep or asyncio.sleep
    placed: list[Order] = []
    clock = 0.0
    for seq, child in enumerate(children):
        wait = child.t_offset - clock
        if wait > 0:
            await sleep_fn(wait)
        clock = child.t_offset
        order = _to_order(child, seq)
        res = await submit_fn(order)
        placed.append(res)
    return placed
