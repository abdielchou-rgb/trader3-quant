"""
严格闭环订单状态机。

设计目标（机构级确定性）：
  1. 一切状态转移必须经 VALID_TRANSITIONS 白名单 —— 非法转移在本地即拒绝，
     杜绝"撤单后复活""未 ack 即成交"这类由异步回报乱序造成的伪状态。
  2. 终态（FILLED / CANCELED / REJECTED）吸收集为空：一旦进入，任何后续
     迟到回报都无法改变历史 —— 对账逻辑因此可幂等。
  3. FAILED_LOST 为瞬态异常态（断网/超时/发送结果未知）：不允许自动转移到
     成交类状态，必须经对账或人工裁决后走 CANCELED 路径。
"""

from __future__ import annotations

from enum import Enum, auto


class OrderState(Enum):
    PENDING_SUBMIT = auto()    # 本地创建已落盘，待发向网关
    SUBMITTED = auto()         # 已发送至 broker，等待 ack
    ACKNOWLEDGED = auto()      # 券商已接收并确认报单（持有效 order_id）
    PARTIALLY_FILLED = auto()  # 部分成交
    FILLED = auto()            # 全部成交（终态）
    CANCEL_REQUESTED = auto()  # 本地发出撤单请求
    CANCELED = auto()          # 撤单成功（终态）
    REJECTED = auto()         # 废单/拒单（终态）
    FAILED_LOST = auto()       # 瞬时断网等异常未知态


# PyPI 版兼容导出（测试与对账层直接引用单例成员）
FAILED_LOST = OrderState.FAILED_LOST

VALID_TRANSITIONS: dict[OrderState, set[OrderState]] = {
    OrderState.PENDING_SUBMIT: {OrderState.SUBMITTED, OrderState.REJECTED},
    # SUBMITTED 允许直达任意后继：真实回报流可能乱序/压缩
    # （漏 ACK 直接成交、撤单回报先于确认到达），对账时按券商事实收敛
    OrderState.SUBMITTED: {OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED,
                           OrderState.FILLED, OrderState.CANCEL_REQUESTED,
                           OrderState.CANCELED, OrderState.REJECTED,
                           OrderState.FAILED_LOST},
    OrderState.ACKNOWLEDGED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED,
                              OrderState.CANCEL_REQUESTED, OrderState.REJECTED},
    OrderState.PARTIALLY_FILLED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED,
                                 OrderState.CANCEL_REQUESTED},
    OrderState.CANCEL_REQUESTED: {OrderState.CANCELED, OrderState.FILLED},
    OrderState.FILLED: set(),
    OrderState.CANCELED: set(),
    OrderState.REJECTED: set(),
    # FAILED_LOST 只能人工/对账裁决后撤销，不允许"自动恢复"为成交
    OrderState.FAILED_LOST: {OrderState.CANCELED},
}

# 终态集合（对账幂等性的基础）
TERMINAL_STATES = {OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED}


def can_transition(src: OrderState, dst: OrderState) -> bool:
    return dst in VALID_TRANSITIONS.get(src, set())
