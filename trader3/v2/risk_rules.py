"""3号交易员 v2.1 — 事前风控规则链 (risk_rules)

吸收 vnpy 事前风控规则链模式（已深读）：
- 每条规则独立类（黑白名单/单笔限额/时间窗流控/持仓上限）
- 下单/触发前逐条全拦，任一命中即记录并阻断

落地目标：在 3hao 触发/下单出口前加一道风控链，防止「三因子触发但风控不过」的误单。

用法：
    from trader3.v2.risk_rules import RiskRuleChain, build_default_chain
    chain = build_default_chain()
    ok, reason = chain.check(code, action="buy", value=5000000, price=12.5)
"""
from __future__ import annotations

import json


class BaseRiskRule:
    """风控规则基类"""

    rule_name: str = ""
    description: str = ""
    enabled: bool = True

    def check(self, **ctx) -> tuple[bool, str]:
        """返回 (通过?, 原因)"""
        raise NotImplementedError


class BlacklistRule(BaseRiskRule):
    """黑白名单：禁止买入/交易 ST、退市整理、黑名单标的"""

    rule_name = "blacklist"
    description = "禁止交易黑名单标的（ST/退市/指定）"

    def __init__(self, blacklist: list[str] | None = None):
        self.blacklist = set(blacklist or [])
        self.st_prefix_block = True  # ST 一律禁买

    def check(self, **ctx) -> tuple[bool, str]:
        code = ctx.get("code", "")
        name = str(ctx.get("name", "") or "")
        action = ctx.get("action", "buy")
        if code in self.blacklist:
            return False, f"黑名单标的: {code}"
        if action in ("buy",):
            upper = name.upper()
            is_risky_name = ("ST" in upper) or ("退" in name)
            if is_risky_name and self.st_prefix_block:
                return False, f"ST/退市整理标的禁买: {name}"
            # fail-closed：买入动作但名称缺失 → 无法核实 ST/退市状态，拦截
            if not name.strip():
                return False, "名称缺失，无法核实 ST/退市状态（fail-closed），请补全行情快照后重试"
        return True, ""


class SingleOrderLimitRule(BaseRiskRule):
    """单笔限额：单笔订单不可超过上限（金额或占比）"""

    rule_name = "single_order_limit"
    description = "单笔订单金额/比例上限"

    def __init__(self, max_value: float = 5e6, max_pct: float = 0.05):
        self.max_value = max_value   # 单笔最大金额（元）
        self.max_pct = max_pct       # 单笔占组合最大比例

    def check(self, **ctx) -> tuple[bool, str]:
        value = float(ctx.get("value", 0) or 0)
        if value > self.max_value:
            return False, f"单笔超限: {value:.0f} > 上限 {self.max_value:.0f}"
        pct = ctx.get("value_pct", 0)
        if pct and pct > self.max_pct:
            return False, f"单笔占比超限: {pct:.1%} > {self.max_pct:.1%}"
        return True, ""


class TimeWindowFlowControlRule(BaseRiskRule):
    """时间窗流控：日内交易笔数/金额窗内上限（vnpy 模式照抄）"""

    rule_name = "time_window_flow_control"
    description = "日内/窗口内成交笔数与金额流控"

    def __init__(self, max_orders_per_day: int = 20, max_value_per_day: float = 3e7,
                 window_seconds: int = 3600, max_orders_in_window: int = 5):
        self.max_orders_per_day = max_orders_per_day
        self.max_value_per_day = max_value_per_day
        self.window_seconds = window_seconds
        self.max_orders_in_window = max_orders_in_window

    def check(self, **ctx) -> tuple[bool, str]:
        day_orders = int(ctx.get("day_orders", 0) or 0)
        day_value = float(ctx.get("day_value", 0) or 0)
        window_orders = int(ctx.get("window_orders", 0) or 0)
        if day_orders >= self.max_orders_per_day:
            return False, f"日内笔数超限: {day_orders} >= {self.max_orders_per_day}"
        if day_value > self.max_value_per_day:
            return False, f"日内金额超限: {day_value:.0f} > {self.max_value_per_day:.0f}"
        if window_orders >= self.max_orders_in_window:
            return False, f"窗口内笔数超限: {window_orders} >= {self.max_orders_in_window}"
        return True, ""


class PositionLimitRule(BaseRiskRule):
    """持仓上限：单票持仓/总仓位上限"""

    rule_name = "position_limit"
    description = "单票/总仓位上限"

    def __init__(self, max_single_pos: float = 0.15, max_total_pos: float = 0.95):
        self.max_single_pos = max_single_pos
        self.max_total_pos = max_total_pos

    def check(self, **ctx) -> tuple[bool, str]:
        pos_pct = float(ctx.get("position_pct", 0) or 0)
        if pos_pct + float(ctx.get("add_pct", 0) or 0) > self.max_single_pos:
            return False, f"单票仓位超限: {pos_pct:.1%} -> {self.max_single_pos:.1%}"
        total_pct = float(ctx.get("total_position_pct", 0) or 0)
        if total_pct > self.max_total_pos:
            return False, f"总仓位超限: {total_pct:.1%} > {self.max_total_pos:.1%}"
        return True, ""


class PriceBandRule(BaseRiskRule):
    """价格波动带：偏离基准价过大拒绝（A股涨跌停 ±10% 相关）"""

    rule_name = "price_band"
    description = "价格偏离基准过大拒绝（涨跌停保护）"

    def __init__(self, max_dev: float = 0.095):
        self.max_dev = max_dev  # 偏离基准价上限（A股涨停一般±10%，留缓冲）

    def check(self, **ctx) -> tuple[bool, str]:
        price = float(ctx.get("price", 0) or 0)
        ref = float(ctx.get("ref_price", 0) or 0)
        if price and ref and ref > 0:
            dev = abs(price - ref) / ref
            if dev > self.max_dev:
                return False, f"价格偏离基准 {dev:.1%} > {self.max_dev:.1%}"
        return True, ""


class RiskRuleChain:
    """
    事前风控规则链（vnpy 模式）：
    规则按序执行，任一失败即记录并阻断。
    """

    def __init__(self, rules: list[BaseRiskRule] | None = None, log_path: str = ""):
        self.rules = rules or []
        self.log_path = log_path or ""
        self._history: list[dict] = []

    def add(self, rule: BaseRiskRule) -> RiskRuleChain:
        self.rules.append(rule)
        return self

    def check(self, **ctx) -> tuple[bool, str]:
        """逐条风控。返回 (整体通过?, 原因)

        ctx 可用字段：code/name/action/value/price/ref_price/value_pct/
                     day_orders/day_value/window_orders/position_pct/add_pct/total_position_pct
        """
        for rule in self.rules:
            if not rule.enabled:
                continue
            ok, reason = rule.check(**ctx)
            self._record(rule.rule_name, ok, reason, ctx)
            if not ok:
                return False, f"[风控-{rule.rule_name}] {reason}"
        return True, "全部通过"

    def _record(self, rule: str, ok: bool, reason: str, ctx: dict):
        self._history.append({
            "rule": rule, "ok": ok, "reason": reason,
            "code": ctx.get("code", ""), "action": ctx.get("action", ""),
        })
        if len(self._history) > 200:
            self._history = self._history[-200:]
        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(self._history[-1], ensure_ascii=False) + "\n")
            except Exception:
                pass

    def summary(self) -> dict:
        total = len(self._history)
        blocked = sum(1 for h in self._history if not h["ok"])
        return {"total": total, "blocked": blocked, "pass": total - blocked}

    def reset(self):
        self._history = []


def build_default_chain(blacklist: list[str] | None = None) -> RiskRuleChain:
    """构建默认风控链（顺序敏感：名单→限额→仓位→价格→流控）"""
    return RiskRuleChain([
        BlacklistRule(blacklist),
        SingleOrderLimitRule(max_value=5e6, max_pct=0.05),
        PositionLimitRule(max_single_pos=0.15, max_total_pos=0.95),
        PriceBandRule(max_dev=0.095),
        TimeWindowFlowControlRule(
            max_orders_per_day=20, max_value_per_day=3e7,
            window_seconds=3600, max_orders_in_window=5,
        ),
    ])
