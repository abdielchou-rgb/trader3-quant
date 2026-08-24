"""3号交易员 v2.1 — 事前风控规则链测试"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trader3.v2.risk_rules import (
    BlacklistRule,
    PositionLimitRule,
    PriceBandRule,
    SingleOrderLimitRule,
    TimeWindowFlowControlRule,
    build_default_chain,
)


class TestRiskRules:
    def test_blacklist(self):
        r = BlacklistRule(["600999"])
        assert r.check(code="600999")[0] is False
        # fail-closed：买入但名称缺失 → 无法核实 ST/退市状态，拦截
        ok, msg = r.check(code="000001")
        assert ok is False and "fail-closed" in msg
        assert r.check(code="000001", name="平安银行")[0] is True  # 有名称且非ST → 通过
        assert r.check(code="000001", name="ST测试")[0] is False  # ST禁买
        assert r.check(code="000001", name="退市XX")[0] is False  # 退市禁买
        assert r.check(code="000001", action="sell")[0] is True    # 卖出不依赖名称

    def test_single_order_limit(self):
        r = SingleOrderLimitRule(max_value=5e6, max_pct=0.05)
        assert r.check(value=6e6)[0] is False      # 金额超限
        assert r.check(value=1e6)[0] is True
        assert r.check(value=1e6, value_pct=0.08)[0] is False  # 占比超限

    def test_position_limit(self):
        r = PositionLimitRule(max_single_pos=0.15, max_total_pos=0.95)
        assert r.check(position_pct=0.08, add_pct=0.10)[0] is False  # 单票 18%>15%
        assert r.check(position_pct=0.05, add_pct=0.05)[0] is True
        assert r.check(total_position_pct=0.97)[0] is False          # 总仓位超

    def test_price_band(self):
        r = PriceBandRule(max_dev=0.095)
        assert r.check(price=11.0, ref_price=10.0)[0] is False  # +10%涨停外
        assert r.check(price=10.8, ref_price=10.0)[0] is True

    def test_flow_control(self):
        r = TimeWindowFlowControlRule(max_orders_per_day=3, max_value_per_day=1e7,
                                      window_seconds=3600, max_orders_in_window=2)
        assert r.check(day_orders=3)[0] is False       # 日内笔数达上限
        assert r.check(day_orders=2, window_orders=2)[0] is False  # 窗口达上限
        assert r.check(day_orders=2, window_orders=1, day_value=5e6)[0] is True

    def test_chain_all_pass(self):
        chain = build_default_chain(blacklist=["600999"])
        ok, reason = chain.check(code="600519", name="贵州茅台", action="buy",
                                 value=1e6, price=12.5, ref_price=12.0,
                                 position_pct=0.02, add_pct=0.03)
        assert ok, reason
        assert reason == "全部通过"

    def test_chain_blocks(self):
        chain = build_default_chain(blacklist=["600999"])
        ok, _ = chain.check(code="600999", action="buy", value=1e6)
        assert ok is False
        stats = chain.summary()
        assert stats["blocked"] >= 1


if __name__ == "__main__":
    t = TestRiskRules()
    for name in dir(t):
        if name.startswith("test_"):
            try:
                getattr(t, name)()
                print(f"  ✅ {name}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
