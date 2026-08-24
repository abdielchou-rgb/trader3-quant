"""3号交易员 v2.0 — 自选股状态机测试"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trader3.v2.watchlist import (
    ALERT, ATTENTION, BUY_ZONE, HOLDING, OBSERVING, SELL_TRIGGERED, UNTRACKED,
    TRANSITIONS, PRIORITY, WatchlistDB,
)


class TestStateMachine:
    """状态机核心规则"""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.db = WatchlistDB(os.path.join(self.tmp, "test_watch.db"))
        self.db.add("000001.SZ", "平安银行", "测试")

    def teardown_method(self):
        self.db.close()

    def test_add_creates_observing(self):
        """加入跟踪 → 状态=观察中"""
        item = self.db.get("000001.SZ")
        assert item is not None
        assert item.status == OBSERVING

    def test_transition_whitelist(self):
        """合法迁移：观察→关注→买入区间→持有"""
        assert self.db.transition("000001.SZ", ATTENTION, "估值进入区间")
        assert self.db.transition("000001.SZ", BUY_ZONE, "三因子触发")
        assert self.db.transition("000001.SZ", HOLDING, "买入执行")
        item = self.db.get("000001.SZ")
        assert item.status == HOLDING

    def test_illegal_transition_blocked(self):
        """非法迁移被拦截：观察→持有（跳过中间状态）"""
        assert self.db.transition("000001.SZ", HOLDING, "跳过") is False
        item = self.db.get("000001.SZ")
        assert item.status == OBSERVING  # 状态未变

    def test_all_transitions_valid(self):
        """TRANSITIONS 白名单完整性：目标状态都存在于 ALL_STATES"""
        from trader3.v2.watchlist import ALL_STATES
        for src, targets in TRANSITIONS.items():
            assert src in ALL_STATES
            for t in targets:
                assert t in ALL_STATES

    def test_event_recorded(self):
        """状态迁移写入事件日志"""
        self.db.transition("000001.SZ", ATTENTION, "催化临近")
        events = self.db.events("000001.SZ")
        assert len(events) >= 1
        assert events[0].from_status == OBSERVING
        assert events[0].to_status == ATTENTION
        assert events[0].reason == "催化临近"

    def test_priority_ordering(self):
        """排序：卖出触发/买入区间/预警在前（经合法路径迁移）"""
        # 卖出触发：观察→关注→买入区间→持有→卖出
        self.db.add("BBB")
        for s, r in [(ATTENTION, "x"), (BUY_ZONE, "x"), (HOLDING, "x"), (SELL_TRIGGERED, "x")]:
            assert self.db.transition("BBB", s, r)
        # 买入区间：观察→关注→买入区间
        self.db.add("AAA")
        for s, r in [(ATTENTION, "x"), (BUY_ZONE, "x")]:
            assert self.db.transition("AAA", s, r)
        # 预警：观察→预警（合法）
        self.db.add("CCC")
        assert self.db.transition("CCC", ALERT, "风险事件")
        items = self.db.list_by_priority()
        assert items[0].status == SELL_TRIGGERED  # 最高优先级
        assert items[1].status == BUY_ZONE
        assert items[2].status == ALERT

    def test_trigger_data_updated(self):
        """三因子触发结果写入"""
        self.db.update_trigger("000001.SZ", 0.85, "催化强×估值低估×技术上破",
                               valuation_anchor=15.0, current_price=12.5)
        item = self.db.get("000001.SZ")
        assert item.trigger_score == 0.85
        assert item.valuation_anchor == 15.0

    def test_remove(self):
        """移除跟踪"""
        assert self.db.remove("000001.SZ", "用户移除")
        assert self.db.get("000001.SZ") is None
        # 事件仍保留
        assert len(self.db.events("000001.SZ")) >= 1


if __name__ == "__main__":
    t = TestStateMachine()
    t.setup_method()
    for name in dir(t):
        if name.startswith("test_"):
            t.setup_method()
            try:
                getattr(t, name)()
                print(f"  ✅ {name}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")