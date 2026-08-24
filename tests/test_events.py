"""3号交易员 v2.0 — 事件库 + 催化评分器测试"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trader3.v2.events import CatalystScorer, Event, get_event_library


class TestCatalystScorer:
    def setup_method(self):
        self.scorer = CatalystScorer()

    def test_strong_positive(self):
        """业绩增长 → 强正向催化"""
        r = self.scorer.score("净利润同比增长30%，业绩超预期")
        assert r["direction"] == "positive"
        assert r["score"] >= 0.6
        assert r["category"] == "业绩"

    def test_strong_negative(self):
        """净利润下降 → 强负向催化"""
        r = self.scorer.score("净利润同比下降50%，预亏")
        assert r["direction"] == "negative"
        assert r["score"] <= 0.5

    def test_neutral(self):
        """例行公告 → 中性低分"""
        r = self.scorer.score("召开股东大会的提示性公告")
        assert r["score"] <= 0.5

    def test_order_contract(self):
        """中标/大单 → 订单类别"""
        r = self.scorer.score("公司中标重大工程项目")
        assert r["category"] == "订单/合同"
        assert r["direction"] == "positive"


class TestEventLibrary:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.db = get_event_library(os.path.join(self.tmp, "events.db"))

    def teardown_method(self):
        self.db.close()

    def test_upsert_dedup(self):
        """同一事件去重"""
        ev = Event(code="600519", source="news", title="测试标题", catalyst_score=0.8,
                   direction="positive", collected_at="2026-08-01")
        assert self.db.upsert(ev) is True
        assert self.db.upsert(Event(code="600519", source="news", title="测试标题",
                                    catalyst_score=0.8, direction="positive",
                                    collected_at="2026-08-01")) is False

    def test_get_recent_sorted(self):
        """取最近事件按催化强度返回"""
        self.db.upsert(Event(code="600519", source="news", title="弱中性",
                             catalyst_score=0.3, direction="neutral"))
        self.db.upsert(Event(code="600519", source="news", title="强催化",
                             catalyst_score=0.9, direction="positive"))
        recent = self.db.get_recent("600519", days=7)
        assert len(recent) >= 2
        strongest = self.db.get_strongest_recent("600519")
        assert strongest.title == "强催化"


if __name__ == "__main__":
    for cls in (TestCatalystScorer, TestEventLibrary):
        t = cls()
        t.setup_method()
        for name in dir(t):
            if name.startswith("test_"):
                t.setup_method()
                try:
                    getattr(t, name)()
                    print(f"  ✅ {name}")
                except Exception as e:
                    print(f"  ❌ {name}: {e}")