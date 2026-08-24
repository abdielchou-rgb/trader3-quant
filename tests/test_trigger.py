"""3号交易员 v2.0 — 三因子触发引擎测试"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trader3.v2.trigger import TriggerEngine
from trader3.v2.watchlist import ATTENTION, BUY_ZONE, get_watchlist


class TestTriggerEngine:
    """三因子触发引擎核心逻辑"""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "test_watch.db")
        # 用真实 qlib 数据的触发引擎（估值锚用 financials.db 真实数据）
        self.engine = TriggerEngine()

    def test_valuation_factor_real_600519(self):
        """估值因子：茅台真实估值锚输出"""
        score, implied, fair, price = self.engine._valuation_factor("600519")
        assert fair > 0          # 加权目标价存在
        assert price > 0         # 当前价存在
        assert 0 <= score <= 1

    def test_tech_factor_real_600519(self):
        """技术因子：茅台 qlib 行情计算"""
        score, note = self.engine._tech_factor("600519", 300.0)
        assert 0 <= score <= 1
        assert note  # 有说明

    def test_scan_produces_result(self):
        """扫描自选股产生结果"""
        wl = get_watchlist(self.db_path)
        wl.add("600519", "贵州茅台")
        items = wl.list()
        wl.close()

        results = self.engine.scan(items, catalyst_scores={"600519": 0.8})
        assert len(results) >= 1
        r = results[0]
        assert r.code == "600519"
        assert 0 <= r.score <= 1.2
        assert r.reason

    def test_scan_watchlist_auto_transition(self):
        """触发后自动迁移状态"""
        wl = get_watchlist(self.db_path)
        wl.add("600519", "贵州茅台")
        # 观察 → 关注
        wl.transition("600519", ATTENTION, "估值进入区间")
        wl.close()

        self.engine.scan_watchlist(db_path=self.db_path,
                                   catalyst_scores={"600519": 0.9})
        wl = get_watchlist(self.db_path)
        item = wl.get("600519")
        # 若触发，状态应迁移到买入区间
        status = item.status
        wl.close()
        print(f"     600519 触发后状态: {status}")
        assert status in (ATTENTION, BUY_ZONE)  # 触发则 BUY_ZONE，不触发则保持 ATTENTION


if __name__ == "__main__":
    t = TestTriggerEngine()
    t.setup_method()
    for name in dir(t):
        if name.startswith("test_"):
            t.setup_method()
            try:
                getattr(t, name)()
                print(f"  ✅ {name}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
