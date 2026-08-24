"""审计修复验证测试（test_fix_v2_audit）

覆盖 10 项修复规格的核心验收点，零联网、零生产库：
  1. 负面事件否决买入（trigger 风控链接通）
  2. events 时效过滤（get_recent 日期下界）
  3. comps TTM 口径（近4单季合计）
  4. SQLite WAL PRAGMA
  5. cookie 合并进 headers
  6. scan dry_run 不改库
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trader3.v2.events import Event, get_event_library
from trader3.v2.trigger import TriggerEngine
from trader3.v2.watchlist import ATTENTION, BUY_ZONE, WatchlistDB


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _make_engine(tmpdir: str) -> TriggerEngine:
    """离线引擎：注入临时事件库，估值/技术因子由各测试按需 monkeypatch。"""
    engine = TriggerEngine()
    engine._event_lib = get_event_library(os.path.join(tmpdir, "events.db"))
    return engine


class TestNegativeEventVeto:
    """修复(1)：负面事件否决买入迁移"""

    def test_negative_event_vetoes_buy(self, monkeypatch):
        tmp = tempfile.mkdtemp()
        engine = _make_engine(tmp)
        # 高分三因子面（估值/技术全强）
        monkeypatch.setattr(engine, "_valuation_factor",
                            lambda code, asof_date=None: (1.0, 0.35, 20.0, 12.0))
        monkeypatch.setattr(engine, "_tech_factor",
                            lambda code, price: (1.0, "测试上破20日高"))
        # 注入 0.8 强负面事件
        lib = engine._event_lib
        lib.upsert(Event(
            code="600519", source="news", title="实控人被立案调查",
            catalyst_score=0.8, direction="negative", collected_at=_iso(datetime.now()),
        ))
        r = engine._scan_one("600519", None)
        assert r.triggered is False
        assert any("否决" in c for c in r.caveats), f"caveats={r.caveats}"
        lib.close()


class TestEventsTimeFilter:
    """修复(2)：events 时效过滤"""

    def test_get_recent_respects_days(self):
        tmp = tempfile.mkdtemp()
        db = get_event_library(os.path.join(tmp, "events.db"))
        try:
            old = datetime.now() - timedelta(days=40)
            db.upsert(Event(code="000001", source="news", title="40天前旧闻",
                            catalyst_score=0.95, direction="positive",
                            collected_at=_iso(old)))
            db.upsert(Event(code="000001", source="news", title="今天新鲜事",
                            catalyst_score=0.60, direction="positive",
                            collected_at=_iso(datetime.now())))
            recent = db.get_recent("000001", days=7)
            titles = [e.title for e in recent]
            assert titles == ["今天新鲜事"], f"got {titles}"
            strongest = db.get_strongest_recent("000001", days=7)
            assert strongest.title == "今天新鲜事"
        finally:
            db.close()


class TestCompsTTM:
    """修复(3)：comps TTM 口径"""

    def test_comps_ttm_math(self, monkeypatch):
        from trader3.v2 import comps as comps_mod

        class FakeFP:
            def get_latest_financials(self, code):
                return {}

            def get_field_history(self, code, field, table=None, n=8):
                return []

        analyzer = comps_mod.CompsAnalyzer(fp=FakeFP(), dp=None)
        # 4 个已知单季值：np 各 1e8 → TTM 4e8；mc=4e9 → pe=10
        known = {
            "MBRevenue": [(f"q{i}", 3e8) for i in range(4)],
            "netProfit": [(f"q{i}", 1e8) for i in range(4)],
            "operateProfit": [(f"q{i}", 2e8) for i in range(4)],
        }
        monkeypatch.setattr(analyzer, "_quarter_history",
                            lambda code, field, n=5: list(known[field]))
        import trader3.v2.market_data as md
        monkeypatch.setattr(md, "get_quote",
                            lambda code: {"name": "测试股", "price": 10.0,
                                          "market_cap": 4e9})
        row = analyzer._peer_snapshot("600519", "600519")
        assert abs(row.net_profit_cny - 4 * 1e8) < 1e-6
        assert abs(row.pe - (4e9 / (4 * 1e8))) < 1e-6, f"pe={row.pe}"


class TestSQLiteHardening:
    """修复(4)：WAL/busy_timeout/synchronous"""

    def test_wal_pragma_set(self):
        tmp = tempfile.mkdtemp()
        ev = get_event_library(os.path.join(tmp, "events.db"))
        wl = WatchlistDB(os.path.join(tmp, "watch.db"))
        try:
            for conn in (ev._conn, wl._conn):
                mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
                busy = conn.execute("PRAGMA busy_timeout").fetchone()[0]
                sync = conn.execute("PRAGMA synchronous").fetchone()[0]
                assert str(mode).lower() == "wal", f"journal_mode={mode}"
                assert int(busy) == 5000
                assert int(sync) == 1  # NORMAL
        finally:
            ev.close()
            wl.close()

    def test_watchlist_add_idempotent(self):
        """INSERT OR IGNORE 单语句：重复 add 返回 False 且不报错"""
        tmp = tempfile.mkdtemp()
        wl = WatchlistDB(os.path.join(tmp, "watch.db"))
        try:
            assert wl.add("600000", "浦发银行") is True
            assert wl.add("600000", "浦发银行") is False
            assert len(wl.list()) == 1
        finally:
            wl.close()


class TestCookieHeader:
    """修复(5)：cookie 合并进 headers 而非 cookies 参数"""

    def test_cookie_sent_as_header(self, monkeypatch):
        import requests

        captured = []

        class FakeResp:
            status_code = 200

            def json(self):
                return {}

        def fake_get(url, **kwargs):
            captured.append({"url": url, **kwargs})
            return FakeResp()

        monkeypatch.setattr(requests, "get", fake_get)
        monkeypatch.setenv("XUEQIU_COOKIE", "xq_a_token=abc123")

        from trader3.v2.sources import XueqiuSource
        items = XueqiuSource().fetch("茅台", limit=5)
        assert items == []
        assert captured, "应至少发起一次请求"
        first = captured[0]
        assert first["headers"].get("Cookie") == "xq_a_token=abc123"
        cookies_kwarg = first.get("cookies")
        assert not cookies_kwarg or "Cookie" not in cookies_kwarg


class TestDryRun:
    """修复(1)配套：dry_run 只报告不改库"""

    def test_scan_dry_run_no_mutation(self, monkeypatch):
        tmp = tempfile.mkdtemp()
        engine = _make_engine(tmp)
        monkeypatch.setattr(engine, "_valuation_factor",
                            lambda code, asof_date=None: (1.0, 0.35, 20.0, 12.0))
        monkeypatch.setattr(engine, "_tech_factor",
                            lambda code, price: (1.0, "测试上破"))

        wl_path = os.path.join(tmp, "watch.db")
        wl = WatchlistDB(wl_path)
        wl.add("600519", "贵州茅台")
        wl.transition("600519", ATTENTION, "预热")
        wl.close()

        # dry_run=True：触发也不迁移
        results = engine.scan_watchlist(db_path=wl_path,
                                        catalyst_scores={"600519": 0.9},
                                        dry_run=True)
        assert results[0].code == "600519"
        wl = WatchlistDB(wl_path)
        assert wl.get("600519").status == ATTENTION
        wl.close()

        # 正式运行：触发 → 买入区间
        engine.scan_watchlist(db_path=wl_path, catalyst_scores={"600519": 0.9})
        wl = WatchlistDB(wl_path)
        assert wl.get("600519").status == BUY_ZONE
        wl.close()
