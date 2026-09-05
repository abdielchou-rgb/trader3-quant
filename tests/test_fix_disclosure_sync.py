"""
公告日历自动化（disclosure_sync）零联网测试：

1. sync 写入显式日历 JSON + 原子性（无 .tmp 残留）
2. financials_asof 显式日历优先于规律推断
3. 幂等覆盖（同 key 后者生效）
4. 主源故障降级（synced=0 不抛出）
5. codes 过滤生效
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import sqlite3

import pandas as pd
import pytest

_COLS = [
    "序号", "股票代码", "股票简称",
    "首次预约时间", "一次变更日期", "二次变更日期", "三次变更日期", "实际披露时间",
]


def _row(code: str, first: str, actual=None) -> dict:
    return {
        "序号": 1,
        "股票代码": code,
        "股票简称": "样例",
        "首次预约时间": pd.Timestamp(first),
        "一次变更日期": pd.NaT,
        "二次变更日期": pd.NaT,
        "三次变更日期": pd.NaT,
        "实际披露时间": pd.Timestamp(actual) if actual else pd.NaT,
    }


def _df(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_COLS)


class FakeFP:
    """替身 FinancialsProvider：临时 SQLite，含 financials 表"""

    def __init__(self, db):
        self.db = db
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS financials"
            "(code TEXT, quarter TEXT, table_name TEXT, field TEXT, value TEXT)"
        )
        conn.execute(
            "INSERT INTO financials VALUES ('600519','2026-06-30','income','eps','5.5')"
        )
        conn.commit()
        conn.close()

    def available_quarters(self, code):
        return ["2026-06-30"]

    def _connect(self):
        return sqlite3.connect(self.db)


@pytest.fixture
def cal_path(tmp_path, monkeypatch):
    p = tmp_path / "disclosure_calendar.json"
    monkeypatch.setenv("TRADER3_DISCLOSURE_CALENDAR", str(p))
    return p


@pytest.fixture
def ds():
    from trader3.v2 import disclosure_sync as mod
    return mod


def _make_calendar(tmp_path):
    from trader3.v2.announcement_calendar import AnnouncementCalendar
    cal = AnnouncementCalendar.__new__(AnnouncementCalendar)
    cal.fp = FakeFP(tmp_path / "fake_events.db")
    return cal


def test_sync_writes_calendar(ds, cal_path, monkeypatch):
    df = _df([
        _row("600519", "2026-07-16", actual="2026-07-15"),
        _row("000858", "2026-08-28"),
    ])
    monkeypatch.setattr(ds.ak, "stock_yysj_em", lambda **kw: df, raising=False)
    res = ds.sync_disclosure_dates("2026-06-30")
    assert res["synced"] == 2
    data = json.loads(cal_path.read_text(encoding="utf-8"))
    # 实际披露时间优先于首次预约时间
    assert data["600519"]["2026-06-30"] == "2026-07-15"
    assert data["000858"]["2026-06-30"] == "2026-08-28"
    # 原子性：无 .tmp 残留
    assert not list(cal_path.parent.glob("*.tmp"))


def test_financials_asof_prefers_explicit(ds, cal_path, tmp_path, monkeypatch):
    # 推断值 2026-08-30 > asof=2026-07-20（推断路径取不到）；显式值 07-16 可命中
    cal_path.write_text(
        json.dumps({"600519": {"2026-06-30": "2026-07-16"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(ds.ak, "stock_yysj_em",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("不应联网")),
                        raising=False)
    from trader3.v2.announcement_calendar import AnnouncementCalendar
    cal = AnnouncementCalendar.__new__(AnnouncementCalendar)
    cal.fp = FakeFP(tmp_path / "fake_events.db")

    fin = cal.financials_asof("600519", "2026-07-20")
    assert fin["quarter"] == "2026-06-30"
    assert fin["announce_date"] == "2026-07-16"
    assert fin.get("announce_source") == "explicit"

    # 无显式条目时回退推断并标记 inferred
    fin2 = cal.financials_asof("000000", "2026-09-01")
    assert fin2.get("announce_source") == "inferred"


def test_idempotent_overwrite(ds, cal_path, monkeypatch):
    df1 = _df([_row("600519", "2026-07-16")])
    df2 = _df([_row("600519", "2026-08-01")])
    monkeypatch.setattr(ds.ak, "stock_yysj_em", lambda **kw: df1, raising=False)
    r1 = ds.sync_disclosure_dates("2026-06-30")
    assert r1["synced"] == 1
    monkeypatch.setattr(ds.ak, "stock_yysj_em", lambda **kw: df2, raising=False)
    r2 = ds.sync_disclosure_dates("2026-06-30")
    assert r2["synced"] == 1
    data = json.loads(cal_path.read_text(encoding="utf-8"))
    assert data["600519"]["2026-06-30"] == "2026-08-01"
    assert not list(cal_path.parent.glob("*.tmp"))


def test_main_source_failure_degrades(ds, cal_path, monkeypatch):
    def boom(**kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(ds.ak, "stock_yysj_em", boom, raising=False)
    res = ds.sync_disclosure_dates("2026-06-30")
    assert res["synced"] == 0
    assert not cal_path.exists()


def test_quarter_filter_codes(ds, cal_path, monkeypatch):
    df = _df([_row("600519", "2026-07-16"), _row("000858", "2026-08-28")])
    monkeypatch.setattr(ds.ak, "stock_yysj_em", lambda **kw: df, raising=False)
    res = ds.sync_disclosure_dates("2026-06-30", codes=["600519"])
    assert res["synced"] == 1
    data = json.loads(cal_path.read_text(encoding="utf-8"))
    assert "600519" in data and "000858" not in data

    res2 = ds.sync_disclosure_dates("2026-06-30", codes=["000858.SZ"])
    data2 = json.loads(cal_path.read_text(encoding="utf-8"))
    assert res2["synced"] == 1
    assert "000858" in data2
