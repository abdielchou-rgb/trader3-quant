"""
模块一：PIT（Point-in-Time）双时间戳存储与检索 — 回归测试
核心契约：
  1. schema：financial_pit 表（symbol/field_name/report_date/publish_timestamp/value/is_restatement）
     + (symbol, field_name, publish_timestamp) 索引
  2. get_asof_cross_section 严格按 publish_timestamp <= asof 切片：
     绝不返回披露时刻晚于 asof 的记录（未来时序穿越=0）
  3. 同一 (symbol, field) 多版本（首次披露 vs 更正公告）→ asof 取最新披露的最新报告期
  4. is_restatement=1 的更正值在更正披露时刻后才可见
  5. 财报期更早但披露更晚的记录（4月30日披露去年年报）在 4月30日 前不可见
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.data.pit_loader import PITFundamentalLoader, ensure_pit_schema  # noqa: E402

DB = ":memory:"


def _ts(s: str) -> int:
    """'2024-04-30 18:00' 或 '2025-01-01' → unix 毫秒（本地时区）"""
    fmt = "%Y-%m-%d %H:%M" if " " in s else "%Y-%m-%d"
    dt = datetime.strptime(s, fmt)
    return int(dt.timestamp() * 1000)


@pytest.fixture
def loader():
    ldr = PITFundamentalLoader(DB)
    ensure_pit_schema(ldr.conn)
    return ldr


def _seed_standard_case(pit: PITFundamentalLoader):
    """标准案例：茅台 epsTTM 三个版本。
    - 2023 三季报（report 2023-09-30）：2023-10-27 披露 eps=25.0
    - 2023 年报（report 2023-12-31）：2024-04-30 披露 eps=33.0 ← 4月末才可见
    - 2023 年报更正（restatement）：2024-06-15 披露 eps=33.5
    另一只票 五粮液 2023 年报：2024-04-25 披露 eps=8.0
    """
    rows = [
        ("600519", "epsTTM", "2023-09-30", _ts("2023-10-27 18:00"), 25.0, 0),
        ("600519", "epsTTM", "2023-12-31", _ts("2024-04-30 18:00"), 33.0, 0),
        ("600519", "epsTTM", "2023-12-31", _ts("2024-06-15 18:00"), 33.5, 1),
        ("000858", "epsTTM", "2023-12-31", _ts("2024-04-25 18:00"), 8.0, 0),
    ]
    pit.insert_many(rows)


def test_pit_schema_and_index(loader):
    cur = loader.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='financial_pit'")
    assert cur.fetchone() is not None
    idx = loader.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_pit_lookup'")
    assert idx.fetchone() is not None


def test_no_lookahead_annual_report_after_april(loader):
    """4月30日披露的年报在 4月15日 asof 不可见（只见三季报）。"""
    _seed_standard_case(loader)
    cs = loader.get_asof_cross_section(
        ["600519", "000858"], "epsTTM", _ts("2024-04-15 00:00"))
    # 茅台：年报未披露 → 只有三季报 25.0
    assert cs["600519"] == pytest.approx(25.0)
    # 五粮液：年报已披露（4月25日 > 4月15日？否——4月25日 > 4月15日，不可见）→ 无值
    assert "000858" not in cs or cs.get("000858") is None


def test_asof_after_disclosure_sees_annual(loader):
    """5月1日 asof：年报可见（茅台 33.0、五粮液 8.0），更正版不可见。"""
    _seed_standard_case(loader)
    cs = loader.get_asof_cross_section(
        ["600519", "000858"], "epsTTM", _ts("2024-05-01 00:00"))
    assert cs["600519"] == pytest.approx(33.0)
    assert cs["000858"] == pytest.approx(8.0)


def test_restatement_visible_only_after_correction(loader):
    """6月15日更正公告：6月20日 asof 取更正值 33.5；6月1日仍取首次值 33.0。"""
    _seed_standard_case(loader)
    before = loader.get_asof_cross_section(
        ["600519"], "epsTTM", _ts("2024-06-01 00:00"))
    after = loader.get_asof_cross_section(
        ["600519"], "epsTTM", _ts("2024-06-20 00:00"))
    assert before["600519"] == pytest.approx(33.0)
    assert after["600519"] == pytest.approx(33.5)


def test_exact_timestamp_inclusive(loader):
    """asof 恰等于披露时刻 → 可见（边界含）。"""
    _seed_standard_case(loader)
    cs = loader.get_asof_cross_section(
        ["600519"], "epsTTM", _ts("2024-04-30 18:00"))
    assert cs["600519"] == pytest.approx(33.0)


def test_missing_symbol_absent(loader):
    _seed_standard_case(loader)
    cs = loader.get_asof_cross_section(["999999"], "epsTTM", _ts("2025-01-01"))
    assert "999999" not in cs


def test_bulk_backfill_from_legacy_financials(loader, tmp_path):
    """旧 financials.db（无公告日列）→ 按 A股法定披露窗口回填 publish_timestamp。
    法定上限：一季报 4/30、半年报 8/31、三季报 10/31、年报次年 4/30。
    回填必须保守（上限=法定最晚日，宁可晚可见，不可早可见）。
    """
    import sqlite3
    legacy = sqlite3.connect(str(tmp_path / "legacy.db"))
    legacy.executescript("""
        CREATE TABLE financials (code TEXT, quarter TEXT, table_name TEXT,
                                field TEXT, value REAL, source TEXT);
        INSERT INTO financials VALUES ('600519', '2023-09-30', 'profit', 'epsTTM', 25.0, 'em');
        INSERT INTO financials VALUES ('600519', '2023-12-31', 'profit', 'epsTTM', 33.0, 'em');
        INSERT INTO financials VALUES ('600519', '2024-03-31', 'profit', 'epsTTM', 9.0, 'em');
    """)
    legacy.commit()
    n = loader.backfill_from_legacy(legacy, conservative_lag_days=0)
    assert n >= 3
    # 年报 2023-12-31 → 法定最晚 2024-04-30；asof 2024-04-15 不可见、5月1日可见
    cs_apr = loader.get_asof_cross_section(
        ["600519"], "epsTTM", _ts("2024-04-15 00:00"))
    # 4/15 应见 2023-09-30 三季报（10/31 前披露）与…… 一季报 2024-03-31 法定 4/30，
    # 4/15 不可见 → 应取 25.0
    assert cs_apr["600519"] == pytest.approx(25.0)
    cs_may = loader.get_asof_cross_section(
        ["600519"], "epsTTM", _ts("2024-05-01 00:00"))
    # 5/1：一季报（法定4/30）与年报（法定4/30）都已"最晚可见"，取报告期最新 2024-03-31 → 9.0
    assert cs_may["600519"] == pytest.approx(9.0)
