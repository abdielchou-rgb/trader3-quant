"""
PIT（Point-in-Time）双时间戳基本面存储与检索。

痛点：财务数据若只记 report_date（会计期截止日），回测在 4 月中旬就能"看到"
4 月底才披露的年报 —— 未来时序穿越。机构级解法是双时间戳：
  - report_date        会计期截止（信息归属期）
  - publish_timestamp  交易所首次披露时刻（信息可用时刻）

一切 asof 检索只认 publish_timestamp <= asof；宁可晚可见（保守回填），
不可早可见（穿越）。

旧 financials.db 无公告日列时的回填策略：按 A 股法定披露截止日
（一季报 4/30、半年报 8/31、三季报 10/31、年报次年 4/30）作为
publish_timestamp 的保守上界 —— 真实披露可能更早，但按最晚日回填
保证零穿越（代价是部分时段取不到本可取到的值，属保守偏差）。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

# A股法定披露截止日：报告期月末 → （截止月, 截止日）
_LEGAL_DEADLINE = {
    "03-31": (4, 30),    # 一季报：次年 4/30 前
    "06-30": (8, 31),    # 半年报：8/31 前
    "09-30": (10, 31),   # 三季报：10/31 前
    "12-31": (4, 30),    # 年报：次年 4/30 前
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS financial_pit (
    symbol TEXT NOT NULL,
    field_name TEXT NOT NULL,
    report_date TEXT NOT NULL,
    publish_timestamp INTEGER NOT NULL,
    value REAL NOT NULL,
    is_restatement INTEGER DEFAULT 0,
    PRIMARY KEY (symbol, field_name, report_date, publish_timestamp)
);
CREATE INDEX IF NOT EXISTS idx_pit_lookup
ON financial_pit (symbol, field_name, publish_timestamp);
"""


def ensure_pit_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def _ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000
               if dt.tzinfo is None else dt.timestamp() * 1000)


def legal_publish_upper_bound(report_date: str) -> int:
    """报告期 → 法定最晚披露日（截止日 00:00 本地时的 unix 毫秒）。

    年报（12-31）：次年 4/30；一季报（03-31）：同年 4/30；
    半年报（06-30）：同年 8/31；三季报（09-30）：同年 10/31。
    """
    year = int(report_date[:4])
    md = report_date[5:10]
    if md not in _LEGAL_DEADLINE:
        raise ValueError(f"非法报告期 {report_date!r}（需 03-31/06-30/09-30/12-31）")
    m, d = _LEGAL_DEADLINE[md]
    deadline_year = year + 1 if md == "12-31" else year
    return _ms(datetime(deadline_year, m, d))


class PITFundamentalLoader:
    """零未来函数的 asof 截面检索器。"""

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    # ── 写入 ──────────────────────────────────────

    def insert_many(
        self,
        rows: Iterable[tuple[str, str, str, int, float, int]],
    ) -> int:
        """批量写入 (symbol, field_name, report_date, publish_ms, value, is_restatement)。

        主键冲突（同披露时刻重复披露）→ REPLACE 语义（后写覆盖）。
        """
        ensure_pit_schema(self.conn)
        cur = self.conn.executemany(
            """
            INSERT OR REPLACE INTO financial_pit
            (symbol, field_name, report_date, publish_timestamp, value, is_restatement)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    def backfill_from_legacy(
        self,
        legacy_conn: sqlite3.Connection,
        conservative_lag_days: int = 0,
    ) -> int:
        """旧 financials.db（code/quarter/table_name/field/value）→ PIT 回填。

        publish_timestamp 取法定最晚披露日 + conservative_lag_days（保守延后）。
        旧库无公告日信息，无法区分首次/更正 → 全部记 is_restatement=0，
        并接受同 (symbol, field, report_date) 单版本（有真实公告日数据源后应重灌）。
        """
        ensure_pit_schema(self.conn)
        cur = legacy_conn.execute(
            "SELECT DISTINCT code, quarter, field, value FROM financials"
        )
        rows: list[tuple[str, str, str, int, float, int]] = []
        for code, quarter, field, value in cur.fetchall():
            if value is None:
                continue
            try:
                pub_ms = legal_publish_upper_bound(str(quarter))
            except ValueError:
                continue
            pub_ms += conservative_lag_days * 86_400_000
            rows.append((str(code), str(field), str(quarter), pub_ms,
                         float(value), 0))
        return self.insert_many(rows)

    # ── 检索 ──────────────────────────────────────

    def get_asof_cross_section(
        self,
        symbols: list[str],
        field: str,
        asof_timestamp: int,
    ) -> dict[str, float]:
        """严格 asof 切片：每标的最新的 (report_date, publish_timestamp) 已披露值。

        排序规则：先取披露时刻 <= asof 的记录，再按 report_date 降序、
        publish_timestamp 降序取第一名 —— 保证同报告期的更正版覆盖首次版，
        且跨报告期时新报告期优先（更近的会计期信息）。
        """
        if not symbols:
            return {}
        placeholders = ",".join(["?"] * len(symbols))
        query = f"""
        WITH RankedFinancials AS (
            SELECT symbol, value,
                   ROW_NUMBER() OVER (
                       PARTITION BY symbol
                       ORDER BY report_date DESC, publish_timestamp DESC
                   ) AS rank
            FROM financial_pit
            WHERE symbol IN ({placeholders})
              AND field_name = ?
              AND publish_timestamp <= ?
        )
        SELECT symbol, value FROM RankedFinancials WHERE rank = 1;
        """
        params: list[Any] = list(symbols) + [field, asof_timestamp]
        cur = self.conn.execute(query, params)
        return {str(r[0]): float(r[1]) for r in cur.fetchall()}

    # ── 审计 ──────────────────────────────────────

    def count_records(self) -> int:
        try:
            return int(
                self.conn.execute("SELECT COUNT(*) FROM financial_pit").fetchone()[0]
            )
        except sqlite3.OperationalError:
            return 0
