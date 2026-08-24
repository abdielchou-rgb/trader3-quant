"""3号交易员 v2.1 — 财务事件表（防前视，fooltrader 模式）

核心：建「报告期→公告日」映射。读财务必须按公告日过滤——
t 日只能用「公告日 ≤ t」的最新财务，不能用报告期当天假设已有数据。

A股财报公告规律（用于合成公告日兜底）：
- 年报（12-31）→ 次年 4 月底前披露
- 一季报（03-31）→ 4 月底前
- 半年报（06-30）→ 8 月底前
- 三季报（09-30）→ 10 月底前

真实公告日（akshare 财务披露接口）优先，规律兜底。

用法：
    from trader3.v2.announcement_calendar import get_calendar
    cal = get_calendar()
    q = cal.latest_asof("600519", "2024-06-15")
    fin = cal.financials_asof("600519", "2024-06-15")
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# 报告期末月 → 披露截止月（A股规律）
_QUARTER_MONTH = {
    "12-31": 4,  # 年报 → 次年 4 月
    "03-31": 4,  # 一季报 → 4 月
    "06-30": 8,  # 半年报 → 8 月
    "09-30": 10, # 三季报 → 10 月
}


def _announce_date(quarter: str) -> str:
    """报告期 → 公告截止日（季度末后推月）"""
    q = quarter[:10]
    if len(q) < 10:
        return q
    y = int(q[:4])
    md = q[5:]
    month = _QUARTER_MONTH.get(md, 6)
    year = y + 1 if md == "12-31" else y
    return f"{year}-{month:02d}-30"


def _norm_code(code: str) -> str:
    c = str(code).replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
    return c.zfill(6)[:6]


class AnnouncementCalendar:
    """报告期→公告日 映射 + 防前视财务快照"""

    def __init__(self):
        from trader3.financials_provider import FinancialsProvider
        self.fp = FinancialsProvider()

    def announce_date(self, code: str, quarter: str) -> str:
        """报告期 quarter → 公告日（规律兜底；真实值可由 akshare 补充）"""
        return _announce_date(quarter)

    def latest_asof(self, code: str, asof_date: str) -> Optional[str]:
        """asof 日前最新的可用 quarter（公告日 ≤ asof）"""
        try:
            quarters = self.fp.available_quarters(code)
        except Exception:
            return None
        usable = []
        for q in quarters:
            ann = self.announce_date(code, q)
            if ann <= asof_date:
                usable.append((ann, q))
        if not usable:
            return None
        usable.sort()
        return usable[-1][1]

    def financials_asof(self, code: str, asof_date: str) -> dict:
        """按公告日对齐的财务（防前视）"""
        q = self.latest_asof(code, asof_date)
        if not q:
            return {}
        conn = self.fp._connect()
        cur = conn.execute(
            "SELECT table_name, field, value FROM financials WHERE code=? AND quarter=?",
            (_norm_code(code), q),
        )
        out = {}
        for row in cur.fetchall():
            out[f"{row[0]}.{row[1]}"] = row[2]
            out[row[1]] = row[2]
        out["quarter"] = q
        out["announce_date"] = self.announce_date(code, q)
        return out

    def summary(self) -> dict:
        return {"module": "announcement_calendar", "mode": "fooltrader-防前视"}


def get_calendar() -> AnnouncementCalendar:
    return AnnouncementCalendar()
