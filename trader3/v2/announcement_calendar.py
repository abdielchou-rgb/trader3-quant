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


# --- 显式公告日历（disclosure_sync 产出）模块级 loader ---
_EXPLICIT_CACHE: dict = {"key": None, "data": None}


def _load_explicit_calendar() -> Dict[str, Dict[str, str]]:
    """读取 data/disclosure_calendar.json（{code:{quarter:announce_date}}），mtime 缓存。

    disclosure_sync 缺失或文件缺失时返回 {}，不影响推断兜底。
    """
    try:
        from trader3.v2.disclosure_sync import default_calendar_path
        path = default_calendar_path()
    except ImportError:
        return {}
    key = (str(path), None)
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        pass
    if _EXPLICIT_CACHE["key"] == key:
        return _EXPLICIT_CACHE["data"]
    data: Dict[str, Dict[str, str]] = {}
    try:
        from trader3.v2.disclosure_sync import load_explicit_calendar
        loaded = load_explicit_calendar()
        if isinstance(loaded, dict):
            data = loaded
    except ImportError:
        pass
    _EXPLICIT_CACHE["key"] = key
    _EXPLICIT_CACHE["data"] = data
    return data


def _resolve_announce(code: str, quarter: str):
    """→ (公告日, 'explicit'|'inferred')；无显式值时规律推断兜底"""
    explicit = _load_explicit_calendar().get(_norm_code(code), {}).get(quarter[:10])
    if explicit:
        return explicit, "explicit"
    return _announce_date(quarter), "inferred"


class AnnouncementCalendar:
    """报告期→公告日 映射 + 防前视财务快照"""

    def __init__(self):
        from trader3.financials_provider import FinancialsProvider
        self.fp = FinancialsProvider()

    def announce_date(self, code: str, quarter: str) -> str:
        """报告期 quarter → 公告日（显式日历优先，规律兜底）"""
        return _resolve_announce(code, quarter)[0]

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
        ann, source = _resolve_announce(code, q)
        if source == "explicit":
            logger.info("financials_asof %s %s 命中显式披露日历（公告日=%s）", code, q, ann)
        out["quarter"] = q
        out["announce_date"] = ann
        out["announce_source"] = source
        return out

    def summary(self) -> dict:
        explicit = _load_explicit_calendar()
        return {
            "module": "announcement_calendar",
            "mode": "fooltrader-防前视",
            "explicit_codes": len(explicit),
        }


def get_calendar() -> AnnouncementCalendar:
    return AnnouncementCalendar()
