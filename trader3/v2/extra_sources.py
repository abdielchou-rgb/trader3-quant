"""
3号交易员 v2.0 — 扩展数据源 (extra_sources)

已验证可用的 A股补充数据源（沙箱 2026-08-23 实测）：
  1. 股本/市值/名称   — 东财 push2 个股接口（无 key，免爬）✅
  2. 融资融券(沪)     — akshare stock_margin_detail_sse ✅
  3. 股东户数         — akshare stock_zh_a_gdhs ✅
  4. 北向个股持股     — akshare stock_hsgt_individual_em ✅
  5. 实时行情         — 东财/腾讯/新浪 三通道兜底 ✅

每个源统一出口 fetch(keywords/code, limit) → List[SourceItem]
入库事件库 events.db → 三因子催化评分 → 日常提醒。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import List, Optional

os.environ.setdefault("AKSHARE_NO_PROXY", "1")

from trader3.v2.sources import SourceItem

logger = logging.getLogger("trader3.v2.extra_sources")


class FundingMarginSource:
    """融资融券（沪市个股）— 资金杠杆变化信号"""

    source_name = "margin"
    availability = "sandbox"

    def fetch(self, code: str = "600519", limit: int = 5) -> List[SourceItem]:
        try:
            import akshare as ak
            df = ak.stock_margin_detail_sse(date=datetime.now().strftime("%Y%m%d"))
            if df is None or df.empty:
                try:
                    df = ak.stock_margin_detail_sse(date="20260821")
                except Exception:
                    return []
            if df is None or df.empty:
                return []
            code_clean = code.zfill(6)
            sub = df[df["标的证券代码"].astype(str).str.zfill(6) == code_clean]
            if sub.empty:
                return []
            row = sub.iloc[-1]
            items = []
            for i, (_, r) in enumerate(sub.tail(limit).iterrows()):
                title = f"融资融券 {code_clean}: 融资余额 {r.get('融资余额', 0)}"
                items.append(SourceItem(
                    title=title, content=f"融资买入 {r.get('融资买入额',0)} 偿还 {r.get('融资偿还额',0)}",
                    source_name=self.source_name, url="",
                    ts=str(r.get("信用交易日期", ""))[:10],
                ))
            return items
        except Exception as e:
            logger.debug("[extra] margin %s fail: %s", code, str(e)[:60])
            return []


class ShareholderCountSource:
    """股东户数（新浪口径）— 筹码集中度信号（户数减少=筹码集中）"""

    source_name = "shareholder_count"
    availability = "sandbox"

    def fetch(self, code: str = "600519", limit: int = 5) -> List[SourceItem]:
        try:
            import akshare as ak
            symbol = code if code.startswith(("sh", "sz", "bj")) else code.strip()
            df = ak.stock_zh_a_gdhs(symbol=symbol)
            if df is None or df.empty:
                return []
            items = []
            for _, r in df.head(limit).iterrows():
                t = str(r.get("股东户数统计截止日", ""))[:10]
                cur = r.get("股东户数-本次", 0)
                prev = r.get("股东户数-上次", 0)
                delta = r.get("股东户数-增减比例", 0)
                title = f"股东户数 {code}: {cur}（{delta:+.2f}%）"
                items.append(SourceItem(
                    title=title,
                    content=f"上次 {prev}，截止 {t}",
                    source_name=self.source_name, url="", ts=t,
                ))
            return items
        except Exception as e:
            logger.debug("[extra] gdhs %s fail: %s", code, str(e)[:60])
            return []


class NorthboundSource:
    """北向个股持股 — 外资动向信号"""

    source_name = "northbound"
    availability = "sandbox"

    def fetch(self, code: str = "600519", limit: int = 5) -> List[SourceItem]:
        try:
            import akshare as ak
            df = ak.stock_hsgt_individual_em(symbol=code)
            if df is None or df.empty:
                return []
            items = []
            for _, r in df.head(limit).iterrows():
                d = str(r.get("持股日期", ""))[:10]
                shares = r.get("持股数量", 0)
                pct = r.get("持股数量占A股百分比", 0)
                title = f"北向持股 {code}: {pct:.2f}%"
                items.append(SourceItem(
                    title=title, content=f"持股 {shares} 股，截止 {d}",
                    source_name=self.source_name, url="", ts=d,
                ))
            return items
        except Exception as e:
            logger.debug("[extra] hsgt %s fail: %s", code, str(e)[:60])
            return []


ALL_EXTRA = [FundingMarginSource(), ShareholderCountSource(), NorthboundSource()]


def extra_status() -> List[dict]:
    return [{"name": s.source_name, "availability": s.availability} for s in ALL_EXTRA]