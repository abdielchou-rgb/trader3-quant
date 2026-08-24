"""
3号交易员 v2.0 — 多源采集适配器 (collector_sources)

按"通道可用性"分层：
    A 层（沙箱可直连）:  新浪财经 / 东财公告 / 东财涨停池 / akshare 新闻+龙虎榜 / Sina 行情
    B 层（需本机环境）:  雪球(需cookie) / 知乎(需认证) / DDG搜索(bot验证) / 路透-彭博-高盛(需代理)

所有适配器统一接口：
    fetch(keywords: str, limit: int) -> List[SourceItem]
    SourceItem(title, content, source_name, url, ts)

设计原则：
- 每个源一个类，可独立启用/禁用
- A 层自动探测并启用；B 层标记 availability="local"，本机运行时可接入
- 采集结果统一交 CatalystScorer 评分 + 大模型解读层(interpret.py)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("trader3.v2.sources")


@dataclass
class SourceItem:
    """单条采集结果"""
    title: str
    content: str = ""
    source_name: str = ""
    url: str = ""
    ts: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title, "content": self.content,
            "source_name": self.source_name, "url": self.url, "ts": self.ts,
        }


class BaseSource:
    """采集源基类"""
    source_name: str = ""
    availability: str = "local"   # sandbox / local / manual

    def fetch(self, keywords: str, limit: int = 10) -> List[SourceItem]:
        raise NotImplementedError


# ──────────────────────────────────────────────
# A 层：沙箱可直连
# ──────────────────────────────────────────────

class SinaNewsSource(BaseSource):
    """新浪财经 A股要闻（已验证沙箱可直连）"""

    source_name = "sina_news"
    availability = "sandbox"

    def __init__(self, http=None):
        self._http = http

    def fetch(self, keywords: str = "", limit: int = 10) -> List[SourceItem]:
        import requests
        # 新浪财经滚动要闻：pageid=153 lid=2516 是国内财经
        url = ("https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2516"
               "&k=&num=%d&page=1" % (limit + 5))
        r = (self._http or requests).get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        data = r.json().get("result", {}) or {}
        items = []
        for row in (data.get("data") or [])[:limit + 10]:
            title = row.get("title", "")
            intro = (row.get("intro") or "")
            if keywords and keywords not in title and keywords not in intro:
                continue
            items.append(SourceItem(
                title=title,
                content=("" + intro)[:200],
                source_name=self.source_name,
                url=row.get("url", ""),
                ts=_to_iso_ts(row.get("ctime", "")),   # 新浪 unix 秒 → ISO
            ))
        return items


class EastmoneyAnnouncementSource(BaseSource):
    """东财个股公告（已验证沙箱可直连）"""

    source_name = "announcement"
    availability = "sandbox"

    def __init__(self, http=None):
        self._http = http

    def fetch(self, code: str, limit: int = 10) -> List[SourceItem]:
        import requests
        url = ("https://np-anotice-stock.eastmoney.com/api/security/ann"
               f"?sr=-1&page_size={limit}&page_index=1&ann_type=A&client_source=web&stock_list={code}")
        r = (self._http or requests).get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        data = r.json().get("data") or {}
        items = []
        for item in (data.get("list") or []):
            title = item.get("title", "")
            if not title:
                continue
            items.append(SourceItem(
                title=title[:200], content="", source_name=self.source_name,
                url="https://data.eastmoney.com/notices/detail/%s/%s.html" % (
                    code, item.get("art_code", "")),
                ts=(item.get("notice_date") or "")[:10],
            ))
        return items


class EastmoneyZTPoolSource(BaseSource):
    """东财涨停池（已验证沙箱可直连）"""

    source_name = "zt_pool"
    availability = "sandbox"

    def fetch(self, keywords: str = "", limit: int = 30) -> List[SourceItem]:
        import requests
        today = datetime.now().strftime("%Y%m%d")
        url = ("https://push2ex.eastmoney.com/getTopicZTPool?"
               "ut=7eea3edcaed734bea9cbfc24409ed989&dpt=wz.ztzt"
               f"&Pageindex=0&pagesize={limit}&sort=fbt:asc&date={today}")
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        pool = (r.json().get("data") or {}).get("pool") or []
        items = []
        for item in pool:
            name = str(item.get("n", ""))
            code = str(item.get("c", ""))
            zdp = float(item.get("zdp") or 0)
            items.append(SourceItem(
                title=f"涨停: {name}({code}) {zdp:.1f}%",
                content=f"{name} 涨幅 {zdp:.1f}% 涨停",
                source_name=self.source_name, url="", ts=today,
            ))
        return items


class SinaQuoteSource(BaseSource):
    """Sina 实时行情（已验证沙箱可直连）"""

    source_name = "sina_quote"
    availability = "sandbox"

    def fetch(self, codes: List[str], limit: int = 10) -> List[SourceItem]:
        import requests
        if isinstance(codes, str):
            codes = [codes]
        items = []
        for code in codes:
            url = f"https://hq.sinajs.cn/list={code}"
            r = requests.get(url, headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://finance.sina.com.cn",
            }, timeout=8)
            # 格式：var hq_str_sh600519="名,今开,昨收,现价,最高,最低,...,成交量,成交额"
            m = re.search(r'="([^"]*)"', r.text)
            if m and m.group(1):
                parts = m.group(1).split(",")
                if len(parts) >= 4 and parts[0]:
                    # 高/低/量等高位字段仅在 parts 长度足够时取，短行填空防 IndexError
                    high = parts[4] if len(parts) > 4 else ""
                    low = parts[5] if len(parts) > 5 else ""
                    vol = parts[8] if len(parts) > 8 else "0"
                    items.append(SourceItem(
                        title=f"行情: {parts[0]}({code})",
                        content=f"开盘{parts[1]} 昨收{parts[2]} 现价{parts[3]} "
                                f"高{high} 低{low} 量{vol}",
                        source_name=self.source_name,
                        url=f"https://finance.sina.com.cn/realstock/company/{code}/nc.shtml",
                        ts=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    ))
        return items


# ──────────────────────────────────────────────
# B 层：需本机环境（读环境变量注入凭据，真实接口实现）
# ──────────────────────────────────────────────

def _get_cookie(name: str) -> str:
    """从环境变量读 cookie/token（本机配置，沙箱无）"""
    import os
    return os.environ.get(name, "")


def _to_iso_ts(raw) -> str:
    """时间戳宽容转 ISO：unix 秒/毫秒 → 'YYYY-MM-DD HH:MM:SS'，其他原样返回"""
    try:
        s = str(raw).strip()
        if s.isdigit():
            v = int(s)
            if v > 10**12:
                v //= 1000
            return datetime.fromtimestamp(v).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    return str(raw or "")


def _fetch_url(url: str, cookie: str = "", referer: str = "", timeout: int = 12):
    """统一请求（带 UA + 可选 cookie 合并进 headers + referer）。

    Cookie 显式放进 headers 而非 requests 的 cookies 参数，
    避免重定向/跨域时会话 cookie 丢失。
    """
    import requests
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
    if referer:
        headers["Referer"] = referer
    if cookie:
        headers["Cookie"] = cookie
    return requests.get(url, headers=headers, timeout=timeout)


class XueqiuSource(BaseSource):
    """雪球（自媒体热度）— 本机注入 cookie 后走真实搜索/热帖接口"""

    source_name = "xueqiu"
    availability = "local"

    def fetch(self, keywords: str = "", limit: int = 10) -> List[SourceItem]:
        cookie = _get_cookie("XUEQIU_COOKIE")
        if not cookie:
            logger.warning("[sources] 雪球需 XUEQIU_COOKIE 环境变量（本机浏览器复制）。"
                           "示例：export XUEQIU_COOKIE='xq_a_token=...'")
            return []
        items = []
        # 1. 搜索用户讨论（q=关键词）
        try:
            url = f"https://xueqiu.com/query/v1/search/status.json?q={keywords}&count={limit}&page=1"
            r = _fetch_url(url, cookie=cookie, referer="https://xueqiu.com/")
            data = r.json() or {}
            for item in (data.get("list") or [])[:limit]:
                title = (item.get("title") or "").replace("<em>", "").replace("</em>", "")
                desc = (item.get("description") or "")[:200]
                items.append(SourceItem(
                    title=title[:120], content=desc,
                    source_name=self.source_name,
                    url=f"https://xueqiu.com/{item.get('id')}",
                    ts=(item.get("created_at") or "")[:10],
                ))
        except Exception as e:
            logger.debug("[sources] xueqiu search fail: %s", str(e)[:60])
        # 若搜索无结果，退热帖热榜（免搜索接口）
        if not items:
            try:
                url = "https://xueqiu.com/statuses/hot/listV2.json?since_id=-1&max_id=-1&size=%d" % limit
                r = _fetch_url(url, cookie=cookie, referer="https://xueqiu.com/")
                data = r.json() or {}
                for item in (data.get("items") or [])[:limit]:
                    title = (item.get("title") or "")
                    if keywords and keywords not in title:
                        continue
                    items.append(SourceItem(
                        title=title[:120],
                        content=(item.get("text") or "")[:200],
                        source_name=self.source_name,
                        url=f"https://xueqiu.com/{item.get('id')}",
                        ts=(item.get("created_at") or "")[:10],
                    ))
            except Exception as e:
                logger.debug("[sources] xueqiu hot fail: %s", str(e)[:60])
        return items


class ZhihuSource(BaseSource):
    """知乎（深度讨论）— 本机注入 token 走热榜接口；无 token 时退 DDG 搜索"""

    source_name = "zhihu"
    availability = "local"

    def fetch(self, keywords: str = "", limit: int = 10) -> List[SourceItem]:
        token = _get_cookie("ZHIHU_TOKEN")
        items = []
        # 1. 有 token → 知乎热榜真实接口
        if token:
            try:
                url = f"https://www.zhihu.com/api/v3/feed/topstory/hot-lists/total?limit={limit}"
                r = _fetch_url(url, cookie=token)
                data = r.json() or {}
                for item in (data.get("data") or [])[:limit]:
                    target = item.get("target") or {}
                    title = target.get("title") or ""
                    if keywords and keywords not in title:
                        continue
                    items.append(SourceItem(
                        title=title[:120],
                        content=(target.get("excerpt") or "")[:200],
                        source_name=self.source_name,
                        url=target.get("url", ""),
                        ts=(item.get("detail_text") or ""),
                    ))
            except Exception as e:
                logger.debug("[sources] zhihu API fail: %s", str(e)[:60])
        # 2. 无 token 或 API 失败 → DDG 搜索兜底（本机可过反爬）
        if not items:
            try:
                from trader3.v2.search import ddg_websearch
                results = ddg_websearch(f"site:zhihu.com {keywords}", limit=limit)
                for title, url in results:
                    items.append(SourceItem(
                        title=title[:120], content="", source_name=self.source_name,
                        url=url, ts="",
                    ))
            except Exception as e:
                logger.debug("[sources] zhihu ddg fail: %s", str(e)[:60])
        return items


class GlobalReutersSource(BaseSource):
    """路透/彭博/高盛（外网消息）— 本机走 RSS 聚合 + DDG 新闻搜索"""

    source_name = "global_news"
    availability = "local"

    RSS_FEEDS = [
        "https://www.reuters.com/rssFeed/worldNews",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",   # 华尔街日报市场
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC 财经
        "https://feeds.content.dowjones.io/public/rss/mw_topstories",  # MarketWatch
    ]

    def fetch(self, keywords: str = "", limit: int = 10) -> List[SourceItem]:
        items = []
        import xml.etree.ElementTree as ET
        for feed_url in self.RSS_FEEDS:
            try:
                r = _fetch_url(feed_url, timeout=8)
                if r.status_code != 200:
                    continue
                root = ET.fromstring(r.text)
                for item in root.iter("item"):
                    title = (item.findtext("title") or "")
                    link = (item.findtext("link") or "")
                    desc = (item.findtext("description") or "")[:200]
                    if keywords and keywords.lower() not in title.lower() \
                            and keywords.lower() not in desc.lower():
                        continue
                    items.append(SourceItem(
                        title=title[:120], content=desc,
                        source_name=self.source_name, url=link, ts="",
                    ))
                    if len(items) >= limit:
                        return items
            except Exception as e:
                logger.debug("[sources] global feed %s fail: %s", feed_url, str(e)[:50])
        # RSS 不全时 → DDG 新闻搜索兜底
        if not items:
            try:
                from trader3.v2.search import ddg_websearch
                results = ddg_websearch(f"{keywords} Reuters OR Bloomberg OR 'Goldman Sachs'",
                                        limit=limit)
                for title, url in results:
                    items.append(SourceItem(
                        title=title[:120], content="", source_name=self.source_name,
                        url=url, ts="",
                    ))
            except Exception as e:
                logger.debug("[sources] global ddg fail: %s", str(e)[:60])
        return items


# ──────────────────────────────────────────────
# 注册表
# ──────────────────────────────────────────────

ALL_SOURCES: List[BaseSource] = [
    SinaNewsSource(),
    EastmoneyAnnouncementSource(),
    EastmoneyZTPoolSource(),
    SinaQuoteSource(),
    XueqiuSource(),
    ZhihuSource(),
    GlobalReutersSource(),
]


def get_sources(level: str = "all") -> List[BaseSource]:
    """按通道级别取源。level: all / sandbox / local"""
    if level == "sandbox":
        return [s for s in ALL_SOURCES if s.availability == "sandbox"]
    if level == "local":
        return [s for s in ALL_SOURCES if s.availability == "local"]
    return ALL_SOURCES


def source_status() -> List[dict]:
    """源可用性概览（供文档/UI）"""
    return [{"name": s.source_name, "availability": s.availability} for s in ALL_SOURCES]