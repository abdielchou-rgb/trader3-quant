"""
3号交易员 v2.0 — 数据采集层（collector）

统一从免费源采集每日事件/数据，写入 events.db + 行情缓存：
  1. 个股新闻    — akshare stock_news_em（东财）→ 催化事件
  2. 龙虎榜      — akshare stock_lhb_detail_em → 游资/资金事件
  3. 资金流      — akshare stock_individual_fund_flow / 东财直连 → 资金事件
  4. 涨停池      — 东财 push2ex getTopicZTPool → 热点事件
  5. 个股公告    — 东财 np-anotice-stock → 公告事件

统一流程：拉取 → CatalystScorer 评分 → EventLibrary.upsert → 供三因子引擎读取催化强度
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta

os.environ.setdefault("AKSHARE_NO_PROXY", "1")

from trader3.v2.events import CatalystScorer, Event, get_event_library

logger = logging.getLogger("trader3.v2.collector")

# ── 限速 / 退避 / UA 轮换（模块常量） ──────────────────
SOURCE_DELAY_SECONDS = 1.5      # 每源串行请求间隔基准（秒）
SOURCE_JITTER_SECONDS = 0.5     # 间隔抖动 ±0.5s
FAILURE_BACKOFF_THRESHOLD = 3   # 单源连续失败≥3次触发指数退避并跳过本轮
BACKOFF_BASE_SECONDS = 2.0      # 退避时长 2^n 秒
BACKOFF_MAX_SECONDS = 60.0      # 退避上限

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
]

# 采集状态（记录每只股票上次采集时间，防重复）
COLLECT_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "collect_state.json",
)


def _load_state() -> dict:
    try:
        with open(COLLECT_STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        os.makedirs(os.path.dirname(COLLECT_STATE_FILE), exist_ok=True)
        with open(COLLECT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
    except Exception as e:
        logger.debug("[collector] state save fail: %s", e)


class DataCollector:
    """每日数据采集器（支持 LLM 解读层，规则评分作兜底）"""

    def __init__(self, skip_akshare: bool = False, use_llm_interpret: bool = True):
        self.scorer = CatalystScorer()
        self.use_llm_interpret = use_llm_interpret
        self._interpreter = None
        try:
            self._library = get_event_library()
        except Exception as e:
            logger.warning("[collector] events.db 初始化失败: %s", e)
            self._library = None

        self._HAS_AKSHARE = False
        if not skip_akshare:
            try:
                import akshare as ak
                self._ak = ak
                self._HAS_AKSHARE = True
            except ImportError:
                logger.warning("[collector] akshare 未安装，跳过 akshare 源")
        self._session = None
        self._ua_cycle = itertools.cycle(USER_AGENTS)
        self._fail_counts: dict[str, int] = {}   # 每源连续失败计数

    def _next_ua(self) -> str:
        """UA 小列表轮换（避免固定单一指纹）"""
        return next(self._ua_cycle)

    # ── 限速 / 退避 ──

    @staticmethod
    def _pace_delay() -> float:
        """每源间隔：SOURCE_DELAY_SECONDS ± 0.5s 抖动"""
        return max(0.0, SOURCE_DELAY_SECONDS + random.uniform(
            -SOURCE_JITTER_SECONDS, SOURCE_JITTER_SECONDS))

    def _pace(self) -> None:
        time.sleep(self._pace_delay())

    def _source_ready(self, source: str) -> bool:
        """单源连续失败≥阈值：指数退避（2^n 秒，上限60s）并跳过该源本轮"""
        fails = self._fail_counts.get(source, 0)
        if fails < FAILURE_BACKOFF_THRESHOLD:
            return True
        wait = min(BACKOFF_BASE_SECONDS ** fails, BACKOFF_MAX_SECONDS)
        logger.warning("[collector] 源 %s 连续失败 %d 次，退避 %.1fs 并跳过本轮",
                       source, fails, wait)
        time.sleep(wait)
        return False

    def _mark_success(self, source: str) -> None:
        self._fail_counts.pop(source, None)

    def _mark_failure(self, source: str) -> None:
        n = self._fail_counts.get(source, 0) + 1
        self._fail_counts[source] = n
        if n >= FAILURE_BACKOFF_THRESHOLD:
            wait = min(BACKOFF_BASE_SECONDS ** n, BACKOFF_MAX_SECONDS)
            logger.warning("[collector] 源 %s 已连续失败 %d 次，后续将退避 %.1fs",
                           source, n, wait)

    def _score_event(self, title: str, content: str = "") -> dict:
        """事件评分：LLM 解读优先，规则得分兜底"""
        text = f"{title} {content}".strip()
        if self.use_llm_interpret:
            try:
                if self._interpreter is None:
                    from trader3.v2.interpret import get_interpreter
                    self._interpreter = get_interpreter()
                interp = self._interpreter.interpret(text)
                s = {
                    "score": interp.catalyst_score,
                    "direction": interp.direction,
                    "category": interp.category,
                    "keywords_hit": [interp.reasoning[:30]] if interp.reasoning else [],
                }
                return s
            except Exception:
                pass
        return self.scorer.score(title, content)

    def _http(self):
        import requests
        if self._session is None:
            self._session = requests.Session()
        self._session.headers.update({"User-Agent": self._next_ua()})
        return self._session

    # ── 公共接口 ──

    def collect_stock(self, code: str, source: str = "all") -> int:
        """采集单只股票的事件，返回新增条数"""
        added = 0
        if not self._library:
            return 0
        if source in ("all", "news") and self._HAS_AKSHARE:
            added += self._collect_news(code)
        if source in ("all", "announcement"):
            added += self._collect_announcements(code)
        return added

    def collect_flow_events(self) -> int:
        """采集全市场资金/龙虎榜/涨停池事件（关注 TOP 异动）"""
        added = 0
        if self._HAS_AKSHARE:
            added += self._collect_lhb()
        added += self._collect_zt_pool()
        return added

    def scan_and_library(self, watchlist_codes: list[str]) -> dict[str, list[Event]]:
        """扫描自选股全部事件并入库，返回 每只股票最近事件"""
        for code in watchlist_codes:
            self.collect_stock(code)
        result = {}
        if self._library:
            for code in watchlist_codes:
                result[code] = self._library.get_recent(code, days=7)
        return result

    # ── 个股新闻 ──

    def _collect_news(self, code: str) -> int:
        """akshare stock_news_em → 新闻事件（失败时降级新浪直连）"""
        added = 0
        if self._HAS_AKSHARE and self._source_ready("akshare_news"):
            self._pace()
            try:
                df = self._ak.stock_news_em(symbol=code)
                self._mark_success("akshare_news")
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        title = str(row.get("新闻标题", ""))
                        content = str(row.get("新闻内容", ""))
                        pub = str(row.get("发布时间", ""))
                        if self._save_event(code, "news", title, content, pub[:16], "", ""):
                            added += 1
                    return added
            except Exception as e:
                self._mark_failure("akshare_news")
                logger.warning("[collector] news %s akshare 失败，降级新浪直连: %s", code, str(e)[:60])
        return added + self._collect_news_direct(code)

    def _collect_news_direct(self, code: str) -> int:
        """新浪个股新闻直连（免费稳定，绕开 akshare+pyarrow 正则兼容问题）"""
        added = 0
        if not self._source_ready("sina_news"):
            return added
        self._pace()
        try:
            prefix = "sh" if code.startswith(("6", "9")) else "sz"
            url = ("https://vip.stock.finance.sina.com.cn/corp/go.php/"
                   f"vCB_AllNewsStock/symbol/{prefix}{code}.phtml")
            r = self._http().get(url, timeout=12, headers={
                "User-Agent": self._next_ua(), "Referer": "https://finance.sina.com.cn"})
            r.encoding = "gbk"
            txt = r.text
            import re as _re
            m = _re.search(r"datelist(.*?)</table>", txt, _re.S)
            if not m:
                self._mark_success("sina_news")
                return 0
            pat = _re.compile(
                r"(\d{4}-\d{2}-\d{2})&nbsp;(\d{2}:\d{2})(?:&nbsp;)+"
                r"<a[^>]*?href='([^']+)'[^>]*>([^<]+)</a>")
            for dt, tm, link, title in pat.findall(m.group(1)):
                title = title.strip()
                if not title:
                    continue
                if self._save_event(code, "news", title, "", f"{dt} {tm}", link, ""):
                    added += 1
            self._mark_success("sina_news")
            return added
        except Exception as e:
            self._mark_failure("sina_news")
            logger.warning("[collector] news %s 新浪直连失败: %s", code, str(e)[:80])
            return 0

    def _save_event(self, code: str, source: str, title: str, content: str,
                    event_time: str, url: str, extra: str) -> bool:
        """评分 + 入库单条事件，返回是否新增"""
        try:
            s = self._score_event(title, content)
            ev = Event(
                code=code, source=source, title=title[:200], content=content[:500],
                event_time=event_time, url=url, catalyst_score=s["score"],
                direction=s["direction"], category=s["category"],
                keywords_hit=s["keywords_hit"],
                collected_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            return bool(self._library.upsert(ev))
        except Exception as e:
            logger.warning("[collector] save event %s/%s fail: %s", code, source, str(e)[:60])
            return False

    # ── 个股公告（东财直连） ──

    def _collect_announcements(self, code: str) -> int:
        """东财 np-anotice-stock → 公告事件"""
        if not self._source_ready("announcement"):
            return 0
        self._pace()
        try:
            url = ("https://np-anotice-stock.eastmoney.com/api/security/ann"
                   f"?sr=-1&page_size=10&page_index=1&ann_type=A"
                   f"&client_source=web&stock_list={code}")
            r = self._http().get(url, timeout=12)
            data = r.json().get("data") or {}
            items = (data.get("list") or [])[:10]
            added = 0
            for item in items:
                title = item.get("title", "") or item.get("art_title", "")
                if not title:
                    continue
                date = item.get("notice_date", "")[:10]
                url2 = item.get("art_code", "")
                s = self._score_event(title, "")
                ev = Event(
                    code=code, source="announcement", title=title[:200], content="",
                    event_time=date, url=url2 or "", catalyst_score=s["score"],
                    direction=s["direction"], category=s["category"],
                    keywords_hit=s["keywords_hit"],
                    collected_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                if self._library.upsert(ev):
                    added += 1
            self._mark_success("announcement")
            return added
        except Exception as e:
            self._mark_failure("announcement")
            logger.warning("[collector] announcements %s fail: %s", code, str(e)[:80])
            return 0

    # ── 龙虎榜（akshare） ──

    def _collect_lhb(self) -> int:
        """akshare stock_lhb_detail_em → 龙虎榜事件（只存上榜标的高强度信号）"""
        if not self._source_ready("lhb"):
            return 0
        self._pace()
        try:
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=3)).strftime("%Y%m%d")
            df = self._ak.stock_lhb_detail_em(start_date=start, end_date=end)
            if df is None or df.empty:
                return 0
            added = 0
            for _, row in df.iterrows():
                code = str(row.get("代码", ""))
                name = str(row.get("名称", ""))
                reason = str(row.get("上榜原因", ""))
                if not code.isdigit():
                    continue
                # 龙虎榜 = 强资金催化（净买入大或著名游资）
                s = self.scorer.score(f"龙虎榜 {reason}", "")
                s["score"] = max(s["score"], 0.62)  # 上榜至少中强催化
                ev = Event(
                    code=code, source="lhb", title=f"龙虎榜: {name} {reason[:60]}",
                    content=reason, event_time=datetime.now().strftime("%Y-%m-%d"),
                    url="", catalyst_score=s["score"], direction="positive" if s["score"] >= 0.6 else s["direction"],
                    category="资金/游资", keywords_hit=s["keywords_hit"],
                    collected_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                if self._library.upsert(ev):
                    added += 1
            self._mark_success("lhb")
            return added
        except Exception as e:
            self._mark_failure("lhb")
            logger.warning("[collector] lhb fail: %s", str(e)[:80])
            return 0

    # ── 涨停池（东财直连） ──

    def _collect_zt_pool(self) -> int:
        """东财 push2ex getTopicZTPool → 涨停热点事件（全市场）"""
        if not self._source_ready("zt_pool"):
            return 0
        self._pace()
        try:
            today = datetime.now().strftime("%Y%m%d")
            url = ("https://push2ex.eastmoney.com/getTopicZTPool?"
                   "ut=7eea3edcaed734bea9cbfc24409ed989&dpt=wz.ztzt"
                   f"&Pageindex=0&pagesize=50&sort=fbt:asc&date={today}")
            r = self._http().get(url, timeout=12)
            data = (r.json().get("data") or {})
            pool = data.get("pool") or []
            added = 0
            for item in pool[:30]:
                code = str(item.get("c", ""))
                name = str(item.get("n", ""))
                zdp = item.get("zdp", 0)
                if not code.isdigit():
                    continue
                ev = Event(
                    code=code, source="zt_pool", title=f"涨停: {name}({zdp:.1f}%)",
                    content=f"涨停池 {name} 涨幅 {zdp:.1f}%",
                    event_time=datetime.now().strftime("%Y-%m-%d"),
                    url="", catalyst_score=0.8, direction="positive",
                    category="资金/游资", keywords_hit=["涨停", "热点"],
                    collected_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                if self._library.upsert(ev):
                    added += 1
            self._mark_success("zt_pool")
            return added
        except Exception as e:
            self._mark_failure("zt_pool")
            logger.warning("[collector] zt_pool fail: %s", str(e)[:80])
            return 0

    # ── 关闭 ──

    def close(self):
        if self._library:
            self._library.close()


def get_collector(skip_akshare: bool = False) -> DataCollector:
    return DataCollector(skip_akshare=skip_akshare)
