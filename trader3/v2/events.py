"""
3号交易员 v2.0 — 事件数据模型 + 催化强度评分器

事件库 (SQLite events.db)：
- 事件来源：新闻(akshare stock_news_em) / 公告(东财) / 龙虎榜 / 资金流
- 催化强度：从事件文本提取，关键词规则 + 基本面事件加权

催化规则（DDM 预期差逻辑——识别"会改变市场对未来现金流预期"的事件）：
  强催化(0.7-1.0): 业绩超预期/大合同/政策利好/重大资产重组/股东增持
  中催化(0.4-0.7): 常规合同/行业景气回暖/机构调研/分红
  弱催化(0.1-0.4): 常规公告/例行新闻/无实质信息
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("trader3.v2.events")


# ── 催化关键词规则（按权重分层） ──
# STRONG 只保留"会实质改变现金流预期"的事件；
# 常规增长/订单/涨价等移入 MEDIUM，避免催化虚高（降噪）。

STRONG_POSITIVE = [
    "净利润", "业绩超预期", "超预期", "预增", "上调", "增持",
    "中标", "签订合同", "大单", "回购", "分红", "重组", "并购",
    "政策支持", "利好", "获批", "签约", "战略合作", "涨停",
    "龙虎榜", "提价", "供不应求", "业绩增长",
]
STRONG_NEGATIVE = [
    "净利润下降", "同比减少", "预亏", "亏损", "下调", "减持",
    "违规", "处罚", "立案", "诉讼", "退市", "质押爆仓",
    "安全事故", "召回", "商誉减值", "业绩变脸",
]
MEDIUM_POSITIVE = [
    "机构调研", "调研", "股东增持", "回购进展", "新产品",
    "产能", "扩产", "景气", "复苏", "回暖", "提价预期",
    "增长", "订单", "突破", "涨价",
]
MEDIUM_NEGATIVE = [
    "毛利率下滑", "应收", "存货积压", "竞争加剧", "降价",
    "需求疲软", "增速放缓",
]
NEUTRAL_MARKERS = [
    "例行", "补充", "更正", "提示", "会议", "审议",
    "公告日期", "披露",
]

# 行业景气通用词（用于行业催化）
INDUSTRY_HEAT = ["景气", "复苏", "回暖", "需求", "供不应求", "涨价", "扩产", "渗透率"]


@dataclass
class Event:
    """单个事件：新闻/公告/龙虎榜/资金流"""
    code: str
    source: str            # news / announcement / lhb / fundflow
    title: str
    content: str = ""
    event_time: str = ""
    url: str = ""
    catalyst_score: float = 0.0     # 0~1 催化强度（正为多，负为空）
    direction: str = "neutral"      # positive / negative / neutral
    category: str = ""              # 业绩/订单/政策/重组/调研/分红
    keywords_hit: list[str] = field(default_factory=list)
    collected_at: str = ""


def _apply_sqlite_pragmas(conn: sqlite3.Connection) -> None:
    """SQLite 加固：WAL（读写不互斥）+ 忙等 5s + NORMAL 同步（并发防 database is locked）"""
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.execute("PRAGMA synchronous=NORMAL")
        conn.commit()
    except Exception as e:
        logger.warning("[events] PRAGMA 设置失败: %s", e)


@dataclass
class EventLibrary:
    """事件持久化 (SQLite events.db)"""

    def __init__(self, db_path: str | None = None):
        if db_path:
            self.db_path = db_path
        else:
            base = Path(__file__).resolve().parent.parent.parent
            data_dir = base / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "events.db")
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        _apply_sqlite_pragmas(self._conn)
        self._init_schema()

    def _init_schema(self):
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT,
                source TEXT,
                title TEXT,
                content TEXT DEFAULT '',
                event_time TEXT,
                url TEXT DEFAULT '',
                catalyst_score REAL DEFAULT 0,
                direction TEXT DEFAULT 'neutral',
                category TEXT DEFAULT '',
                keywords TEXT DEFAULT '[]',
                collected_at TEXT,
                UNIQUE(code, source, title)
            )
        """)
        self._conn.commit()

    def upsert(self, ev: Event) -> bool:
        """插入事件（按 code+source+title 去重）"""
        cur = self._conn.cursor()
        try:
            cur.execute("""
                INSERT OR IGNORE INTO events
                (code, source, title, content, event_time, url, catalyst_score,
                 direction, category, keywords, collected_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """, (
                ev.code, ev.source, ev.title, ev.content, ev.event_time, ev.url,
                ev.catalyst_score, ev.direction, ev.category,
                json.dumps(ev.keywords_hit, ensure_ascii=False),
                ev.collected_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ))
            self._conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            logger.warning("[events] upsert fail: %s", e)
            return False

    # ── 时间解析（老数据宽容） ──

    @staticmethod
    def _parse_ts(raw) -> datetime | None:
        """宽容解析 collected_at/event_time：ISO / unix秒 / unix毫秒，失败返回 None"""
        if raw is None:
            return None
        s = str(raw).strip()
        if not s:
            return None
        if s.isdigit():  # 新浪等源的 unix 时间戳
            try:
                ts = int(s)
                if ts > 10**12:
                    ts //= 1000
                return datetime.fromtimestamp(ts)
            except Exception:
                return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S",
                    "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s[:19], fmt)
            except Exception:
                continue
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return None

    def get_recent(self, code: str, days: int = 7, source: str | None = None) -> list[Event]:
        """取某只股票最近事件（days 天时效窗内，按催化强度排序）

        SQL 层先按 ISO 日期下界粗筛；unix 时间戳/异格式老数据走宽容解析再过滤。
        """
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        limit = max(days * 40, 40)
        cur = self._conn.cursor()
        # ISO 格式直接 SQL 下界；非 ISO（unix 老数据等）放行后由 Python 宽容解析过滤
        base_where = (
            "code=? AND (substr(collected_at,1,19) >= ? OR "
            "collected_at IS NULL OR collected_at='' OR "
            "collected_at NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-*')"
        )
        params: list = [code]
        if source:
            base_where += " AND source=?"
            params.append(source)
        params.extend([cutoff, limit])
        cur.execute(f"SELECT * FROM events WHERE {base_where} ORDER BY id DESC LIMIT ?",
                    params)
        cutoff_dt = datetime.now() - timedelta(days=days)
        out = []
        for r in cur.fetchall():
            ev = self._row_to_event(r)
            dt = self._parse_ts(ev.collected_at)
            if dt is not None and dt >= cutoff_dt:
                out.append(ev)
        out.sort(key=lambda e: abs(e.catalyst_score), reverse=True)
        return out

    def get_strongest_recent(self, code: str, days: int = 7) -> Event | None:
        """最近一周内催化最强的事件"""
        events = self.get_recent(code, days)
        if not events:
            return None
        return max(events, key=lambda e: abs(e.catalyst_score))

    def list_today(self, limit: int = 200) -> list[Event]:
        """今日采集到的全部事件"""
        cur = self._conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        cur.execute("""
            SELECT * FROM events WHERE collected_at LIKE ? ORDER BY catalyst_score DESC LIMIT ?
        """, (f"{today}%", limit))
        return [self._row_to_event(r) for r in cur.fetchall()]

    def count(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM events")
        return cur.fetchone()[0]

    def close(self):
        self._conn.close()

    @staticmethod
    def _row_to_event(row) -> Event:
        return Event(
            code=row["code"], source=row["source"], title=row["title"],
            content=row["content"], event_time=row["event_time"], url=row["url"],
            catalyst_score=row["catalyst_score"], direction=row["direction"],
            category=row["category"],
            keywords_hit=json.loads(row["keywords"] or "[]"),
            collected_at=row["collected_at"],
        )


# ── 催化评分器 ──

class CatalystScorer:
    """从事件文本提取催化强度（关键词规则）"""

    def score(self, title: str, content: str = "") -> dict:
        """返回 {score, direction, category, keywords_hit}"""
        text = f"{title} {content}"

        strong_pos = [k for k in STRONG_POSITIVE if k in text]
        strong_neg = [k for k in STRONG_NEGATIVE if k in text]
        med_pos = [k for k in MEDIUM_POSITIVE if k in text]
        med_neg = [k for k in MEDIUM_NEGATIVE if k in text]

        keywords_hit = strong_pos + strong_neg + med_pos + med_neg

        # 方向判定（负面优先——先排除）
        if strong_neg:
            direction = "negative"
            score = 0.15 + 0.1 * min(len(strong_neg), 3)  # 0.25~0.45 负面强度
        elif strong_pos:
            direction = "positive"
            score = 0.65 + 0.08 * min(len(strong_pos), 4)  # 0.73~0.97
        elif med_neg:
            direction = "negative"
            score = 0.3
        elif med_pos:
            direction = "positive"
            score = 0.55
        else:
            # 检查是否纯例行
            if any(m in text for m in NEUTRAL_MARKERS):
                direction = "neutral"
                score = 0.25
            else:
                direction = "neutral"
                score = 0.4

        # 类别判断
        category = self._classify(text)

        return {
            "score": round(min(score, 1.0), 4),
            "direction": direction,
            "category": category,
            "keywords_hit": keywords_hit[:10],
        }

    def _classify(self, text: str) -> str:
        if any(k in text for k in ["中标", "签订合同", "大单", "订单", "签约", "战略合作"]):
            return "订单/合同"
        if any(k in text for k in ["净利润", "业绩", "预增", "预亏", "营收", "利润"]):
            return "业绩"
        if any(k in text for k in ["重组", "并购", "收购"]):
            return "重组"
        if any(k in text for k in ["政策", "获批", "监管", "规划"]):
            return "政策"
        if any(k in text for k in ["调研", "增持", "减持", "回购"]):
            return "股东/机构"
        if any(k in text for k in ["分红", "送转", "股利"]):
            return "分红"
        return "行业/其他"

    # 行业景气催化（宏观看板用）
    def industry_heat(self, industry_text: str) -> float:
        hits = [k for k in INDUSTRY_HEAT if k in industry_text]
        n = len(hits)
        if n >= 4:
            return 0.9
        if n == 3:
            return 0.75
        if n == 2:
            return 0.6
        if n == 1:
            return 0.45
        return 0.3


def get_event_library(db_path: str | None = None) -> EventLibrary:
    return EventLibrary(db_path)
