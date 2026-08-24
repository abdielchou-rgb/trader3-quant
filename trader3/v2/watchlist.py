"""
3号交易员 v2.0 — 自选股状态机

核心创新：每只跟踪股在状态间迁移，状态变化才是提醒。

状态流转：
    未覆盖 ─加入跟踪→ 观察中 ────────关注（估值进入区间+催化临近）─────▶ 买入区间
                      │  ▲             │                              │
                      ▼  │             ▼                              ▼
                   预警（风险事件/财务不符）                         持有中
                   退出跟踪                                        │  ▲
                                                                   ▼  │
                                                                卖出触发（目标/止损/逻辑破坏）
                                                                   ▼
                                                               退出跟踪 / 重新观察

状态迁移规则（白名单）：
    未覆盖:  -> 观察
    观察中:  -> 关注 / 预警 / 退出
    关注:    -> 买入区间 / 观察(降级) / 预警 / 退出
    买入区间: -> 持有(执行) / 关注(触发失效) / 预警
    持有中:  -> 卖出触发 / 预警
    预警:    -> 观察 / 退出
    卖出触发: -> 退出 / 观察(再评估)

持久化：SQLite（watchlist.db）
"""

from __future__ import annotations

import builtins
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("trader3.v2.watchlist")


def _apply_sqlite_pragmas(conn: sqlite3.Connection) -> None:
    """SQLite 加固：WAL + busy_timeout 5s + synchronous NORMAL（并发防锁库）"""
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.execute("PRAGMA synchronous=NORMAL")
        conn.commit()
    except Exception as e:
        logger.warning("[watchlist] PRAGMA 设置失败: %s", e)

# ── 状态枚举（常量） ──
UNTRACKED = "未覆盖"
OBSERVING = "观察中"
ATTENTION = "关注"
BUY_ZONE = "买入区间"
HOLDING = "持有中"
ALERT = "预警"
SELL_TRIGGERED = "卖出触发"

ALL_STATES = [UNTRACKED, OBSERVING, ATTENTION, BUY_ZONE, HOLDING, ALERT, SELL_TRIGGERED]

# 状态迁移白名单：{当前状态: 允许迁移到的状态列表}
TRANSITIONS: dict[str, list[str]] = {
    UNTRACKED: [OBSERVING],
    OBSERVING: [ATTENTION, ALERT, UNTRACKED],
    ATTENTION: [BUY_ZONE, OBSERVING, ALERT, UNTRACKED],
    BUY_ZONE: [HOLDING, ATTENTION, ALERT],
    HOLDING: [SELL_TRIGGERED, ALERT],
    ALERT: [OBSERVING, UNTRACKED],
    SELL_TRIGGERED: [UNTRACKED, OBSERVING],
}

# 状态 → 提示动作优先级（用于提醒排序）
PRIORITY = {
    SELL_TRIGGERED: 1,   # 最高：必须立即动作
    BUY_ZONE: 2,         # 高：买入提醒
    ALERT: 3,            # 高：风险预警
    HOLDING: 4,
    ATTENTION: 5,
    OBSERVING: 6,
    UNTRACKED: 7,
}


@dataclass
class WatchItem:
    """自选股条目"""
    code: str
    name: str = ""
    status: str = OBSERVING
    add_date: str = ""
    last_change: str = ""
    note: str = ""
    # 触发相关信息（三因子引擎写入）
    trigger_score: float = 0.0
    trigger_reason: str = ""
    valuation_anchor: float = 0.0     # 估值锚（加权目标价）
    current_price: float = 0.0


@dataclass
class StatusChange:
    """状态迁移记录"""
    code: str
    from_status: str
    to_status: str
    reason: str = ""
    event_time: str = ""
    trigger_data: dict = field(default_factory=dict)


class WatchlistDB:
    """
    自选股持久化（SQLite）。
    表:
        watch_items(code TEXT PRIMARY KEY, name, status, add_date, last_change, note,
                    trigger_score, trigger_reason, valuation_anchor, current_price)
        watch_events(id INTEGER PK AUTOINCREMENT, code, from_status, to_status,
                     reason, event_time, trigger_data TEXT)
    """

    def __init__(self, db_path: str | None = None):
        if db_path:
            self.db_path = db_path
        else:
            base = Path(__file__).resolve().parent.parent.parent  # 3号交易员/
            data_dir = base / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "watchlist.db")
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        _apply_sqlite_pragmas(self._conn)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS watch_items (
                code TEXT PRIMARY KEY,
                name TEXT DEFAULT '',
                status TEXT DEFAULT '观察中',
                add_date TEXT DEFAULT '',
                last_change TEXT DEFAULT '',
                note TEXT DEFAULT '',
                trigger_score REAL DEFAULT 0,
                trigger_reason TEXT DEFAULT '',
                valuation_anchor REAL DEFAULT 0,
                current_price REAL DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS watch_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT,
                from_status TEXT,
                to_status TEXT,
                reason TEXT,
                event_time TEXT,
                trigger_data TEXT DEFAULT '{}'
            )
        """)
        self._conn.commit()

    def add(self, code: str, name: str = "", note: str = "") -> bool:
        """加入跟踪（状态=观察中），单语句 INSERT OR IGNORE 原子去重，返回是否新增"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO watch_items(code,name,status,add_date,last_change,note) "
            "VALUES(?,?,?,?,?,?)",
            (code, name, OBSERVING, now, now, note),
        )
        added = cur.rowcount > 0
        self._conn.commit()
        if added:
            self._record_event(code, UNTRACKED, OBSERVING, "加入跟踪", {"note": note})
            self._conn.commit()
        else:
            logger.info("[watchlist] %s 已在跟踪，跳过", code)
        return added

    def remove(self, code: str, reason: str = "用户移除") -> bool:
        """移除跟踪（状态→未覆盖），返回是否移除"""
        cur = self._conn.cursor()
        cur.execute("SELECT status FROM watch_items WHERE code=?", (code,))
        row = cur.fetchone()
        if not row:
            return False
        self._record_event(code, row[0], UNTRACKED, reason)
        cur.execute("DELETE FROM watch_items WHERE code=?", (code,))
        self._conn.commit()
        return True

    def get(self, code: str) -> WatchItem | None:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM watch_items WHERE code=?", (code,))
        row = cur.fetchone()
        return self._row_to_item(row) if row else None

    def list(self, status: str | None = None) -> builtins.list[WatchItem]:
        cur = self._conn.cursor()
        if status:
            cur.execute("SELECT * FROM watch_items WHERE status=? ORDER BY code", (status,))
        else:
            cur.execute("SELECT * FROM watch_items ORDER BY code")
        return [self._row_to_item(r) for r in cur.fetchall()]

    def list_by_priority(self) -> builtins.list[WatchItem]:
        """按状态优先级排序（卖出触发/买入区间/预警在前）"""
        items = self.list()
        items.sort(key=lambda x: PRIORITY.get(x.status, 99))
        return items

    def transition(self, code: str, to_status: str, reason: str = "",
                   trigger_data: dict | None = None) -> bool:
        """状态迁移（校验白名单），返回是否成功"""
        item = self.get(code)
        if not item:
            logger.warning("[watchlist] %s 不在跟踪中", code)
            return False
        allowed = TRANSITIONS.get(item.status, [])
        if to_status not in allowed:
            logger.warning("[watchlist] 非法迁移 %s: %s->%s（允许: %s）",
                           code, item.status, to_status, allowed)
            return False
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        cur = self._conn.cursor()
        cur.execute("UPDATE watch_items SET status=?, last_change=? WHERE code=?",
                    (to_status, now, code))
        self._record_event(code, item.status, to_status, reason, trigger_data)
        self._conn.commit()
        logger.info("[watchlist] %s: %s -> %s（%s）", code, item.status, to_status, reason)
        return True

    def update_trigger(self, code: str, score: float, reason: str,
                       valuation_anchor: float, current_price: float) -> None:
        """写入三因子触发结果（不改变状态）"""
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE watch_items SET trigger_score=?, trigger_reason=?, "
            "valuation_anchor=?, current_price=? WHERE code=?",
            (score, reason, valuation_anchor, current_price, code),
        )
        self._conn.commit()

    def _record_event(self, code: str, from_status: str, to_status: str,
                      reason: str = "", trigger_data: dict | None = None) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO watch_events(code,from_status,to_status,reason,event_time,trigger_data) "
            "VALUES(?,?,?,?,?,?)",
            (code, from_status, to_status, reason,
             datetime.now().strftime("%Y-%m-%d %H:%M"),
             json.dumps(trigger_data or {}, ensure_ascii=False)),
        )

    def events(self, code: str | None = None, limit: int = 20) -> builtins.list[StatusChange]:
        cur = self._conn.cursor()
        if code:
            cur.execute(
                "SELECT * FROM watch_events WHERE code=? ORDER BY id DESC LIMIT ?",
                (code, limit),
            )
        else:
            cur.execute("SELECT * FROM watch_events ORDER BY id DESC LIMIT ?", (limit,))
        result = []
        for r in cur.fetchall():
            result.append(StatusChange(
                code=r["code"], from_status=r["from_status"], to_status=r["to_status"],
                reason=r["reason"], event_time=r["event_time"],
                trigger_data=json.loads(r["trigger_data"] or "{}"),
            ))
        return result

    @staticmethod
    def _row_to_item(row) -> WatchItem:
        return WatchItem(
            code=row["code"], name=row["name"], status=row["status"],
            add_date=row["add_date"], last_change=row["last_change"],
            note=row["note"], trigger_score=row["trigger_score"],
            trigger_reason=row["trigger_reason"],
            valuation_anchor=row["valuation_anchor"],
            current_price=row["current_price"],
        )

    def close(self) -> None:
        self._conn.close()


# ── 便捷入口 ──

def get_watchlist(db_path: str | None = None) -> WatchlistDB:
    return WatchlistDB(db_path)
