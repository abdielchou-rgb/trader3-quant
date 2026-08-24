"""
3号交易员 — 财务数据提供者 (M8)

读取 2hao-analyst/data/financials.db（560万行真实 A股财务数据）。

Schema:
    financials(code TEXT, quarter TEXT, table_name TEXT, field TEXT, value REAL, source TEXT)
    - table_name: profit / balance / cashflow
    - code: 600519 等裸代码（无 SH/SZ 前缀）
"""

from __future__ import annotations

import os
import sqlite3

import numpy as np

# 默认财务数据库路径
DEFAULT_FINANCIALS_DB = os.environ.get(
    "T3_FINANCIALS_DB",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "2hao-analyst", "data", "financials.db")
    ),
)
_ALT_DBS: list[str] = []


class FinancialsProvider:
    """
    真实 A股财务数据读取器（financials.db）。

    用法:
        fp = FinancialsProvider()
        fin = fp.get_financials('600519')   # -> {field: value} 最新一期
        hist = fp.get_history('600519', 'profit', 'roeAvg')
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = self._resolve_db_path(db_path)
        self._conn: sqlite3.Connection | None = None

    def _resolve_db_path(self, db_path: str | None) -> str:
        if db_path and os.path.exists(db_path):
            return db_path
        if os.path.exists(DEFAULT_FINANCIALS_DB):
            return DEFAULT_FINANCIALS_DB
        for alt in _ALT_DBS:
            if os.path.exists(alt):
                return alt
        raise FileNotFoundError(
            "未找到财务数据库 financials.db。请设置 T3_FINANCIALS_DB 指向该文件。"
        )

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    @staticmethod
    def _normalize_code(code: str) -> str:
        """'SH600519' / '600519.SH' / '600519' -> '600519'"""
        c = code.upper()
        c = c.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
        if c.startswith("SH") or c.startswith("SZ") or c.startswith("BJ"):
            c = c[2:]
        return c

    def available_quarters(self, code: str) -> list[str]:
        """某股票可用的财报期（降序）"""
        conn = self._connect()
        cur = conn.execute(
            "SELECT DISTINCT quarter FROM financials WHERE code=? ORDER BY quarter DESC",
            (self._normalize_code(code),),
        )
        return [row[0] for row in cur.fetchall()]

    def get_latest_financials(self, code: str) -> dict[str, float]:
        """
        获取某股票最新一期的财务数据（跨 profit/balance/cashflow 表）。

        Returns
        -------
        {field: value}
        """
        quarters = self.available_quarters(code)
        if not quarters:
            return {}
        latest_q = quarters[0]

        conn = self._connect()
        cur = conn.execute(
            "SELECT table_name, field, value FROM financials "
            "WHERE code=? AND quarter=?",
            (self._normalize_code(code), latest_q),
        )
        result = {}
        for row in cur.fetchall():
            result[f"{row[0]}.{row[1]}"] = row[2]
            result[row[1]] = row[2]  # 无前缀别名
        result["quarter"] = latest_q
        return result

    def get_field_history(self, code: str, field: str, table: str | None = None, n: int = 8) -> list[dict]:
        """
        获取某字段的历史序列（降序，最新在前）。

        Returns
        -------
        [{"quarter": ..., "value": ...}, ...]
        """
        conn = self._connect()
        if table:
            cur = conn.execute(
                "SELECT quarter, value FROM financials "
                "WHERE code=? AND table_name=? AND field=? "
                "ORDER BY quarter DESC LIMIT ?",
                (self._normalize_code(code), table, field, n),
            )
        else:
            cur = conn.execute(
                "SELECT quarter, value FROM financials "
                "WHERE code=? AND field=? "
                "ORDER BY quarter DESC LIMIT ?",
                (self._normalize_code(code), field, n),
            )
        return [{"quarter": row[0], "value": row[1]} for row in cur.fetchall()]

    def get_metric(self, code: str, field: str) -> float | None:
        """获取最新值"""
        fin = self.get_latest_financials(code)
        return fin.get(field)

    def get_current_price_with_date(self, code: str) -> tuple:
        """从 qlib 行情获取最新收盘价及其交易日 (price, date)；无数据返回 (None, None)"""
        try:
            from trader3.data_provider import QlibDataProvider

            dp = QlibDataProvider()
            norm = self._normalize_code(code)
            for prefix in ("sh", "sz", "bj"):
                close, dates = dp.load_stock(f"{prefix}{norm}", "close")
                if len(close) > 0:
                    valid_idx = np.where(close > 0)[0]
                    if len(valid_idx) > 0:
                        i = valid_idx[-1]
                        d = dates[i] if dates is not None and len(dates) > i else None
                        return float(close[i]), d
            return None, None
        except Exception:
            return None, None

    def get_current_price(self, code: str) -> float | None:
        """从 qlib 行情获取最新收盘价（配套 QlibDataProvider）"""
        price, _ = self.get_current_price_with_date(code)
        return price

    def to_valuation_input(self, code: str) -> dict:
        """
        将真实财务数据转换为 valuation_anchor 需要的输入字典。

        返回包含 current_price/eps/roe/bvps/revenue/fcf 等的 dict。
        缺少的字段保持缺失（由调用方决定是否回退合成）。
        """
        fin = self.get_latest_financials(code)
        if not fin:
            return {}

        result = {}
        # 每股指标（统一以总股本为口径，与市值=价格×总股本一致）
        total_share = fin.get("totalShare") or fin.get("profit.totalShare")

        price, price_date = self.get_current_price_with_date(code)
        if price:
            result["current_price"] = price
            if price_date:
                result["price_date"] = price_date

        eps = fin.get("epsTTM")
        if eps:
            result["eps"] = eps

        roe = fin.get("roeAvg")
        if roe:
            result["roe"] = roe

        if price and eps and eps > 0:
            result["pe"] = price / eps

        bvps = None
        total_equity = fin.get("totalEquity")
        if total_equity and total_share and total_share > 0:
            bvps = total_equity / total_share
            result["bvps"] = bvps

        if price and bvps and bvps > 0:
            result["pb"] = price / bvps

        # 营收/净利/FCF（单位：元）
        revenue = fin.get("MBRevenue")  # 可能单位是百万
        net_profit = fin.get("netProfit")
        fcf = fin.get("FCF")

        if revenue and total_share and total_share > 0:
            result["revenue_per_share"] = revenue / total_share

        if net_profit and total_share and total_share > 0:
            result["net_profit_per_share"] = net_profit / total_share

        if fcf and total_share and total_share > 0:
            result["fcf_per_share"] = fcf / total_share

        # 利润率
        gp = fin.get("gpMargin")
        np_margin = fin.get("npMargin")
        if gp:
            result["gross_margin"] = gp
        if np_margin:
            result["net_margin"] = np_margin

        # 资产负债表
        goodwill = fin.get("goodwill")
        total_assets = fin.get("totalAssets")
        total_liab = fin.get("totalLiab")
        if goodwill and total_assets and total_assets > 0:
            result["goodwill_to_assets"] = goodwill / total_assets
        if total_liab and total_equity and total_equity > 0:
            result["debt_to_equity"] = total_liab / total_equity

        # 现金流
        ocf = fin.get("OCF")
        if net_profit and ocf:
            result["ocf_net_income_ratio"] = ocf / net_profit if net_profit != 0 else None

        result["_source"] = "financials.db"
        result["_quarter"] = fin.get("quarter", "")
        return {k: v for k, v in result.items() if v is not None}

    def describe(self) -> dict:
        """数据库概览"""
        conn = self._connect()
        cur = conn.execute("SELECT COUNT(DISTINCT code) FROM financials")
        n_codes = cur.fetchone()[0]
        cur = conn.execute("SELECT COUNT(*) FROM financials")
        n_rows = cur.fetchone()[0]
        cur = conn.execute("SELECT MAX(quarter) FROM financials")
        latest = cur.fetchone()[0]
        return {
            "db_path": self.db_path,
            "total_rows": n_rows,
            "distinct_codes": n_codes,
            "latest_quarter": latest,
        }


def find_financials_db() -> str | None:
    """探测可用的财务数据库"""
    for candidate in [DEFAULT_FINANCIALS_DB, *_ALT_DBS]:
        if candidate and os.path.exists(candidate):
            return candidate
    return None
