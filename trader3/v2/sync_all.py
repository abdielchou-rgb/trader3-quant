"""
3号交易员 v2.1 — 通用增量同步入口 sync_all.py

对齐 QUANTAXIS/QASU/save_tdx.py 的「查尾续拉」思想，但 K 线缓存例外：
前复权（qfq）因子会随分红送转对全历史重述，尾部增量拼接会把新旧口径
混进同一个 CSV 造成价格断层，因此 kline 同步采用「全窗口拉取 + 临时文件 +
os.replace 原子替换整个 CSV」，宁可多拉不可拼错。事件/财务仍走查尾增量：

  1. kline      行情 K 线（akshare 日线 → data/kline_cache/{code}.csv，qfq 全窗口原子替换）
  2. events     事件/公告增量（复用 DataCollector，collect_state.json 查尾）
  3. financials 财务数据（检测 financials.db；存在则可用，不存在则提示指向 2hao-analyst）

用法：
    python -m trader3.v2.sync_all --mode all
    python -m trader3.v2.sync_all --mode kline --codes 600519,000858 --days 120
    python -m trader3.v2.sync_all --mode events
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta

logger = logging.getLogger("trader3.v2.sync_all")

# 行情 K 线本地缓存目录
KLINE_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "kline_cache",
)


# ── K线同步（qfq 全窗口 + 原子替换） ──────────────────────

def _replace_cache(code: str, rows: list[list]) -> int:
    """整文件原子替换缓存：写临时文件 → os.replace 覆盖正式 CSV。

    qfq 前复权全历史重述，禁止追加拼接；os.replace 在同目录内为
    原子操作，进程崩溃/断电也不会留下半写的正式缓存。
    返回写入行数（按日期去重，重述值以后到为准）。
    """
    os.makedirs(KLINE_CACHE_DIR, exist_ok=True)
    path = os.path.join(KLINE_CACHE_DIR, f"{code}.csv")
    tmp = path + ".tmp"
    merged: dict[str, list] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    parts = ln.split(",")
                    merged[parts[0]] = parts   # 旧行先入，新数据覆盖
    for row in rows:
        if not row:
            continue
        merged[str(row[0])] = [str(x) for x in row]
    ordered = sorted(merged.items(), key=lambda kv: kv[0])
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for _, parts in ordered:
            f.write(",".join(parts) + "\n")
    os.replace(tmp, path)
    return len(rows)


def sync_kline(codes: list[str], days: int = 120, tail: bool = True) -> dict:
    """行情 K 线同步（qfq 全窗口拉取 + 原子替换，不做查尾续拉）。

    qfq 前复权因子随分红送转对全历史重述：尾部增量拼接会让除权日前的
    旧价格与新口径断层，技术因子（MA20/20日高）会算错。因此每次全窗口
    拉取最近 days 天，写临时文件后 os.replace 原子替换整个 CSV。
    tail 参数保留兼容旧调用但不再参与续拉判断。
    数据源自动降级：东方财富 stock_zh_a_hist → 新浪 stock_zh_a_daily。
    缓存统一 schema：date,open,close,high,low,volume
    """
    import akshare as ak

    os.makedirs(KLINE_CACHE_DIR, exist_ok=True)
    result = {}
    today = datetime.now().date()
    start = (today - timedelta(days=days)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")

    def _to_rows(df, source: str) -> list[list]:
        """统一映射为 date,open,close,high,low,volume"""
        if source == "eastmoney":
            return [
                [r["日期"], r["开盘"], r["收盘"], r["最高"], r["最低"], r["成交量"]]
                for _, r in df.iterrows()
            ]
        # sina: date/open/high/low/close/volume
        return [
            [str(r["date"])[:10], r["open"], r["close"], r["high"], r["low"], r["volume"]]
            for _, r in df.iterrows()
        ]

    for code in codes:
        norm = code.upper().replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
        try:
            rows, source = [], ""
            try:  # 主源：东方财富
                df = ak.stock_zh_a_hist(
                    symbol=norm, period="daily",
                    start_date=start, end_date=end, adjust="qfq",
                )
                if df is not None and not df.empty:
                    rows, source = _to_rows(df, "eastmoney"), "eastmoney"
            except Exception as e:
                logger.debug("[sync_all] %s 东财源失败，降级新浪: %s", norm, e)
            if not rows:  # 降级源：新浪
                try:
                    df = ak.stock_zh_a_daily(
                        symbol=("sh" if norm.startswith("6") else "sz") + norm,
                        start_date=start, end_date=end, adjust="qfq",
                    )
                    if df is not None and not df.empty:
                        rows, source = _to_rows(df, "sina"), "sina"
                except Exception as e:
                    logger.warning("[sync_all] %s 新浪源也失败: %s", norm, e)
            if not rows:
                result[norm] = {"mode": "full", "added": 0, "msg": "双源均无数据"}
                continue
            added = _replace_cache(norm, rows)
            result[norm] = {
                "mode": "full(qfq全历史重述安全)", "source": source, "added": added,
                "tail": rows[-1][0] if rows else None,
            }
        except Exception as e:
            logger.warning("[sync_all] %s K线同步失败: %s", norm, e)
            result[norm] = {"mode": "error", "added": 0, "msg": str(e)}
    return result


# ── 事件增量同步（复用 collector 查尾） ───────────────────

def sync_events(codes: list[str] | None = None) -> dict:
    """事件/公告/龙虎榜/资金流增量采集（collect_state.json 查尾防重）"""
    from trader3.v2.collector import DataCollector
    from trader3.v2.watchlist import get_watchlist

    collector = DataCollector()
    wl = get_watchlist()
    target = codes or [c.code for c in wl.list()] or ["600519", "000858"]
    stock_added = 0
    for code in target:
        stock_added += collector.collect_stock(code, source="news")
    flow_added = collector.collect_flow_events()
    collector.close()
    wl.close()
    return {"stock_events_added": stock_added, "flow_events_added": flow_added, "codes": target}


# ── 财务数据检查 ─────────────────────────────────────

def sync_financials() -> dict:
    """财务数据检查：financials.db 由 2hao-analyst 侧维护，本入口负责检测可达性。"""
    try:
        from trader3.financials_provider import find_financials_db
        db = find_financials_db()
        if db:
            return {"db": db, "status": "ok"}
        return {"db": None, "status": "missing", "hint": "设置 T3_FINANCIALS_DB 或同步 2hao-analyst/data/financials.db"}
    except Exception as e:
        return {"db": None, "status": "error", "hint": str(e)}


# ── 扩展数据源注册表（extra_sources 薄适配器） ────────
# 默认全部关闭（避免误联网）；参数 enabled 或环境变量 T3_SYNC_<NAME> 开启。

_EXTRA_ENV_PREFIX = "T3_SYNC_"
_EXTRA_REGISTRY: list[dict] = []


class ExtraSourceAdapter:
    """把 extra_sources 源（fetch(code, limit)->List[SourceItem]）
    适配为与本模块同步函数一致的协议：sync(codes, limit) -> {code: {...}}。
    仅在注册项 enabled=True 时才会被真正调用（联网）。"""

    def __init__(self, src):
        self._src = src
        self.name = src.source_name
        self.availability = getattr(src, "availability", "local")

    def sync(self, codes: list[str], limit: int = 5) -> dict:
        out = {}
        for code in codes:
            try:
                items = self._src.fetch(code, limit=limit)
            except Exception as e:
                logger.warning("[sync_all] 扩展源 %s 拉取 %s 失败: %s",
                               self.name, code, str(e)[:80])
                items = []
            out[code] = {"added": len(items), "source": self.name}
        return out


def _env_enabled(name: str, override: bool | None = None) -> bool:
    if override is not None:
        return override
    return os.environ.get(_EXTRA_ENV_PREFIX + name.upper(), "").strip().lower() \
        in ("1", "true", "yes")


def register_extra_sources(enabled: bool | None = None) -> list[dict]:
    """把 extra_sources 的额外源注册进同步源注册表（重建注册表）。

    每个源带名称与开关：默认关；enabled 参数或环境变量 T3_SYNC_<NAME> 开启，
    避免默认跑联网。
    """
    from trader3.v2.extra_sources import ALL_EXTRA
    _EXTRA_REGISTRY.clear()
    for src in ALL_EXTRA:
        adapter = ExtraSourceAdapter(src)
        _EXTRA_REGISTRY.append({
            "name": adapter.name,
            "availability": adapter.availability,
            "enabled": _env_enabled(adapter.name, override=enabled),
            "adapter": adapter,
        })
    return list_sync_sources()


def list_sync_sources() -> list[dict]:
    """已注册扩展源概览：[{name, availability, enabled}]"""
    return [{k: v for k, v in e.items() if k != "adapter"} for e in _EXTRA_REGISTRY]


def sync_extra_sources(codes: list[str], limit: int = 5) -> dict:
    """执行所有已启用扩展源的增量拉取；未启用的跳过并注明开关方法。"""
    results = {}
    for entry in _EXTRA_REGISTRY:
        if not entry["enabled"]:
            results[entry["name"]] = {
                "skipped": f"disabled（设 {_EXTRA_ENV_PREFIX}{entry['name'].upper()}=1 或 --extra 开启）",
            }
            continue
        results[entry["name"]] = entry["adapter"].sync(codes, limit=limit)
    return results


register_extra_sources()   # 导入即注册；默认全关，不产生任何网络请求


# ── CLI ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="3号交易员 v2.1 通用增量同步")
    parser.add_argument("--mode", choices=["all", "kline", "events", "financials"], default="all")
    parser.add_argument("--codes", default="", help="逗号分隔股票代码，如 600519,000858")
    parser.add_argument("--days", type=int, default=120, help="全窗口回拉天数")
    parser.add_argument("--no-tail", action="store_true", help="(已废弃，兼容保留) kline 本就全窗口替换")
    parser.add_argument("--extra", action="store_true",
                        help="同时启用 extra_sources 扩展源同步（默认关闭，避免误联网；"
                             "亦可按源设 T3_SYNC_MARGIN / T3_SYNC_SHAREHOLDER_COUNT / T3_SYNC_NORTHBOUND=1）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] if args.codes else []

    if args.mode in ("all", "kline"):
        if not codes:
            from trader3.v2.watchlist import get_watchlist
            wl = get_watchlist()
            codes = [c.code for c in wl.list()] or ["600519", "000858"]
            wl.close()
        print("== K线 全窗口同步（qfq 原子替换） ==")
        res = sync_kline(codes, days=args.days, tail=not args.no_tail)
        for k, v in res.items():
            print(f"  {k}: {v}")

    if args.extra or any(os.environ.get(_EXTRA_ENV_PREFIX + n.upper())
                         for n in ("margin", "shareholder_count", "northbound")):
        if not codes:
            codes = ["600519", "000858"]
        register_extra_sources(enabled=True if args.extra else None)
        print("== 扩展源增量同步（margin/shareholder_count/northbound） ==")
        print(" ", sync_extra_sources(codes))

    if args.mode in ("all", "events"):
        print("== 事件增量采集 ==")
        print(" ", sync_events(codes or None))

    if args.mode in ("all", "financials"):
        print("== 财务数据检查 ==")
        print(" ", sync_financials())


if __name__ == "__main__":
    main()
