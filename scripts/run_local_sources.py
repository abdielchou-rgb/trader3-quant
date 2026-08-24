"""3号交易员 v2.0 — 本机多源采集运行脚本（在 Windows 本机运行）

用途：采集 雪球(自媒体/热度) + 知乎(深度讨论) + 外网(路透/彭博/高盛/WSJ) 信息
→ 评分入库 events.db → 供三因子引擎消费。

前置条件（本机）：
  1. 安装依赖：pip install requests akshare
  2. 浏览器登录雪球，复制 cookie：
     打开 xueqiu.com → F12 → Network → 任意请求 → Cookie 头 → 复制整个值
     设置环境变量： set XUEQIU_COOKIE=...
  3.（可选）知乎：登录 zhihu.com → 复制 cookie 到 ZHIHU_TOKEN
  4.（可选，外网加速）设置代理：HTTPS_PROXY=http://127.0.0.1:7890

用法：
  python scripts/run_local_sources.py --codes 600519,000858,300750 --all
  python scripts/run_local_sources.py --codes 600519 --xueqiu --zhihu --global-news
  python scripts/run_local_sources.py --status
  python scripts/run_local_sources.py --check-env

说明：本脚本需在本机运行（沙箱被反爬/无代理）。采集结果写入 data/events.db，
沙箱/本机均可读取供三因子引擎使用。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("AKSHARE_NO_PROXY", "1")

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("run_local_sources")

from trader3.v2.collector import CatalystScorer
from trader3.v2.events import Event, get_event_library
from trader3.v2.sources import GlobalReutersSource, XueqiuSource, ZhihuSource


def _persist(srccode: str, source: str, items, lib) -> int:
    """把采集结果评分并入库 events.db"""
    scorer = CatalystScorer()
    added = 0
    for it in items:
        s = scorer.score(it.title, it.content)
        ev = Event(
            code=srccode, source=source, title=it.title[:200],
            content=it.content[:500], event_time=it.ts, url=it.url,
            catalyst_score=s["score"], direction=s["direction"],
            category=s["category"], keywords_hit=s["keywords_hit"],
            collected_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        if lib.upsert(ev):
            added += 1
    return added


def run_all(codes: list, do_xueqiu=True, do_zhihu=True, do_global=True) -> dict:
    results = {}
    lib = get_event_library()

    if do_xueqiu:
        xq = XueqiuSource()
        total_added = 0
        for code in codes:
            items = xq.fetch(code, limit=8)
            added = _persist(code, "xueqiu", items, lib)
            total_added += added
            logger.info("雪球 %s: 采到 %d 条，新增 %d", code, len(items), added)
        results["xueqiu"] = total_added

    if do_zhihu:
        zh = ZhihuSource()
        total_added = 0
        for code in codes:
            items = zh.fetch(code, limit=6)
            added = _persist(code, "zhihu", items, lib)
            total_added += added
            logger.info("知乎 %s: 采到 %d 条，新增 %d", code, len(items), added)
        results["zhihu"] = total_added

    if do_global:
        g = GlobalReutersSource()
        total_added = 0
        # 外网用英文关键词（公司名）检索
        name_map = {"600519": "Kweichow Moutai", "000858": "Wuliangye",
                    "300750": "CATL", "601318": "Ping An"}
        for code in codes:
            keyword = name_map.get(code, code)
            items = g.fetch(keyword, limit=6)
            added = _persist(code, "global_news", items, lib)
            total_added += added
            logger.info("外网 %s(%s): 采到 %d 条，新增 %d", code, keyword, len(items), added)
        results["global_news"] = total_added

    lib.close()
    return results


def status() -> None:
    """查看现有事件库状态"""
    lib = get_event_library()
    print(f"事件库 events.db: {lib.count()} 条")
    for src in ["xueqiu", "zhihu", "global_news", "news", "announcement", "lhb", "zt_pool"]:
        today_events = lib.list_today(limit=9999)
        n = sum(1 for e in today_events if e.source == src)
        print(f"  {src}: 今日 {n} 条")
    lib.close()


def main():
    parser = argparse.ArgumentParser(description="3hao v2.0 本机多源采集")
    parser.add_argument("--codes", default="600519,000858,300750", help="自选股代码，逗号分隔")
    parser.add_argument("--all", action="store_true", help="采集全部三个源")
    parser.add_argument("--xueqiu", action="store_true")
    parser.add_argument("--zhihu", action="store_true")
    parser.add_argument("--global-news", action="store_true")
    parser.add_argument("--status", action="store_true", help="查看事件库状态")
    parser.add_argument("--check-env", action="store_true", help="检查 cookie 环境变量")
    flags = parser.parse_args()

    if flags.status:
        status()
        return

    if flags.check_env:
        for var in ["XUEQIU_COOKIE", "ZHIHU_TOKEN", "HTTPS_PROXY"]:
            v = os.environ.get(var, "")
            print(f"{var}: {'已配置('+v[:20]+'...)' if v else '未配置'}")
        return

    codes = [c.strip() for c in flags.codes.split(",") if c.strip()]
    do_all = flags.all or not (flags.xueqiu or flags.zhihu or flags.global_news)

    print("=" * 60)
    print("3hao v2.0 本机多源采集")
    print(f"自选股: {codes}")
    srcs = [n for f, n in [(flags.xueqiu, "雪球"), (flags.zhihu, "知乎"),
                           (flags.global_news, "外网")] if f]
    print("源: " + ("雪球/知乎/外网" if do_all else ",".join(srcs)))
    print("=" * 60)

    results = run_all(
        codes,
        do_xueqiu=do_all or flags.xueqiu,
        do_zhihu=do_all or flags.zhihu,
        do_global=do_all or flags.global_news,
    )

    print()
    print("采集完成，新增入库:")
    for k, v in results.items():
        print(f"  {k}: {v} 条")
    print()
    print("事件已写入 data/events.db，可运行：")
    print(f"  python -m trader3.v2.daily_pipeline --codes {','.join(codes)}")
    print()
    print("提示：若某源返回 0 条，先看 --check-env 是否缺 cookie，或用本机浏览器登录后重试。")


if __name__ == "__main__":
    main()
