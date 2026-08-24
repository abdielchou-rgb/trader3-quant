"""
3号交易员 v2.0 — 统一行情/股本数据模块 (market_data)

提供：真实市价（腾讯/新浪）、总股本/流通股本/总市值（东财 push2）、公司名称。

解决 comps.py 的股本缺失问题：东财 push2 单股接口直连，
不依赖 financials.db 的 totalShare（该字段部分股票缺失）。

已验证（沙箱 2026-08-23）：
- 东财 push2 secid 接口：f57代码 f58名称 f84总股本 f85流通股本 f116总市值 f117流通市值 ✅
- 腾讯 qt.gtimg.cn：现价/涨跌 ✅
- 新浪 hq.sinajs.cn：现价 ✅
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Dict

import requests

logger = logging.getLogger("trader3.v2.market")

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
_TENCENT_UA = {"User-Agent": "Mozilla/5.0"}


def em_secid(code: str) -> str:
    """股票代码 → 东财 secid（1.=沪 0.=深/北）"""
    c = code.upper().replace(".", "").replace("SH", "").replace("SZ", "").replace("BJ", "").zfill(6)
    return ("1." if c[:1] in ("6", "9") else "0.") + c


def _exchange_prefix(c6: str) -> str:
    """6位纯数字代码 → 腾讯/新浪通道前缀。

    北交所：43/83/87/88 开头 + 920 开头（2024 起 920 新段），统一 bj 前缀；
    沪市：6/9 开头；其余（0/2/3）深市。
    """
    if c6.startswith(("4", "8", "92")):
        return "bj"
    if c6.startswith(("6", "9")):
        return "sh"
    return "sz"


def em_quote(code: str) -> Optional[Dict]:
    """
    东财 push2 单股接口。
    返回 {name, total_share, float_share, market_cap, float_market_cap, price, change_pct}
    """
    secid = em_secid(code)
    url = ("https://push2.eastmoney.com/api/qt/stock/get"
           f"?secid={secid}&fields=f57,f58,f43,f44,f84,f85,f116,f117,f170")
    try:
        r = requests.get(url, headers={**_UA, "Referer": "https://quote.eastmoney.com/"}, timeout=10)
        d = r.json().get("data") or {}
        if not d:
            return None
        return {
            "name": d.get("f58") or "",
            "price": (d.get("f43") or 0) / 100 if isinstance(d.get("f43"), (int, float)) else 0,
            "change_pct": (d.get("f170") or 0) / 100 if isinstance(d.get("f170"), (int, float)) else 0,
            "total_share": (d.get("f84") or 0),            # 股本（股，东财已是原始单位）
            "float_share": (d.get("f85") or 0),
            "market_cap": (d.get("f116") or 0),            # 市值（元，东财已是原始单位）
            "float_market_cap": (d.get("f117") or 0),
        }
    except Exception as e:
        logger.debug("[market] em_quote %s fail: %s", code, str(e)[:60])
        return None


def tencent_quote(code: str) -> Optional[Dict]:
    """腾讯实时行情：v_sh600519="1~贵州茅台~600519~现价~昨收~今开~..."""
    c = code.upper().replace(".", "").replace("SH", "sh").replace("SZ", "sz").replace("BJ", "bj")
    if c[:2] not in ("sh", "sz", "bj"):
        c = _exchange_prefix(c.zfill(6)) + c.zfill(6)
    try:
        r = requests.get(f"https://qt.gtimg.cn/q={c}", headers=_TENCENT_UA, timeout=8)
        m = re.search(r'="([^"]+)"', r.text)
        if not m or not m.group(1):
            return None
        p = m.group(1).split("~")
        if len(p) < 6:
            return None

        def _f(idx):
            try:
                return float(p[idx])
            except (IndexError, ValueError):
                return 0.0

        out = {"name": p[1], "price": _f(3), "prev_close": _f(4), "open": _f(5)}
        # 腾讯 qt.gtimg.cn 字段：44=流通市值(亿) 45=总市值(亿) 46=市净率 38=换手率(%)
        float_mv = _f(44) * 1e8
        total_mv = _f(45) * 1e8
        if out["price"] > 0:
            out["float_share"] = int(float_mv / out["price"]) if float_mv > 0 else 0
            out["total_share"] = int(total_mv / out["price"]) if total_mv > 0 else 0
        out["market_cap"] = total_mv
        out["float_market_cap"] = float_mv
        out["pb"] = _f(46)
        out["turnover_rate"] = _f(38) / 100.0 if _f(38) else 0.0
        return out
    except Exception as e:
        logger.debug("[market] tencent %s fail: %s", code, str(e)[:60])
        return None


def sina_quote(code: str) -> Optional[Dict]:
    """新浪实时行情：var hq_str_sh600519="名,今开,昨收,现价,最高,最低,...,量,额" """
    c = code.upper().replace(".", "").replace("SH", "sh").replace("SZ", "sz").replace("BJ", "bj")
    if c[:2] not in ("sh", "sz", "bj"):
        c = _exchange_prefix(c.zfill(6)) + c.zfill(6)
    try:
        r = requests.get(f"https://hq.sinajs.cn/list={c}",
                         headers={**_UA, "Referer": "https://finance.sina.com.cn/"}, timeout=8)
        m = re.search(r'="([^"]*)"', r.text)
        if not m or not m.group(1):
            return None
        p = m.group(1).split(",")
        if len(p) < 6 or not p[0]:
            return None
        return {"name": p[0], "open": p[1], "prev_close": p[2], "price": float(p[3]),
                "high": p[4], "low": p[5], "volume": p[8] if len(p) > 8 else ""}
    except Exception as e:
        logger.debug("[market] sina %s fail: %s", code, str(e)[:60])
        return None


def get_quote(code: str) -> Dict:
    """
    综合取数（东财优先 → 腾讯 → 新浪）。返回标准化快照。
    """
    q = em_quote(code)
    if q:
        return {**{"code": code, "source": "eastmoney"}, **q}
    t = tencent_quote(code)
    if t:
        base = {"code": code, "source": "tencent"}
        base.update({k: t.get(k, 0) for k in
                     ("total_share", "float_share", "market_cap", "float_market_cap")})
        return {**base, **t}
    s = sina_quote(code)
    if s:
        return {**{"code": code, "source": "sina"}, **s,
                **{"total_share": 0, "float_share": 0, "market_cap": 0, "float_market_cap": 0}}
    return {"code": code, "source": "none", "name": "", "price": 0.0,
            "total_share": 0, "float_share": 0, "market_cap": 0}