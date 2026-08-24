"""
3号交易员 v2.0 — 搜索通道（search.py）

DDG 网页搜索（本机可过反爬；沙箱受限）。供知乎/外网源兜底使用。

通道说明：
- primary: 直接 requests → html.duckduckgo.com/html（本机通常可用）
- fallback: WebFetch 通道（由上层调用，此处预留接口说明）

用法：
    from trader3.v2.search import ddg_websearch
    results = ddg_websearch("茅台 库存", limit=5)
"""

from __future__ import annotations

import html as html_mod
import logging
import re
from typing import List, Tuple
from urllib.parse import urlencode

logger = logging.getLogger("trader3.v2.search")


def _clean_title(raw: str) -> str:
    return html_mod.unescape(re.sub(r"<[^>]+>", "", raw)).strip()


def ddg_websearch(query: str, limit: int = 8, timeout: int = 12) -> List[Tuple[str, str]]:
    """
    DDG HTML 端点网页搜索。返回 [(title, url)]。

    注意：可能遇到 bot 验证（返回 challenge 页）。此时返回空，
    由上层决定是否回退到 WebFetch（本机浏览器渲染通道）。
    """
    import requests
    # query 统一 urlencode：& 空格 中文等特殊字符不再裸拼进 URL
    url = "https://html.duckduckgo.com/html/?" + urlencode({"q": query})
    try:
        r = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        }, timeout=timeout)
        if r.status_code != 200:
            logger.debug("[search] DDG status %s", r.status_code)
            return []
        # bot 验证检测
        if "anomaly" in r.text or "challenge" in r.text.lower():
            logger.warning("[search] DDG 触发 bot 验证，返回空（本机浏览器通道可过）")
            return []
        results = re.findall(
            r'class="result__a" href="([^"]+)"[^>]*>(.*?)</a>', r.text
        )
        out = []
        for url_raw, title_raw in results[:limit]:
            url = html_mod.unescape(url_raw).replace("/l/?uddg=", "")
            # 从 redirect 链接里提取真实 URL
            m = re.search(r"(https?://[^&]+)", url)
            if m:
                url = m.group(1)
            title = _clean_title(title_raw)
            if title:
                out.append((title, url))
        logger.info("[search] DDG '%s' → %d 条", query[:30], len(out))
        return out
    except Exception as e:
        logger.debug("[search] DDG fail: %s", str(e)[:60])
        return []


if __name__ == "__main__":
    for q, lim in [("site:zhihu.com 白酒行业", 4), ("Reuters Moutai Kweichow", 4)]:
        print(f"--- {q} ---")
        for t, u in ddg_websearch(q, limit=lim):
            print(" ", t[:70], "|", u[:50])