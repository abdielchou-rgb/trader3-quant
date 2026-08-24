"""3号交易员 v2.1 — 统一行情列访问器（QUANTAXIS QADataStruct 风格）

吸收 QUANTAXIS data_fq.py 的列别名语义：外部代码通过统一字段名
（open/high/low/close/volume/amount）访问行情，不关心底层是
dict / pandas.DataFrame / 序列。

用法：
    from trader3.v2.qa_accessor import BarAccessor
    bar = BarAccessor({"open": 12.5, "close": 12.8, "volume": 10000})
    bar.close()     # 12.8
    bar["volume"]   # 10000（兼容别名 vol / turnover）
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class BarAccessor:
    """单根 Bar 行情列访问器：字段别名 + 缺失兜底（None）"""

    # 统一字段 → 常见别名（QUANTAXIS / qlib / 东财 等）
    ALIASES: Dict[str, tuple] = {
        "open": ("open", "o", "开盘", "open_price"),
        "high": ("high", "h", "最高", "high_price"),
        "low": ("low", "l", "最低", "low_price"),
        "close": ("close", "c", "收盘", "close_price", "last"),
        "volume": ("volume", "vol", "成交量", "volume_ratio"),
        "amount": ("amount", "turnover", "成交额", "money"),
        "prev_close": ("prev_close", "pre_close", "昨收", "last_close"),
    }

    def __init__(self, bar: Any):
        self._bar = bar

    def _raw(self, field: str) -> Any:
        """按别名链取原始值（支持 dict / 属性 / 下标）"""
        if self._bar is None:
            return None
        for name in self.ALIASES.get(field, (field,)):
            try:
                if isinstance(self._bar, dict):
                    if name in self._bar:
                        return self._bar[name]
                else:
                    if hasattr(self._bar, name):
                        return getattr(self._bar, name)
                    try:
                        if name in self._bar:
                            return self._bar[name]
                    except (TypeError, KeyError, IndexError):
                        pass
            except Exception:
                continue
        return None

    def get(self, field: str) -> Optional[float]:
        val = self._raw(field)
        if val is None:
            return None
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        return f if f == f else None  # 过滤 NaN

    def open(self) -> Optional[float]:
        return self.get("open")

    def high(self) -> Optional[float]:
        return self.get("high")

    def low(self) -> Optional[float]:
        return self.get("low")

    def close(self) -> Optional[float]:
        return self.get("close")

    def volume(self) -> Optional[float]:
        return self.get("volume")

    def amount(self) -> Optional[float]:
        return self.get("amount")

    def prev_close(self) -> Optional[float]:
        return self.get("prev_close")

    def __getitem__(self, field: str) -> Optional[float]:
        return self.get(field)

    def fields(self) -> list:
        """返回当前 bar 中实际存在的统一字段列表"""
        return [f for f in self.ALIASES if self._raw(f) is not None]

    def to_dict(self) -> dict:
        return {f: self.get(f) for f in self.fields()}


def bar_close(bar: Any) -> Optional[float]:
    """快捷函数：取 bar 的收盘价（None 兜底）"""
    return BarAccessor(bar).close()


def get_quote_snapshot(code: str) -> Dict[str, Any]:
    """统一行情快照访问口：name/price 等快照一律走这里（换数据源只改一处）。

    委托 market_data.get_quote（东财→腾讯→新浪 三通道兜底），
    返回归一化 dict：{code, name, price, ...原始字段}。
    任何失败降级为 price=0.0 / name=""，调用方按无行情处理。
    """
    try:
        from trader3.v2.market_data import get_quote
        raw = dict(get_quote(code) or {})
    except Exception:
        raw = {}
    snap = {"code": code}
    snap.update(raw)
    snap.setdefault("name", "")
    try:
        snap["price"] = float(snap.get("price") or 0.0)
    except (TypeError, ValueError):
        snap["price"] = 0.0
    return snap
