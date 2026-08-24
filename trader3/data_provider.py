"""
3号交易员 — 轻量级 Qlib 数据提供者 (M8)

直接读取 qlib bin 格式（numpy fromfile），无需安装完整 qlib 依赖。

qlib bin 格式:
- features/{instrument}/{field}.day.bin — float32 little-endian 数组
- calendars/day.txt — 交易日历（每行一个日期 YYYY-MM-DD）
- instruments/{name}.txt — 股票列表（instrument<TAB>start<TAB>end，逐段覆盖）

对齐逻辑（已钉死，勿改回"文件长度=区间交易日数"假设）:
- bin 数组可能含首/尾 price<=0 的占位值（如 close[0]=0.0），加载时先剥离（记录 n_lead/n_tail）
- 日期映射 = cal[start_idx + n_lead : start_idx + n_lead + len(stripped)]
  其中 start_idx 由 instruments 的上市日期二分定位
- 契约校验（违反则 warning + ValueError，调用方应逐股容错）：
  * close 场景剥离后首值 > 10000 → 明显非价格
  * len(stripped) 与「上市日→日历末」交易日数偏差 > 5
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── 可配置数据目录 ──
DEFAULT_QLIB_DATA_DIR = os.environ.get(
    "T3_QLIB_DATA_DIR",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "2hao-analyst", "data", "qlib_bin")
    ),
)

# 若在 VM/沙箱中，回退到已挂载路径
_ALT_DIRS = [
    "/sessions/focused-great-pasteur/mnt/2hao-analyst/data/qlib_bin",
]


class QlibDataProvider:
    """
    轻量级 qlib bin 数据读取器。

    用法:
        dp = QlibDataProvider()
        df = dp.load_panel(['SH600000', 'SZ000001'], ['close', 'volume'],
                           start='2020-01-01', end='2024-12-31')
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = self._resolve_data_dir(data_dir)
        self._calendars: Optional[List[str]] = None
        self._instruments_cache: Dict[Tuple[str, Optional[str]], List[str]] = {}

    def _resolve_data_dir(self, data_dir: Optional[str]) -> str:
        if data_dir and os.path.isdir(data_dir):
            return data_dir
        if os.path.isdir(DEFAULT_QLIB_DATA_DIR):
            return DEFAULT_QLIB_DATA_DIR
        for alt in _ALT_DIRS:
            if os.path.isdir(alt):
                return alt
        raise FileNotFoundError(
            "未找到 qlib 数据目录。请设置环境变量 T3_QLIB_DATA_DIR "
            "或传入 data_dir 指向 qlib_bin 目录。"
        )

    # ── 日历 ──

    def calendar(self) -> List[str]:
        """交易日历（升序日期列表）"""
        if self._calendars is None:
            path = os.path.join(self.data_dir, "calendars", "day.txt")
            with open(path, "r") as f:
                self._calendars = [line.strip() for line in f if line.strip()]
        return self._calendars

    def trading_days_between(self, start: str, end: str) -> List[str]:
        """返回 [start, end] 内的交易日"""
        cal = self.calendar()
        # 二分定位
        lo = self._lower_bound(cal, start)
        hi = self._upper_bound(cal, end)
        return cal[lo:hi]

    @staticmethod
    def _lower_bound(arr: List[str], target: str) -> int:
        lo, hi = 0, len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @staticmethod
    def _upper_bound(arr: List[str], target: str) -> int:
        lo, hi = 0, len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] <= target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # ── 股票列表 ──

    def instruments(self, universe: str = "all", asof_date: Optional[str] = None) -> List[str]:
        """
        返回股票代码列表（去重，按代码排序）。
        universe: all / csi300 / csi500 / csi800 / csi1000
        asof_date: 给定时（YYYY-MM-DD）仅返回该日仍在成分内的代码
                   （逐段 code/start/end 判定，防幸存者偏差）；
                   None 时返回文件中出现过的全部代码（历史并集，旧行为，兼容旧调用）。

        行格式: instrument<TAB>start<TAB>end（缺失日期视为无限开放区间）。
        """
        cache_key = (universe, asof_date)
        if cache_key in self._instruments_cache:
            return self._instruments_cache[cache_key]

        path = os.path.join(self.data_dir, "instruments", f"{universe}.txt")
        if not os.path.exists(path):
            return []

        instruments = set()
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    parts = line.split()
                code = parts[0].upper()
                if asof_date is not None:
                    seg_start = parts[1] if len(parts) > 1 and parts[1] else "1900-01-01"
                    seg_end = parts[2] if len(parts) > 2 and parts[2] else "2999-12-31"
                    if not (seg_start <= asof_date <= seg_end):
                        continue
                instruments.add(code)
        result = sorted(instruments)
        self._instruments_cache[cache_key] = result
        return result

    # ── 单只股票 ──

    def load_stock(
        self, instrument: str, field: str, start: str = "2000-01-01", end: str = "2100-01-01"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        读取单只股票单字段。

        对齐规则（已钉死）:
        1. 剥离 bin 数组首尾 price<=0 / 非有限值的占位（如 close[0]=0.0），记录 n_lead/n_tail
        2. 日期映射 = cal[start_idx + n_lead : start_idx + n_lead + len(stripped)]
           start_idx 由 instruments 上市日二分定位
        3. 契约校验，违反则 warnings.warn 并 raise ValueError
           （调用方应逐股 try/except 容错）:
           - close 场景剥离后首值 > 10000 → 明显非价格（错位/脏数据）
           - len(stripped) 与「上市日→日历末」交易日数偏差 > 5

        Returns
        -------
        values : np.ndarray — float64 数组（已剥离占位）
        dates  : List[str] — 对齐的日期列表
        """
        instrument = instrument.upper()
        # 统一代码格式: BJ430017 -> bj430017, SH600000 -> sh600000
        dir_name = self._dir_name(instrument)

        field_path = os.path.join(self.data_dir, "features", dir_name, f"{field}.day.bin")
        if not os.path.exists(field_path):
            return np.array([]), []

        values = np.fromfile(field_path, dtype="<f4").astype(np.float64)

        # 确定上市日期
        listing_start = self._listing_start(instrument)
        cal = self.calendar()
        start_idx = self._lower_bound(cal, listing_start) if listing_start else 0

        # 剥离首尾占位值（price<=0 或 NaN/Inf）
        n_lead = 0
        while n_lead < len(values) and not (np.isfinite(values[n_lead]) and values[n_lead] > 0):
            n_lead += 1
        n_tail = 0
        while (
            n_tail < len(values) - n_lead
            and not (np.isfinite(values[len(values) - 1 - n_tail]) and values[len(values) - 1 - n_tail] > 0)
        ):
            n_tail += 1
        stripped = values[n_lead : len(values) - n_tail]

        if len(stripped) == 0:
            return np.array([]), []

        # ── 契约校验 ──
        problems = []
        if field == "close" and stripped[0] > 10000:
            problems.append(f"剥离后首值 {stripped[0]:.1f} 明显非价格(close>10000)")
        expected_len = max(len(cal) - start_idx, 0)
        if abs(len(stripped) - expected_len) > 5:
            problems.append(
                f"bin 有效长度 {len(stripped)} 与上市区间交易日数 {expected_len} 偏差 > 5"
            )
        if problems:
            msg = f"{instrument}.{field}: " + "; ".join(problems)
            warnings.warn(f"数据契约校验失败(该股将被跳过): {msg}", stacklevel=2)
            raise ValueError(msg)

        dates = cal[start_idx + n_lead : start_idx + n_lead + len(stripped)]
        return stripped, dates

    def _dir_name(self, instrument: str) -> str:
        """SH600000 -> sh600000; BJ430017 -> bj430017"""
        if instrument.startswith("SH"):
            return instrument.lower()
        if instrument.startswith("SZ"):
            return instrument.lower()
        if instrument.startswith("BJ"):
            return instrument.lower()
        # 已经是小写目录形式
        return instrument.lower()

    def _listing_start(self, instrument: str) -> str:
        """返回该股票最早的上市日期（或空）"""
        path = os.path.join(self.data_dir, "instruments", "all.txt")
        if not os.path.exists(path):
            return ""
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts and parts[0].upper() == instrument:
                    return parts[1] if len(parts) > 1 else ""
        return ""

    # ── 面板数据 ──

    def load_panel(
        self,
        instruments: List[str],
        fields: List[str],
        start: str = "2020-01-01",
        end: str = "2025-12-31",
    ) -> Dict[str, np.ndarray]:
        """
        读取多股票多字段面板。

        Returns
        -------
        {instrument: {field: np.ndarray}} — 每只股票按自身上市区间对齐
        """
        result: Dict[str, Dict[str, np.ndarray]] = {}
        for inst in instruments:
            stock = {}
            for fld in fields:
                values, _ = self.load_stock(inst, fld, start, end)
                stock[fld] = values
            result[inst] = stock
        return result

    # ── 便捷方法 ──

    def load_returns(
        self, instruments: List[str], start: str = "2020-01-01", end: str = "2025-12-31"
    ) -> Dict[str, np.ndarray]:
        """读取日收益率（用 close 差分）"""
        returns = {}
        for inst in instruments:
            close, _ = self.load_stock(inst, "close", start, end)
            if len(close) < 2:
                returns[inst] = np.array([])
                continue
            r = np.diff(close) / close[:-1]
            # 清理 inf/nan
            r = np.where(np.isfinite(r), r, 0.0)
            returns[inst] = r
        return returns

    def load_index_returns(
        self, index_code: str = "SH000300", start: str = "2020-01-01", end: str = "2025-12-31"
    ) -> Tuple[np.ndarray, List[str]]:
        """读取指数收益率（作为基准）"""
        close, dates = self.load_stock(index_code, "close", start, end)
        if len(close) < 2:
            return np.array([]), dates
        r = np.diff(close) / close[:-1]
        r = np.where(np.isfinite(r), r, 0.0)
        return r, dates[1:]

    def available_fields(self, instrument: str) -> List[str]:
        """查看某股票可用的字段"""
        dir_name = self._dir_name(instrument)
        path = os.path.join(self.data_dir, "features", dir_name)
        if not os.path.isdir(path):
            return []
        return sorted(f.split(".")[0] for f in os.listdir(path) if f.endswith(".day.bin"))

    def describe(self) -> Dict[str, int]:
        """数据目录概览"""
        cal = self.calendar()
        instruments = self.instruments("all")
        return {
            "trading_days": len(cal),
            "start_date": cal[0] if cal else "",
            "end_date": cal[-1] if cal else "",
            "instruments": len(instruments),
            "data_dir": self.data_dir,
        }


def find_qlib_dir() -> Optional[str]:
    """探测可用的 qlib 数据目录"""
    for candidate in [
        DEFAULT_QLIB_DATA_DIR,
        *_ALT_DIRS,
        "/sessions/focused-great-pasteur/mnt/2hao-analyst/data/qlib_bin",
    ]:
        if candidate and os.path.isdir(os.path.join(candidate, "calendars")):
            return candidate
    return None