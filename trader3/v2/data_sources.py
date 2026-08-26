"""真实数据源接入：解析 Qlib 二进制行情目录（calendars/features/instruments）。

Qlib bin 格式（本项目所用数据集）：
  - 每个 `features/<code>/<field>.day.bin` 为纯 float32 数组；
  - 数组第 0 个元素是该标的的统一头部常量（需丢弃）；
  - 剩余元素按 `instruments/<instr>.txt` 给出的 [start,end] 窗口，
    与 `calendars/day.txt` 一一对齐（长度 = 窗口交易日数）。

这样即可把 3号交易员 的 quant_pipeline 直接接到真实行情上，
无需安装 qlib 本体（本环境 `qlib` 为占位包，无 `init`）。
"""

from __future__ import annotations

import os
import struct
from collections.abc import Callable

import numpy as np
import pandas as pd

QLIB_FIELDS = ["open", "high", "low", "close", "volume", "vwap", "amount"]


class QlibDataSource:
    """可被 `build_panel(universe, ..., source=)` 直接使用的 callable 数据源。

    调用签名： ``source(code, lookback_days, end) -> DataFrame | None``
    返回的 DataFrame 以日期为索引、列为 ``QLIB_FIELDS``。
    """

    def __init__(
        self,
        provider_uri: str,
        use_adjusted: bool = True,
        calendar: str = "day",
    ) -> None:
        self.root = provider_uri
        self.use_adjusted = use_adjusted
        cal_path = os.path.join(provider_uri, "calendars", f"{calendar}.txt")
        with open(cal_path, encoding="utf-8") as fh:
            self.dates = [d.strip() for d in fh if d.strip()]
        self.date_idx = {d: i for i, d in enumerate(self.dates)}
        instr_path = os.path.join(provider_uri, "instruments", "all.txt")
        self.instr: dict[str, tuple[str, str]] = {}
        with open(instr_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    self.instr[parts[0].upper()] = (parts[1], parts[2])

    def _read(self, code: str, field: str) -> np.ndarray | None:
        path = os.path.join(self.root, "features", code.lower(), f"{field}.day.bin")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            buf = fh.read()
        n = len(buf) // 4
        arr = struct.unpack(f"<{n}f", buf[: 4 * n])
        return np.asarray(arr[1:], dtype=np.float64)  # 丢弃统一头部常量

    def __call__(
        self, code: str, lookback_days: int, end: pd.Timestamp | None = None
    ) -> pd.DataFrame | None:
        code = code.upper()
        if code not in self.instr:
            return None
        start, last = self.instr[code]
        if end is not None:
            end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
        else:
            end_s = last
        if end_s > last:
            end_s = last
        if end_s < start:
            return None
        si = self.date_idx[start]
        end_i = self.date_idx[end_s]
        start_i = max(si, end_i - lookback_days + 1)
        win = self.dates[start_i : end_i + 1]
        if not win:
            return None

        def seg(arr: np.ndarray | None) -> np.ndarray:
            if arr is None:
                return np.full(len(win), np.nan)
            return arr[start_i - si : end_i - si + 1]

        raw = {f: seg(self._read(code, f)) for f in QLIB_FIELDS}
        if self.use_adjusted:
            adj = seg(self._read(code, "adjclose"))
            rc = seg(self._read(code, "close"))
            raw["close"] = adj
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where((rc != 0) & ~np.isnan(rc), adj / rc, 1.0)
            ratio = np.nan_to_num(ratio, nan=1.0)
            for f in ("open", "high", "low", "vwap", "amount"):
                raw[f] = raw[f] * ratio
        df = pd.DataFrame(raw, index=pd.to_datetime(win))[QLIB_FIELDS]
        return df


def make_panel_source(
    data_source: str,
    qlib_uri: str = "",
    use_adjusted: bool = True,
) -> Callable | None:
    """按配置构造 panel_builder 的 source。

    - ``synthetic``：返回 None（build_panel 用合成数据兜底）；
    - ``qlib``：返回 :class:`QlibDataSource`，要求 ``qlib_uri`` 指向合法目录。
    """
    if data_source == "qlib":
        if not qlib_uri or not os.path.isdir(qlib_uri):
            raise FileNotFoundError(f"QLIB_URI 无效: {qlib_uri}")
        return QlibDataSource(qlib_uri, use_adjusted=use_adjusted)
    return None
