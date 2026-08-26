"""Qlib 二进制行情数据源解析测试（真实数据在 2hao-analyst 项目下，缺失则跳过）。"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from trader3.v2.data_sources import QlibDataSource, make_panel_source
from trader3.v2.panel_builder import build_panel

QLIB_BIN = os.environ.get(
    "QLIB_BIN", r"D:\Claude\projects\2hao-analyst\data\qlib_bin"
)
HAVE_QLIB = os.path.isdir(QLIB_BIN)


def _ds():
    return QlibDataSource(QLIB_BIN, use_adjusted=True)


def test_make_panel_source_synthetic():
    assert make_panel_source("synthetic") is None


def test_make_panel_source_qlib_invalid_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        make_panel_source("qlib", qlib_uri="D:/does/not/exist")


def test_make_panel_source_qlib_ok():
    if not HAVE_QLIB:
        raise pytest.skip("QLIB_BIN 不存在")
    src = make_panel_source("qlib", qlib_uri=QLIB_BIN)
    assert isinstance(src, QlibDataSource)


def test_qlib_source_shape_and_no_header():
    if not HAVE_QLIB:
        raise pytest.skip("QLIB_BIN 不存在")
    ds = _ds()
    df = ds("bj430017", 250, pd.Timestamp("2026-08-27"))
    assert df.shape[1] == 7
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "vwap", "amount"]
    # 头部常量 5671.0 已被丢弃；收盘价应为真实量级而非 5671
    assert df["close"].abs().max() < 1000
    assert df["close"].notna().all()
    # vwap 与 close 同量级（已做复权对齐）
    assert 0.5 < (df["vwap"] / df["close"]).median() < 2.0


def test_qlib_source_unknown_code_returns_none():
    if not HAVE_QLIB:
        raise pytest.skip("QLIB_BIN 不存在")
    assert _ds()("ZZ999999", 250, pd.Timestamp("2026-08-27")) is None


def test_build_panel_with_qlib():
    if not HAVE_QLIB:
        raise pytest.skip("QLIB_BIN 不存在")
    panel = build_panel(
        ["bj430017", "bj430047", "bj430090"], 200,
        source=make_panel_source("qlib", qlib_uri=QLIB_BIN),
    )
    assert not panel.empty
    assert panel.columns.get_level_values(0).nunique() == 3
    assert panel.columns.get_level_values(1).nunique() == 7
