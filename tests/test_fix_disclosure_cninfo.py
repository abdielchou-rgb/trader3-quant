"""
披露日历 cninfo 备源零联网测试：

1. 主源 stock_yysj_em 抛错 + 巨潮备源 stock_report_disclosure 返回样例 → synced>0，
   source 标记为备源，参数（market/period）正确，列名归一化后解析正确
2. 主源+备源双挂 → synced=0 不抛出，备源尝试过程记 logger.info
3. akshare 无备源接口（delattr 模拟旧版本）→ synced=0 不抛出并记录跳过日志
4. 备源开关默认关闭：主源挂 + 未启用开关 → 不触碰备源（零联网），保持 synced=0
"""
import json
import logging
import os
import sys
import types
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

_COLS_CNINFO = ["股票代码", "股票简称", "首次预约", "初次变更", "二次变更", "三次变更", "实际披露"]


@pytest.fixture
def cal_path(tmp_path, monkeypatch):
    p = tmp_path / "disclosure_calendar.json"
    monkeypatch.setenv("TRADER3_DISCLOSURE_CALENDAR", str(p))
    return p


@pytest.fixture
def ds():
    from trader3.v2 import disclosure_sync as mod
    return mod


def _cninfo_df() -> pd.DataFrame:
    """模拟巨潮接口返回：datetime.date 值 + NaT（与真实接口 .dt.date 输出一致）"""
    return pd.DataFrame(
        [
            ["600519", "贵州茅台", date(2026, 7, 16), pd.NaT, pd.NaT, pd.NaT, date(2026, 7, 15)],
            ["000858", "五粮液", date(2026, 8, 28), pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        ],
        columns=_COLS_CNINFO,
    )


def test_primary_fails_cninfo_backup_succeeds(cal_path, monkeypatch):
    monkeypatch.setenv("TRADER3_DISCLOSURE_BACKUP", "1")

    # 直接替换模块级 ak 属性，避免 SimpleNamespace 的 _missing 限制
    import types
    import trader3.v2.disclosure_sync as ds_mod
    mock_ak = types.SimpleNamespace(
        stock_yysj_em=lambda **kw: (_ for _ in ()).throw(RuntimeError("em down")),
        stock_report_disclosure=lambda **kw: _cninfo_df(),
    )
    monkeypatch.setattr(ds_mod, "ak", mock_ak)

    res = ds_mod.sync_disclosure_dates("2026-06-30")
    assert res["synced"] == 2
    assert res["source"] == "stock_report_disclosure"  # BACKUP 常量值

    data = json.loads(cal_path.read_text(encoding="utf-8"))
    # 列名归一化后实际披露优先于首次预约
    assert data["600519"]["2026-06-30"] == "2026-07-15"
    assert data["000858"]["2026-06-30"] == "2026-08-28"
    assert not list(cal_path.parent.glob("*.tmp"))


def test_both_sources_fail_zero_no_raise(ds, cal_path, monkeypatch, caplog):
    monkeypatch.setenv("TRADER3_DISCLOSURE_BACKUP", "1")

    def boom(**kw):
        raise RuntimeError("all down")

    monkeypatch.setattr(ds.ak, "stock_yysj_em", boom, raising=False)
    monkeypatch.setattr(ds.ak, "stock_report_disclosure", boom, raising=False)

    with caplog.at_level(logging.INFO, logger="trader3.v2.disclosure_sync"):
        res = ds.sync_disclosure_dates("2026-06-30")

    assert res["synced"] == 0
    assert res["source"] is None
    assert not cal_path.exists()
    infos = [r.getMessage() for r in caplog.records]
    assert any("stock_report_disclosure" in m for m in infos), infos


def test_backup_interface_missing_degrades(ds, cal_path, monkeypatch, caplog):
    monkeypatch.setenv("TRADER3_DISCLOSURE_BACKUP", "1")

    def boom(**kw):
        raise RuntimeError("em down")

    monkeypatch.setattr(ds.ak, "stock_yysj_em", boom, raising=False)
    monkeypatch.delattr(ds.ak, "stock_report_disclosure", raising=False)

    with caplog.at_level(logging.INFO, logger="trader3.v2.disclosure_sync"):
        res = ds.sync_disclosure_dates("2026-06-30")

    assert res["synced"] == 0
    assert res["source"] is None
    assert not cal_path.exists()
    infos = [r.getMessage() for r in caplog.records]
    assert any("不存在" in m for m in infos), infos


def test_backup_disabled_by_default_no_network(ds, cal_path, monkeypatch):
    """默认关闭：未设 TRADER3_DISCLOSURE_BACKUP 时主源失败直接降级，不触碰备源"""
    def boom(**kw):
        raise RuntimeError("em down")

    def must_not_call(**kw):
        raise AssertionError("备源默认关闭，不应联网调用")

    monkeypatch.delenv("TRADER3_DISCLOSURE_BACKUP", raising=False)
    monkeypatch.setattr(ds.ak, "stock_yysj_em", boom, raising=False)
    monkeypatch.setattr(ds.ak, "stock_report_disclosure", must_not_call, raising=False)

    res = ds.sync_disclosure_dates("2026-06-30")
    assert res["synced"] == 0
    assert res["source"] is None
    assert not cal_path.exists()


def test_akshare_missing_module_still_importable_and_degrades(ds, cal_path, monkeypatch, caplog):
    """akshare 未安装时：模块可导入（本测试得以收集即证明），
    sync 诚实降级 synced=0 且日志说明缺源，而非 AttributeError 崩溃。

    零联网：主源被 monkeypatch 为不可调用（模拟 akshare 缺接口/缺库），
    备源开关默认关闭 → 不触网。
    """
    import trader3.v2.disclosure_sync as mod

    assert hasattr(mod, "ak")  # 延迟导入占位对象存在
    monkeypatch.delattr(ds.ak, "stock_yysj_em", raising=False)
    monkeypatch.delenv("TRADER3_DISCLOSURE_BACKUP", raising=False)

    with caplog.at_level(logging.INFO, logger="trader3.v2.disclosure_sync"):
        res = mod.sync_disclosure_dates("2026-06-30")
    assert res["synced"] == 0
    assert res["source"] is None
    assert not cal_path.exists()
    infos = [r.getMessage() for r in caplog.records]
    assert any(("不可用" in m) or ("未启用" in m) for m in infos), infos
