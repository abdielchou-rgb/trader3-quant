"""
批次2 防前视验证：历史财务按公告日对齐（financials_asof）。

覆盖：
1. valuation._try_real_financials / execute 的 asof_date 防前视
2. comps.analyze 的 asof_date 防前视（_company_snapshot 按公告日读取）
3. backtest.get_financials_asof 辅助查询
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

CODE = "600519"


@pytest.fixture(scope="module")
def cal():
    from trader3.v2.announcement_calendar import AnnouncementCalendar

    return AnnouncementCalendar()


def test_asof_returns_different_quarters(cal):
    """同一代码不同时点应看到不同财报期（防前视核心）"""
    q1 = cal.financials_asof(CODE, "2024-01-15")
    q2 = cal.financials_asof(CODE, "2024-05-15")
    assert q1 and q2
    assert q1["quarter"] == "2023-09-30"
    assert q2["quarter"] == "2024-03-31"
    assert q1["quarter"] != q2["quarter"]


def test_valuation_asof_uses_announce_aligned_eps():
    """valuation 的 asof 路径应使用公告日对齐财务，而非最新一期"""
    from trader3.tools.valuation import ValuationAnchorTool

    early = ValuationAnchorTool._asof_valuation_input(CODE, "2024-01-15")
    later = ValuationAnchorTool._asof_valuation_input(CODE, "2024-05-15")
    assert early and later
    assert early.get("_quarter") == "2023-09-30"
    assert later.get("_quarter") == "2024-03-31"
    assert early["eps"] != later["eps"]


def test_valuation_execute_accepts_asof_date():
    """execute 签名应支持 asof_date 且不抛错"""
    from trader3.tools.valuation import ValuationAnchorTool

    tool = ValuationAnchorTool()
    resp = tool.execute(codes=[CODE], methods=["dcf"], asof_date="2024-01-15")
    assert resp is not None and resp.success
    # caveats 应标注使用了 2023-09-30 财务（asof 生效）
    assert any("2023-09-30" in c for c in resp.caveats)


def test_comps_analyze_asof():
    """comps.analyze 支持 asof_date，财务快照按公告日对齐"""
    from trader3.v2.comps import CompsAnalyzer

    analyzer = CompsAnalyzer()
    table = analyzer.analyze(CODE, industry="白酒", n_peers=3, asof_date="2024-01-15")
    assert table is not None
    assert len(table.peers) >= 1
    # 目标行财务数据应来自 2023Q3（防前视生效）
    target = next(r for r in table.peers if r.is_target)
    assert target.data_source == "2023-09-30" or "2023-09-30" in str(target.data_source)


def test_backtest_financials_asof_helper():
    """backtest 防前视辅助查询返回公告日对齐快照"""
    from trader3.tools.backtest import get_financials_asof

    fin = get_financials_asof(CODE, "2024-01-15")
    assert fin and fin["quarter"] == "2023-09-30"


def test_trigger_scan_accepts_asof_date():
    """trigger.scan 支持 asof_date 透传，估值因子按公告日对齐财务"""
    from trader3 import Trader3
    from trader3.v2.trigger import TriggerEngine

    eng = TriggerEngine(Trader3())
    r = eng.scan([{"code": CODE}], asof_date="2024-01-15")
    assert len(r) == 1
    assert r[0].code == CODE
    # 触发/未触发均可，关键是执行不抛错且 fair_value 来自公告日对齐估值
    assert r[0].fair_value != 0.0
