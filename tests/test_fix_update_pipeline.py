"""
增量更新管线回归测试 — 全离线（tmp 假 qlib 结构 + monkeypatch 抓取函数）
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import update_market_data as umd  # noqa: E402

CAL = [f"2026-01-{d:02d}" for d in range(1, 23)]  # 22 个交易日
NEW_DAYS = ["2026-01-23", "2026-01-26"]


def _make_qlib(tmp_path: Path, cal=None, with_index=True):
    """构造最小 qlib 目录结构：两只对齐股票 + （可选）指数。"""
    data_dir = tmp_path / "qlib_bin"
    (data_dir / "calendars").mkdir(parents=True)
    (data_dir / "instruments").mkdir()
    cal = cal or CAL
    (data_dir / "calendars" / "day.txt").write_text("\n".join(cal) + "\n", encoding="utf-8")

    lines = [f"SH600519\t{cal[0]}\t{cal[-1]}", f"SZ000001\t{cal[5]}\t{cal[-1]}"]
    if with_index:
        lines.append(f"SH000300\t{cal[0]}\t{cal[-1]}")
    (data_dir / "instruments" / "all.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # csi300 成分文件（run_qc 需要；注意测试原文件缺失，护栏测试需补齐）
    with open(data_dir / "instruments" / "csi300.txt", "w", encoding="utf-8") as f:
        for c in ("SH600519", "SZ000001"):
            f.write(f"{c}\t{cal[0]}\t{cal[-1]}\n")

    for code, start_day, base in (("sh600519", cal[0], 10.0), ("sz000001", cal[5], 20.0)):
        d = data_dir / "features" / code
        d.mkdir(parents=True, exist_ok=True)
        i0 = cal.index(start_day)
        arr = np.arange(len(cal) - i0, dtype=np.float64) + base
        for f in ("open", "close", "volume"):
            umd.write_bin_atomic(str(d / f"{f}.day.bin"), arr)

    if with_index:
        d = data_dir / "features" / "sh000300"
        d.mkdir(parents=True)
        idx_close = np.arange(len(cal), dtype=np.float64) * 0.1 + 4000.0
        umd.write_bin_atomic(str(d / "close.day.bin"), idx_close)
        umd.write_bin_atomic(str(d / "volume.day.bin"), np.full(len(cal), 1e9))
    return data_dir


def test_index_rebuild_extends_calendar_and_bins(tmp_path):
    data_dir = _make_qlib(tmp_path)
    from trader3.data_provider import QlibDataProvider
    dp = QlibDataProvider(data_dir=str(data_dir))

    # 模拟真实全历史抓取：覆盖全部日历（含新增2天），值递增
    all_days = CAL + NEW_DAYS
    rows = [(d, 4000.0 + i) for i, d in enumerate(all_days)]
    plan = umd.plan_index(dp, rows)
    assert plan["new_cal_len"] == len(CAL) + 2
    assert plan["new_tail_days"] == NEW_DAYS

    result = umd.apply_index(dp, rows, plan, str(tmp_path / "bk"))
    ok, msg = umd.verify_index(dp, rows, result["new_cal_len"])
    assert ok, msg

    # 与 verify 同理：必须用全新实例读取（旧实例持有写前的日历缓存）
    fresh = QlibDataProvider(data_dir=str(data_dir))
    close, dates = fresh.load_stock("sh000300", "close")
    assert len(close) == plan["new_cal_len"]
    assert dates[-1] == NEW_DAYS[-1]
    assert float(close[-1]) == pytest.approx(4023.0, abs=0.01)


def test_restore_recovers_corrupted_bin(tmp_path):
    """写坏后用备份恢复 → 内容与写前逐字节一致。"""
    data_dir = _make_qlib(tmp_path)
    close_path = data_dir / "features" / "sh000300" / "close.day.bin"
    day_path = data_dir / "calendars" / "day.txt"
    before_bin = close_path.read_bytes()
    before_cal = day_path.read_bytes()

    backup_dir = tmp_path / "bk"
    umd._backup([str(day_path), str(close_path)], str(backup_dir), str(data_dir))
    # 篡改现场：日历截短、bin 写垃圾
    day_path.write_text("2026-01-01\n", encoding="utf-8")
    umd.write_bin_atomic(str(close_path), np.array([1.0, 2.0]))
    umd._restore(str(tmp_path / "nonexistent_bk"), str(data_dir))  # 无备份目录 → no-op
    umd._restore(str(backup_dir), str(data_dir))

    assert close_path.read_bytes() == before_bin
    assert day_path.read_bytes() == before_cal


def test_stock_append_strict_rules(tmp_path):
    data_dir = _make_qlib(tmp_path)
    # 日历 +2 后（期望=24）：
    #  sh600519 gap=2   → 追加
    #  sh600000 bin 超长（26 > 24）→ 拒绝（负缺口）
    #  sz000001 大缺口（short 40 行，gap=42 > 窗口30）→ 拒绝需人工重建
    over_dir = data_dir / "features" / "sh600000"
    over_dir.mkdir(parents=True)
    long_arr = np.arange(len(CAL) + 4, dtype=np.float64) + 5.0
    for f in ("open", "close"):
        umd.write_bin_atomic(str(over_dir / f"{f}.day.bin"), long_arr)

    big_gap_dir = data_dir / "features" / "sz000001"
    for f in ("open", "close", "volume"):
        umd.write_bin_atomic(str(big_gap_dir / f"{f}.day.bin"), np.arange(1, dtype=np.float64))
    with open(data_dir / "instruments" / "all.txt", "a", encoding="utf-8") as f:
        f.write(f"SH600000\t{CAL[0]}\t{CAL[-1]}\n")

    cal_new = CAL + NEW_DAYS
    (data_dir / "calendars" / "day.txt").write_text("\n".join(cal_new) + "\n", encoding="utf-8")

    from trader3.data_provider import QlibDataProvider
    dp = QlibDataProvider(data_dir=str(data_dir))
    plans = {p["code"]: p for p in
             umd.plan_stocks(dp, ["SH600519", "SH600000", "SZ000001"], k_new=2)}

    assert plans["SH600519"]["action"] == "append"
    assert plans["SH600519"]["new_dates"] == NEW_DAYS
    assert plans["SH600000"]["action"] == "skip" and "超长" in plans["SH600000"]["reason"]
    # sz000001 上市日 cal[5]，raw=1 → gap=24-5-1=18 ≤30 → 治愈性追加
    assert plans["SZ000001"]["action"] == "append"


def test_stock_append_window_boundary(tmp_path):
    """缺口 > 治愈窗口(30) → 拒绝并提示人工重建。"""
    data_dir = _make_qlib(tmp_path, with_index=False)
    cal_huge = [f"2026-{m:02d}-{d:02d}" for m in range(1, 13) for d in (1, 15)]  # 62→24? 构造62天
    cal_huge = ([f"2026-01-{d:02d}" for d in range(1, 32)]
                + [f"2026-02-{d:02d}" for d in range(1, 32)])  # 62 天
    (data_dir / "calendars" / "day.txt").write_text("\n".join(cal_huge) + "\n", encoding="utf-8")
    d = data_dir / "features" / "sh600519"
    tiny = np.arange(10, dtype=np.float64) + 10.0
    for f in ("open", "close", "volume"):
        umd.write_bin_atomic(str(d / f"{f}.day.bin"), tiny)

    from trader3.data_provider import QlibDataProvider
    dp = QlibDataProvider(data_dir=str(data_dir))
    plans = umd.plan_stocks(dp, ["SH600519"])
    assert plans[0]["action"] == "skip"
    assert "超过治愈窗口" in plans[0]["reason"]


def test_stock_append_heals_legacy_drift(tmp_path):
    """基础漂移≤5 的股票追加后：原始长度回到锚点段长，边界与新日期都是真实价。"""
    data_dir = _make_qlib(tmp_path)
    cal_new = CAL + NEW_DAYS
    (data_dir / "calendars" / "day.txt").write_text("\n".join(cal_new) + "\n", encoding="utf-8")

    # 模拟遗留漂移：raw=20（旧期望22，漂移2），扩展后期望24 → 需补 4 行
    d = data_dir / "features" / "sh600519"
    for f in ("open", "close", "volume"):
        o = umd.read_bin(str(d / f"{f}.day.bin"))
        umd.write_bin_atomic(str(d / f"{f}.day.bin"), o[:-2])

    from trader3.data_provider import QlibDataProvider
    dp = QlibDataProvider(data_dir=str(data_dir))
    hist = {d_: 10.0 + i for i, d_ in enumerate(CAL[-2:] + NEW_DAYS)}

    def monkeypatched(code, start):
        return list(hist.items())

    orig_fetch = umd.fetch_stock_close
    umd.fetch_stock_close = monkeypatched
    try:
        plans = umd.plan_stocks(dp, ["SH600519"], k_new=2)
        assert plans[0]["action"] == "append"
        assert plans[0]["k"] == 4  # 边界2行 + 新增2行
        summary = umd.apply_stock_append(dp, plans, NEW_DAYS[0], str(tmp_path / "bk"))
        assert summary["appended"]
    finally:
        umd.fetch_stock_close = orig_fetch

    close, dates = dp.load_stock("sh600519", "close")
    assert len(close) == len(cal_new)
    assert dates[-1] == NEW_DAYS[-1]
    # 边界回填：最后四行（CAL[-2:] + NEW_DAYS）均为真实价
    for back, d_ in ((1, NEW_DAYS[-1]), (2, NEW_DAYS[0]),
                     (3, CAL[-1]), (4, CAL[-2])):
        assert float(close[-back]) == pytest.approx(hist[d_], abs=1e-6), (
            f"close[-{back}] ({d_}) 回填错误"
        )


def test_stock_append_writes_real_prices(tmp_path, monkeypatch):
    data_dir = _make_qlib(tmp_path)
    cal_new = CAL + NEW_DAYS
    (data_dir / "calendars" / "day.txt").write_text("\n".join(cal_new) + "\n", encoding="utf-8")

    from trader3.data_provider import QlibDataProvider
    dp = QlibDataProvider(data_dir=str(data_dir))

    hist = {"2026-01-23": 11.5, "2026-01-26": 11.8}
    monkeypatch.setattr(umd, "fetch_stock_close", lambda code, start: list(hist.items()))
    plans = [{"code": "SH600519", "action": "append", "k": 2, "new_dates": NEW_DAYS}]
    summary = umd.apply_stock_append(dp, plans, NEW_DAYS[0], str(tmp_path / "bk"))
    assert summary["appended"] and summary["appended"][0]["code"] == "SH600519"

    close, dates = dp.load_stock("sh600519", "close")
    assert len(close) == len(cal_new)
    assert dates[-1] == NEW_DAYS[-1]
    assert float(close[-1]) == pytest.approx(11.8, abs=1e-6)
    assert float(close[-2]) == pytest.approx(11.5, abs=1e-6)


def test_data_version_stamp(tmp_path):
    from trader3.shared_state import SharedState

    ss = SharedState(state_dir=str(tmp_path / "ss"))
    ss.set_data_version({"qlib_bin": "2026-01-26"})
    got = ss.read_json("data_version")
    assert got["versions"]["qlib_bin"] == "2026-01-26"
    assert got["metadata"]["updated_at"]


# ── 收尾 QC 护栏（2026-09-03 新增）────────────────────────


def _make_clean_qlib(tmp_path, with_jump: bool = False) -> Path:
    """迷你 qlib：两只无跳变股票。with_jump=True 在 sh600519 注入 1 个 100% 跳变。"""
    data_dir = _make_qlib(tmp_path)
    if with_jump:
        # 在 sh600519 close 中部制造一个非停牌单点跳变（idx 5→6 ×2），不应被豁免
        p = data_dir / "features" / "sh600519" / "close.day.bin"
        arr = umd.read_bin(str(p)).copy()
        arr[6:] = arr[5] * 2.0
        umd.write_bin_atomic(str(p), arr)
    return data_dir


def test_qc_guard_passes_on_clean_data(tmp_path, monkeypatch):
    """干净数据 → QC 护栏 ok=True。"""
    data_dir = _make_clean_qlib(tmp_path)
    # post_qc_guard 内部走 data_qc.save_report —— patch 到真实引用处
    import trader3.v2.data_qc as dqc

    def fake_save(report, path=None):
        return str(tmp_path / "qc_report.json")

    monkeypatch.setattr(dqc, "save_report", fake_save)
    res = umd.post_qc_guard(str(data_dir), universe="csi300", tolerance_critical=0)
    assert res["ok"] is True
    assert res["critical"] == 0
    assert res["stock_contract"] == 0


def test_qc_guard_flags_jumped_data(tmp_path, monkeypatch):
    """带 100% 跳变股 → critical>0 → ok=False（护栏应告警）。"""
    data_dir = _make_clean_qlib(tmp_path, with_jump=True)
    import trader3.v2.data_qc as dqc

    def fake_save(report, path=None):
        return str(tmp_path / "qc_report.json")

    monkeypatch.setattr(dqc, "save_report", fake_save)
    res = umd.post_qc_guard(str(data_dir), universe="csi300", tolerance_critical=0)
    assert res["critical"] >= 1
    assert res["ok"] is False
