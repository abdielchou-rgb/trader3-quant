"""
数据质检管道回归测试 — 全离线（tmp 迷你 qlib 结构 + 注入坏数据断言各检查触发）
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.v2 import data_qc  # noqa: E402

CAL = [f"2026-01-{d:02d}" for d in range(1, 32)]  # 31 个交易日
FIELDS = ("open", "close", "high", "low", "volume")
TODAY_NEAR = "2026-02-05"  # 距日历末 5 天（新鲜）


def _write_bin(path: Path, arr) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_bytes(np.asarray(arr, dtype=np.float64).astype("<f4").tobytes())
    tmp.replace(path)


def _make_qlib(tmp_path: Path, cal=None, codes=("sh600001", "sz000002")):
    """构造迷你 qlib 结构：两只平滑无缺陷成分股，返回 (data_dir, {code: close 序列})。"""
    data_dir = tmp_path / "qlib_bin"
    (data_dir / "calendars").mkdir(parents=True)
    (data_dir / "instruments").mkdir()
    cal = list(cal or CAL)
    (data_dir / "calendars" / "day.txt").write_text("\n".join(cal) + "\n", encoding="utf-8")

    with open(data_dir / "instruments" / "all.txt", "w", encoding="utf-8") as f:
        for c in codes:
            f.write(f"{c.upper()}\t{cal[0]}\t{cal[-1]}\n")
    with open(data_dir / "instruments" / "csi300.txt", "w", encoding="utf-8") as f:
        for c in codes:
            f.write(f"{c.upper()}\t{cal[0]}\t{cal[-1]}\n")

    series = {}
    for i, c in enumerate(codes):
        d = data_dir / "features" / c
        d.mkdir(parents=True)
        arr = np.arange(len(cal), dtype=np.float64) * 0.01 + 10.0 + i * 5.0
        series[c] = arr.copy()
        for fld in FIELDS:
            if fld == "high":
                vals = arr + 1.0
            elif fld == "low":
                vals = np.maximum(arr - 1.0, 0.5)
            elif fld == "volume":
                vals = np.full(len(cal), 1e6)
            else:
                vals = arr
            _write_bin(d / f"{fld}.day.bin", vals)
    return data_dir, series


# ── 基线：干净数据 PASS ──────────────────────────────────

def test_clean_fixture_passes(tmp_path):
    data_dir, _ = _make_qlib(tmp_path)
    report = data_qc.run_qc(str(data_dir), today=TODAY_NEAR)

    assert report["passed"] is True
    assert report["critical"] == 0
    assert report["offenders"] == []
    assert report["generated_at"]
    ck = report["checks"]["calendar"]
    assert ck["monotonic"] is True and ck["duplicates"] == []
    assert ck["days"] == len(CAL)
    assert report["checks"]["stocks"]["sampled"] == 2
    json.dumps(report, ensure_ascii=False)  # 必须可 JSON 化


def test_expired_constituent_excluded_from_sample(tmp_path):
    """csi300 中已在 asof 前退出的成分不参与抽查（防幸存者偏差的反向验证）。"""
    data_dir, _ = _make_qlib(tmp_path, codes=("sh600001",))
    with open(data_dir / "instruments" / "csi300.txt", "a", encoding="utf-8") as f:
        f.write(f"SH600999\t{CAL[0]}\t{CAL[9]}\n")  # 已退出成分
    report = data_qc.run_qc(str(data_dir))
    assert report["checks"]["stocks"]["sampled"] == 1
    assert all(o["code"] != "SH600999" for o in report["offenders"])


def test_stale_calendar_is_warning_not_critical(tmp_path):
    data_dir, _ = _make_qlib(tmp_path)
    report = data_qc.run_qc(str(data_dir), today="2026-08-25")  # 距末 >10 天
    ck = report["checks"]["calendar"]
    assert ck["stale"] is True and ck["stale_days"] > 10
    stale_hits = [o for o in report["offenders"]
                  if o["check"] == "calendar_stale" and o["severity"] == "warning"]
    assert len(stale_hits) == 1
    assert report["critical"] == 0 and report["passed"] is True


def test_fresh_calendar_no_stale_flag(tmp_path):
    data_dir, _ = _make_qlib(tmp_path)
    report = data_qc.run_qc(str(data_dir), today=TODAY_NEAR)
    assert report["checks"]["calendar"]["stale"] is False


# ── 日历检查 ────────────────────────────────────────────

def test_duplicate_calendar_date_detected(tmp_path):
    data_dir, _ = _make_qlib(tmp_path, cal=CAL + ["2026-01-15"])  # 尾部重复日
    report = data_qc.run_qc(str(data_dir))
    ck = report["checks"]["calendar"]
    assert ck["duplicates"] == ["2026-01-15"]
    assert ck["monotonic"] is False
    order_hits = [o for o in report["offenders"] if o["check"] == "calendar_order"]
    assert order_hits and order_hits[0]["severity"] == "critical"
    assert report["passed"] is False


def test_non_monotonic_calendar_detected(tmp_path):
    bad = list(CAL)
    bad[5], bad[7] = bad[7], bad[5]  # 乱序两天
    data_dir, _ = _make_qlib(tmp_path, cal=bad)
    report = data_qc.run_qc(str(data_dir))
    assert report["checks"]["calendar"]["monotonic"] is False
    assert any(o["check"] == "calendar_order" for o in report["offenders"])
    assert report["passed"] is False


# ── 每股抽查 ────────────────────────────────────────────

def test_close_jump_flagged_critical(tmp_path):
    """前复权拼接断裂特征：相邻有效值 |ret|>25% → critical。"""
    data_dir, _ = _make_qlib(tmp_path)
    arr = np.arange(len(CAL), dtype=np.float64) + 10.0
    arr[20:] = arr[19] * 1.6  # 拼接断裂：尾部整体抬升，边界单点 +60%
    _write_bin(data_dir / "features" / "sh600001" / "close.day.bin", arr)

    report = data_qc.run_qc(str(data_dir))
    hits = [o for o in report["offenders"]
            if o["check"] == "close_jump" and o["code"] == "SH600001"]
    assert len(hits) == 1
    assert hits[0]["severity"] == "critical"
    assert hits[0]["value"] == pytest.approx(0.6, abs=1e-6)
    assert report["checks"]["stocks"]["jump_events"] >= 1
    assert report["passed"] is False


def test_zero_ratio_warns_but_not_critical(tmp_path):
    """内部零值占比>30% → warning；零值不得诱发假跳变/契约失败。"""
    data_dir, series = _make_qlib(tmp_path)
    arr = series["sz000002"].copy()
    arr[10:22] = 0.0  # 12/31 ≈ 38.7% 零值
    _write_bin(data_dir / "features" / "sz000002" / "close.day.bin", arr)

    report = data_qc.run_qc(str(data_dir), today=TODAY_NEAR)
    hits = [o for o in report["offenders"]
            if o["check"] == "zero_ratio" and o["code"] == "SZ000002"]
    assert len(hits) == 1 and hits[0]["severity"] == "warning"
    assert hits[0]["value"] > 0.30
    assert not any(o["check"] == "close_jump" for o in report["offenders"])
    assert report["checks"]["stocks"]["contract_failed"] == 0
    assert report["passed"] is True  # 纯告警不影响通过口径


def test_ohlc_high_low_inconsistency_warns(tmp_path):
    data_dir, series = _make_qlib(tmp_path)
    low = np.maximum(series["sh600001"] - 1.0, 0.5)
    low[8:11] = 99.0  # 三行 high < low
    _write_bin(data_dir / "features" / "sh600001" / "low.day.bin", low)

    report = data_qc.run_qc(str(data_dir), today=TODAY_NEAR)
    hits = [o for o in report["offenders"]
            if o["check"] == "ohlc_high_low" and o["code"] == "SH600001"]
    assert len(hits) == 1 and hits[0]["severity"] == "warning"
    assert hits[0]["value"] == 3
    assert report["passed"] is True


def test_truncated_bin_contract_failure(tmp_path):
    """退市截断/锚点错位特征：bin 有效长度与上市区间偏差>5 → critical。"""
    data_dir, _ = _make_qlib(tmp_path)
    short = np.arange(5, dtype=np.float64) + 10.0
    for fld in ("open", "close"):
        _write_bin(data_dir / "features" / "sh600001" / f"{fld}.day.bin", short)

    report = data_qc.run_qc(str(data_dir), today=TODAY_NEAR)
    hits = [o for o in report["offenders"] if o["check"] == "stock_contract"]
    assert hits and hits[0]["severity"] == "critical"
    assert report["checks"]["stocks"]["contract_failed"] >= 1
    assert report["passed"] is False


def test_asof_membership_lag_falls_back_with_warning(tmp_path):
    """成分文件段末日期滞后于日历（asof 无在册）→ 告警并回退全历史并集，不空洞通过。"""
    data_dir, _ = _make_qlib(tmp_path)
    with open(data_dir / "instruments" / "csi300.txt", "w", encoding="utf-8") as f:
        for c in ("SH600001", "SZ000002"):
            f.write(f"{c}\t{CAL[0]}\t{CAL[9]}\n")  # 段末早于 asof
    report = data_qc.run_qc(str(data_dir), today=TODAY_NEAR)
    fb = [o for o in report["offenders"] if o["check"] == "sample_fallback"]
    assert len(fb) == 1 and fb[0]["severity"] == "warning"
    assert report["checks"]["stocks"]["sampled"] == 2
    assert report["passed"] is True


def test_empty_universe_warns_not_critical(tmp_path):
    data_dir, _ = _make_qlib(tmp_path)
    report = data_qc.run_qc(str(data_dir), today=TODAY_NEAR, universe="nonexistent")
    hits = [o for o in report["offenders"] if o["check"] == "sample_empty"]
    assert len(hits) == 1 and hits[0]["severity"] == "warning"
    assert report["checks"]["stocks"]["sampled"] == 0
    assert report["passed"] is True


# ── 抽样与截断口径 ────────────────────────────────────────

def _make_many_jumped(tmp_path: Path, n: int):
    codes = tuple(f"sh6{i:04d}" for i in range(n))
    data_dir, _ = _make_qlib(tmp_path, codes=codes[:2])
    # 重写 instruments 为全部 n 只
    with open(data_dir / "instruments" / "all.txt", "w", encoding="utf-8") as f:
        for c in codes:
            f.write(f"{c.upper()}\t{CAL[0]}\t{CAL[-1]}\n")
    with open(data_dir / "instruments" / "csi300.txt", "w", encoding="utf-8") as f:
        for c in codes:
            f.write(f"{c.upper()}\t{CAL[0]}\t{CAL[-1]}\n")
    base = np.arange(len(CAL), dtype=np.float64) + 10.0
    jumped = base.copy()
    jumped[20:] = jumped[19] * 2.0
    for c in codes:  # 第一只干净，其余各含一个跳变事件
        arr = base if c == codes[0] else jumped
        d = data_dir / "features" / c
        d.mkdir(parents=True, exist_ok=True)
        _write_bin(d / "close.day.bin", arr)
    return data_dir


def test_sample_limit_caps_scope(tmp_path):
    data_dir = _make_many_jumped(tmp_path, 120)
    report = data_qc.run_qc(str(data_dir), sample_limit=50, today=TODAY_NEAR)
    assert report["checks"]["stocks"]["universe_total"] == 120
    assert report["checks"]["stocks"]["sampled"] == 50
    assert report["critical"] == 49  # 抽样内 49 只带跳变


def test_offenders_truncated_to_100(tmp_path):
    data_dir = _make_many_jumped(tmp_path, 120)
    report = data_qc.run_qc(str(data_dir), today=TODAY_NEAR)  # 默认抽样 200 ≥ 120
    assert report["critical"] == 119
    assert len(report["offenders"]) == 100
    assert report["offenders_truncated"] is True


# ── 汇总行与落盘 ────────────────────────────────────────

def test_summary_line_pass_format():
    line = data_qc.qc_summary_line({"passed": True, "critical": 0})
    assert line == "QC: PASS(0 critical)"


def test_summary_line_fail_format_picks_top_offender():
    report = {
        "passed": False,
        "critical": 3,
        "offenders": [
            {"code": "SZ000003", "check": "close_jump", "value": 0.3, "severity": "critical"},
            {"code": "SH600001", "check": "close_jump", "value": 0.6, "severity": "critical"},
            {"code": "-", "check": "zero_ratio", "value": 0.4, "severity": "warning"},
        ],
    }
    line = data_qc.qc_summary_line(report)
    assert line.startswith("QC: FAIL(critical=3, top=")
    assert "SH600001" in line and "close_jump" in line and "+60.0%" in line
    assert "zero_ratio" not in line.split("top=", 1)[1]


def test_summary_line_fail_without_numeric_value():
    report = {
        "passed": False,
        "critical": 1,
        "offenders": [
            {"code": "SH600001", "check": "stock_contract",
             "value": "bin 有效长度 5 与上市区间交易日数 31 偏差 > 5", "severity": "critical"},
        ],
    }
    line = data_qc.qc_summary_line(report)
    assert line.startswith("QC: FAIL(critical=1, top=")
    assert "stock_contract" in line


def test_save_report_atomic_json_roundtrip(tmp_path):
    data_dir, _ = _make_qlib(tmp_path)
    report = data_qc.run_qc(str(data_dir), today=TODAY_NEAR)
    out = tmp_path / "nested" / "qc_report.json"
    saved = data_qc.save_report(report, str(out))
    assert Path(saved) == out
    assert json.loads(out.read_text(encoding="utf-8")) == report
    assert not out.with_suffix(out.suffix + ".tmp").exists()


def test_default_report_path_points_to_project_data():
    p = data_qc._default_report_path()
    assert Path(p).name == "qc_report.json"
    assert Path(p).parent.name == "data"


def test_daily_routine_wires_qc_before_factor_watch():
    """daily_routine 必须在因子监控 Step 前调用质检，并把 critical>0 的 QC 行并入推送文本。"""
    src = (_ROOT / "tools" / "daily_routine.py").read_text(encoding="utf-8")
    qc_pos = src.find("trader3.v2.data_qc")
    fw_pos = src.find("trader3.v2.factor_watch")
    assert qc_pos != -1 and fw_pos != -1
    assert qc_pos < fw_pos
    assert "run_qc(" in src and "save_report(" in src
