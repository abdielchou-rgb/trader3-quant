"""
5分钟线采集试点回归测试 — 全离线（monkeypatch 抓取函数 + tmp 存储）

覆盖：akshare/baostock 列归一化、去重排序、双源降级、
parquet 与 CSV.gz 双后端往返、QC 跳变/倒序/重复检测、
原子替换无 .tmp 残留、CLI dry-run 不落盘。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import fetch_minute_pilot as pilot  # noqa: E402

from trader3.v2 import minute_data as md  # noqa: E402

# ── 工厂 ────────────────────────────────────────────────

def _rows(n=6, base=10.0, day="2026-08-25"):
    """n 根干净 5 分钟线：09:35 起，close 缓涨（涨幅远低于 11% 阈值）。"""
    out = []
    for i in range(n):
        minutes = 9 * 60 + 35 + 5 * i
        out.append({
            "datetime": f"{day} {minutes // 60:02d}:{minutes % 60:02d}",
            "open": base + i,
            "high": base + i + 0.5,
            "low": base + i - 0.5,
            "close": base + i + 0.2,
            "volume": 1000.0 * (i + 1),
        })
    return out


# ── 列归一化 ────────────────────────────────────────────

def test_rows_from_ak_normalizes_columns():
    """探针契约：东财接口中文列名，时间截断到分钟，多余列丢弃，数值转 float。"""
    df = pd.DataFrame({
        "时间": ["2026-08-25 09:35:00", "2026-08-25 09:40:00"],
        "开盘": [10.0, 11.0],
        "最高": [10.5, 11.5],
        "最低": [9.8, 10.9],
        "收盘": [10.2, 11.2],
        "成交量": [100, 200],
        "涨跌幅": [0.1, 0.2],  # 冗余列必须被丢弃
    })
    rows = md._rows_from_ak(df)
    assert rows[0] == {"datetime": "2026-08-25 09:35", "open": 10.0, "high": 10.5,
                       "low": 9.8, "close": 10.2, "volume": 100.0}
    assert all(isinstance(r[k], float) for r in rows
               for k in ("open", "high", "low", "close", "volume"))


def test_rows_from_bs_normalizes_time_and_fills_nan():
    """探针实测：baostock 返回 date/time/code/close/volume 字符串，
    time 形如 20260825093500000；备源无 OHLC → NaN 占位（不伪造数据）。"""
    raw = [
        ["2026-08-25", "20260825093500000", "sh.600519", "1325.0000000000", "313600"],
        ["2026-08-25", "20260825094000000", "sh.600519", "1324.5000000000", "250000"],
    ]
    rows = md._rows_from_bs(raw)
    assert [r["datetime"] for r in rows] == ["2026-08-25 09:35", "2026-08-25 09:40"]
    assert rows[0]["close"] == pytest.approx(1325.0)
    assert rows[0]["volume"] == pytest.approx(313600.0)
    for r in rows:
        for c in ("open", "high", "low"):
            assert np.isnan(r[c])


# ── 抓取：排序/去重/降级 ─────────────────────────────────

def test_fetch_minute_sorts_dedups_coerces_float(monkeypatch):
    messy = [
        {**_rows(1)[0], "close": "10.20", "volume": "1000"},       # 字符串数值
        _rows(1)[0],                                                # 同时刻重复
        _rows(3)[2],
        _rows(3)[1],
    ]
    monkeypatch.setattr(md, "fetch_minute_ak", lambda code, start_date: messy)
    out = md.fetch_minute("600519")
    dts = [r["datetime"] for r in out]
    assert dts == sorted(set(dts))          # 升序且无重复
    assert len(out) == 3
    assert all(isinstance(r["close"], float) and isinstance(r["volume"], float)
               for r in out)


def test_fetch_minute_falls_back_to_baostock(monkeypatch):
    def boom(code, start_date):
        raise RuntimeError("eastmoney down")

    monkeypatch.setattr(md, "fetch_minute_ak", boom)
    bs_rows = _rows(3)
    monkeypatch.setattr(md, "fetch_minute_bs", lambda code, start_date: bs_rows)
    assert md.fetch_minute("600519") == bs_rows


def test_fetch_minute_both_sources_fail(monkeypatch):
    def make(name):
        def f(code, start_date):
            raise RuntimeError(name)
        return f

    monkeypatch.setattr(md, "fetch_minute_ak", make("ak"))
    monkeypatch.setattr(md, "fetch_minute_bs", make("bs"))
    with pytest.raises(RuntimeError, match="双源失败"):
        md.fetch_minute("600519")


# ── 存储往返 ────────────────────────────────────────────

def test_save_load_roundtrip_active_backend(tmp_path):
    rows = _rows()
    path = md.save_minute_atomic("600519", rows, base_dir=str(tmp_path))
    back = md.load_minute("600519", base_dir=str(tmp_path))
    assert path.endswith(md.backend_ext())
    assert back is not None and md.to_frame(rows).equals(back)


def test_parquet_roundtrip(tmp_path):
    if not md.HAVE_PYARROW:
        pytest.skip("pyarrow 未安装（本环境自动回退 CSV.gz）")
    path = md.save_minute_atomic("600519", _rows(), base_dir=str(tmp_path),
                                 backend="parquet")
    assert path.endswith(".parquet")
    back = md.load_minute("600519", base_dir=str(tmp_path))
    assert md.to_frame(_rows()).equals(back)


def test_csv_gz_roundtrip_even_without_pyarrow(tmp_path):
    path = md.save_minute_atomic("000858", _rows(), base_dir=str(tmp_path),
                                 backend="csv.gz")
    assert path.endswith(".csv.gz")
    back = md.load_minute("000858", base_dir=str(tmp_path))
    assert back is not None and md.to_frame(_rows()).equals(back)


def test_load_minute_missing_returns_none(tmp_path):
    assert md.load_minute("999999", base_dir=str(tmp_path)) is None


# ── 原子替换 ────────────────────────────────────────────

def test_atomic_overwrite_leaves_no_tmp_residue(tmp_path):
    md.save_minute_atomic("600519", _rows(), base_dir=str(tmp_path))
    md.save_minute_atomic("600519", _rows(4), base_dir=str(tmp_path))  # 整文件替换
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []
    assert len(md.load_minute("600519", base_dir=str(tmp_path))) == 4


def test_failed_write_cleans_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(pd.DataFrame, "to_csv",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        md.save_minute_atomic("600519", _rows(), base_dir=str(tmp_path),
                              backend="csv.gz")
    assert list(tmp_path.iterdir()) == []      # 无 .tmp 残留


# ── QC ──────────────────────────────────────────────────

def test_qc_clean_series_has_no_findings():
    qc = md.qc_minute(md.to_frame(_rows(8)))
    assert qc == {"rows": 8, "dup": 0, "inverted": 0, "jump_offenders": []}


def test_qc_flags_single_bar_jump_above_threshold():
    rows = _rows(5)
    rows[3]["close"] = rows[2]["close"] * 1.20          # 单根 +20%：A股日内不可能
    rows[4]["close"] = rows[3]["close"] * 1.10          # +10%：恰在阈值内不算
    qc = md.qc_minute(md.to_frame(rows))
    assert qc["rows"] == 5
    assert len(qc["jump_offenders"]) == 1
    off = qc["jump_offenders"][0]
    assert off["datetime"] == rows[3]["datetime"]
    assert abs(off["ret"]) > 0.11


def test_qc_flags_inverted_order_and_duplicates():
    df = md.to_frame(_rows(4))
    dup = pd.concat([df, df.iloc[[1]]], ignore_index=True)              # 重复一根
    shuffled = dup.sort_values("datetime", ascending=False).reset_index(drop=True)
    qc = md.qc_minute(shuffled)
    assert qc["dup"] == 1
    assert qc["inverted"] == len(shuffled) - 1           # 全程倒序：每对相邻均逆序


def test_qc_empty_frame_is_safe():
    qc = md.qc_minute(md.to_frame([]))
    assert qc["rows"] == 0 and qc["dup"] == 0 and qc["jump_offenders"] == []


# ── CLI ─────────────────────────────────────────────────

def test_cli_dry_run_does_not_write(tmp_path, monkeypatch, capsys):
    outdir = tmp_path / "minute_5"
    monkeypatch.setattr(md, "fetch_minute_verbose",
                        lambda code, days=20: (_rows(), "fake-src"))
    rc = pilot.main(["--codes", "600519,000858", "--days", "20",
                     "--out-dir", str(outdir)])
    assert rc == 0
    assert not outdir.exists()                            # dry-run 绝不落盘
    assert "[dry-run]" in capsys.readouterr().out


def test_cli_apply_respects_limit(tmp_path, monkeypatch):
    calls = []

    def fake(code, days=20):
        calls.append(code)
        return _rows(), "fake-src"

    monkeypatch.setattr(md, "fetch_minute_verbose", fake)
    outdir = tmp_path / "m5"
    rc = pilot.main(["--codes", "600519,000858,300750", "--limit", "2",
                     "--apply", "--out-dir", str(outdir)])
    assert rc == 0
    assert calls == ["600519", "000858"]                  # --limit 截断生效
    ext = md.backend_ext()
    assert sorted(p.name for p in outdir.iterdir()) == \
        [f"000858{ext}", f"600519{ext}"]
