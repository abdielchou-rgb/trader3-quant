"""
成分段重建回归测试 — 全离线（tmp qlib fixture + monkeypatch akshare 接口）

背景（data_qc 实跑发现）：csi300.txt 半年度切分的段末滞后日历末约一个月，
导致 instruments(universe, asof_date=近期) 返回空。
接口实测（akshare 1.18.81）：
  主源 index_stock_cons_csindex(symbol="000300") → 列含 "成分券代码"（裸 6 位码）
  备源 index_stock_cons(symbol="000300")         → 列含 "品种代码"
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import update_market_data as umd  # noqa: E402

CAL = [f"2026-01-{d:02d}" for d in range(1, 23)]  # 22 个交易日
NEW_DAYS = ["2026-01-23", "2026-01-26"]
FULL = CAL + NEW_DAYS
CAL_LAST = FULL[-1]
MAX_END = CAL[-1]  # 成分段全局最大段末（滞后日历末 2 天，模拟实跑滞后一个月）
NEW_START = NEW_DAYS[0]  # max_end 后首个交易日（保守起点）

# 初态行：(code, start, end)
#   SH600000/SZ000001 当前成分且段末==max_end → 应延长
#   SH600999 历史成员（非当前成分）→ 不动
#   SZ000002 非当前成分但段末==max_end → 规则5：不动
#   SH600555 当前成分但末段早于 max_end → 保持不动
ROWS = [
    ("SH600000", "2026-01-01", MAX_END),
    ("SH600999", "2026-01-01", "2026-01-15"),
    ("SZ000001", "2026-01-10", MAX_END),
    ("SZ000002", "2026-01-05", MAX_END),
    ("SH600555", "2026-01-01", "2026-01-15"),
]
CONS_NOW = {"600000", "000001", "600333", "600555"}  # 600333 为文件中不存在的新成分


def _make_qlib(tmp_path: Path, cal=None, rows=None):
    """构造迷你 qlib 结构：仅日历 + instruments/csi300.txt（本功能不触碰 bin）。"""
    data_dir = tmp_path / "qlib_bin"
    (data_dir / "calendars").mkdir(parents=True)
    (data_dir / "instruments").mkdir()
    cal = list(cal if cal is not None else FULL)
    (data_dir / "calendars" / "day.txt").write_text("\n".join(cal) + "\n", encoding="utf-8")
    inst = data_dir / "instruments" / "csi300.txt"
    rows = ROWS if rows is None else rows
    inst.write_text("".join(f"{c}\t{s}\t{e}\n" for c, s, e in rows), encoding="utf-8")
    return data_dir, inst


def _read_rows(inst: Path) -> list[tuple[str, ...]]:
    return [tuple(ln.split("\t")) for ln in
            inst.read_text(encoding="utf-8").splitlines() if ln]


@pytest.fixture
def fake_cons(monkeypatch):
    """按探针实测列名 monkeypatch akshare 双接口（全离线）。"""
    ak = pytest.importorskip("akshare", reason="akshare 未安装（成分股接线测试需要）")

    monkeypatch.setattr(ak, "index_stock_cons_csindex",
                        lambda symbol="000300": pd.DataFrame({"成分券代码": sorted(CONS_NOW)}))
    monkeypatch.setattr(ak, "index_stock_cons",
                        lambda symbol="000300": pd.DataFrame({"品种代码": sorted(CONS_NOW)}))
    return CONS_NOW


# ── 核心行为 ────────────────────────────────────────────

def test_extends_trailing_segments(tmp_path, fake_cons):
    data_dir, inst = _make_qlib(tmp_path)
    from trader3.data_provider import QlibDataProvider
    dp = QlibDataProvider(data_dir=str(data_dir))

    stats = umd.refresh_membership(dp, universe="csi300", cal_last=CAL_LAST)

    got = {r[0]: r for r in _read_rows(inst)}
    assert got["SH600000"] == ("SH600000", "2026-01-01", CAL_LAST)
    assert got["SZ000001"] == ("SZ000001", "2026-01-10", CAL_LAST)
    assert stats["changed"] is True
    assert set(stats["extended"]) >= {"SH600000", "SZ000001"}
    assert stats["max_end"] == MAX_END and stats["cal_last"] == CAL_LAST


def test_appends_new_members(tmp_path, fake_cons):
    data_dir, inst = _make_qlib(tmp_path)
    from trader3.data_provider import QlibDataProvider
    dp = QlibDataProvider(data_dir=str(data_dir))

    stats = umd.refresh_membership(dp, universe="csi300", cal_last=CAL_LAST)

    rows = _read_rows(inst)
    assert len(rows) == len(ROWS) + 1
    new = [r for r in rows if r[0] == "SH600333"]
    assert len(new) == 1
    assert tuple(new[0]) == ("SH600333", NEW_START, CAL_LAST)  # 保守起点=max_end后首个交易日
    assert stats["appended"] == ["SH600333"]


def test_keeps_historical_members(tmp_path, fake_cons):
    data_dir, inst = _make_qlib(tmp_path)
    from trader3.data_provider import QlibDataProvider
    dp = QlibDataProvider(data_dir=str(data_dir))

    umd.refresh_membership(dp, universe="csi300", cal_last=CAL_LAST)

    orig = {r[0]: r for r in ROWS}
    rows = _read_rows(inst)
    got = {r[0]: r for r in rows}
    # 历史成员与非当前成分逐字段不变
    for code in ("SH600999", "SZ000002"):
        assert got[code] == orig[code], f"{code} 被意外改动"
    # 当前成分但末段早于 max_end → 保持不动
    assert got["SH600555"] == orig["SH600555"]
    # 原有行相对顺序保留
    assert [r[0] for r in rows][:len(ROWS)] == [r[0] for r in ROWS]


def test_noop_when_fresh(tmp_path, fake_cons):
    """max_end >= cal_last 时零改动：内容不变、无备份目录产生。"""
    fresh_rows = [(c, s, CAL_LAST) for c, s, _e in ROWS]
    data_dir, inst = _make_qlib(tmp_path, rows=fresh_rows)
    before = inst.read_bytes()
    from trader3.data_provider import QlibDataProvider
    dp = QlibDataProvider(data_dir=str(data_dir))

    stats = umd.refresh_membership(dp, universe="csi300", cal_last=CAL_LAST)

    assert stats["changed"] is False
    assert inst.read_bytes() == before
    assert not list(data_dir.glob("_backup_*"))


def test_atomic_and_backup(tmp_path, fake_cons):
    data_dir, inst = _make_qlib(tmp_path)
    original = inst.read_text(encoding="utf-8")
    from trader3.data_provider import QlibDataProvider
    dp = QlibDataProvider(data_dir=str(data_dir))

    stats = umd.refresh_membership(dp, universe="csi300", cal_last=CAL_LAST)

    # 无 .tmp 残留（原子写）
    assert list((data_dir / "instruments").glob("*.tmp")) == []
    # 备份存在且内容为写前原文
    bks = list(data_dir.glob("_backup_*"))
    assert len(bks) == 1
    saved = bks[0] / "instruments__csi300.txt"
    assert saved.is_file()
    assert saved.read_text(encoding="utf-8") == original
    assert stats["backup_dir"]


# ── 闭环：asof 过滤非空 ─────────────────────────────────

def test_real_asof_nonempty_after_refresh(tmp_path, fake_cons):
    data_dir, _inst = _make_qlib(tmp_path)
    from trader3.data_provider import QlibDataProvider
    dp = QlibDataProvider(data_dir=str(data_dir))
    # 写前复现线上病灶：近期 asof 过滤为空
    assert dp.instruments("csi300", asof_date=CAL_LAST) == []

    umd.refresh_membership(dp, universe="csi300", cal_last=CAL_LAST)

    fresh = QlibDataProvider(data_dir=str(data_dir))  # 全新实例绕过进程内缓存
    codes = fresh.instruments("csi300", asof_date=CAL_LAST)
    assert codes  # 非空（问题闭环）
    assert {"SH600000", "SZ000001", "SH600333"} <= set(codes)
    assert "SH600999" not in codes  # asof 过滤语义未被破坏
    assert "SZ000002" not in codes  # 未动的过期段依旧被过滤


# ── 接口接线（双源降级 + 清洗） ─────────────────────────

def test_fetch_constituents_primary(monkeypatch):
    ak = pytest.importorskip("akshare", reason="akshare 未安装（成分股接线测试需要）")

    monkeypatch.setattr(ak, "index_stock_cons_csindex",
                        lambda symbol="000300": pd.DataFrame(
                            {"成分券代码": ["000001", "600000", "688123"]}))
    def _must_not_call(symbol="000300"):
        raise AssertionError("主源成功时不应触备源")
    monkeypatch.setattr(ak, "index_stock_cons", _must_not_call)
    assert umd.fetch_constituents("csi300") == {"000001", "600000", "688123"}


def test_fetch_constituents_fallback_and_clean(monkeypatch):
    ak = pytest.importorskip("akshare", reason="akshare 未安装（成分股接线测试需要）")

    def _boom(symbol="000300"):
        raise RuntimeError("官网接口不可用")
    monkeypatch.setattr(ak, "index_stock_cons_csindex", _boom)
    # 备源列名 品种代码；混入 int 与未补零码，验证清洗（zfill+isdigit 过滤）
    monkeypatch.setattr(ak, "index_stock_cons",
                        lambda symbol="000300": pd.DataFrame({"品种代码": [600519, "1", "ab12cd"]}))
    assert umd.fetch_constituents("csi300") == {"600519", "000001"}


def test_fetch_constituents_double_failure(monkeypatch):
    ak = pytest.importorskip("akshare", reason="akshare 未安装（成分股接线测试需要）")

    def _boom(symbol="000300"):
        raise RuntimeError("down")
    monkeypatch.setattr(ak, "index_stock_cons_csindex", _boom)
    monkeypatch.setattr(ak, "index_stock_cons", _boom)
    with pytest.raises(RuntimeError):
        umd.fetch_constituents("csi300")


def test_fetch_constituents_unknown_universe():
    with pytest.raises(ValueError):
        umd.fetch_constituents("nope100")
