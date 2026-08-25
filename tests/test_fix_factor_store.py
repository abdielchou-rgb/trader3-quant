"""
FactorStore 回归测试 — 全离线（tmp_path + 独立 SharedState 目录）。

覆盖：
- save→load 往返一致（matrix/dates/codes/meta）
- data_version 变化 → load 返回 None
- load_or_compute miss 时调用 compute_fn，二次调用不再触发
- invalidate 后 list_entries 为空
- 损坏的 npz → load 安全返回 None
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.shared_state import SharedState  # noqa: E402
from trader3.v2.factor_store import FactorStore  # noqa: E402


def _make_data(T=30, N=5, seed=42):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(T, N))
    dates = [f"2024-01-{i+1:02d}" for i in range(T)]
    codes = [f"SH6000{i:02d}" for i in range(N)]
    return matrix, dates, codes


@pytest.fixture()
def store(tmp_path):
    """独立 state_dir + factor_store 根目录的仓库实例。"""
    tmp_state = tmp_path / "state"
    tmp_store_root = tmp_path / "factor_store"
    SharedState(str(tmp_state)).write_json(
        "data_version", {"versions": {"qlib_bin": "v1"}}
    )
    return FactorStore(root=str(tmp_store_root), state_dir=str(tmp_state))


def _set_data_version(store: FactorStore, version: str) -> None:
    SharedState(store.state_dir).write_json(
        "data_version", {"versions": {"qlib_bin": version}}
    )


def test_save_load_roundtrip(store):
    """save→load 往返一致：matrix/dates/codes/meta 全量还原。"""
    matrix, dates, codes = _make_data()
    meta = store.save("rank(ts_mean(close,20))", "csi300", "2020-01-01",
                      "2020-12-31", matrix, dates, codes)

    assert meta["expr"] == "rank(ts_mean(close,20))"
    assert meta["universe"] == "csi300"
    assert meta["period"] == ["2020-01-01", "2020-12-31"]
    assert meta["data_version"] == "v1"
    assert meta["sha1"]

    got = store.load("rank(ts_mean(close,20))", "csi300", "2020-01-01", "2020-12-31")
    assert got is not None, "刚写入的条目应命中"
    np.testing.assert_array_equal(got["matrix"], matrix)
    assert got["dates"] == dates
    assert got["codes"] == codes
    for k in ("expr", "universe", "period", "data_version", "created_at", "sha1"):
        assert got["meta"][k] == meta[k]


def test_data_version_change_is_miss(store):
    """data_version 变化后旧缓存视为 miss（key 与 meta 双重失配）。"""
    matrix, dates, codes = _make_data()
    store.save("alpha1", "csi300", "2020-01-01", "2020-06-30", matrix, dates, codes)
    assert store.load("alpha1", "csi300", "2020-01-01", "2020-06-30") is not None

    _set_data_version(store, "v2")
    assert store.load("alpha1", "csi300", "2020-01-01", "2020-06-30") is None

    # 新版本下重新 save 走新 key，两份缓存并存且互不串扰
    store.save("alpha1", "csi300", "2020-01-01", "2020-06-30", matrix, dates, codes)
    assert len(store.list_entries()) == 2


def test_load_or_compute_caches_once(store):
    """miss 时触发 compute_fn 并落盘；二次调用直接命中，compute_fn 不再执行。"""
    matrix, dates, codes = _make_data()
    calls = {"n": 0}

    def compute_fn():
        calls["n"] += 1
        return {"matrix": matrix, "dates": dates, "codes": codes}

    first = store.load_or_compute("alpha2", "csi300", "2021-01-01",
                                  "2021-12-31", compute_fn)
    assert calls["n"] == 1, "首次 miss 应调用 compute_fn"
    second = store.load_or_compute("alpha2", "csi300", "2021-01-01",
                                   "2021-12-31", compute_fn)
    assert calls["n"] == 1, "第二次应命中缓存，不得重复计算"

    np.testing.assert_array_equal(first["matrix"], matrix)
    np.testing.assert_array_equal(second["matrix"], matrix)
    assert first["meta"]["sha1"] == second["meta"]["sha1"]
    # tuple 形式的 compute_fn 同样支持
    calls2 = {"n": 0}

    def compute_tuple():
        calls2["n"] += 1
        return matrix + 1.0, dates, codes

    store.load_or_compute("alpha3", "csi300", "2021-01-01", "2021-12-31", compute_tuple)
    store.load_or_compute("alpha3", "csi300", "2021-01-01", "2021-12-31", compute_tuple)
    assert calls2["n"] == 1


def test_invalidate_clears_entries(store):
    """invalidate 后 list_entries 为空，且 load 不再命中。"""
    matrix, dates, codes = _make_data()
    store.save("a", "csi300", "2020-01-01", "2020-06-30", matrix, dates, codes)
    store.save("b", "csi300", "2020-07-01", "2020-12-31", matrix, dates, codes)
    assert len(store.list_entries()) == 2

    removed = store.invalidate()
    assert removed == 4  # 2 × (npz + meta.json)
    assert store.list_entries() == []
    assert store.load("a", "csi300", "2020-01-01", "2020-06-30") is None


def test_corrupt_npz_returns_none(store):
    """npz 被截断/损坏时 load 安全返回 None，不抛异常。"""
    matrix, dates, codes = _make_data()
    store.save("bad", "csi300", "2020-01-01", "2020-03-31", matrix, dates, codes)
    sha1 = store.key_for("bad", "csi300", "2020-01-01", "2020-03-31")
    npz_path = Path(store.root) / f"{sha1}.npz"
    npz_path.write_bytes(b"this is not an npz file")

    assert store.load("bad", "csi300", "2020-01-01", "2020-03-31") is None
    # 损坏条目不影响其他键的正常读取
    m2, d2, c2 = _make_data(seed=7)
    store.save("good", "csi300", "2020-01-01", "2020-03-31", m2, d2, c2)
    assert store.load("good", "csi300", "2020-01-01", "2020-03-31") is not None


def test_missing_data_version_file_falls_back_to_unknown(tmp_path):
    """data_version.json 缺失时回退 'unknown'，且缓存仍可往返。"""
    tmp_state = tmp_path / "empty_state"
    tmp_state.mkdir()
    store = FactorStore(root=str(tmp_path / "fs"), state_dir=str(tmp_state))
    assert store.data_version() == "unknown"

    m, d, c = _make_data(T=5, N=3)
    store.save("x", "all", "2020-01-01", "2020-01-31", m, d, c)
    got = store.load("x", "all", "2020-01-01", "2020-01-31")
    assert got is not None and got["meta"]["data_version"] == "unknown"
