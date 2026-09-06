"""
R4 唯一性权重 — 回归测试

契约（AFML Ch.4）：标签区间 [t0,t1] 含两端；重叠样本共享价格路径 →
按独占度加权。
  1. concurrency：每根 bar 上同时活跃样本数
  2. average_uniqueness：每样本独占份额；非重叠=1，重叠<1
  3. sample_weights：唯一性权重（可叠时间衰减）
  4. 间隔样本（不共享 bar）→ 全 1
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.research.sample_weights import (  # noqa: E402
    average_uniqueness,
    concurrency,
    sample_weights,
)


def _idx(n=10):
    return pd.date_range("2024-01-01", periods=n, freq="B")


def _t1_sparse(start_t0, t1_offsets, full_idx=None):
    """从指定 bar 起的稀疏样本：t1 = t0 位置 + offset（交易日）。"""
    if full_idx is None:
        full_idx = start_t0
    vals = []
    for i, o in enumerate(t1_offsets):
        pos = full_idx.get_indexer([start_t0[i]])[0]
        vals.append(full_idx[min(pos + o, len(full_idx) - 1)])
    return pd.Series(vals, index=start_t0)


def test_concurrency_sparse_non_overlap():
    """间隔样本（每样本占 [t0,t0+1]，t0 间隔 2）→ 无共享 bar → 全 1。"""
    idx = _idx(10)
    # 样本 t0=0,2,4,6，各持 1 根（t1=t0+1）→ 区间 [0,1],[2,3],[4,5],[6,7]
    t1 = _t1_sparse(idx[[0, 2, 4, 6]], [1, 1, 1, 1], idx)
    cc = concurrency(t1, idx)
    assert (cc.iloc[:8] == 1).all()
    assert (cc.iloc[8:] == 0).all()  # 无样本的尾部 bar 并发=0


def test_concurrency_overlap_counts():
    """相邻样本（t0=0,1,2 各持 1 根）→ 共享 bar 并发>1。"""
    idx = _idx(8)
    # 样本覆盖 [0,1],[1,2],[2,3] → bar1、bar2 并发=2
    t1 = _t1_sparse(idx[[0, 1, 2]], [1, 1, 1], idx)
    cc = concurrency(t1, idx)
    assert cc.iloc[0] == 1
    assert cc.iloc[1] == 2  # 样本0(到1)+样本1(从1起)
    assert cc.iloc[3] == 1


def test_average_uniqueness_sparse_is_one():
    idx = _idx(10)
    t1 = _t1_sparse(idx[[0, 2, 4, 6]], [1, 1, 1, 1], idx)
    u = average_uniqueness(t1, idx)
    assert (u == 1.0).all()


def test_overlap_reduces_uniqueness():
    idx = _idx(8)
    t1 = _t1_sparse(idx[[0, 1, 2]], [1, 1, 1], idx)
    u = average_uniqueness(t1, idx)
    # 中间样本 [1,2] 两根 bar 都并发=2 → uniqueness = (1/2+1/2)/2 = 0.5
    assert u.iloc[0] < 1.0
    assert u.iloc[1] == pytest.approx(0.5)
    # 端点样本 [0,1]：bar0 并发1、bar1 并发2 → (1 + 0.5)/2 = 0.75
    assert u.iloc[0] == pytest.approx(0.75)


def test_weights_positive_and_decay():
    idx = _idx(10)
    t1 = _t1_sparse(idx[[0, 2, 4, 6, 8]], [1, 1, 1, 1, 1], idx)
    w = sample_weights(t1, idx)  # 无衰减 → = 唯一性 = 全 1
    assert (w > 0).all()
    assert (w == 1.0).all()
    # 时间衰减：早样本权低
    wd = sample_weights(t1, idx, time_decay=0.8)
    assert wd.iloc[0] < wd.iloc[-1]
