"""HRP 组合层测试 — 离线零外部依赖（仅 numpy/pytest）。

覆盖：权重归一非负、两资产逆方差二叉性质、高相关资产配额相近、
NaN/常数列剔除、小样本等权、名称映射、纯 numpy 回退一致性、
quasi-diagonal 完备性、输入校验、riskfolio 缺失时的报错文案。
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import trader3.portfolio.hrp as hrp_mod
from trader3.portfolio.hrp import hrp_weights, riskfolio_weights


def _random_returns(t: int = 750, n: int = 8, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.01, size=(t, n))


def test_weights_sum_one_nonneg() -> None:
    w = hrp_weights(_random_returns())
    assert set(w) == {f"asset_{i}" for i in range(8)}
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(v >= 0.0 for v in w.values())


def test_two_assets_inverse_vol_consistency() -> None:
    # HRP 二叉性质：两资产时根节点按子簇方差的反比分配质量，
    # 故 w_lo / w_hi = σ_hi² / σ_lo²（逆方差，即"逆波动率"的平方形式）。
    rng = np.random.default_rng(11)
    t = 6000
    sigma_lo, sigma_hi = 0.01, 0.03
    r = np.column_stack(
        [rng.normal(0.0, sigma_lo, t), rng.normal(0.0, sigma_hi, t)]
    )
    w = hrp_weights(r, names=["low_vol", "high_vol"])
    assert w["low_vol"] > w["high_vol"]
    expected = (sigma_hi / sigma_lo) ** 2
    assert w["low_vol"] / w["high_vol"] == pytest.approx(expected, rel=0.10)


def test_correlated_pair_gets_similar_allocation() -> None:
    # 共同因子驱动的两高相关资产聚成同簇，簇内逆方差分配 ⇒ 权重彼此接近。
    rng = np.random.default_rng(3)
    t = 2000
    factor = rng.normal(0.0, 0.01, t)
    r = np.column_stack(
        [
            factor + rng.normal(0.0, 0.002, t),
            factor + rng.normal(0.0, 0.002, t),
            rng.normal(0.0, 0.01, t),
        ]
    )
    w = hrp_weights(r, names=["a", "b", "mkt"])
    assert abs(w["a"] - w["b"]) < 0.05
    assert 0.30 < w["a"] + w["b"] < 0.70


def test_nan_column_excluded() -> None:
    r = _random_returns(n=5, seed=21)
    r[:, 2] = np.nan
    w = hrp_weights(r, names=[f"s{i}" for i in range(5)])
    assert "s2" not in w
    assert set(w) == {"s0", "s1", "s3", "s4"}
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)


def test_constant_column_excluded() -> None:
    # 常数列方差为 0，相关系数无定义 ⇒ 与 NaN 列同样剔除（文档化行为）。
    r = _random_returns(n=5, seed=22)
    r[:, 1] = 0.003
    w = hrp_weights(r, names=[f"s{i}" for i in range(5)])
    assert "s1" not in w
    assert set(w) == {"s0", "s2", "s3", "s4"}


def test_small_sample_equal_weight() -> None:
    # T<3（收益期数不足）⇒ 协方差/相关阵无统计意义，退化为有效资产等权。
    rng = np.random.default_rng(5)
    r = rng.normal(0.0, 0.01, size=(2, 5))
    w = hrp_weights(r, names=[f"s{i}" for i in range(5)])
    assert w == {f"s{i}": 0.2 for i in range(5)}
    # 单资产：全部权重
    assert hrp_weights(rng.normal(0.0, 0.01, size=(100, 1))) == {"asset_0": 1.0}


def test_names_propagated() -> None:
    r = _random_returns(n=6, seed=9)
    names = [f"stock_{i}" for i in range(6)]
    w_named = hrp_weights(r, names=names)
    w_auto = hrp_weights(r)
    assert list(w_named) == names  # dict 保序：键序与列序一致
    assert w_named == {names[i]: v for i, v in enumerate(w_auto.values())}


def test_all_invalid_returns_empty() -> None:
    r = _random_returns(n=4, seed=31)
    r[:] = np.nan
    assert hrp_weights(r) == {}


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="二维"):
        hrp_weights(np.zeros(10))
    with pytest.raises(ValueError, match="长度"):
        hrp_weights(_random_returns(n=3), names=["a", "b"])
    with pytest.raises(ValueError, match="重复"):
        hrp_weights(_random_returns(n=3), names=["a", "a", "b"])


def test_pure_numpy_fallback_matches_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    # 连续随机距离几乎不可能并列 ⇒ 两条聚类路径应产出同一棵树与权重。
    r = _random_returns(seed=13)
    ref = hrp_weights(r)
    monkeypatch.setattr(hrp_mod, "_scipy_linkage", None)
    w = hrp_weights(r)
    assert set(w) == set(ref)
    for k in ref:
        assert w[k] == pytest.approx(ref[k], rel=1e-7, abs=1e-9)


def test_quasi_diag_covers_all_leaves() -> None:
    r = _random_returns(n=7, seed=17)
    corr = np.corrcoef(r, rowvar=False)
    dist = hrp_mod._corr_distance(corr)
    children, _ = hrp_mod._build_tree(dist)
    order = hrp_mod._quasi_diag(children, 7)
    assert sorted(order) == list(range(7))


def test_riskfolio_import_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "riskfolio", None)
    with pytest.raises(ImportError, match=r"trader3\[portfolio\]"):
        riskfolio_weights(_random_returns(n=4), method="HRP")
