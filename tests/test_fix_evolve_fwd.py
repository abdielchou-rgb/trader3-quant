"""
审计修复回归测试 — evolve 工厂
覆盖：fwd 前向性、交易成本生效、负 delay 拒绝、窗口不相交校验、
适应度符号一致性、相关性去重。
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "evolve"))

from core.gp import _delay, compute_fitness  # noqa: E402
from core.selection import StrategySelector  # noqa: E402


def _make_panel(close: np.ndarray) -> dict:
    return {
        "close": close,
        "open": close.copy(),
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": np.full_like(close, 1e6),
    }


def _random_walk(T=260, N=8, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, size=(T, N))
    close = 100.0 * np.cumprod(1.0 + rets, axis=0)
    return close, rets


def test_fwd_is_forward_looking():
    """同期泄露已消除：IC(close_t, fwd_t) 接近 0；完美因子（次日收益）IC 接近 1。"""
    close, _ = _random_walk()
    T, N = close.shape
    fwd = np.full_like(close, np.nan)
    fwd[:-1] = close[1:] / close[:-1] - 1.0
    panel = _make_panel(close)

    fit_close = compute_fitness("close", panel, fwd)
    assert abs(fit_close["ic"]) < 0.25, f"同期泄露未消除: ic={fit_close['ic']}"

    fit_perfect = compute_fitness(fwd, panel, fwd)
    assert fit_perfect["ic"] > 0.90, f"完美因子 IC 异常: {fit_perfect['ic']}"


def test_trading_cost_applied():
    """成本>0 的净值必须低于成本=0。"""
    from validate_evolved_factors import monthly_rebalance

    rng = np.random.default_rng(3)
    T, N = 200, 10
    signal = rng.normal(size=(T, N))
    close = 100.0 * np.cumprod(1 + rng.normal(0.0004, 0.015, size=(T, N)), axis=0)

    eq_free, _, turn_free, _ = monthly_rebalance(signal, close, 5, cost=0.0)
    eq_cost, _, turn_cost, _ = monthly_rebalance(signal, close, 5, cost=0.0015)
    assert turn_cost > 0 and turn_free > 0, "换手为 0，成本模型无从验证"
    assert eq_cost[-1] < eq_free[-1], "交易成本未生效"


def test_negative_delay_raises():
    a = np.ones((10, 3))
    with pytest.raises(ValueError):
        _delay(a, -1)


def test_window_overlap_rejected():
    """验证起点 <= 训练末 → argparse error，退出码 2，且发生在数据加载前。"""
    script = _ROOT / "evolve" / "validate_evolved_factors.py"
    result = subprocess.run(
        [sys.executable, str(script),
         "--start", "2020-01-01", "--train-end", "2023-12-31"],
        capture_output=True, cwd=str(_ROOT),
    )
    assert result.returncode == 2, f"应拒绝重叠窗口: rc={result.returncode}"
    err = result.stderr or b""
    text = None
    for enc in ("utf-8", "gbk"):
        try:
            text = err.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    assert text and "样本外" in text


def test_fitness_sign_consistency():
    """强负 IC 因子的 fitness 必须显著低于零信号。"""
    close, _ = _random_walk(T=300, N=8, seed=11)
    T, N = close.shape
    fwd = np.full_like(close, np.nan)
    fwd[:-1] = close[1:] / close[:-1] - 1.0
    panel = _make_panel(close)

    f_neg = compute_fitness(-np.nan_to_num(fwd, nan=0.0), panel, fwd)
    rng = np.random.default_rng(99)
    f_noise = compute_fitness(rng.normal(size=(T, N)), panel, fwd)
    assert f_neg["fitness"] < f_noise["fitness"], (
        f"负IC因子({f_neg['fitness']}) 不应高于噪声信号({f_noise['fitness']})"
    )


def test_dedup_by_correlation():
    """因子值 ρ≈1 但表达式不同的两个候选，第二个被去重拒绝；处理顺序按 fitness 降序。"""
    rng = np.random.default_rng(5)
    vals = rng.normal(size=100)
    cand_a = {"expr": "rank(ts_mean(close, 10))", "ic": 0.05, "icir": 0.30,
              "monotonicity": 0.60, "long_short": 0.20, "fitness": 1.0, "values": vals}
    cand_b = {"expr": "rank(ts_mean(close, 60))", "ic": 0.05, "icir": 0.30,
              "monotonicity": 0.60, "long_short": 0.20, "fitness": 0.90,
              "values": vals + rng.normal(0, 1e-9, size=100)}

    selector = StrategySelector()
    results = selector.select([cand_a, cand_b])
    assert results[0].passed, "第一个候选应通过"
    assert not results[1].passed, "ρ≈1 的第二个候选应被去重拒绝"
