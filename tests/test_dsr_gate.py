"""
DSR 验收闸门接入 StrategySelector — 回归测试

解剖结论 S2 落地：GP 候选入库前必须过 Deflated Sharpe（惩罚整个进化实验
的累计试验次数），否则"完美回测"大概率是挖掘噪声。

契约：
  1. StrategySelector 新增可选 n_trials（进化累计个体评估数），注入后启用
     dsr 门禁；未注入保持历史行为（向后兼容）
  2. 候选携带 ic_series（逐期 IC 时序）时计算 DSR；缺 ic_series 则跳过该门禁
  3. 高 DSR（真实 alpha）候选通过；噪声候选在大量试验后 DSR 低 → 拒
  4. 纯噪声序列 n_trials 大 → 拒；同序列 n_trials=1 → 放行（验证惩罚有效性）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "evolve"))

from core.selection import StrategySelector  # noqa: E402


def _noisy_candidate(cid="c1", ic_series=None, expr="rank(close)"):
    """构造带 IC 时序的候选。"""
    return {
        "expr": expr, "ic": 0.05, "icir": 0.3, "monotonicity": 0.5,
        "long_short": 0.2, "fitness": 1.0,
        "values": np.random.default_rng(1).normal(size=100),
        "ic_series": ic_series,
    }


def _real_alpha_ic_series(n=80, seed=1):
    """真实 alpha：逐期 IC 均值为正且稳定。"""
    rng = np.random.default_rng(seed)
    return rng.normal(0.05, 0.08, n)  # mean=0.05, std=0.08 → 正显著


def test_no_dsr_gate_by_default():
    """未注入 n_trials → 候选正常评估，无 dsr 键（向后兼容）。"""
    sel = StrategySelector()
    cand = _noisy_candidate(ic_series=_real_alpha_ic_series())
    res = sel._evaluate_one(cand)
    assert "dsr" not in res.gates
    assert res.passed  # 既有 IC/ICIR 门槛能过


def test_dsr_gate_passes_real_alpha():
    """真实 alpha：DSR 高 → 通过。"""
    sel = StrategySelector(n_trials=100, dsr_min=0.9)
    cand = _noisy_candidate(ic_series=_real_alpha_ic_series(seed=3))
    res = sel._evaluate_one(cand)
    assert "dsr" in res.gates
    assert res.gates["dsr"]["passed"] is True


def test_dsr_gate_rejects_noise_after_many_trials():
    """纯噪声 IC 序列 + 大量试验 → DSR 低 → 拒。"""
    rng = np.random.default_rng(9)
    noise = rng.normal(0.0, 0.08, 100)  # mean≈0 噪声
    sel = StrategySelector(n_trials=500, dsr_min=0.9)
    cand = _noisy_candidate(ic_series=noise)
    res = sel._evaluate_one(cand)
    assert res.gates["dsr"]["passed"] is False
    assert res.passed is False  # 被 DSR 闸门拦住


def test_trials_penalty_monotonic():
    """同一弱 alpha 序列：试验越多越难通过。"""
    rng = np.random.default_rng(11)
    weak = rng.normal(0.02, 0.08, 100)  # 弱 alpha
    sel_1 = StrategySelector(n_trials=1, dsr_min=0.5)
    sel_many = StrategySelector(n_trials=1000, dsr_min=0.5)
    r1 = sel_1._evaluate_one(_noisy_candidate(ic_series=weak))
    rmany = sel_many._evaluate_one(_noisy_candidate(ic_series=weak))
    # n_trials=1 通过时，n_trials=1000 可能更严；验证 DSR 值单调递减方向
    assert r1.gates["dsr"]["value"] >= rmany.gates["dsr"]["value"]


def test_missing_ic_series_skips_dsr():
    """候选无 ic_series → DSR 门禁跳过（不误杀只带聚合值的候选）。"""
    sel = StrategySelector(n_trials=100, dsr_min=0.9)
    cand = _noisy_candidate()  # 无 ic_series
    cand["ic_series"] = None
    res = sel._evaluate_one(cand)
    assert "dsr" not in res.gates or res.gates["dsr"]["passed"] is True
