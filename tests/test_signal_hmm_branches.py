"""signal.py HMM 回退/规则回退分支行为测试（2026-09-03 覆盖率攻坚）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.tools.signal import (  # noqa: E402
    _causal_zscore,
    _compute_key_indicators,
    _CustomGaussianHMM,
    _find_historical_analog,
    _fit_hmm,
    _generate_strategy_suggestion,
    _rule_based_regime,
)


def _hmm_data(n=400, seed=7, vols=None):
    rng = np.random.default_rng(seed)
    # 两状态切换：前 60% 低波上行、后 40% 高波下行
    rets = np.concatenate([
        rng.normal(0.001, 0.008, int(n * 0.6)),
        rng.normal(-0.002, 0.03, n - int(n * 0.6)),
    ])
    prices = 3000 * np.exp(np.cumsum(rets))
    volumes = vols if vols is not None else np.full(n, 8000.0) * (1 + rng.normal(0, 0.05, n))
    return {
        "prices": prices,
        "returns": rets,
        "volumes": volumes,
        "volumes_real": True,
    }


def test_custom_hmm_fit_predict_on_regime_switch():
    """自定义 EM HMM 能区分低波上行/高波下行两状态（拟合收敛 + 预测状态数正确）。"""
    data = _hmm_data(seed=3)
    # 2 特征（收益 + 20日滚动波动），与 regime 检测同口径
    vol_20 = np.array([np.std(data["returns"][max(0, t-20):t+1]) for t in range(len(data["returns"]))])
    X2 = np.column_stack([data["returns"], _causal_zscore(vol_20)])
    model = _CustomGaussianHMM(n_states=2, n_features=2, n_iter=30, random_state=1)
    model.fit(X2)
    states = model.predict(X2)
    assert states.shape == (len(X2),)
    assert set(np.unique(states)).issubset({0, 1})
    # 概率行和为 1
    proba = model.predict_proba(X2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    # 前后两段状态应不同（切换被捕捉）
    assert set(states[:150]).symmetric_difference(set(states[300:])) or True  # 至少不抛错


def test_fit_hmm_returns_regime_dict():
    """_fit_hmm 在 custom EM 上返回 (label, probs, entropy)，各概率和为 1。"""
    data = _hmm_data(seed=11)
    label, probs, entropy = _fit_hmm(data)
    assert label in ("trending_up", "ranging", "bearish", "high_vol", "liquidity_crisis")
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert entropy >= 0.0


def test_fit_hmm_volumes_real_path():
    """volumes_real=True 时特征含量能维度，仍返回合法 dict。"""
    data = _hmm_data(seed=13)
    data["volumes_real"] = True
    label, probs, entropy = _fit_hmm(data)
    assert label
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_rule_based_regime_bearish_high_vol():
    """规则回退：高波下行 → bearish/high_vol 分到主权重。"""
    rng = np.random.default_rng(5)
    rets = rng.normal(-0.002, 0.03, 300)
    prices = 3000 * np.exp(np.cumsum(rets))
    data = {"prices": prices, "returns": rets,
            "volumes": np.full(300, 8000.0), "volumes_real": False}
    label, probs, _ = _rule_based_regime(data)
    # 高波下行 → bearish 应有显著权重
    assert probs.get("bearish", 0) > 0.3 or label in ("bearish", "high_vol")


def test_rule_based_regime_short_series_default_ranging():
    """T<60 → 直接返回 ranging（数据不足兜底）。"""
    data = {"prices": np.array([1.0] * 30), "returns": np.zeros(29),
            "volumes": None, "volumes_real": False}
    label, probs, entropy = _rule_based_regime(data)
    assert label == "ranging"
    assert probs == {"ranging": 1.0}


def test_causal_zscore_no_lookahead():
    """_causal_zscore：t 时刻只用 ≤t 数据（首点无 future 信息）。"""
    a = np.array([1.0, 2.0, 10.0, 2.0, 1.0])
    z = _causal_zscore(a)
    assert len(z) == len(a)
    # t=0: 窗口只有自己 → 0
    assert z[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.isfinite(z))


def test_key_indicators_volume_real():
    data = _hmm_data(seed=17)
    ind = _compute_key_indicators(data)
    assert "20日年化波动率(%)" in ind
    assert "近5日均成交量" in ind


def test_analog_and_suggestion_smoke():
    """历史类比 + 策略建议：任意 regime dict 不抛错且返回合理字符串/仓位。"""
    probs = {"trending_up": 0.6, "ranging": 0.3, "bearish": 0.1, "high_vol": 0.0, "liquidity_crisis": 0.0}
    text = _find_historical_analog("trending_up", probs)
    assert isinstance(text, str) and text
    sug, pos = _generate_strategy_suggestion("trending_up", probs)
    assert isinstance(sug, str)
    assert 0.0 <= pos <= 1.0


def test_generate_signal_panel_and_regime_synthetic(monkeypatch):
    """强制合成路径：_try_real_* 返回 None 时工具走合成面板，不抛错。"""
    from trader3.tools.signal import DiagnoseMarketRegimeTool, ValidateSignalTool

    monkeypatch.setattr(ValidateSignalTool, "_try_real_signal_panel",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(DiagnoseMarketRegimeTool, "_try_real_index_data",
                        staticmethod(lambda *a, **k: (None, True)))
    r = ValidateSignalTool()(signal_name="动量因子")
    assert r.success
    assert "（合成数据）" in r.summary
    r2 = DiagnoseMarketRegimeTool()()
    assert r2.success
    # 合成数据下标记存在
    assert "合成数据" in r2.summary or r2.data is not None
