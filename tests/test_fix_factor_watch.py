"""因子衰减监控回归测试 — 全离线（注入合成面板构建器）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.v2 import factor_watch as fw  # noqa: E402


def _make_builder(T=120, N=10, seed=7, ic_strength=0.0):
    """合成面板：信号 = ic_strength * 真实次日IC + 噪声。ic_strength=1 → IC≈1。"""
    rng = np.random.default_rng(seed)

    def builder(lookback_days, universe, n_stocks):
        closes = 100.0 * np.cumprod(
            1 + rng.normal(0.0004, 0.015, size=(T, N)), axis=0)
        scores = np.full((T, N), np.nan)
        rets = np.zeros((T, N))
        rets[1:] = closes[1:] / closes[:-1] - 1.0
        for t in range(T - 1):
            noise = rng.normal(size=N)
            scores[t] = ic_strength * rets[t + 1] * 20 + np.sqrt(max(1 - ic_strength**2, 0.01)) * noise
        dates = [f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(T)]
        return dates, scores, closes, np.ones((T, N), dtype=bool)

    return builder


def test_perfect_signal_ic_near_one():
    res = fw.compute_ic_series(panel_builder=_make_builder(ic_strength=1.0))
    assert res["n_obs"] > 50
    assert res["ic_mean"] > 0.85


def test_noise_signal_ic_near_zero():
    res = fw.compute_ic_series(panel_builder=_make_builder(ic_strength=0.0, seed=11))
    assert abs(res["ic_mean"]) < 0.25


def test_decay_alert_threshold():
    assert fw.decay_alert(-0.05, threshold=0.0) is True
    assert fw.decay_alert(0.03, threshold=0.0) is False
    assert fw.decay_alert(None, threshold=0.0) is False


def test_append_history_and_recent_mean(tmp_path):
    rec1 = {"last_date": "2026-08-20", "ic_last": -0.10}
    fw.append_ic_history(str(tmp_path), "f1", rec1, decay_window=3)
    rec2 = {"last_date": "2026-08-21", "ic_last": -0.06}
    r2 = fw.append_ic_history(str(tmp_path), "f1", rec2, decay_window=3)
    # 仅两条负值 → 近窗均值为负 → 告警
    assert r2["alert"] is True
    assert r2["ic_mean_recent"] == pytest.approx(-0.08, abs=1e-6)

    # 追加正值修复后告警解除
    fw.append_ic_history(str(tmp_path), "f1",
                         {"last_date": "2026-08-22", "ic_last": 0.30},
                         decay_window=3)
    r3 = fw.append_ic_history(str(tmp_path), "f1",
                              {"last_date": "2026-08-23", "ic_last": 0.25},
                              decay_window=3)
    assert r3["alert"] is False
    assert r3["ic_mean_recent"] > 0


def test_real_expr_default_builder_smoke():
    """默认 qlib 构建器冒烟（真实数据，小窗口）。"""
    import os
    qlib_bin = os.environ.get(
        "QLIB_BIN",
        str(Path(__file__).resolve().parent.parent.parent / "2hao-analyst" / "data" / "qlib_bin"),
    )
    if not Path(qlib_bin).exists():
        import pytest
        pytest.skip("无本地 qlib 数据")
    res = fw.compute_ic_series(lookback_days=140, n_stocks=20)
    assert res["n_obs"] >= 60
    assert res["ic_mean"] is not None and abs(res["ic_mean"]) < 0.5


