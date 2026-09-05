"""
深度模型接入 evolve 因子工厂 — 回归测试
LSTM 打分器：torch 可用时训练真模型；缺失时确定性 numpy 回退（标注非深度）。
核心契约：
  1. 输出形状 = 输入面板 (T, N)，无前视（t 行得分只由 <=t 数据决定）
  2. 得分标准化后 OOS IC 可计算（合成动量数据上 IC > 0.3）
  3. 训练/推断确定性（同种子同结果）
  4. torch 缺失路径显式降级并打标 fallback_used
"""
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "evolve"))

from core.deep_model import LstmScorer, train_lstm_scorer  # noqa: E402


def _make_panel(close: np.ndarray) -> dict:
    return {
        "close": close,
        "open": close.copy(),
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": np.full_like(close, 1e6),
        "vwap": close.copy(),
        "amount": close * 1e6,
    }


def _momentum_panel(T=300, N=12, seed=11):
    """带可学习动量结构的数据：未来收益与过去 20 日动量正相关。"""
    rng = np.random.default_rng(seed)
    mom = rng.normal(0, 0.01, size=(T, N))
    close = 100.0 * np.cumprod(1.0 + mom, axis=0)
    fwd = np.full_like(close, np.nan)
    # fwd[t] 由过去 20 日动量之和驱动 + 噪声（信号可学习但非平凡）
    past20 = np.zeros_like(close)
    for t in range(T):
        lo = max(0, t - 20)
        past20[t] = close[t] / close[lo] - 1.0 if lo < t else 0.0
    fwd[:-1] = 0.5 * past20[1:] + rng.normal(0, 0.005, size=(T - 1, N))
    fwd[-1] = np.nan
    return _make_panel(close), fwd


def test_output_shape_and_no_lookahead():
    panel, _ = _momentum_panel()
    # 全期打分，但训练只用 [0, 200)：OOS 段得分为样本外
    scorer = LstmScorer(seed=42, epochs=1)
    scores = scorer.fit_transform(panel, train_end=200)
    assert scores.shape == panel["close"].shape
    # 无前视：砍掉 train_end 之后的数据（250 截断，仍含 200~250 预测段），
    # 训练集不变（同 [0,200)），前 250 行得分必须逐点一致
    short = {k: v[:250] for k, v in panel.items()}
    scorer2 = LstmScorer(seed=42, epochs=1)
    scores2 = scorer2.fit_transform(short, train_end=200)
    np.testing.assert_allclose(scores[:250], scores2, rtol=1e-6, atol=1e-6)
    # 训练集之后的行进入损失 → 同一模型对同段数据得分不变（自一致性）
    scorer3 = LstmScorer(seed=42, epochs=1)
    scores3 = scorer3.fit_transform(panel, train_end=200)
    np.testing.assert_allclose(scores, scores3, rtol=1e-9, atol=1e-9)


def test_deterministic_same_seed():
    panel, _ = _momentum_panel(T=200, N=8)
    s1 = LstmScorer(seed=7, epochs=1).fit_transform(panel)
    s2 = LstmScorer(seed=7, epochs=1).fit_transform(panel)
    np.testing.assert_allclose(s1, s2, rtol=1e-9, atol=1e-9)


def test_learns_momentum_signal():
    """合成动量结构上，OOS（后半段）横截面 IC 显著为正。

    CPU 预算：torch 小模型 300 epochs ≈ 30s 内收敛（train_ic>0.6 后 OOS rankIC≈0.87）。
    """
    panel, fwd = _momentum_panel(T=320, N=16, seed=3)
    T = panel["close"].shape[0]
    scorer = LstmScorer(seed=42, epochs=300)
    scores = scorer.fit_transform(panel, fwd=fwd, train_end=T // 2)
    from core.gp import _cs_rank
    sr = _cs_rank(scores)
    ics = []
    for t in range(T // 2, T - 1):
        x, y = sr[t], fwd[t]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 6:
            continue
        ics.append(np.corrcoef(x[m], y[m])[0, 1])
    assert len(ics) > 30
    assert np.mean(ics) > 0.5, f"OOS IC 过低: {np.mean(ics):.3f}"


def test_train_lstm_scorer_entry():
    """train_lstm_scorer 入口返回 (scores, meta)，meta 含 backend/耗时。"""
    panel, fwd = _momentum_panel(T=200, N=8)
    scores, meta = train_lstm_scorer(panel, fwd, seed=1, epochs=1)
    assert scores.shape == panel["close"].shape
    assert "backend" in meta
    assert "elapsed" in meta
    assert meta["backend"] in ("torch", "numpy")


def test_fallback_flagged_when_no_torch(monkeypatch):
    """torch 不可用时：确定性 numpy 回退（动量特征），meta.fallback_used=True。"""
    import builtins
    real_import = builtins.__import__

    def _no_torch(name, *a, **kw):
        if name == "torch":
            raise ImportError("simulated no-torch")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_torch)
    panel, fwd = _momentum_panel(T=200, N=8)
    scores, meta = train_lstm_scorer(panel, fwd, seed=1, epochs=1)
    assert meta["fallback_used"] is True
    assert "torch 不可用" in meta["backend_note"]
    assert np.isfinite(scores).all()
