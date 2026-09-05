"""
3号交易员 — 深度模型打分器（LSTM）

evolve 因子工厂的深度学习层：把量价面板 (T, N) 训练成横截面打分模型，
输出与 GP 因子同语义的得分面板，供 StrategySelector / 回测管线消费。

设计约束：
  1. 无前视：t 行得分只由 <=t 的数据决定（训练只用 [0, train_end) 段）
  2. 确定性：同种子同输入同输出（torch 设 seed + 单线程）
  3. CPU 友好：小模型（hidden=16, layers=1），纯 CPU 可训
  4. torch 可用时训练真 LSTM；缺失时确定性 numpy 动量特征回退（显式打标）
  5. 输出横截面 rank 标准化（与 gp._cs_rank 同语义），NaN 保留

接口对齐 core.gp.compute_fitness 的面板语义：panel = {field: (T, N)}。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .gp import _cs_rank

# ── 特征工程（两条路径共用）─────────────────────────────


def build_features(panel: dict[str, np.ndarray]) -> np.ndarray:
    """
    量价面板 → 特征张量 (T, N, F)。

    特征（全部由 <=t 数据构成，无前视）：
      0. 20日动量 close/close[t-20]-1
      1. 5日动量
      2. 20日波动率（日收益 std）
      3. 量能变化（5日均量/20日均量 - 1）
      4. 振幅（high-low）/close 20日均值
    """
    close = np.asarray(panel["close"], dtype=np.float64)
    volume = np.asarray(panel.get("volume", close), dtype=np.float64)
    high = np.asarray(panel.get("high", close), dtype=np.float64)
    low = np.asarray(panel.get("low", close), dtype=np.float64)
    T, N = close.shape

    def _shift(a: np.ndarray, k: int) -> np.ndarray:
        out = np.full_like(a, np.nan)
        if k < T:
            out[k:] = a[:-k] if k > 0 else a
        return out

    def _rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
        out = np.full_like(a, np.nan)
        for t in range(w - 1, T):
            out[t] = np.nanmean(a[t - w + 1 : t + 1], axis=0)
        return out

    def _rolling_std(a: np.ndarray, w: int) -> np.ndarray:
        out = np.full_like(a, np.nan)
        for t in range(w - 1, T):
            out[t] = np.nanstd(a[t - w + 1 : t + 1], axis=0)
        return out

    rets = np.full_like(close, np.nan)
    rets[1:] = close[1:] / close[:-1] - 1.0

    feats = np.stack(
        [
            close / _shift(close, 20) - 1.0,
            close / _shift(close, 5) - 1.0,
            _rolling_std(rets, 20),
            _rolling_mean(volume, 5) / _rolling_mean(volume, 20) - 1.0,
            _rolling_mean((high - low) / close, 20),
        ],
        axis=-1,
    )
    # 逐特征横截面 zscore（rank 鲁棒化），NaN 安全（暖机期整行 NaN → 置 0）
    F = feats.shape[-1]
    with np.errstate(invalid="ignore"):
        for f in range(F):
            sl = feats[:, :, f]
            mu = np.nan_to_num(np.nanmean(sl, axis=1, keepdims=True), nan=0.0)
            sd = np.nan_to_num(np.nanstd(sl, axis=1, keepdims=True), nan=0.0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            feats[:, :, f] = (sl - mu) / sd
    # 剩余 NaN（前 20 日暖机）置 0：暖机期不携带信息，不引入未来
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def _mask_from_fwd(fwd: np.ndarray | None, T: int) -> np.ndarray:
    """fwd[t] 有效（非 NaN，行内至少一半有限）的行参与训练；缺 fwd 时全 True。

    无前视：fwd[t] 是 t→t+1 的收益，训练时仅用于 [0, train_end) 段，
    train_end 之后的行不进入损失，未来数据不影响模型参数，
    因而也不影响 [0, train_end) 段的得分（截尾重训一致性由此保证）。
    """
    if fwd is None:
        return np.ones(T, dtype=bool)
    with np.errstate(invalid="ignore"):
        row_valid = np.isfinite(fwd).sum(axis=1) >= max(1, fwd.shape[1] // 2)
    return np.asarray(row_valid, dtype=bool)


# ── torch 路径：真 LSTM ──────────────────────────────


class _TorchLstm:
    """torch LSTM 打分器：输入 (T,N,F) → 逐日横截面打分 (T,N)。"""

    def __init__(self, n_feat: int, hidden: int, seed: int):
        import torch  # 延迟导入；仅本路径需要

        torch.manual_seed(seed)
        torch.set_num_threads(1)
        self.torch = torch
        self.lstm = torch.nn.LSTM(n_feat, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, 1)

    def fit(self, feats: np.ndarray, target: np.ndarray, mask: np.ndarray,
            train_end: int, epochs: int, lr: float, seed: int) -> None:
        import torch
        torch.manual_seed(seed)
        X = torch.from_numpy(feats[:train_end]).float()          # (T1, N, F)
        y = torch.from_numpy(target[:train_end]).float()          # (T1, N)
        w = torch.from_numpy(mask[:train_end]).float()            # (T1,)
        opt = torch.optim.Adam(list(self.lstm.parameters()) + list(self.head.parameters()), lr=lr)
        lossf = torch.nn.MSELoss()
        rows = w > 0
        if rows.sum() < 10:
            return
        for _ in range(epochs):
            opt.zero_grad()
            out, _ = self.lstm(X)                                   # (T1, N, H)
            pred = self.head(out).squeeze(-1)                       # (T1, N)
            loss = lossf(pred[rows], y[rows])
            loss.backward()
            opt.step()
        self._fitted = True

    def predict(self, feats: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            X = torch.from_numpy(feats).float()
            out, _ = self.lstm(X)
            return self.head(out).squeeze(-1).numpy()


# ── numpy 回退路径：正则化动量回归（确定性）────────────


class _NumpyFallback:
    """torch 缺失时的确定性回退：ridge 回归（闭式解，无随机性）。"""

    def __init__(self, seed: int):
        self.seed = seed
        self.w: np.ndarray | None = None

    def fit(self, feats: np.ndarray, target: np.ndarray, mask: np.ndarray,
            train_end: int, epochs: int = 0, lr: float = 0.0, seed: int = 0) -> None:
        X = feats[:train_end][mask[:train_end]]                    # (M, N, F) → 展平
        y = target[:train_end][mask[:train_end]]
        Xf = X.reshape(-1, X.shape[-1])
        yf = y.reshape(-1)
        keep = np.isfinite(yf)
        Xf, yf = Xf[keep], yf[keep]
        lam = 1e-2
        gram = Xf.T @ Xf + lam * np.eye(Xf.shape[1])
        self.w = np.linalg.solve(gram, Xf.T @ yf)

    def predict(self, feats: np.ndarray) -> np.ndarray:
        T, N, F = feats.shape
        return (feats @ self.w).reshape(T, N)


# ── 统一入口 ─────────────────────────────────────────


class LstmScorer:
    """LSTM 横截面打分器（torch 优先，numpy 确定性回退）。

    Parameters
    ----------
    seed : 随机种子（torch 路径与回退路径均确定性）
    epochs : torch 训练轮数
    hidden : LSTM 隐层数
    """

    def __init__(self, seed: int = 42, epochs: int = 3, hidden: int = 16):
        self.seed = seed
        self.epochs = epochs
        self.hidden = hidden
        self.backend: str = ""
        self.fallback_used: bool = False

    def fit_transform(
        self,
        panel: dict[str, np.ndarray],
        fwd: np.ndarray | None = None,
        train_end: int | None = None,
    ) -> np.ndarray:
        """训练（只用 [0, train_end) 与有效 fwd 行）→ 输出全期得分 (T, N)。

        无前视保证：训练目标取自 fwd 的有效行，且 train_end 之后的行
        不参与任何参数更新；预测阶段逐日独立打分。
        """
        feats = build_features(panel)
        T, N, F = feats.shape
        if fwd is None:
            # 无目标时用次日收益自举（与 data_loader 的 fwd 语义一致）
            close = np.asarray(panel["close"], dtype=np.float64)
            fwd = np.full_like(close, np.nan)
            fwd[:-1] = close[1:] / close[:-1] - 1.0
        mask = _mask_from_fwd(fwd, T)
        train_end = T if train_end is None else int(train_end)
        target = np.nan_to_num(np.asarray(fwd, dtype=np.float64),
                               nan=0.0, posinf=0.0, neginf=0.0)

        try:
            model = _TorchLstm(F, self.hidden, self.seed)
            model.fit(feats, target, mask, train_end, self.epochs, 1e-3, self.seed)
            raw = model.predict(feats)
            self.backend = "torch"
            self.fallback_used = False
        except ImportError:
            fb = _NumpyFallback(self.seed)
            fb.fit(feats, target, mask, train_end)
            raw = fb.predict(feats)
            self.backend = "numpy"
            self.fallback_used = True
        return _cs_rank(np.asarray(raw, dtype=np.float64))


def train_lstm_scorer(
    panel: dict[str, np.ndarray],
    fwd: np.ndarray | None = None,
    seed: int = 42,
    epochs: int = 3,
    train_end: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """evolve 管线入口：返回 (得分面板, 元信息)。元信息含 backend/耗时/回退标注。"""
    t0 = time.time()
    scorer = LstmScorer(seed=seed, epochs=epochs)
    scores = scorer.fit_transform(panel, fwd=fwd, train_end=train_end)
    meta: dict[str, Any] = {
        "backend": scorer.backend,
        "fallback_used": scorer.fallback_used,
        "elapsed": round(time.time() - t0, 3),
    }
    if scorer.fallback_used:
        meta["backend_note"] = "torch 不可用 → 确定性 numpy ridge 回退（非深度模型，禁标 LSTM）"
    return scores, meta
