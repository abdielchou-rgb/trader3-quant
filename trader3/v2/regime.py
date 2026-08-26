"""
市场状态（Regime）检测 — 高斯隐马尔可夫模型。

纯 numpy/scipy 实现 Baum-Welch (EM) 训练 + Viterbi 解码，
无 hmmlearn 等外部依赖。

典型用法：对指数/组合日收益序列训练 K=2~3 状态 HMM，
输出各状态均值/波动率及每日最可能状态，供仓位调节与风控开关使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

MIN_VARIANCE = 1e-8


@dataclass
class RegimeResult:
    """Regime 检测结果。"""
    states: np.ndarray            # 每期解码状态 (T,)
    state_probs: np.ndarray       # 后验概率 (T, K)
    means: np.ndarray             # 各状态收益均值 (K,)
    vols: np.ndarray              # 各状态波动率 (K,)
    transition: np.ndarray        # 转移矩阵 (K, K)
    stationary: np.ndarray        # 平稳分布 (K,)
    log_likelihood: float         # 最终 log-likelihood
    n_iter: int                   # EM 迭代次数
    labels_by_vol: dict[int, str] # 状态 -> 语义标签（low/mid/high vol）


def _init_params(returns: np.ndarray, n_states: int, seed: int):
    rng = np.random.default_rng(seed)
    len(returns)
    mu = np.quantile(returns, np.linspace(0.1, 0.9, n_states))
    sigma = np.full(n_states, returns.std() + MIN_VARIANCE)
    pi = np.full(n_states, 1.0 / n_states)
    A = np.full((n_states, n_states), 0.9 / max(n_states - 1, 1))
    np.fill_diagonal(A, 0.5)
    A /= A.sum(axis=1, keepdims=True)
    return mu, sigma, pi, A, rng


def _forward_backward(returns: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                      pi: np.ndarray, A: np.ndarray):
    """前向-后向算法，返回 alpha, beta, loglik, gamma。数值稳定版（log域归一化）。"""
    t = len(returns)
    k = len(mu)
    log_b = -0.5 * ((returns[:, None] - mu[None, :]) / sigma[None, :]) ** 2 \
            - np.log(sigma)[None, :] - 0.5 * np.log(2 * np.pi)

    log_alpha = np.full((t, k), -np.inf)
    log_alpha[0] = np.log(pi + 1e-300) + log_b[0]
    for i in range(1, t):
        m = log_alpha[i - 1][:, None] + np.log(A + 1e-300)
        log_alpha[i] = np.logaddexp.reduce(m, axis=0) + log_b[i]

    log_beta = np.zeros((t, k))
    for i in range(t - 2, -1, -1):
        m = np.log(A + 1e-300) + (log_beta[i + 1] + log_b[i + 1])[None, :]
        log_beta[i] = np.logaddexp.reduce(m, axis=1)

    loglik = np.logaddexp.reduce(log_alpha[-1])
    log_gamma = log_alpha + log_beta - loglik
    gamma = np.exp(log_gamma)

    xi_sum = np.zeros((k, k))
    for i in range(t - 1):
        m = log_alpha[i][:, None] + np.log(A + 1e-300) \
            + (log_b[i + 1] + log_beta[i + 1])[None, :] - loglik
        xi_sum += np.exp(m)
    return loglik, gamma, xi_sum


class GaussianHMM:
    """高斯HMM，支持 fit/predict/predict_proba。"""

    def __init__(self, n_states: int = 3, max_iter: int = 200, tol: float = 1e-6,
                 n_init: int = 5, random_state: int = 42,
                 vol_floor_quantile: float = 0.05):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.vol_floor_quantile = vol_floor_quantile
        self.mu_: np.ndarray | None = None
        self.sigma_: np.ndarray | None = None
        self.pi_: np.ndarray | None = None
        self.A_: np.ndarray | None = None
        self.loglik_: float = -np.inf
        self.n_iter_: int = 0

    def fit(self, returns: np.ndarray) -> GaussianHMM:
        r = np.asarray(returns, dtype=float).ravel()
        r = r[~np.isnan(r)]
        if len(r) < 10 * self.n_states:
            raise ValueError(f"样本不足: {len(r)} < {10 * self.n_states}")

        best_ll, best_params, best_niter = -np.inf, None, 0
        for init in range(self.n_init):
            ll, params, n_iter = self._fit_single(r, self.random_state + init)
            if ll > best_ll:
                best_ll, best_params, best_niter = ll, params, n_iter
        self.loglik_ = best_ll
        self.mu_, self.sigma_, self.pi_, self.A_ = best_params
        self.n_iter_ = best_niter
        return self

    def _fit_single(self, r: np.ndarray, seed: int):
        mu, sigma, pi, A, _ = _init_params(r, self.n_states, seed)
        floor = max(np.quantile(np.abs(r), self.vol_floor_quantile), MIN_VARIANCE)
        prev_ll = -np.inf
        n_iter = 0
        for it in range(self.max_iter):
            ll, gamma, xi_sum = _forward_backward(r, mu, sigma, pi, A)
            n_iter = it + 1
            # M-step
            s = gamma.sum(axis=0) + 1e-300
            mu = (gamma * r[:, None]).sum(axis=0) / s
            var = (gamma * (r[:, None] - mu[None, :]) ** 2).sum(axis=0) / s
            sigma = np.sqrt(np.clip(var, floor**2, None))
            pi = gamma[0] / gamma[0].sum()
            A = xi_sum / xi_sum.sum(axis=1, keepdims=True)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return ll, (mu, sigma, pi, A), n_iter

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        _, gamma, _ = _forward_backward(
            np.asarray(returns, dtype=float).ravel(), self.mu_, self.sigma_, self.pi_, self.A_
        )
        return gamma

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Viterbi 最优路径。"""
        r = np.asarray(returns, dtype=float).ravel()
        t, k = len(r), self.n_states
        log_b = -0.5 * ((r[:, None] - self.mu_[None, :]) / self.sigma_[None, :]) ** 2 \
                - np.log(self.sigma_)[None, :] - 0.5 * np.log(2 * np.pi)
        log_a = np.log(self.A_ + 1e-300)
        v = np.full((t, k), -np.inf)
        ptr = np.zeros((t, k), dtype=int)
        v[0] = np.log(self.pi_ + 1e-300) + log_b[0]
        for i in range(1, t):
            cand = v[i - 1][:, None] + log_a
            ptr[i] = np.argmax(cand, axis=0)
            v[i] = cand[ptr[i], np.arange(k)] + log_b[i]
        states = np.zeros(t, dtype=int)
        states[-1] = int(np.argmax(v[-1]))
        for i in range(t - 2, -1, -1):
            states[i] = ptr[i + 1, states[i + 1]]
        return states

    @property
    def stationary_(self) -> np.ndarray:
        A = self.A_
        vals, vecs = np.linalg.eig(A.T)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        p = np.real(vecs[:, idx])
        return np.abs(p) / np.abs(p).sum()


def detect_regimes(returns: np.ndarray | pd.Series, n_states: int = 3,
                   random_state: int = 42,
                   label_scheme: Literal["vol", "mean"] = "vol") -> RegimeResult:
    """
    一站式 regime 检测。returns 为日收益序列。

    label_scheme="vol":  按波动率排序标注 low/mid/high-vol
    label_scheme="mean": 按均值排序标注 bear/flat/bull
    """
    if isinstance(returns, pd.Series):
        r_arr = returns.dropna().values
    else:
        r_arr = np.asarray(returns, dtype=float)
        r_arr = r_arr[~np.isnan(r_arr)]

    model = GaussianHMM(n_states=n_states, random_state=random_state).fit(r_arr)
    states = model.predict(r_arr)
    probs = model.predict_proba(r_arr)

    order = np.argsort(model.sigma_)
    names_vol = ["low-vol", "mid-vol", "high-vol"]
    labels = {}
    for rank, st in enumerate(order):
        if n_states == 2:
            labels[int(st)] = ["calm", "turbulent"][min(rank, 1)]
        elif n_states == 3:
            labels[int(st)] = names_vol[min(rank, 2)]
        else:
            labels[int(st)] = f"vol-q{rank + 1}"
    if label_scheme == "mean":
        order_m = np.argsort(model.mu_)
        names_mean = ["bear", "flat", "bull"]
        for rank, st in enumerate(order_m):
            if n_states == 3:
                labels[int(st)] = names_mean[min(rank, 2)]
            else:
                labels[int(st)] = f"mean-q{rank + 1}"

    return RegimeResult(
        states=states,
        state_probs=probs,
        means=model.mu_,
        vols=model.sigma_,
        transition=model.A_,
        stationary=model.stationary_,
        log_likelihood=model.loglik_,
        n_iter=model.n_iter_,
        labels_by_vol={int(k): v for k, v in labels.items()},
    )


def current_regime(result: RegimeResult, lookback: int = 5) -> tuple[str, float]:
    """最近 lookback 天的主导状态及其平均置信度。返回 (label, confidence)。"""
    tail = result.states[-lookback:]
    st = int(np.bincount(tail, minlength=len(result.means)).argmax())
    conf = float(result.state_probs[-lookback:, st].mean())
    return result.labels_by_vol.get(st, f"state-{st}"), conf
