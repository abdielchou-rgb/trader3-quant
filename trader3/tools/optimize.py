"""
3号交易员 — 组合优化 Tools (M2: 真实组合优化器)

M2 升级:
1. 风险平价 (risk_budget): scipy.optimize.minimize SLSQP 求解最小方差
2. 均值-方差 (mean_variance): 最大化 风险调整后收益
3. Black-Litterman (简化版): 市场等权先验 + 信号观点 + 后验优化
4. 情境路由 (regime_aware_allocation): 概率加权动态分配
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.optimize import minimize

from trader3.base_tool import BaseTool, ChartSpec, Trader3Response
from trader3.models import OptimizationResult, PortfolioConstraints

# ═══════════════════════════════════════════
# Solver backend (cvxpy 可用则升级凸优化路径)
# ═══════════════════════════════════════════

try:
    import cvxpy as cp
except Exception:  # noqa: BLE001 - cvxpy 为可选依赖
    cp = None

SOLVER_BACKEND = "cvxpy" if cp is not None else "scipy"
CVXPY_FALLBACK_NOTE = "cvxpy失败回退SLSQP"


def _refresh_solver_backend() -> str:
    """重新探测 cvxpy 可用性并刷新模块级后端标志（测试隔离用）。"""
    global cp, SOLVER_BACKEND
    import importlib

    try:
        cp = importlib.import_module("cvxpy")
        SOLVER_BACKEND = "cvxpy"
    except Exception:  # noqa: BLE001
        cp = None
        SOLVER_BACKEND = "scipy"
    return SOLVER_BACKEND


# ═══════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════

TRADING_DAYS = 252
DEFAULT_LAMBDA = 1.0            # 风险厌恶系数 (均值-方差)
ANNUAL_ALPHA_SCALE = 0.08       # 信号 Z-score → 年化 alpha 的缩放
BASE_CORRELATION = 0.30         # 假设的资产间相关系数
DEFAULT_MAX_SINGLE = 0.05       # 单票权重上限 5%
TURNOVER_COST_PER_UNIT_BP = 17.0  # 每单位换手的成本 bp
BL_TAU = 0.05                   # Black-Litterman 先验置信度
SLSQP_OPTIONS = {"maxiter": 2000, "ftol": 1e-12, "disp": False}


# ═══════════════════════════════════════════
# 数据构建 Helpers
# ═══════════════════════════════════════════

def _signals_to_mu_sigma(
    signals: dict[str, float],
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    将信号分数转换为期望收益和协方差矩阵。

    信号分数 → Z-score (0 均值 1 标准差) → 年化 alpha
    协方差从基础波动率 + 信号调整 + 相关结构构建。

    Parameters
    ----------
    signals : {ticker: score}
    rng_seed : int, 随机种子保证可复现

    Returns
    -------
    mu : (N,) 年化期望收益
    Sigma : (N, N) 年化协方差矩阵
    tickers : 代码列表
    """
    tickers = list(signals.keys())
    N = len(tickers)
    if N == 0:
        raise ValueError("No signals provided")

    scores = np.array([signals[t] for t in tickers], dtype=np.float64)

    # Z-score 标准化: 让信号在截面可比
    score_std = float(np.std(scores, ddof=1)) if N > 1 else 1.0
    if score_std < 1e-10:
        score_std = 1.0
    z_scores = (scores - np.mean(scores)) / score_std

    # 期望收益 = Z-score * 年化 alpha 缩放
    # 正 Z-score → 预期正 alpha, 负 Z-score → 预期负 alpha
    mu = z_scores * ANNUAL_ALPHA_SCALE

    # 波动率: 基础 20% + 信号极端程度调整
    # 信号越极端 → 不确定性越高 → 波动率越大
    base_vol = 0.20
    max_abs_z = max(np.max(np.abs(z_scores)), 1.0)
    vol_adj = 0.05 * np.abs(z_scores) / max_abs_z
    vols = base_vol + vol_adj

    # 相关矩阵: 基础相关系数 + 随机扰动保证非平凡解
    rng = np.random.default_rng(rng_seed)
    corr = np.full((N, N), BASE_CORRELATION, dtype=np.float64)
    np.fill_diagonal(corr, 1.0)

    noise = rng.uniform(-0.10, 0.10, (N, N))
    noise = (noise + noise.T) / 2  # 对称化
    np.fill_diagonal(noise, 0.0)
    corr += noise
    corr = np.clip(corr, -0.5, 0.95)
    np.fill_diagonal(corr, 1.0)

    # Sigma = diag(vols) @ corr @ diag(vols)
    vol_diag = np.diag(vols)
    Sigma = vol_diag @ corr @ vol_diag

    return mu, Sigma, tickers


def _parse_constraints(
    constraints: PortfolioConstraints | None,
    N: int,
) -> dict:
    """
    解析 PortfolioConstraints 为优化器友好格式。

    用户显式给出的 max_single 不可行（< 1/N）时抛错；
    未显式给出时按可行性自动放宽为 1/N 并在返回值中标注。
    """
    if constraints is None:
        user_max = DEFAULT_MAX_SINGLE
        explicit = False
    else:
        user_max = constraints.max_single_weight or DEFAULT_MAX_SINGLE
        explicit = bool(constraints.max_single_weight)

    min_feasible = 1.0 / N
    adjusted_note = None
    if user_max < min_feasible:
        if explicit:
            raise ValueError(
                f"约束不可行: max_single_weight={user_max:.4f} < 1/N={min_feasible:.4f}"
                f"（{N} 只资产在 sum(w)=1 下无法全部 ≤ 上限），请放宽上限或减少资产数"
            )
        adjusted_note = (
            f"默认单票上限 {user_max:.2%} 在 N={N} 下不可行，"
            f"已自动放宽至 1/N={min_feasible:.2%}"
        )
        user_max = min_feasible

    return {
        "max_single": user_max,
        "user_max_single": user_max,
        "long_only": True if constraints is None or constraints.long_only is None else constraints.long_only,
        "max_sector": constraints.max_sector if constraints and constraints.max_sector else 0.25,
        "min_weight": 0.0,
        "cap_adjusted_note": adjusted_note,
    }


def _verify_constraints(weights: np.ndarray, cons: dict) -> list[str]:
    """对最终权重向量做实测复检，返回结构性违规描述（不信任求解器自报）。"""
    violations: list[str] = []
    cap = cons["max_single"]
    if len(weights) and float(np.max(weights)) > cap + 1e-6:
        violations.append(
            f"单票超限: max_w={float(np.max(weights)):.4f} > {cap:.4f}（归一化后仍破限）"
        )
    if np.any(weights < -1e-9):
        violations.append("出现负权重")
    s = float(np.sum(weights))
    if abs(s - 1.0) > 1e-6:
        violations.append(f"权重和偏离 1: {s:.6f}")
    return violations


# ═══════════════════════════════════════════
# 优化器
# ═══════════════════════════════════════════

def _solve_cvxpy_mean_variance(
    mu: np.ndarray,
    Sigma: np.ndarray,
    N: int,
    risk_aversion: float,
    max_single: float,
) -> np.ndarray:
    """maximize μᵀw − (λ/2)·wᵀΣw s.t. sum(w)=1, 0≤w≤max_single。失败抛异常。"""
    w = cp.Variable(N)
    sigma_sym = 0.5 * (Sigma + Sigma.T)
    objective = cp.Minimize(
        0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(sigma_sym)) - mu @ w
    )
    problem = cp.Problem(objective, [cp.sum(w) == 1, w >= 0, w <= max_single])
    problem.solve()
    vals = np.asarray(w.value, dtype=np.float64).ravel()
    if vals.size != N or not bool(np.all(np.isfinite(vals))):
        raise RuntimeError(f"cvxpy 返回无效解 (size={vals.size}, N={N})")
    return vals


def _solve_cvxpy_risk_budget(
    Sigma: np.ndarray,
    N: int,
    max_single: float,
) -> np.ndarray:
    """Spinu 风险平价凸式: min Σᵢ(wᵢ(Σw)ᵢ − bᵢ·log wᵢ)，b=等权预算，后归一化。"""
    w = cp.Variable(N)
    b = np.full(N, 1.0 / N)
    sigma_sym = 0.5 * (Sigma + Sigma.T)
    objective = cp.Minimize(
        cp.sum(cp.multiply(w, sigma_sym @ w)) - cp.sum(cp.multiply(b, cp.log(w)))
    )
    problem = cp.Problem(objective, [cp.sum(w) == 1, w >= 1e-9, w <= max_single])
    problem.solve()
    vals = np.asarray(w.value, dtype=np.float64).ravel()
    if vals.size != N or not bool(np.all(np.isfinite(vals))):
        raise RuntimeError(f"cvxpy 返回无效解 (size={vals.size}, N={N})")
    return vals


def _risk_budget_optimize(
    Sigma: np.ndarray,
    tickers: list[str],
    constraints: dict,
    meta: dict | None = None,
) -> tuple[np.ndarray, bool, list[str]]:
    """
    风险平价 / 最小方差优化。

    cvxpy 可用时走 Spinu 凸式（min Σᵢ(wᵢ(Σw)ᵢ − bᵢ log wᵢ)）；
    否则/失败时回退 SLSQP 最小方差。

    minimize: 0.5 * w' * Sigma * w
    subject to: sum(w) = 1, 0 <= w_i <= max_single, long_only
    """
    N = len(tickers)
    max_single = constraints["max_single"]
    violations: list[str] = []
    backend_used = "scipy-SLSQP"

    weights: np.ndarray | None = None
    if SOLVER_BACKEND == "cvxpy":
        try:
            weights = _solve_cvxpy_risk_budget(Sigma, N, max_single)
            backend_used = "cvxpy"
        except Exception as exc:
            violations.append(f"{CVXPY_FALLBACK_NOTE}（{type(exc).__name__}: {exc}）")
            weights = None

    if weights is not None:
        success = True
    else:
        def objective(w):
            return 0.5 * w @ Sigma @ w

        cons_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, max_single) for _ in range(N)]

        w0 = np.full(N, 1.0 / N)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective, w0, method="SLSQP",
                bounds=bounds, constraints=cons_list,
                options=SLSQP_OPTIONS,
            )

        if not result.success:
            violations.append(f"SLSQP 收敛警告: {result.message}")

        weights = result.x if result.success else w0
        success = result.success

    # 数值安全: clip + renormalize
    weights = np.clip(weights, 0.0, max_single)
    w_sum = np.sum(weights)
    if w_sum > 1e-10:
        weights = weights / w_sum
    else:
        weights = np.full(N, 1.0 / N)

    if meta is not None:
        meta["backend"] = backend_used
    violations.extend(_verify_constraints(weights, constraints))

    return weights, success, violations


def _mean_variance_optimize(
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: list[str],
    constraints: dict,
    risk_aversion: float = DEFAULT_LAMBDA,
    meta: dict | None = None,
) -> tuple[np.ndarray, bool, list[str]]:
    """
    均值-方差优化。

    cvxpy 可用时走凸式 QP（maximize μᵀw − (λ/2)wᵀΣw），
    否则/失败时回退 SLSQP。

    maximize: w'mu - 0.5*lambda*w'Sigma*w
    subject to: sum(w)=1, 0<=w_i<=max_single, long_only

    风险厌恶系数 lambda 越大 → 越倾向于低波动组合。
    """
    N = len(tickers)
    max_single = constraints["max_single"]
    violations: list[str] = []
    backend_used = "scipy-SLSQP"

    weights: np.ndarray | None = None
    if SOLVER_BACKEND == "cvxpy":
        try:
            weights = _solve_cvxpy_mean_variance(mu, Sigma, N, risk_aversion, max_single)
            backend_used = "cvxpy"
        except Exception as exc:
            violations.append(f"{CVXPY_FALLBACK_NOTE}（{type(exc).__name__}: {exc}）")
            weights = None

    if weights is not None:
        success = True
    else:
        def objective(w):
            return -(w @ mu) + 0.5 * risk_aversion * (w @ Sigma @ w)

        cons_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, max_single) for _ in range(N)]

        w0 = np.full(N, 1.0 / N)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective, w0, method="SLSQP",
                bounds=bounds, constraints=cons_list,
                options=SLSQP_OPTIONS,
            )

        weights = result.x if result.success else w0
        success = result.success
        if not result.success:
            violations.append(f"SLSQP未收敛，已回退等权: {result.message}")

    weights = np.clip(weights, 0.0, max_single)
    w_sum = np.sum(weights)
    weights = weights / w_sum if w_sum > 1e-10 else np.full(N, 1.0 / N)

    if meta is not None:
        meta["backend"] = backend_used
    violations.extend(_verify_constraints(weights, constraints))

    return weights, success, violations


def _black_litterman_optimize(
    signals: dict[str, float],
    Sigma: np.ndarray,
    tickers: list[str],
    constraints: dict,
    risk_aversion: float = DEFAULT_LAMBDA,
    tau: float = BL_TAU,
) -> tuple[np.ndarray, bool, list[str]]:
    """
    简化版 Black-Litterman。

    1. 先验: 市场等权 (无信息先验)
    2. 反向优化: pi = delta * Sigma * w_mkt → 隐含均衡收益
    3. 观点: 信号 Z-score 作为绝对收益观点
    4. 观点不确定性: 信号越极端 → 不确定性越高
    5. 后验: BL 公式融合先验和观点
    6. 后验均值-方差优化
    """
    N = len(tickers)
    scores = np.array([signals[t] for t in tickers], dtype=np.float64)
    score_std = float(np.std(scores, ddof=1)) if N > 1 else 1.0
    if score_std < 1e-10:
        score_std = 1.0
    z_scores = (scores - np.mean(scores)) / score_std

    # Step 1: 市场等权作为先验权重
    w_mkt = np.full(N, 1.0 / N)

    # Step 2: 反向优化隐含收益 pi = delta * Sigma * w_mkt
    pi = risk_aversion * Sigma @ w_mkt

    # Step 3: 观点矩阵 Q (观点向量) 和 P (观点连接矩阵)
    # 绝对观点: 每个资产的超额收益方向由其 Z-score 决定
    Q = z_scores * ANNUAL_ALPHA_SCALE
    P = np.eye(N)  # 每个观点对应一个资产

    # Step 4: 观点不确定性 Omega (对角矩阵)
    # 观点不确定性 = 基础不确定性 + 信号极端性调整
    max_abs_z = max(np.max(np.abs(z_scores)), 1.0)
    view_uncertainty = 0.02 + 0.03 * np.abs(z_scores) / max_abs_z
    Omega = np.diag(view_uncertainty ** 2)

    # Step 5: 后验期望收益 (Black-Litterman 公式)
    # mu_bl = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
    #         * [(tau*Sigma)^-1*pi + P'*Omega^-1*Q]
    tau_sigma = tau * Sigma
    tau_sigma_inv = np.linalg.inv(tau_sigma)
    omega_inv = np.linalg.inv(Omega)
    P_omega_inv = P.T @ omega_inv @ P
    posterior_cov_inv = tau_sigma_inv + P_omega_inv

    rhs = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
    mu_bl = np.linalg.solve(posterior_cov_inv, rhs)

    # Step 6: 用后验收益做均值-方差优化
    def objective(w):
        return -(w @ mu_bl) + 0.5 * risk_aversion * (w @ Sigma @ w)

    cons_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, constraints["max_single"]) for _ in range(N)]

    w0 = np.full(N, 1.0 / N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            objective, w0, method="SLSQP",
            bounds=bounds, constraints=cons_list,
            options=SLSQP_OPTIONS,
        )

    violations = []
    weights = result.x if result.success else w0
    weights = np.clip(weights, 0.0, constraints["max_single"])
    w_sum = np.sum(weights)
    weights = weights / w_sum if w_sum > 1e-10 else np.full(N, 1.0 / N)

    if not result.success:
        violations.append(f"BL SLSQP未收敛，已回退等权: {result.message}")
    violations.extend(_verify_constraints(weights, constraints))

    return weights, result.success, violations


# ═══════════════════════════════════════════
# 指标计算 Helpers
# ═══════════════════════════════════════════

def _compute_factor_exposures(
    weights: np.ndarray,
    tickers: list[str],
    signals: dict[str, float],
) -> dict[str, float]:
    """
    估算组合因子暴露。

    因子暴露反映组合对不同风格因子的倾向性:
    - momentum:  加权 Z-score (正 = 追涨, 负 = 逆势)
    - size:      有效持仓数 (正 = 大盘, 负 = 小盘)
    - value:     与 momentum 负相关 (正 = 价值, 负 = 成长)
    - quality:   与 momentum 负相关 (正 = 质量)
    - volatility: 分散度 (正 = 高波, 负 = 低波)
    """
    scores = np.array([signals.get(t, 0.0) for t in tickers], dtype=np.float64)
    score_mean = np.mean(scores)
    N = len(tickers)
    if N > 1:
        score_var = float(np.var(scores, ddof=1))
    else:
        score_var = 1.0
    score_var = max(score_var, 1e-10)
    score_std = math.sqrt(score_var)
    z_scores = (scores - score_mean) / score_std

    momentum = float(np.clip(weights @ z_scores, -1.0, 1.0))

    effective_n = 1.0 / max(np.sum(weights ** 2), 1e-10)

    # Sise: 有效持仓 vs 总股票数, 映射到 [-0.5, 0.3]
    size_norm = (effective_n - 1.0) / max(N - 1.0, 1.0)
    size = -0.3 + 0.6 * size_norm

    value = -0.15 * momentum + 0.05
    quality = 0.10 - 0.05 * momentum
    vol_factor = -0.10 + 0.20 * (1.0 - effective_n / N)
    vol_factor = max(-0.3, min(0.1, vol_factor))

    return {
        "momentum": round(float(momentum), 4),
        "size": round(float(size), 4),
        "value": round(float(value), 4),
        "quality": round(float(quality), 4),
        "volatility": round(float(vol_factor), 4),
    }


def _estimate_turnover_cost(
    weights: np.ndarray,
    tickers: list[str],
    current_weights: dict[str, float] | None = None,
) -> float:
    """
    估算换手成本 (bp)。

    基准: 等权起始 → 目标权重的换手 × 单位换手成本。
    """
    if current_weights is None:
        current = np.full(len(tickers), 1.0 / len(tickers))
    else:
        current = np.array([current_weights.get(t, 0.0) for t in tickers])

    turnover = np.sum(np.abs(weights - current))
    cost_bp = turnover * TURNOVER_COST_PER_UNIT_BP
    return round(cost_bp, 1)


# ═══════════════════════════════════════════
# OptimizePortfolioTool
# ═══════════════════════════════════════════

class OptimizePortfolioTool(BaseTool):
    """组合优化 (M2: 真实优化引擎)"""

    tool_name = "optimize_portfolio"
    tool_description = (
        "根据信号和约束求解最优组合权重, "
        "支持 risk_budget(风险平价)/mean_variance(均值-方差)/black_litterman(Black-Litterman) 三种方法"
    )
    tool_version = "2.0.0"
    tool_category = "optimize"

    def execute(
        self,
        signals: dict[str, float] = None,
        method: str = "risk_budget",
        constraints: PortfolioConstraints = None,
        risk_model: dict = None,
    ) -> Trader3Response:
        """执行组合优化 (M2: 真实优化引擎)"""
        if not signals:
            return Trader3Response.error("需提供信号字典 {symbol: score}")

        tickers = list(signals.keys())
        N = len(tickers)

        # ── 构建期望收益和协方差矩阵（外部 risk_model 优先） ──
        mu, Sigma, _ = _signals_to_mu_sigma(signals)
        cov_source = "信号派生合成估计"
        if risk_model:
            ext_cov = risk_model.get("cov") or risk_model.get("sigma") or risk_model.get("covariance")
            if ext_cov is not None:
                ext_cov = np.asarray(ext_cov, dtype=np.float64)
                if ext_cov.shape == (N, N):
                    Sigma = ext_cov
                    cov_source = "外部 risk_model 协方差"
                else:
                    raise ValueError(
                        f"risk_model 协方差形状 {ext_cov.shape} 与资产数 {N} 不符"
                    )
            ext_mu = risk_model.get("mu")
            if ext_mu is not None:
                ext_mu = np.asarray(ext_mu, dtype=np.float64)
                if ext_mu.shape != (N,):
                    raise ValueError(f"risk_model mu 形状 {ext_mu.shape} 与资产数 {N} 不符")
                mu = ext_mu
        cons = _parse_constraints(constraints, N)

        # ── 选择优化方法 ──
        method = method or "risk_budget"
        meta_backend: dict = {"backend": "scipy-SLSQP"}
        method_labels = {
            "risk_budget": "风险平价",
            "mean_variance": "均值-方差",
            "black_litterman": "Black-Litterman",
        }
        method_label = method_labels.get(method, method)

        if method == "risk_budget":
            weights, success, violations = _risk_budget_optimize(
                Sigma, tickers, cons, meta=meta_backend
            )
        elif method == "mean_variance":
            weights, success, violations = _mean_variance_optimize(
                mu, Sigma, tickers, cons, meta=meta_backend
            )
        elif method == "black_litterman":
            weights, success, violations = _black_litterman_optimize(
                signals, Sigma, tickers, cons,
            )
        else:
            return Trader3Response.error(f"未知优化方法: {method}")

        # ── 计算结果指标 ──
        expected_return = float(weights @ mu)
        expected_risk = float(math.sqrt(max(weights @ Sigma @ weights, 1e-30)))
        expected_sharpe = (
            expected_return / expected_risk if expected_risk > 1e-10 else 0.0
        )

        factor_exposure = _compute_factor_exposures(weights, tickers, signals)
        turnover_cost_bp = _estimate_turnover_cost(weights, tickers)

        target_weights_all = {
            t: round(float(w), 6) for t, w in zip(tickers, weights, strict=False)
        }
        target_weights_sorted = dict(
            sorted(target_weights_all.items(), key=lambda x: -x[1])[: min(N, 20)]
        )

        structural = [v for v in violations if any(k in v for k in ("超限", "负权重", "偏离"))]
        result = OptimizationResult(
            target_weights=target_weights_sorted,
            expected_return=round(expected_return, 4),
            expected_risk=round(expected_risk, 4),
            expected_sharpe=round(expected_sharpe, 4),
            factor_exposure=factor_exposure,
            turnover_cost_bp=turnover_cost_bp,
            constraints_satisfied=len(structural) == 0,
            constraint_violations=violations,
            max_single_weight_cap=float(cons["max_single"]),
        )

        caveats = [
            f"求解后端: {meta_backend.get('backend', SOLVER_BACKEND)}",
            f"协方差矩阵来源: {cov_source}（合成估计仅供流程演示，不构成风险预测）"
            if cov_source.startswith("信号") else
            f"协方差矩阵来源: {cov_source}",
        ]
        if cons.get("cap_adjusted_note"):
            caveats.append(cons["cap_adjusted_note"])
        if not result.constraints_satisfied:
            caveats.append(f"约束违规 {len(structural)} 项: {'; '.join(structural)}")
        if any("未收敛" in v or "收敛警告" in v for v in violations):
            caveats.append("求解器存在未收敛警告，权重为回退值")

        return Trader3Response(
            success=True,
            data=result,
            summary=(
                f"优化完成 [{method_label}]: "
                f"预期年化 {result.expected_return:.1%}, "
                f"波动 {result.expected_risk:.1%}, "
                f"夏普 {result.expected_sharpe:.2f}, "
                f"换手成本 {result.turnover_cost_bp:.0f}bp"
                + ("" if result.constraints_satisfied else " ⚠约束违规")
            ),
            caveats=caveats,
            key_metrics={
                "预期收益": result.expected_return,
                "预期风险": result.expected_risk,
                "预期夏普": result.expected_sharpe,
                "换手成本(bp)": result.turnover_cost_bp,
                "持仓数量": len(result.target_weights),
                "有效持仓数": round(float(1.0 / sum(w ** 2 for w in weights)), 1),
            },
            charts=[
                ChartSpec(
                    chart_type="bar",
                    title="目标权重分布",
                    data=target_weights_sorted,
                    description=f"Top {len(target_weights_sorted)} 持仓目标权重",
                ),
                ChartSpec(
                    chart_type="bar",
                    title="因子暴露",
                    data=factor_exposure,
                    description="组合的五因子暴露 (动量/规模/价值/质量/波动)",
                ),
            ],
        )


# ═══════════════════════════════════════════
# RegimeAwareAllocationTool
# ═══════════════════════════════════════════

class RegimeAwareAllocationTool(BaseTool):
    """情境路由分配 (M2: 真实概率加权路由)"""

    tool_name = "regime_aware_allocation"
    tool_description = (
        "根据市场状态概率和状态-信号权重映射, "
        "加权计算动态分配, 返回调整后的组合"
    )
    tool_version = "2.0.0"
    tool_category = "optimize"

    def execute(
        self,
        signals: dict[str, float] = None,
        regime_probs: dict[str, float] = None,
        regime_weights: dict[str, dict] = None,
        constraints: PortfolioConstraints = None,
    ) -> Trader3Response:
        """
        情境路由 (M2: 真实概率加权路由)。

        算法:
            composite_score[asset] = sum_r P(r) * w_r(asset)
            target_weight[asset] = composite_score[asset] / sum(composite_score)

        parameters
        ----------
        signals : {asset: signal_score} — 用于计算指标
        regime_probs : {regime: probability}
        regime_weights : {regime: {asset: weight}}
        constraints : PortfolioConstraints (用于 max_single 限制)
        """
        if not signals:
            return Trader3Response.error("需提供信号字典 {symbol: score}")
        if not regime_probs:
            return Trader3Response.error("需提供市场状态概率 regime_probs")
        if not regime_weights:
            return Trader3Response.error("需提供状态权重映射 regime_weights")

        tickers = list(signals.keys())

        # ── 计算概率加权综合分配 ──
        composite_scores: dict[str, float] = {t: 0.0 for t in tickers}
        total_prob = sum(regime_probs.values())

        if total_prob <= 0:
            return Trader3Response.error("状态概率之和必须大于 0")

        for regime, prob in regime_probs.items():
            rw_map = regime_weights.get(regime, {})
            if not rw_map:
                continue
            # 状态内权重归一化
            rw_sum = sum(rw_map.values()) or 1.0
            for asset, rw in rw_map.items():
                if asset in composite_scores:
                    composite_scores[asset] += (prob / total_prob) * (rw / rw_sum)

        # ── 转换综合分数为目标权重 ──
        score_values = np.array(
            [composite_scores.get(t, 0.0) for t in tickers], dtype=np.float64
        )
        score_values = np.maximum(score_values, 0.0)  # long-only
        total_score = np.sum(score_values)

        if total_score <= 1e-10:
            weights = np.full(len(tickers), 1.0 / len(tickers))
        else:
            weights = score_values / total_score

        # ── 单票上限约束 ──
        if constraints and constraints.max_single_weight:
            max_single = constraints.max_single_weight
        else:
            max_single = DEFAULT_MAX_SINGLE

        cons_dict = {"max_single": max_single}
        if np.any(weights > max_single + 1e-10):
            weights = np.minimum(weights, max_single)
            weights = weights / np.sum(weights)
        cap_violations = _verify_constraints(weights, cons_dict)

        # ── 计算指标 (用 composite_scores 作为信号重建 mu/Sigma) ──
        composite_signals = {t: composite_scores[t] * 100 for t in tickers}
        mu, Sigma, _ = _signals_to_mu_sigma(composite_signals, rng_seed=123)

        expected_return = float(weights @ mu)
        expected_risk = float(math.sqrt(max(weights @ Sigma @ weights, 1e-30)))
        expected_sharpe = (
            expected_return / expected_risk if expected_risk > 1e-10 else 0.0
        )
        factor_exposure = _compute_factor_exposures(weights, tickers, composite_signals)
        turnover_cost_bp = _estimate_turnover_cost(weights, tickers)

        # ── 主导状态与信息 ──
        current_regime = max(regime_probs, key=lambda k: regime_probs.get(k, 0))
        dominant_prob = regime_probs[current_regime]

        entropy = -sum(p * math.log(p) for p in regime_probs.values() if p > 0)

        # 风险状态概率 → 建议仓位 (0.3~1.0)
        risk_regimes = {"bearish", "high_vol", "liquidity_crisis"}
        risk_prob = sum(regime_probs.get(r, 0) for r in risk_regimes)
        suggested_position = max(0.3, min(1.0, 1.0 - risk_prob * 1.5))

        target_weights_all = {
            t: round(float(w), 6) for t, w in zip(tickers, weights, strict=False)
        }
        target_weights_sorted = dict(
            sorted(target_weights_all.items(), key=lambda x: -x[1])[
                : min(len(tickers), 20)
            ]
        )

        result = OptimizationResult(
            target_weights=target_weights_sorted,
            expected_return=round(expected_return, 4),
            expected_risk=round(expected_risk, 4),
            expected_sharpe=round(expected_sharpe, 4),
            factor_exposure=factor_exposure,
            turnover_cost_bp=turnover_cost_bp,
            constraints_satisfied=len(cap_violations) == 0,
            constraint_violations=cap_violations,
            max_single_weight_cap=float(max_single),
        )

        regime_caveats = [f"求解后端: {SOLVER_BACKEND}"]
        if cap_violations:
            regime_caveats.insert(0, f"约束违规: {'; '.join(cap_violations)}")

        return Trader3Response(
            success=True,
            data=result,
            summary=(
                f"情境路由: 当前状态「{current_regime}」(P={dominant_prob:.0%}), "
                f"综合 {len(tickers)} 个标的权重, "
                f"预期夏普 {result.expected_sharpe:.2f}, "
                f"建议仓位 {suggested_position:.0%}"
                + ("" if result.constraints_satisfied else " ⚠约束违规")
            ),
            caveats=regime_caveats,
            key_metrics={
                "当前状态": current_regime,
                "主导概率": round(dominant_prob, 4),
                "建议仓位": round(suggested_position, 2),
                "预期夏普": result.expected_sharpe,
                "预期风险": result.expected_risk,
                "持仓数量": len(target_weights_all),
                "状态熵": round(entropy, 4),
            },
            charts=[
                ChartSpec(
                    chart_type="bar",
                    title="状态概率分布",
                    data=dict(sorted(regime_probs.items(), key=lambda x: -x[1])),
                    description="M2 情境路由: 输入的市场状态概率",
                ),
                ChartSpec(
                    chart_type="bar",
                    title="路由后权重分布",
                    data=target_weights_sorted,
                    description="概率加权后的目标权重",
                ),
            ],
        )
