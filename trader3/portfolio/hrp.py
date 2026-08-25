"""层次风险平价 (Hierarchical Risk Parity) 组合层 — 原生实现。

算法遵循 López de Prado (2016)《Building Diversified Portfolios that
Outperform Out-of-Sample》：

    a) 相关距离   d_ij = sqrt(0.5 * (1 - ρ_ij))
    b) 层次聚类   单连锁凝聚聚类；scipy 可用时走 scipy.cluster.hierarchy.linkage
                  加速，否则纯 numpy 朴素凝聚法（两者仅在并列距离的合并顺序上
                  允许差异）
    c) quasi-diagonal 重排（树的中序展开）
    d) 递归二分   沿树自顶向下，按子簇方差的反比分配权重质量
                  （子簇方差按簇内逆方差组合 IV-P 度量）

设计决定（数值健壮性）：
- 收益序列含 NaN/Inf 或零方差（常数列）的资产被剔除，不进入计算，也不出现
  在返回权重中 —— 其相关系数无定义，强行纳入会污染整张相关阵；
- 收益期数 T < 3 时协方差阵无统计意义，退化为有效资产等权；
- 二分沿真实树结构（父子簇）进行而非对排序列表中点切分：中点切分在奇数
  资产时会割裂真实簇、把高相关资产拆到两侧；两资产极限下本实现给出
  w_i ∝ 1/σ_i²（逆方差，即 HRP 的标准二叉性质）。

依赖：仅 numpy 为硬依赖；scipy 为可选加速项；riskfolio-lib 见
riskfolio_weights（可选依赖组 `portfolio`，安装：pip install "trader3[portfolio]"）。
"""

from __future__ import annotations

import numpy as np

__all__ = ["hrp_weights", "riskfolio_weights"]

_INSTALL_HINT = 'pip install "trader3[portfolio]"'

try:  # 可选加速路径：scipy 单连锁
    from scipy.cluster.hierarchy import linkage as _scipy_linkage
    from scipy.spatial.distance import squareform as _scipy_squareform
except Exception:  # noqa: BLE001 - scipy 为可选依赖
    _scipy_linkage = None
    _scipy_squareform = None


# ═══════════════════════════════════════════
# 步骤 a) 相关距离
# ═══════════════════════════════════════════


def _corr_distance(corr: np.ndarray) -> np.ndarray:
    """d_ij = sqrt(0.5 * (1 - ρ_ij))；clip 防御浮点漂移产生的微负值。"""
    d = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(d, 0.0)
    return d


# ═══════════════════════════════════════════
# 步骤 b) 层次聚类 → 树结构
# ═══════════════════════════════════════════


def _naive_single_linkage_tree(
    dist: np.ndarray,
) -> tuple[dict[int, tuple[int, int]], dict[int, list[int]]]:
    """纯 numpy 单连锁凝聚聚类（O(N³) 量级，组合规模 N≤数百足够快）。

    返回 (children, members)：叶节点 id = 0..N-1，内部节点 id 从 N 起，
    children[node]=(左子,右子)，members[node] 为该簇叶索引列表。
    """
    n = dist.shape[0]
    active: dict[int, list[int]] = {i: [i] for i in range(n)}
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    children: dict[int, tuple[int, int]] = {}
    next_id = n
    while len(active) > 1:
        keys = sorted(active)
        best_pair: tuple[int, int] | None = None
        best_d = np.inf
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                dd = float(np.min(dist[np.ix_(active[a], active[b])]))
                if dd < best_d:  # 严格小于 ⇒ 并列距离取最先遍历对（确定性）
                    best_d, best_pair = dd, (a, b)
        assert best_pair is not None  # 循环不变量：len(active) ≥ 2
        a, b = best_pair
        children[next_id] = (a, b)
        active[next_id] = active.pop(a) + active.pop(b)
        members[next_id] = list(active[next_id])
        next_id += 1
    return children, members


def _build_tree(
    dist: np.ndarray,
) -> tuple[dict[int, tuple[int, int]], dict[int, list[int]]]:
    n = dist.shape[0]
    if _scipy_linkage is not None:
        z = _scipy_linkage(_scipy_squareform(dist, checks=False), method="single")
        children = {n + i: (int(a), int(b)) for i, (a, b, _, _) in enumerate(z)}
        members: dict[int, list[int]] = {i: [i] for i in range(n)}
        for node in sorted(children):
            left, right = children[node]
            members[node] = members[left] + members[right]
        return children, members
    return _naive_single_linkage_tree(dist)


# ═══════════════════════════════════════════
# 步骤 c) quasi-diagonal 重排
# ═══════════════════════════════════════════


def _quasi_diag(children: dict[int, tuple[int, int]], n: int) -> list[int]:
    """从根做深度优先展开，得到相似资产相邻的叶子排序。"""
    order: list[int] = []
    stack: list[int] = [max(children)]  # id 最大者即根
    while stack:
        node = stack.pop()
        if node < n:
            order.append(int(node))
        else:
            left, right = children[node]
            stack.append(right)  # 先压右后压左 ⇒ 左子树先出栈
            stack.append(left)
    return order


# ═══════════════════════════════════════════
# 步骤 d) 沿树递归二分 + 逆方差分配
# ═══════════════════════════════════════════


def _cluster_var(cov: np.ndarray, items: list[int]) -> float:
    """簇内逆方差组合 (IV-P) 的方差。"""
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def _allocate(
    children: dict[int, tuple[int, int]],
    members: dict[int, list[int]],
    n: int,
    cov: np.ndarray,
) -> np.ndarray:
    """自根向下按子簇方差反比分配质量：低方差（风险低）子簇得高份额。"""
    w = np.empty(n, dtype=float)
    stack: list[tuple[int, float]] = [(max(children), 1.0)]
    while stack:
        node, mass = stack.pop()
        if node < n:
            w[node] = mass
            continue
        left, right = children[node]
        var_l = _cluster_var(cov, members[left])
        var_r = _cluster_var(cov, members[right])
        denom = var_l + var_r
        alpha = var_r / denom if denom > 0.0 else 0.5
        stack.append((left, alpha * mass))
        stack.append((right, (1.0 - alpha) * mass))
    return w


def _validate_returns(
    returns: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        raise ValueError(f"returns 须为二维 (T,N)，收到 shape={r.shape}")
    t, n = r.shape
    return r, t, n


def _valid_columns(r: np.ndarray, names: list[str]) -> tuple[list[str], list[int]]:
    """剔除含 NaN/Inf 或零方差（常数列）的资产 —— 相关系数对其无定义。"""
    valid = [
        j
        for j in range(r.shape[1])
        if bool(np.isfinite(r[:, j]).all()) and float(r[:, j].std(ddof=0)) > 0.0
    ]
    return [names[j] for j in valid], valid


# ═══════════════════════════════════════════
# 对外接口
# ═══════════════════════════════════════════


def hrp_weights(
    returns: np.ndarray,
    names: list[str] | None = None,
) -> dict[str, float]:
    """计算层次风险平价 (HRP) 权重。

    参数
    ----
    returns : (T, N) 收益矩阵，行=时间，列=资产。
    names : 资产名；None 时自动命名 asset_0 .. asset_{N-1}。

    返回
    ----
    dict[str, float]：键序与有效资产列序一致，值非负且和为 1。

    设计决定
    --------
    1. 含 NaN/Inf 或常数列收益序列的资产被剔除且不出现在返回结果中
       （相关系数无定义）；全部无效时返回空 dict。
    2. 收益期数 T < 3 或剔除后有效资产数 < 2 时退化为等权。
    3. scipy 存在时用 scipy.cluster.hierarchy.linkage(method="single") 加速，
       否则纯 numpy 凝聚法；两者在并列距离处的合并顺序可不同，权重允许差异
       （均为单连锁语义下的合法结果）。
    """
    r, t, n = _validate_returns(returns)
    if names is None:
        names = [f"asset_{i}" for i in range(n)]
    if len(names) != n:
        raise ValueError(f"names 长度 {len(names)} ≠ 资产数 {n}")
    if len(set(names)) != n:
        raise ValueError("names 存在重复，将导致权重字典键冲突")
    kept_names, valid = _valid_columns(r, names)
    m = len(valid)
    if m == 0:
        return {}
    if t < 3 or m < 2:
        eq = 1.0 / m
        return {nm: eq for nm in kept_names}

    rv = r[:, valid]
    cov = np.cov(rv, rowvar=False, ddof=1)
    corr = np.corrcoef(rv, rowvar=False)
    dist = _corr_distance(corr)
    children, members = _build_tree(dist)
    w = _allocate(children, members, m, cov)
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= 0.0:  # 理论不可达；防御浮点极端情形
        return {nm: 1.0 / m for nm in kept_names}
    w /= total
    return dict(zip(kept_names, (float(x) for x in w), strict=True))


def riskfolio_weights(
    returns: np.ndarray,
    method: str = "HRP",
    names: list[str] | None = None,
) -> dict[str, float]:
    """基于 riskfolio-lib 的组合优化（可选依赖路径）。

    method="HRP" → riskfolio 层次风险平价 (model="HRP")
    method="MV"  → 均值-方差最小风险 (model="Classic", rm="MV", obj="MinRisk")

    riskfolio-lib 未安装时抛 ImportError 并提示安装命令：
    pip install "trader3[portfolio]"
    """
    if method not in {"HRP", "MV"}:
        raise ValueError(f"method 须为 'HRP' 或 'MV'，收到 {method!r}")
    try:
        import riskfolio as rp
    except Exception as exc:
        raise ImportError(
            f"riskfolio-lib 未安装，无法使用 method={method!r}；"
            f"请先执行：{_INSTALL_HINT}"
        ) from exc
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - 安装 riskfolio 必带 pandas
        raise ImportError(f"riskfolio 依赖 pandas；请执行：{_INSTALL_HINT}") from exc

    r, _, n = _validate_returns(returns)
    if r.shape[0] < 2 or n < 1:
        raise ValueError("returns 须为 (T,N) 且 T≥2")
    if names is None:
        names = [f"asset_{i}" for i in range(n)]
    if len(set(names)) != n:
        raise ValueError("names 存在重复，将导致权重字典键冲突")
    kept_names, valid = _valid_columns(r, names)
    if not valid:
        return {}

    port = rp.Portfolio(returns=pd.DataFrame(r[:, valid], columns=kept_names))
    port.assets_stats()
    model = "HRP" if method == "HRP" else "Classic"
    w_df = port.optimization(model=model, rm="MV", rf=0.0, obj="MinRisk")
    if w_df is None:
        raise RuntimeError("riskfolio 优化未收敛（返回 None）")
    return {str(k): float(v) for k, v in w_df["weights"].items()}
