"""
Alpha158 骨架因子库（R5）—— 从 qlib Alpha158 六类因子族提炼的向量化实现。

吸收点（见 docs/quant-knowledge/projects/qlib/dissection.md）：
  - 6 大类骨架：K线单日形态 / 趋势动量 / 波动 / 价量相关 / RSI / 量能
  - 元设计：归一化除以"当期值"（close 或 volume+ε）→ 无量纲、跨标的可比
  - 输入 (T,N) 面板，无 qlib 依赖；纯 numpy

无前视保证：所有窗口因子只用 <=t 数据（本实现不"回看未来"；
测试以"截尾重算一致性"验证）。

注意：这是"骨架"而非 Alpha158 全 158 因子复刻 —— 覆盖核心家族，
每族取代表因子；追求可独立验证的正确性与覆盖，非逐列克隆。
"""

from __future__ import annotations

import numpy as np

# ── 滚动工具（时间轴窗口，NaN 感知）─────────────────

def _rolling(arr: np.ndarray, w: int, fn) -> np.ndarray:
    """沿 axis=0 滚动窗口应用 fn(窗口切片 2D)→(N,) 向量。返回同形状。
    只用 <=t 的数据；t<w 前为 NaN。
    """
    T, N = arr.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    for t in range(w - 1, T):
        out[t] = fn(arr[t - w + 1 : t + 1])
    return out


def _rm(arr: np.ndarray, w: int) -> np.ndarray:
    return _rolling(arr, w, lambda x: np.nanmean(x, axis=0))


def _rsd(arr: np.ndarray, w: int) -> np.ndarray:
    return _rolling(arr, w, lambda x: np.nanstd(x, axis=0, ddof=0))


def _delay(arr: np.ndarray, d: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=np.float64)
    if d < 0:
        raise ValueError("delay 需非负")
    if d == 0:
        return arr.copy()
    out[d:] = arr[:-d]
    return out


# ── 因子实现 ─────────────────────────────────────

def _kbar_family(panel: dict) -> dict[str, np.ndarray]:
    """1. K线单日形态（1 日窗口，无回看）。"""
    open_ = panel["open"].astype(np.float64)
    high = panel["high"].astype(np.float64)
    low = panel["low"].astype(np.float64)
    close = panel["close"].astype(np.float64)
    out = {}
    out["KMID"] = (close - open_) / open_          # 收盘相对开盘
    out["KLEN"] = (high - low) / open_             # 振幅
    out["KMID2"] = (close - open_) / (high - low + 1e-12)  # 收在区间位置
    out["KUP"] = (high - np.maximum(open_, close)) / (high - low + 1e-12)  # 上影线
    out["KLOW"] = (np.minimum(open_, close) - low) / (high - low + 1e-12)  # 下影线
    return out


def _trend_family(panel: dict) -> dict[str, np.ndarray]:
    """2. 趋势/动量（ROC/MA/RSV，相对收益无量纲）。"""
    close = panel["close"].astype(np.float64)
    out = {}
    for d in (5, 10, 20):
        prev = _delay(close, d)
        out[f"ROC{d}"] = close / prev - 1.0
        out[f"MA{d}"] = close / _rm(close, d) - 1.0   # 偏离均线
    # RSV（威廉指标基础，归一化到 0-1）
    w = 20
    hh = _rolling(panel["high"].astype(np.float64), w,
                  lambda x: np.nanmax(x, axis=0))
    ll = _rolling(panel["low"].astype(np.float64), w,
                  lambda x: np.nanmin(x, axis=0))
    out["RSV20"] = (close - ll) / (hh - ll + 1e-12)
    return out


def _volatility_family(panel: dict) -> dict[str, np.ndarray]:
    """3. 波动（日收益滚动 std / ATR 归一）。"""
    close = panel["close"].astype(np.float64)
    ret = np.full_like(close, np.nan)
    ret[1:] = close[1:] / close[:-1] - 1.0
    out = {}
    for w in (5, 10, 20):
        out[f"STD{w}"] = _rsd(ret, w)
    # ATR 归一化
    high = panel["high"].astype(np.float64)
    low = panel["low"].astype(np.float64)
    tr = high - low
    out["ATR10"] = _rm(tr, 10) / _rm(close, 10)
    return out


def _volprice_family(panel: dict) -> dict[str, np.ndarray]:
    """4. 价量相关（量与价变化的滚动相关）。"""
    close = panel["close"].astype(np.float64)
    volume = panel["volume"].astype(np.float64)
    ret = np.full_like(close, np.nan)
    ret[1:] = close[1:] / close[:-1] - 1.0
    vchg = np.full_like(volume, np.nan)
    vchg[1:] = volume[1:] / volume[:-1] - 1.0
    out = {}
    w = 10

    def _corr_pair(a: np.ndarray, b: np.ndarray, ww: int) -> np.ndarray:
        T, N = a.shape
        out_c = np.full_like(a, np.nan, dtype=np.float64)
        for t in range(ww - 1, T):
            aa = a[t - ww + 1 : t + 1]
            bb = b[t - ww + 1 : t + 1]
            for j in range(N):
                av, bv = aa[:, j], bb[:, j]
                m = np.isfinite(av) & np.isfinite(bv)
                if m.sum() >= 5 and np.std(av[m]) > 1e-12 and np.std(bv[m]) > 1e-12:
                    out_c[t, j] = np.corrcoef(av[m], bv[m])[0, 1]
        return out_c

    out["CORR_CR"] = _corr_pair(close, vchg, w)    # 价 vs 量变
    out["CORD"] = _corr_pair(ret, vchg, w)          # 价变化率 vs 量变化率
    return out


def _rsi_family(panel: dict) -> dict[str, np.ndarray]:
    """5. RSI（窗口内上涨/下跌强度比）。"""
    close = panel["close"].astype(np.float64)
    diff = np.full_like(close, np.nan)
    diff[1:] = close[1:] - close[:-1]
    out = {}
    for w in (5, 14):
        up = np.where(diff > 0, diff, 0.0)
        dn = np.where(diff < 0, -diff, 0.0)
        aup = _rm(up, w)
        adn = _rm(dn, w)
        out[f"RSI{w}"] = aup / (aup + adn + 1e-12)
    return out


def _volume_family(panel: dict) -> dict[str, np.ndarray]:
    """6. 量能统计（量 MA/std/RSI 同构 + 量比）。"""
    volume = panel["volume"].astype(np.float64)
    out = {}
    for w in (5, 10):
        out[f"VMA{w}"] = volume / (_rm(volume, w) + 1e-12) - 1.0  # 量比
        out[f"VSTD{w}"] = _rsd(volume, w) / (_rm(volume, w) + 1e-12)
    return out


# ── 统一入口 ─────────────────────────────────────


def compute_alpha_factors(panel: dict) -> dict[str, np.ndarray]:
    """面板 → {factor_name: (T,N) ndarray}（六族骨架）。"""
    out: dict[str, np.ndarray] = {}
    for fam in (_kbar_family, _trend_family, _volatility_family,
                _volprice_family, _rsi_family, _volume_family):
        out.update(fam(panel))
    return out


def factor_names() -> list[str]:
    """全部可用因子名（用合成面板推导，保持稳定顺序）。"""
    rng = np.random.default_rng(0)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, (40, 3)), axis=0)
    panel = {
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.full_like(close, 1e6),
        "vwap": close, "amount": close * 1e6,
    }
    return list(compute_alpha_factors(panel).keys())
