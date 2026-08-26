"""
多资产工具层（Multi-Asset Instruments）。

统一股票 / ETF / 股指期货的合约规格与仓位换算：
- InstrumentSpec: 合约乘数、保证金率、最小变动价、T+N、涨跌停
- position_sizing: 目标名义 → 合约数/股数（含保证金占用校验）
- portfolio_margin: 组合保证金占用汇总
- risk_parity_weights: 逆波动率加权的跨资产目标权重（无协方差依赖版）

CN 股指期货规格按中金所公开参数（IF/IH/IC/IM），其余为通用模板。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class AssetType(str, Enum):
    STOCK = "stock"
    ETF = "etf"
    INDEX_FUTURES = "index_futures"
    COMMODITY_FUTURES = "commodity_futures"


@dataclass(frozen=True)
class InstrumentSpec:
    """合约规格。"""
    symbol: str
    name: str
    asset_type: AssetType
    multiplier: float = 1.0        # 每点/每股对应金额（期货=合约乘数）
    margin_rate: float = 1.0       # 保证金比例（现货=1.0）
    tick_size: float = 0.01        # 最小变动价位
    lot_size: int = 1              # 一手数量（股票100，期货1）
    t_plus: int = 0                # T+0/T+1
    price_limit: float = 0.10      # 涨跌停幅度（期货按常规保证金口径近似）
    commission_bp: float = 2.5     # 单边费率 bp（近似）
    currency: str = "CNY"

    def notional(self, price: float, quantity: float) -> float:
        """名义市值 = 价格 × 数量 × 乘数 × 手数基准。quantity 统一为『手』。"""
        return abs(price * quantity * self.multiplier * self.lot_size)

    def margin_required(self, price: float, quantity: float) -> float:
        return self.notional(price, quantity) * self.margin_rate


# ── 内置合约库 ────────────────────────────────────────

def _stock(code: str, name: str = "") -> InstrumentSpec:
    return InstrumentSpec(symbol=code, name=name, asset_type=AssetType.STOCK,
                          multiplier=1.0, margin_rate=1.0, tick_size=0.01,
                          lot_size=100, t_plus=1, price_limit=0.10)


def _etf(code: str, name: str = "") -> InstrumentSpec:
    return InstrumentSpec(symbol=code, name=name, asset_type=AssetType.ETF,
                          multiplier=1.0, margin_rate=1.0, tick_size=0.001,
                          lot_size=100, t_plus=1, price_limit=0.10)


def _index_fut(sym: str, name: str, mult: float, limit: float) -> InstrumentSpec:
    return InstrumentSpec(symbol=sym, name=name,
                          asset_type=AssetType.INDEX_FUTURES,
                          multiplier=mult, margin_rate=limit,
                          tick_size=0.2, lot_size=1, t_plus=0,
                          price_limit=limit, commission_bp=0.23)


INSTRUMENTS: dict[str, InstrumentSpec] = {
    # 股指期货（中金所；保证金率≈交易所+期货公司常见口径）
    "IF": _index_fut("IF", "沪深300", 300, 0.12),
    "IH": _index_fut("IH", "上证50", 300, 0.12),
    "IC": _index_fut("IC", "中证500", 200, 0.14),
    "IM": _index_fut("IM", "中证1000", 200, 0.15),
    # 常见 ETF
    "510300": _etf("510300", "沪深300ETF"),
    "510500": _etf("510500", "中证500ETF"),
    "512100": _etf("512100", "中证1000ETF"),
    "588000": _etf("588000", "科创50ETF"),
}


def get_spec(symbol: str, name: str = "") -> InstrumentSpec:
    """
    取合约规格：内置库命中直接返回；
    未命中按代码规则推断（6位数字→股票/ETF，字母开头→股指期货模板）。
    """
    if symbol in INSTRUMENTS:
        return INSTRUMENTS[symbol]
    s = symbol.upper()
    if len(s) == 6 and s.isdigit():
        prefix = s[0]
        if prefix in ("5", "1") and (s.startswith("51") or s.startswith("15")):
            return _etf(s, name)
        return _stock(s, name)
    if s[:2] in ("IF", "IH", "IC", "IM"):
        base = INSTRUMENTS[s[:2]]
        return InstrumentSpec(**{**base.__dict__, "symbol": symbol})
    raise ValueError(f"未知合约类型: {symbol}")


# ── 仓位换算 ──────────────────────────────────────────

@dataclass
class PositionPlan:
    symbol: str
    lots: int                  # 下单手数（股票=百股手，期货=张）
    notional: float            # 名义敞口
    margin: float              # 占用保证金
    reason: str = ""


def position_sizing(spec: InstrumentSpec, target_notional: float,
                    price: float, available_cash: float,
                    round_lots: bool = True) -> PositionPlan:
    """把目标名义换算成手数，并做保证金可行性检查。"""
    if price <= 0 or spec.multiplier <= 0:
        return PositionPlan(spec.symbol, 0, 0.0, 0.0, "invalid price/multiplier")
    raw_lots = target_notional / (price * spec.multiplier * spec.lot_size)
    lots = int(raw_lots) if round_lots else int(np.ceil(raw_lots))
    if lots < 1:
        return PositionPlan(spec.symbol, 0, 0.0, 0.0,
                            f"target {target_notional:.0f} < 1 lot "
                            f"({price * spec.multiplier * spec.lot_size:.0f})")
    notional = spec.notional(price, lots)
    margin = spec.margin_required(price, lots)
    if margin > available_cash:
        affordable = int(available_cash /
                         (price * spec.multiplier * spec.lot_size * spec.margin_rate))
        if affordable < 1:
            return PositionPlan(spec.symbol, 0, 0.0, 0.0, "insufficient cash for 1 lot")
        notional = spec.notional(price, affordable)
        margin = spec.margin_required(price, affordable)
        return PositionPlan(spec.symbol, affordable, notional, margin,
                            f"cash-capped from {lots} to {affordable} lots")
    return PositionPlan(spec.symbol, lots, notional, margin)


# ── 组合层 ────────────────────────────────────────────

@dataclass
class PortfolioMargin:
    total_notional: float
    total_margin: float
    gross_exposure_ratio: float   # 名义 / 保证金后总权益（由调用方传 equity）
    by_symbol: dict[str, dict]


def portfolio_margin(holdings: list[tuple[str, float, float]],
                     equity: float) -> PortfolioMargin:
    """holdings: [(symbol, price, lots)] → 保证金汇总。"""
    total_n = total_m = 0.0
    detail: dict[str, dict] = {}
    for sym, price, lots in holdings:
        spec = get_spec(sym)
        n = spec.notional(price, lots)
        m = spec.margin_required(price, lots)
        total_n += n
        total_m += m
        detail[sym] = {"notional": round(n, 2), "margin": round(m, 2),
                       "type": spec.asset_type.value}
    return PortfolioMargin(
        total_notional=round(total_n, 2),
        total_margin=round(total_m, 2),
        gross_exposure_ratio=round(total_n / equity, 4) if equity > 0 else float("inf"),
        by_symbol=detail,
    )


def inverse_vol_weights(vol_by_asset: pd.Series | dict[str, float],
                        floor: float = 0.05) -> pd.Series:
    """
    逆波动率加权（风险平价的简化版）：w_i ∝ 1/vol_i，带下限防止单资产归零。
    vol 缺失或 ≤0 的资产用截面中位数兜底。
    """
    v = pd.Series(vol_by_asset, dtype=float)
    med = float(v[v > 0].median()) if (v > 0).any() else 1.0
    v_safe = v.where(v > 0, med).fillna(med)
    inv = 1.0 / v_safe
    w = inv / inv.sum()
    # 地板约束后再归一
    w = w.clip(lower=floor)
    return w / w.sum()
