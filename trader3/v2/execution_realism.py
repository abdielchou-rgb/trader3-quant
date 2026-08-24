"""3号交易员 v2.1 — A股撮合真实度（涨跌停 + T+1 两态即时成交）

吸收 QUANTAXIS(QAMarket/QAPosition.py) T+1 持仓模型 与 vnpy 涨跌停拦截：

- 持仓两态：his（历史可卖） / today（当日买入，T+1 冻结）
  （原第三态 frozen 为"委托挂单冻结"，在即时成交模型下恒为 0，
   属死路径，已删除——若未来引入挂单撮合再恢复）
- 卖单可用量 = volume_long_his（当日买入当日不可卖）
- 日终 settle：today 并入 his，并复位当日流控计数器（跨日额度恢复）
- 卖出与买入行为一致：超量请求部分成交至可卖上限，而非整单拒绝
- 买入成本基础含费用：佣金摊入 avg_cost
- 费率唯一事实来源：costs.DEFAULT_COSTS（bp/10000 换算），本模块不定义常量
- 涨跌停拦截：涨停禁买入（一字板无买盘），跌停禁卖出（一字板无卖盘）
- 买入按 buy_frozen_coeff 冻资再成交（防超买）

用法：
    from trader3.v2.execution_realism import PositionT1Account, PriceLimitMatcher
    acct = PositionT1Account(cash=1e7)
    ok, msg = acct.buy("600519", 100, 12.5)      # 实时成交模拟（T+1 两态）
    ok, msg = acct.sell("600519", 50, 13.0)      # 仅可卖 his 量
    acct.settle()                                  # 日终：today->his + 计数复位
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from trader3.v2.costs import DEFAULT_COSTS
from trader3.v2.fillers import FixedSizeFiller

logger = logging.getLogger("trader3.v2.execution_realism")


@dataclass
class PositionT1Stock:
    """单票 T+1 两态持仓（his/today 全部为股数）"""

    code: str
    his_volume: float = 0.0    # 历史可卖（T-1 及之前买入）
    today_volume: float = 0.0  # 当日买入（T+1 冻结，当日不可卖）
    avg_cost: float = 0.0      # 摊薄成本（含摊入佣金）

    @property
    def sellable(self) -> float:
        """当日可卖 = 历史可卖（即时成交模型：无挂单冻结）"""
        return max(self.his_volume, 0.0)

    @property
    def total(self) -> float:
        return self.his_volume + self.today_volume


class PositionT1Account:
    """T+1 两态账户：买入冻资+today 冻结，卖出仅可卖量，日终结算复位"""

    def __init__(self, cash: float = 1e7, buy_frozen_coeff: float = 1.0,
                 commission_rate: Optional[float] = None,
                 stamp_tax_rate: Optional[float] = None,
                 filler: Optional[object] = None):
        self.cash = cash
        self.buy_frozen_coeff = buy_frozen_coeff  # 买入冻资系数（A股一般全额，1.0）
        # 费率唯一事实来源 costs.DEFAULT_COSTS：bp/10000 换算，显式传参可覆盖
        self.commission_rate = (DEFAULT_COSTS.commission_bp / 10000.0
                                if commission_rate is None else commission_rate)
        self.stamp_tax_rate = (DEFAULT_COSTS.stamp_tax_bp / 10000.0
                               if stamp_tax_rate is None else stamp_tax_rate)
        self.min_commission = DEFAULT_COSTS.min_commission  # 最低佣金（元/笔）
        self.filler = filler or FixedSizeFiller() # 成交量填充器（backtrader Fillers 风格）
        self.stocks: Dict[str, PositionT1Stock] = {}
        self._today_value = 0.0   # 当日成交金额（供 风控流控 用）
        self._today_orders = 0

    # ── 内部：找/建持仓 ──

    def _get(self, code: str) -> PositionT1Stock:
        if code not in self.stocks:
            self.stocks[code] = PositionT1Stock(code=code)
        return self.stocks[code]

    # ── 买入 ──

    def buy(self, code: str, volume: float, price: float,
            prev_close: Optional[float] = None, bar_volume: Optional[float] = None,
            limit_pct: Optional[float] = None) -> Tuple[bool, str]:
        """买入：冻结资金 → 成交（T+1 当日不可卖）

        传入 prev_close/bar_volume 时按 Filler 限制成交量：
        涨停一字板买入量趋近 0；常规行情受当根 Bar 成交量约束。
        """
        if volume <= 0 or price <= 0:
            return False, "非法买入参数"
        filled = self.filler.fill("buy", volume, price=price,
                                  prev_close=prev_close, limit_pct=limit_pct,
                                  bar_volume=bar_volume)
        if filled <= 0:
            return False, f"涨停/流动性不足拒单: 买入 {volume}股 @ {price:.2f} 可成交 0"
        if filled < volume:
            logger.info("部分成交: 买入 %s 请求 %s -> 成交 %s（流动性约束）", code, volume, filled)
            volume = filled
        gross = volume * price
        commission = max(gross * self.commission_rate, self.min_commission)
        frozen = gross + commission                           # 买入需冻结全额+佣金
        if frozen > self.cash:
            return False, f"现金不足: 需 {frozen:.2f} > 现金 {self.cash:.2f}"
        # 扣钱、建仓
        self.cash -= frozen
        st = self._get(code)
        old_total = st.his_volume + st.today_volume
        # 成本基础含费用：佣金摊入 avg_cost（backtrader 同口径）
        new_cost = st.avg_cost * old_total + gross + commission
        st.avg_cost = new_cost / (old_total + volume) if (old_total + volume) else 0
        st.today_volume += volume          # 当日买入 → today 冻结（T+1）
        # 记录流控
        self._today_value += gross
        self._today_orders += 1
        return True, f"买入 {code} {volume}股 @ {price:.2f}, 冻结 {frozen:.2f}"

    # ── 卖出 ──

    def sell(self, code: str, volume: float, price: float,
             prev_close: Optional[float] = None, bar_volume: Optional[float] = None,
             limit_pct: Optional[float] = None) -> Tuple[bool, str]:
        """卖出：仅允许 可卖量=his；超量请求部分成交至可卖上限

        与买入部分成交行为一致：请求量 > 可卖量时按可卖上限成交，
        仅在可卖量为 0 时整单拒绝。
        传入 prev_close/bar_volume 时按 Filler 限制成交量：
        跌停一字板卖出量趋近 0；常规行情受当根 Bar 成交量约束。
        """
        st = self._get(code)
        sellable = st.sellable
        if volume <= 0:
            return False, f"非法卖出参数: {volume}"
        if sellable <= 0:
            return False, (f"可卖量不足: 无可用持仓"
                           f"（his={st.his_volume:.0f}，当日买入 T+1 不可卖）")
        requested = volume
        if volume > sellable:
            logger.info("部分成交: 卖出 %s 请求 %s -> 成交 %s（可卖上限）",
                        code, volume, sellable)
            volume = sellable
        filled = self.filler.fill("sell", volume, price=price,
                                  prev_close=prev_close, limit_pct=limit_pct,
                                  bar_volume=bar_volume)
        if filled <= 0:
            return False, f"跌停/流动性不足拒单: 卖出 {requested}股 @ {price:.2f} 可成交 0"
        if filled < volume:
            logger.info("部分成交: 卖出 %s 请求 %s -> 成交 %s（流动性约束）", code, volume, filled)
            volume = filled
        # 模拟即时成交（无挂单，frozen 死路径已删除——真实度细节见 matcher）
        self._fill_sell(code, volume, price)
        self._today_value += volume * price
        self._today_orders += 1
        msg = f"卖出 {code} {volume}股 @ {price:.2f}"
        if volume < requested:
            msg = f"部分成交: {msg}（请求 {requested}股）"
        return True, msg

    def _fill_sell(self, code: str, volume: float, price: float):
        st = self._get(code)
        gross = volume * price
        stamp = gross * self.stamp_tax_rate          # 印花税卖出单边
        commission = max(gross * self.commission_rate, self.min_commission)
        self.cash += gross - stamp - commission
        st.his_volume -= volume
        if st.his_volume <= 1e-9 and st.today_volume <= 1e-9:
            del self.stocks[code]

    # ── 日终结算 ──

    def settle(self):
        """日终：today → his（T+1 到期），并复位当日流控计数器（跨日额度恢复）"""
        for st in self.stocks.values():
            st.his_volume += st.today_volume
            st.today_volume = 0.0
        self._today_value = 0.0
        self._today_orders = 0

    # ── 查询 ──

    def total_assets(self) -> float:
        cash = self.cash
        for st in self.stocks.values():
            cash += st.total * st.avg_cost * 1.0  # 简化以成本计
        return cash

    def summary(self) -> dict:
        return {
            "cash": round(self.cash, 2),
            "positions": [
                {"code": k, "his": v.his_volume, "today": v.today_volume,
                 "sellable": v.sellable, "avg_cost": round(v.avg_cost, 3)}
                for k, v in self.stocks.items()
            ],
            "today_value": round(self._today_value, 2),
            "today_orders": self._today_orders,
        }

    # ── 状态序列化（纸面账户落盘/恢复用） ──

    def to_state(self) -> dict:
        """序列化为可持久化状态（供 daily_pipeline 纸面账户写盘）"""
        return {
            "cash": self.cash,
            "positions": [
                {"code": s.code, "his": s.his_volume,
                 "today": s.today_volume, "avg_cost": s.avg_cost}
                for s in self.stocks.values()
            ],
        }

    @classmethod
    def from_state(cls, state: dict, **kwargs) -> "PositionT1Account":
        """从 to_state 的字典恢复账户（缺省字段安全兜底）"""
        acct = cls(**kwargs)
        acct.cash = float(state.get("cash") or acct.cash)
        for p in state.get("positions") or []:
            st = acct._get(str(p.get("code", "")))
            st.his_volume = float(p.get("his") or 0)
            st.today_volume = float(p.get("today") or 0)
            st.avg_cost = float(p.get("avg_cost") or 0)
        return acct


class PriceLimitMatcher:
    """涨跌停撮合拦截：涨停禁买、跌停禁卖（vnpy 模式照抄）"""

    def __init__(self, limit_pct: float = 0.10, tolerance: float = 0.001):
        self.limit_pct = limit_pct
        self.tolerance = tolerance   # 撮合容差（float 误差）

    def can_buy(self, price: float, prev_close: float) -> bool:
        """涨停（price ≈ prev_close×1.1）时禁买（一字板无卖盘）"""
        limit = prev_close * (1 + self.limit_pct)
        return price < limit - self.tolerance

    def can_sell(self, price: float, prev_close: float) -> bool:
        """跌停（price ≈ prev_close×0.9）时禁卖（一字板无买盘）"""
        limit = prev_close * (1 - self.limit_pct)
        return price > limit + self.tolerance

    def check_order(self, action: str, price: float, prev_close: float) -> Tuple[bool, str]:
        if action == "buy" and not self.can_buy(price, prev_close):
            return False, f"涨停中禁买: {price:.2f} >= 涨 停价 {prev_close*(1+self.limit_pct):.2f}"
        if action == "sell" and not self.can_sell(price, prev_close):
            return False, f"跌停中禁卖: {price:.2f} <= 跌 停价 {prev_close*(1-self.limit_pct):.2f}"
        return True, ""