"""
3号交易员 v2.0 — 三因子买卖点触发引擎

触发模型（DDM 预期差升级版）：
    信号强度 = 0.4×催化强度 + 0.3×估值分位 + 0.3×技术确认

三因子同时成立（均>阈值）才触发提醒，任一不满足 → 只进日报不打扰。

因子定义：
  1. 催化强度 (0.4) — 事件/新闻是否会改变市场对未来现金流的预期
     手动输入 score（0~1），或由 2hao 新闻分析注入
  2. 估值分位 (0.3) — 当前价 vs 估值锚（valuation_anchor 输出的加权目标价）
     低估越多，分位越高：隐含收益 > 15% → 强；0~15% → 中；<0 → 弱
  3. 技术确认 (0.3) — 价格是否与预期差修正共振
     价格上穿关键位 / 成交放量（从 qlib 行情实时计算）
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger("trader3.v2.trigger")

# ── 阈值 ──
CATALYST_THRESHOLD = 0.6      # 催化 ≥0.6 算强
VALUATION_IR_MIN = 0.15       # 隐含收益 ≥15% 算强低估
TECH_BREAKOUT = True          # 技术上破关键位
TRIGGER_MIN_SCORE = 0.65      # 综合信号强度 ≥0.65 才触发
TRIGGER_MIN_FACTOR = 2        # 至少 2 个因子达标（另一因子可略低于阈值）

NEGATIVE_VETO_THRESHOLD = 0.3  # 负面事件强度 ≥0.3 否决买入


@dataclass
class TriggerResult:
    """单个股票的触发结果"""
    code: str
    triggered: bool
    score: float = 0.0                    # 综合信号强度
    catalyst_score: float = 0.0           # 催化强度 0~1
    valuation_score: float = 0.0          # 估值分位 0~1
    tech_score: float = 0.0               # 技术确认 0~1
    reason: str = ""                      # 触发/未触发理由
    direction: str = "buy"                # buy / sell
    # 触发时的关键数字（写作规范要求带来源）
    implied_return: float = 0.0           # 隐含收益率（来自 valuation_anchor）
    fair_value: float = 0.0               # 估值锚（加权目标价）
    current_price: float = 0.0
    stop_loss: float = 0.0                # 止损价
    key_technical: str = ""               # 技术确认依据（上破XX位/放量）
    catalyst_note: str = ""               # 催化事件说明
    caveats: list[str] = field(default_factory=list)   # 风控否决/警示（含负面事件否决）
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class NegativeEventRule:
    """负面事件否决规则：negative_event_score ≥ 阈值时禁止买入迁移

    接口兼容 risk_rules.BaseRiskRule（check(**ctx) -> (ok, reason)），
    不继承以避免循环依赖，可加入 RiskRuleChain。
    """
    rule_name = "negative_event_veto"
    description = "负面事件否决买入"

    def __init__(self, threshold: float = NEGATIVE_VETO_THRESHOLD):
        self.threshold = threshold
        self.enabled = True

    def check(self, **ctx) -> tuple[bool, str]:
        action = ctx.get("action", "buy")
        neg = float(ctx.get("negative_event_score", 0) or 0)
        if action == "buy" and neg >= self.threshold:
            title = str(ctx.get("negative_event_title", "") or "")
            return False, f"负面事件否决：{title}"
        return True, ""


def build_scan_risk_chain():
    """构建扫描判定用风控链：默认链（名单/限额/仓位/价格/流控）+ 负面事件否决"""
    from trader3.v2.risk_rules import build_default_chain
    chain = build_default_chain()
    chain.add(NegativeEventRule())
    return chain


class TriggerEngine:
    """
    三因子触发引擎。

    估值分位复用 valuation_anchor（真实财务+qlib行情），
    技术确认从 qlib 行情实时计算（上破20日高/成交放量），
    催化强度由调用方注入（2hao 新闻分析或手动评分）。
    """

    def __init__(self, t3=None):
        from trader3 import Trader3
        self._t3 = t3 or Trader3()
        self._dp = None
        self._event_lib = None   # 可注入事件库（测试/复用），None 时用默认库
        self._init_data_provider()

    def _init_data_provider(self):
        try:
            from trader3.data_provider import QlibDataProvider
            self._dp = QlibDataProvider()
        except Exception as e:
            logger.warning("数据提供者初始化失败（技术确认将用估值近似）: %s", e)

    # ── 主入口 ──

    def scan(self, items: list[dict], catalyst_scores: dict[str, float] | None = None,
             asof_date: str | None = None, dry_run: bool = False) -> list[TriggerResult]:
        """
        扫描自选股列表，逐只算三因子+风控链，返回触发/未触发结果。

        Parameters
        ----------
        items : [{"code": str, ...}] — 自选股列表（来自 watchlist）
        catalyst_scores : {code: score(0~1)} — 每只股票的催化强度（外部注入）
        dry_run : True 时只报告不写库（scan 本身无副作用，参数供管线透传）

        Returns
        -------
        results : 按触发与否+信号强度排序
        """
        results = []
        for item in items:
            code = item["code"] if isinstance(item, dict) else item.code
            name = (item.get("name") if isinstance(item, dict)
                    else getattr(item, "name", "")) or ""
            try:
                r = self._evaluate(code, (catalyst_scores or {}).get(code),
                                   asof_date=asof_date, name=name)
                results.append(r)
            except Exception as e:
                logger.warning("[trigger] %s 扫描失败: %s", code, e)
                results.append(TriggerResult(
                    code=code, triggered=False,
                    reason=f"扫描失败: {str(e)[:60]}",
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                ))
        # 排序：触发在前，未触发按分数
        results.sort(key=lambda r: (not r.triggered, -r.score))
        return results

    # ── 单股判定（公共函数：_scan_one 与 daily_pipeline.run_daily 共用） ──

    def _evaluate(self, code: str, catalyst_score: float | None = None,
                  asof_date: str | None = None, action: str = "buy",
                  day_orders: int = 0, day_value: float = 0.0,
                  position_pct: float = 0.0, name: str = "") -> TriggerResult:
        """三因子综合判定 + 事前风控链门禁。

        风控 ctx 注入：code/name/negative_event_score/negative_event_title/
                      day_orders/day_value(默认0)/position_pct(默认0)。
        规则：negative_event_score ≥ 0.3 → veto 买入，caveat 写"负面事件否决：<事件标题>"；
             chain.check() 被阻断 → 同样 veto，violations 全部进 caveats。
        """
        # 1. 估值分位（复用 valuation_anchor）
        valuation_score, implied_return, fair_value, current_price = \
            self._valuation_factor(code, asof_date=asof_date)

        # 2. 技术确认（qlib 实时计算）
        tech_score, tech_note = self._tech_factor(code, current_price)

        # 3. 催化强度（外部注入或事件库自动）
        catalyst = catalyst_score if catalyst_score is not None else self._default_catalyst(code)

        # 综合
        score = 0.4 * catalyst + 0.3 * valuation_score + 0.3 * tech_score
        triggered = (
            score >= TRIGGER_MIN_SCORE
            and valuation_score >= 0.6            # 估值必须达标（低估）
            and (catalyst >= CATALYST_THRESHOLD or tech_score >= 0.7)  # 催化或技术至少一个强
        )

        caveats: list[str] = []
        if triggered:
            neg_score, neg_title = self._strongest_negative_event(code)
            chain = build_scan_risk_chain()
            ok, reason_msg = chain.check(
                code=code, name=name or self._code_name(code), action=action,
                negative_event_score=neg_score, negative_event_title=neg_title,
                day_orders=day_orders, day_value=day_value, position_pct=position_pct,
            )
            if not ok:
                triggered = False
                caveats.append(reason_msg)

        reason = self._build_reason(catalyst, valuation_score, tech_score, score, triggered,
                                    implied_return)
        if caveats:
            reason += " ⛔" + "；".join(caveats[:1])
        return TriggerResult(
            code=code, triggered=triggered, score=round(score, 4),
            catalyst_score=round(catalyst, 4), valuation_score=round(valuation_score, 4),
            tech_score=round(tech_score, 4), reason=reason,
            implied_return=round(implied_return, 4), fair_value=round(fair_value, 2),
            current_price=round(current_price, 2),
            stop_loss=round(current_price * 0.92, 2) if triggered else 0.0,
            key_technical=tech_note,
            catalyst_note="催化强度由外部注入/事件库自动",
            caveats=caveats,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )

    def _scan_one(self, code: str, catalyst_score: float | None,
                  asof_date: str | None = None) -> TriggerResult:
        """兼容入口 → 公共判定 _evaluate"""
        return self._evaluate(code, catalyst_score, asof_date=asof_date)

    def _strongest_negative_event(self, code: str, days: int = 7) -> tuple[float, str]:
        """近 days 天最强负面事件的（强度, 标题）；无则 (0.0, "")"""
        try:
            lib = self._event_lib
            if lib is None:
                from trader3.v2.events import get_event_library
                lib = get_event_library()
            evs = lib.get_recent(code, days=days)
            negs = [e for e in evs if getattr(e, "direction", "") == "negative"]
            if not negs:
                return 0.0, ""
            strongest = max(negs, key=lambda e: abs(e.catalyst_score))
            return abs(strongest.catalyst_score), strongest.title
        except Exception as e:
            logger.debug("[trigger] negative event lookup %s: %s", code, e)
            return 0.0, ""

    def _code_name(self, code: str) -> str:
        """尽力取证券名称（黑名单/ST 规则用），失败返回空。

        统一经 qa_accessor.get_quote_snapshot 行情快照口取名称
        （换数据源只改 qa_accessor 一处）。
        """
        try:
            from trader3.v2.qa_accessor import get_quote_snapshot
            return str(get_quote_snapshot(code).get("name") or "")
        except Exception:
            return ""

    # ── 估值因子 ──

    def _valuation_factor(self, code: str, asof_date: str | None = None) -> tuple[float, float, float, float]:
        """
        估值分位：隐含收益率（估值锚 vs 现价）。

        Returns (valuation_score, implied_return, fair_value, current_price)
        """
        try:
            resp = self._t3.valuation_anchor(codes=[code], asof_date=asof_date)
            if not (resp and resp.success):
                return 0.5, 0.0, 0.0, 0.0
            km = resp.key_metrics
            fair_value = km.get("加权目标价", 0) or 0
            implied = km.get("隐含收益率", 0) or 0
            price = km.get("当前价", 0) or 0
        except Exception:
            return 0.5, 0.0, 0.0, 0.0

        # 隐含收益 → 估值分位（0~1）
        if implied >= 0.30:
            val = 1.0
        elif implied >= 0.15:
            val = 0.8
        elif implied >= 0.0:
            val = 0.5
        elif implied >= -0.10:
            val = 0.3
        else:
            val = 0.1  # 高估
        return val, implied, fair_value, price

    # ── 技术因子 ──

    def _tech_factor(self, code: str, current_price: float) -> tuple[float, str]:
        """
        技术确认：价格上破20日高 + 成交放量（qlib 实时）。

        Returns (tech_score, note)
        """
        if self._dp is None or current_price <= 0:
            return 0.5, "无行情数据（技术中性）"
        try:
            code_key = code.upper().replace(".", "")
            # 兼容多种代码格式：SH600519 / 600519 / sh600519
            lookup = None
            for fmt in (f"sh{code_key[-6:]}", f"sz{code_key[-6:]}", code_key.lower()):
                close, dates = self._dp.load_stock(fmt, "close")
                if len(close) > 20:
                    lookup = (close, dates)
                    break
            if not lookup:
                return 0.5, "无可映射行情"

            close = lookup[0]
            valid = close[close > 0]
            if len(valid) < 25:
                return 0.5, "行情样本不足"

            latest = float(valid[-1])
            ma20 = float(np.mean(valid[-20:]))
            high20 = float(np.max(valid[-20:]))

            # 上破20日高 + 站上MA20
            breakout = latest >= high20 * 0.995
            above_ma = latest > ma20
            # 放量（从 qlib volume 实时计算）
            vol_up = False
            try:
                instrument = self._guess_instrument(code)
                volume, _ = self._dp.load_stock(instrument, "volume")
                if len(volume) > 20:
                    vvalid = volume[volume > 0]
                    if len(vvalid) >= 2:
                        vol_up = float(vvalid[-1]) > 1.2 * float(np.mean(vvalid[-20:]))
            except Exception:
                pass

            if breakout and above_ma and vol_up:
                return 1.0, f"上破20日高并放量，站上MA20（现价 {latest:.2f} vs 20日高 {high20:.2f}）"
            if breakout or (above_ma and vol_up):
                return 0.7, f"技术上破（现价 {latest:.2f} 上破20日高/放量）"
            if above_ma:
                return 0.5, f"站上MA20但未放量（现价 {latest:.2f}）"
            return 0.3, f"技术上未确认（现价 {latest:.2f} 低于MA20 {ma20:.2f}）"
        except Exception as e:
            logger.debug("[trigger] tech factor %s: %s", code, e)
            return 0.5, "技术计算失败（中性）"

    def _guess_instrument(self, code: str) -> str:
        """把股票代码映射到 qlib instrument 目录名"""
        c = code.upper().replace(".", "")
        if c.startswith(("SH", "SZ", "BJ")):
            return c[:2].lower() + c[2:]
        if c.isdigit():
            c = c.zfill(6)
            if c.startswith(("4", "8", "92")):   # 北交所（43/83/87/88/920）
                return "bj" + c
            if c.startswith(("6", "9")):
                return "sh" + c
            if c.startswith(("0", "2", "3")):
                return "sz" + c
        return c.lower()

    # ── 催化因子 ──

    def _default_catalyst(self, code: str) -> float:
        """
        默认催化强度：从事件库自动读取（新闻/公告/龙虎榜/涨停池）。
        降噪：仅 positive 抬底至 0.5；negative 压低；neutral 取原始分不保底
        （例行事件不应被抬成"中性偏强"）。
        """
        try:
            lib = self._event_lib
            if lib is None:
                from trader3.v2.events import get_event_library
                lib = get_event_library()
            ev = lib.get_strongest_recent(code, days=7)
            if ev is None:
                return 0.5  # 无事件 → 中性
            if ev.direction == "negative":
                return max(ev.catalyst_score, 0.1)  # 负向催化本身低分
            if ev.direction == "positive":
                return max(ev.catalyst_score, 0.5)  # 正向至少中性，强事件上浮
            return min(max(ev.catalyst_score, 0.0), 1.0)  # neutral：原始分，无保底
        except Exception:
            return 0.5

    # ── 理由构建 ──

    def _build_reason(self, catalyst, val_score, tech_score, score, triggered, implied) -> str:
        c = "强" if catalyst >= 0.6 else "中" if catalyst >= 0.4 else "弱"
        v = "低估" if implied >= 0.15 else "合理" if implied >= 0 else "高估"
        t = "确认" if tech_score >= 0.7 else "待确认" if tech_score >= 0.5 else "未确认"
        head = "✅ 触发" if triggered else "观察"
        return (f"{head}：催化{c}({catalyst:.2f}) × 估值{v}(隐含{implied:+.1%}) × "
                f"技术{t}({tech_score:.2f}) → 信号 {score:.2f}")

    # ── CLI 便捷 ──

    def scan_watchlist(self, db_path=None, catalyst_scores=None, asof_date: str | None = None,
                       dry_run: bool = False) -> list[TriggerResult]:
        """扫描 watchlist 数据库里的全部自选股。

        dry_run=True：只报告不改库（不迁移状态、不写触发结果）。
        """
        from trader3.v2.watchlist import get_watchlist
        wl = get_watchlist(db_path)
        items = wl.list()
        results = self.scan(items, catalyst_scores, asof_date=asof_date)
        if not dry_run:
            for r in results:
                apply_trigger_transition(wl, r)
        wl.close()
        return results


def apply_trigger_transition(wl, result: TriggerResult) -> bool:
    """统一两段式白名单迁移（_scan_one 判定后由 watchlist 调用方共用）：

    观察中 → 关注 → 买入区间（仅当 result.triggered 且风控未被否决）。
    daily_pipeline.run_daily 与 TriggerEngine.scan_watchlist 共用本函数。
    """
    if not result or not result.triggered:
        return False
    from trader3.v2.watchlist import ATTENTION, BUY_ZONE, OBSERVING
    item = wl.get(result.code)
    if not item:
        return False
    if item.status == ATTENTION:
        return wl.transition(result.code, BUY_ZONE, "三因子触发", result.to_dict())
    if item.status == OBSERVING:
        return wl.transition(result.code, ATTENTION, "估值进入区间", result.to_dict())
    return False


def get_trigger_engine(t3=None) -> TriggerEngine:
    return TriggerEngine(t3)
