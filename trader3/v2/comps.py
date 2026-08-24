"""
3号交易员 v2.0 — 可比公司分析模块 (comps_analysis)

吸收 Anthropic comps-analysis Skill 启发 + 国内落地改造路线：

流程（对齐文章蓝图）：
  1. 建可比池（同行业/同商业模式/同规模/同市场）
  2. 拉数（financials.db 营收/利润/市值 + qlib 价格）
  3. 算倍数（EV/Revenue、EV/EBITDA、P/E）+ 四分位统计基准
  4. 异常标红（2 标准差外的离群倍数）
  5. 每格带来源注释 → 可审计 comps 表

产出：可审计可比公司表（CompsTable），供 2hao 写报告时"比价"引用。
铁律：真实数据优先；无数据的可比公司标注"未验证"，不硬算。
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("trader3.v2.comps")


# ── A股行业 → 市值分档（简单口径，够用即可） ──
SIZE_TIERS = [("大盘", 500e8), ("中盘", 100e8), ("小盘", 0)]  # 流通市值（元）

# 行业默认估值倍数（国内常用）——用于可比池不足时的"行业基准"标注
INDUSTRY_DEFAULT_MULTIPLES = {
    "白酒": {"pe": 25.0, "ev_ebitda": 18.0, "ev_revenue": 10.0},
    "银行": {"pe": 6.0, "ev_ebitda": 8.0, "ev_revenue": 3.0},
    "医药": {"pe": 30.0, "ev_ebitda": 20.0, "ev_revenue": 5.0},
    "科技": {"pe": 35.0, "ev_ebitda": 25.0, "ev_revenue": 6.0},
    "制造": {"pe": 25.0, "ev_ebitda": 15.0, "ev_revenue": 3.0},
    "消费": {"pe": 30.0, "ev_ebitda": 20.0, "ev_revenue": 5.0},
    "新能源": {"pe": 30.0, "ev_ebitda": 20.0, "ev_revenue": 4.0},
}


@dataclass
class CompsRow:
    """可比公司单行"""
    code: str
    name: str
    market_cap_cny: float = 0.0          # 市值（元）
    revenue_cny: float = 0.0             # 营收 TTM（元）
    ebitda_cny: float = 0.0              # EBITDA TTM（元，用营业利润近似）
    net_profit_cny: float = 0.0          # 净利润 TTM（元）
    pe: float = 0.0                      # 市值 / 净利润TTM
    ev_ebitda: float = 0.0               # 真 EV / EBITDA（仅有息负债可得时）
    p_ebitda: float = 0.0                # 市值 / EBITDA（EV 不可得时的退化口径）
    ev_revenue: float = 0.0              # 真 EV / 营收（仅有息负债可得时）
    has_ev: bool = False                 # 是否取到了真 EV（有息负债字段可得）
    data_verified: bool = False          # 真实数据（financials.db）
    data_source: str = ""                # 来源标注（财务期+口径）
    is_target: bool = False              # 是否目标公司
    outlier: bool = False                # 是否离群（2 标准差外）
    source_note: str = ""                # 每格来源注释


@dataclass
class CompsTable:
    """可比公司分析结果"""
    target_code: str = ""
    target_name: str = ""
    industry: str = ""
    peers: List[CompsRow] = field(default_factory=list)
    # 统计基准（四分位）
    stats: Dict[str, Dict[str, float]] = field(default_factory=dict)  # {metric: {p25,p50,p75,mean,std}}
    # 异常标红
    outliers: List[str] = field(default_factory=list)   # [code.metric]
    # 结论
    conclusion: str = ""                 # 目标公司溢价/折价
    caveats: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target_code": self.target_code, "target_name": self.target_name,
            "industry": self.industry,
            "peers": [asdict(p) for p in self.peers],
            "stats": self.stats, "outliers": self.outliers,
            "conclusion": self.conclusion, "caveats": self.caveats,
        }


class CompsAnalyzer:
    """可比公司分析器（复用 financials.db + qlib 行情）

    财务口径：TTM（最近4个单季合计）。financials.db 存的是年内累计值
    （如 600519 2025年报营收1688亿 vs 2025Q3累计1284亿），先用差额法
    还原单季再求和；不足4期按期数年化并在 source_note 标注。
    """

    # balance 表有息负债/现金候选字段（PRAGMA 探测确认存在的才生效）
    DEBT_KEYS = ("shortLoan", "longLoan", "bondPayable")
    CASH_KEYS = ("moneyFunds", "cashAssets")

    def __init__(self, fp=None, dp=None):
        if fp is None:
            try:
                from trader3.financials_provider import FinancialsProvider
                self._fp = FinancialsProvider()
            except Exception as e:
                logger.warning("[comps] financials 不可用: %s", e)
                self._fp = None
        else:
            self._fp = fp
        if dp is None:
            try:
                from trader3.data_provider import QlibDataProvider
                self._dp = QlibDataProvider()
            except Exception:
                self._dp = None
        else:
            self._dp = dp

    # ── 主入口 ──

    def analyze(self, target_code: str, industry: str = "",
                peer_codes: Optional[List[str]] = None, n_peers: int = 8,
                asof_date: Optional[str] = None) -> CompsTable:
        """对目标公司做可比分析"""
        # 1. 目标公司财务快照
        target_name, target_fin = self._company_snapshot(target_code, asof_date=asof_date)
        target_name = target_name or target_code

        # 2. 建可比池（传入 > 自动按行业补）
        peers = []
        if peer_codes:
            for c in peer_codes:
                peers.append(self._peer_snapshot(c, target_code, asof_date=asof_date))
        else:
            # 用 industry 默认 + 同市值附近补位（简化：先只取传入/默认，行业名匹配留 2hao 知识库）
            peers = self._auto_peers(target_code, industry, n_peers, asof_date=asof_date)

        # 3. 统计基准 + 异常标红（EBITDA 倍数按 EV 可得性选口径）
        ebitda_metric, ebitda_caveat = self._ebitda_metric_choice(peers)
        metrics = ["pe", ebitda_metric, "ev_revenue"]
        stats = self._compute_stats(peers, metrics)
        outliers = self._mark_outliers(peers, stats, metrics)

        # 4. 结论（目标 vs 中位数）
        conclusion, caveats = self._conclude(peers, stats, target_code, industry, metrics)
        if ebitda_metric == "p_ebitda" and any(p.ebitda_cny > 0 for p in peers):
            caveats.append(ebitda_caveat)

        return CompsTable(
            target_code=target_code, target_name=target_name, industry=industry,
            peers=peers, stats=stats, outliers=outliers,
            conclusion=conclusion, caveats=caveats,
        )

    # ── 公司快照 ──

    def _real_price(self, code: str) -> float:
        """从 Sina 拉真实市价（code 自动映射 sh/sz 前缀）"""
        try:
            from trader3.v2.sources import SinaQuoteSource
            items = SinaQuoteSource().fetch([self._sina_code(code)], limit=1)
            if items:
                # content 形如 "开盘.. 昨收.. 现价X.. 高.. 低.. 量.."
                txt = items[0].content
                m = re.search(r"现价([\d.]+)", txt)
                if m:
                    return float(m.group(1))
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _sina_code(code: str) -> str:
        c = code.upper().replace(".", "")
        if c.startswith(("SH", "SZ", "BJ")):
            return c[:2].lower() + c[2:]
        if c.isdigit():
            c = c.zfill(6)
            if c.startswith(("4", "8", "92")):   # 北交所
                return "bj" + c
            if c.startswith(("6", "9")):
                return "sh" + c
            return "sz" + c
        return c.lower()

    def _company_snapshot(self, code: str, asof_date: Optional[str] = None) -> tuple:
        """从 financials.db 读公司财务 + 实时行情/股本/市值（东财→腾讯→新浪）"""
        fin = {}
        if self._fp:
            if asof_date:
                # 防前视：按公告日对齐读取历史财务
                from trader3.v2.announcement_calendar import AnnouncementCalendar
                fin = AnnouncementCalendar().financials_asof(code, asof_date) or {}
            else:
                fin = self._fp.get_latest_financials(code) or {}

        # 实时行情 + 股本 + 市值 + 名称（统一 get_quote）
        from trader3.v2.market_data import get_quote
        q = get_quote(code)
        name = q.get("name", "")
        price = float(q.get("price") or 0)
        market_cap = float(q.get("market_cap") or 0)

        snap = {
            "market_cap_cny": market_cap,
            "revenue_cny": (fin.get("MBRevenue") or 0),
            "ebitda_cny": (fin.get("operateProfit") or 0),
            "net_profit_cny": (fin.get("netProfit") or 0),
            "quarter": fin.get("quarter", ""),
        }
        return name, snap

    # ── TTM 口径 ──

    def _quarter_history(self, code: str, field: str, n: int = 5) -> List[tuple]:
        """取利润表字段最近 n 期原始值 [(quarter, value)] 降序。

        接缝方法：测试可 monkeypatch 本方法注入已知序列。
        """
        if not self._fp:
            return []
        try:
            hist = self._fp.get_field_history(code, field, table="profit", n=n)
            return [(h["quarter"], h["value"]) for h in hist if h.get("value") is not None]
        except Exception as e:
            logger.debug("[comps] quarter history %s/%s fail: %s", code, field, e)
            return []

    @staticmethod
    def _looks_cumulative(pairs: List[tuple]) -> bool:
        """年内逐季递增 → 判定为累计值口径（如 Q1<H1<Q3<年报）"""
        by_year: Dict[str, List[tuple]] = {}
        for q, v in pairs:
            by_year.setdefault(str(q)[:4], []).append((str(q), float(v)))
        for _, items in by_year.items():
            items.sort()
            vals = [v for _, v in items]
            if len(vals) >= 2 and all(b > a for a, b in zip(vals, vals[1:])):
                return True
        return False

    @staticmethod
    def _cumulative_to_singles(pairs: List[tuple]) -> List[tuple]:
        """累计值差额法还原单季值（同年内逐季差分，首期为本身），降序返回"""
        asc = sorted(pairs, key=lambda x: str(x[0]))
        out: List[tuple] = []
        prev_q, prev_v = None, None
        for q, v in asc:
            if prev_q is not None and str(prev_q)[:4] == str(q)[:4]:
                out.append((q, float(v) - float(prev_v)))
            else:
                out.append((q, float(v)))
            prev_q, prev_v = q, v
        return list(reversed(out))

    def _ttm_financials(self, code: str, max_quarter: Optional[str] = None) -> Dict[str, object]:
        """最近4个单季 MBRevenue/netProfit/operateProfit 求和为 TTM。

        max_quarter：防前视封顶（只使用 <= 该财报期的数据）。
        Returns {revenue_ttm, net_profit_ttm, operate_profit_ttm,
                 quarters(有效单季期数), note(口径标注)}
        """
        fields = {"revenue_ttm": "MBRevenue", "net_profit_ttm": "netProfit",
                  "operate_profit_ttm": "operateProfit"}
        singles_map: Dict[str, List[float]] = {}
        for key, fld in fields.items():
            pairs = self._quarter_history(code, fld, n=5)
            if max_quarter:
                pairs = [(q, v) for q, v in pairs if str(q) <= max_quarter]
            if not pairs:
                singles_map[key] = []
                continue
            singles = (self._cumulative_to_singles(pairs)
                       if self._looks_cumulative(pairs) else pairs)
            singles_map[key] = [float(v) for _, v in singles[:4]]

        min_len = min((len(v) for v in singles_map.values()), default=0)
        scale = (4.0 / min_len) if 0 < min_len < 4 else 1.0
        result: Dict[str, object] = {
            key: sum(vals[:min_len]) * scale if min_len else 0.0
            for key, vals in singles_map.items()
        }
        result["quarters"] = min_len
        if min_len >= 4:
            result["note"] = "TTM口径：近4单季合计"
        elif min_len > 0:
            result["note"] = f"仅{min_len}个单季，按期数年化(×{scale:.2f})近似TTM"
        else:
            result["note"] = "无季度财务数据（未验证）"
        return result

    def _balance_latest(self, code: str) -> Dict[str, float]:
        """balance 表最新一期有息负债/现金候选字段；拿不到返回空 dict"""
        if not self._fp:
            return {}
        try:
            fin = self._fp.get_latest_financials(code) or {}
            keys = set(self.DEBT_KEYS) | set(self.CASH_KEYS)
            return {k: float(fin[k]) for k in keys if fin.get(k)}
        except Exception as e:
            logger.debug("[comps] balance %s fail: %s", code, e)
            return {}

    @classmethod
    def enterprise_value(cls, market_cap: float, bal: Dict[str, float]) -> Optional[float]:
        """EV = 市值 + 有息负债 − 现金。有息负债字段全部缺失时返回 None。"""
        debt = sum(bal.get(k, 0) for k in cls.DEBT_KEYS)
        cash = sum(bal.get(k, 0) for k in cls.CASH_KEYS)
        if debt > 0:
            return market_cap + debt - cash
        return None

    def _peer_snapshot(self, peer_code: str, target_code: str,
                       asof_date: Optional[str] = None) -> CompsRow:
        """单只可比公司快照（财务统一 TTM 口径；asof 时按可见财报期封顶防前视）"""
        fname, fin = self._company_snapshot(peer_code, asof_date=asof_date)
        mc = float(fin.get("market_cap_cny", 0) or 0)
        visible_q = fin.get("quarter") or None
        ttm = self._ttm_financials(peer_code, max_quarter=visible_q)
        rev = float(ttm["revenue_ttm"])
        ebitda = float(ttm["operate_profit_ttm"])
        np_ = float(ttm["net_profit_ttm"])
        has_data = rev > 0 or np_ > 0
        ds_parts = ([visible_q] if visible_q else []) + [f"TTM({ttm['quarters']}期)"]
        row = CompsRow(
            code=peer_code, name=fname or peer_code,
            market_cap_cny=mc, revenue_cny=rev, ebitda_cny=ebitda,
            net_profit_cny=np_,
            data_verified=has_data,
            data_source=" / ".join(ds_parts),
            is_target=(peer_code == target_code),
            source_note=str(ttm["note"]),
        )
        # EV：有息负债/现金从 balance 取，拿得到算真 EV，拿不到退化 P/EBITDA
        # （asof 防前视路径下 balance 无历史对齐，直接退化为 P/EBITDA）
        bal = {} if asof_date else self._balance_latest(peer_code)
        ev = self.enterprise_value(mc, bal) if mc > 0 else None
        row.has_ev = ev is not None
        if rev > 0 and mc > 0 and row.has_ev:
            row.ev_revenue = ev / rev
        if ebitda > 0:
            if mc > 0:
                row.p_ebitda = mc / ebitda          # 退化口径：市值/EBITDA
            if row.has_ev:
                row.ev_ebitda = ev / ebitda         # 真口径：EV/EBITDA
        if np_ > 0 and mc > 0:
            row.pe = mc / np_
        return row

    def _auto_peers(self, target_code: str, industry: str, n: int, asof_date: Optional[str] = None) -> List[CompsRow]:
        """自动补可比：用 industry 默认基准构建虚拟可比（诚实标注）"""
        _, tfin = self._company_snapshot(target_code, asof_date=asof_date)
        t_mc = tfin.get("market_cap_cny", 0)
        # 在 peers 里至少放目标本身（用于对比），再用行业默认做"行业基准行"
        peers = [self._peer_snapshot(target_code, target_code, asof_date=asof_date)]
        if industry and industry in INDUSTRY_DEFAULT_MULTIPLES:
            # 行业基准行（默认倍数）
            base = INDUSTRY_DEFAULT_MULTIPLES[industry]
            peers.append(CompsRow(
                code="INDUSTRY", name=f"{industry}-行业基准", market_cap_cny=t_mc,
                pe=base["pe"], ev_ebitda=base["ev_ebitda"], ev_revenue=base["ev_revenue"],
                data_verified=False, data_source="行业默认", is_target=False,
                source_note=f"{industry} 行业默认倍数（非具体公司）",
            ))
        return peers

    # ── 统计 + 离群 ──

    @staticmethod
    def _ebitda_metric_choice(rows: List[CompsRow]) -> tuple:
        """≥2 家取到真 EV → 用 EV/EBITDA；否则退化 P/EBITDA 并给出 caveat"""
        n_ev = sum(1 for r in rows if getattr(r, "ev_ebitda", 0) > 0)
        if n_ev >= 2:
            return "ev_ebitda", ""
        return ("p_ebitda",
                "有息负债字段缺失，真 EV 不可得，EV/EBITDA 退化为 P/EBITDA"
                "（市值/营业利润TTM）口径")

    def _compute_stats(self, rows: List[CompsRow],
                       metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        metrics = metrics or ["pe", "ev_ebitda", "p_ebitda", "ev_revenue"]
        stats = {}
        for metric in metrics:
            vals = [getattr(r, metric) for r in rows
                    if getattr(r, metric, 0) > 0 and not r.is_target]
            if len(vals) >= 3:
                arr = np.array(vals)
                stats[metric] = {
                    "p25": float(np.percentile(arr, 25)),
                    "p50": float(np.percentile(arr, 50)),
                    "p75": float(np.percentile(arr, 75)),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                }
            else:
                stats[metric] = {}
        return stats

    def _mark_outliers(self, rows: List[CompsRow], stats,
                       metrics: Optional[List[str]] = None) -> List[str]:
        metrics = metrics or ["pe", "ev_ebitda", "p_ebitda", "ev_revenue"]
        outliers = []
        for r in rows:
            if r.is_target or r.code == "INDUSTRY":
                continue
            for metric in metrics:
                s = stats.get(metric, {})
                v = getattr(r, metric)
                if s and v > 0 and "mean" in s and "std" in s and s["std"] > 0:
                    if abs(v - s["mean"]) > 2 * s["std"]:
                        r.outlier = True
                        outliers.append(f"{r.code}.{metric}")
        return outliers

    def _conclude(self, rows, stats, target_code, industry,
                  metrics: Optional[List[str]] = None) -> tuple:
        """目标 vs 行业中位数 → 溢价/折价结论"""
        metrics = metrics or ["pe", "ev_ebitda", "p_ebitda", "ev_revenue"]
        label_map = {"pe": "P/E", "ev_ebitda": "EV/EBITDA", "p_ebitda": "P/EBITDA",
                     "ev_revenue": "EV/Revenue"}
        target = next((r for r in rows if r.is_target), None)
        if not target:
            return "无目标数据（未验证）", ["目标公司财务不可用"]
        parts = []
        caveats = []
        for metric in metrics:
            label = label_map.get(metric, metric)
            s = stats.get(metric, {})
            tv = getattr(target, metric, 0)
            base = s.get("p50") if (s and s.get("p50", 0) > 0) else None
            use_default = False
            if base is None:
                # 可比池过薄时回退到行业默认基准（诚实标注）
                ind = next((r for r in rows if r.code == "INDUSTRY"), None)
                bv = getattr(ind, metric, 0) if ind else 0
                if bv == 0 and ind is not None and metric == "p_ebitda":
                    bv = getattr(ind, "ev_ebitda", 0)  # 行业默认只有 EV 口径时的近似
                if bv > 0 and tv > 0:
                    base, use_default = bv, True
            if base is None or not tv > 0:
                continue
            ratio = tv / base
            pos = "溢价" if ratio > 1.1 else "折价" if ratio < 0.9 else "合理"
            tag = "行业默认基准" if use_default else f"行业中位 {base:.1f}"
            parts.append(f"{label} {pos}（{tv:.1f} vs {tag} {base:.1f}）")
            if use_default:
                caveats.append(f"{label}：可比池过薄，采用行业默认基准，未经真实同业验证")
        if not parts:
            return "可比数据不足（未验证）", caveats + ["可比池太薄，需 2hao 知识库补行业可比"]
        return "；".join(parts), caveats


def get_comps_analyzer() -> CompsAnalyzer:
    return CompsAnalyzer()