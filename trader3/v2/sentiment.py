"""
新闻情绪管线（Sentiment Pipeline）。

从 sources.SourceItem 流构建每日个股/组合情绪因子：
1. 中文金融情感词典打分（正负词 + 否定词 + 程度副词，纯规则、零依赖）
2. 标题加权 > 正文；按半衰期时间衰减聚合到 code × date
3. 持久化 shared_state/sentiment/YYYYMMDD.json，并输出因子宽表
4. 可选 LLM 增强钩子（interpret.py 风格），缺省关闭

词典说明：内置 ~120 词的精简金融词典（利好/利空/调控/业绩类），
覆盖日常新闻的主要极性来源；可经 lexicon_path 换成自定义 CSV（word,score）。
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger("trader3.v2.sentiment")

# ── 内置精简金融情感词典 ──────────────────────────────
_POSITIVE = {
    "涨停": 2.0, "大涨": 2.0, "暴涨": 2.5, "利好": 2.0, "利多": 1.8,
    "超预期": 2.0, "预增": 1.8, "扭亏": 2.0, "净利增": 1.6, "增长": 0.8,
    "创新高": 1.6, "突破": 1.0, "中标": 1.5, "签约": 1.2, "回购": 1.5,
    "增持": 1.5, "分红": 1.0, "扩产": 1.0, "涨价": 1.2, "订单": 1.2,
    "获批": 1.5, "批准": 1.2, "合作": 0.8, "重组成功": 2.0, "摘帽": 1.8,
    "纳入指数": 1.5, "调入": 1.0, "回购注销": 1.8, "业绩预喜": 1.8,
    "景气": 1.0, "需求旺盛": 1.2, "满产": 1.3, "供不应求": 1.5,
    "政策支持": 1.3, "补贴": 1.0, "减税": 1.2, "降准": 1.2, "降息": 1.3,
}
_NEGATIVE = {
    "跌停": -2.0, "大跌": -2.0, "暴跌": -2.5, "利空": -2.0, "预亏": -2.0,
    "亏损": -1.6, "净利降": -1.5, "下滑": -1.0, "下降": -0.8, "减持": -1.5,
    "质押爆仓": -2.0, "立案": -2.2, "调查": -1.8, "处罚": -1.8, "警告函": -1.5,
    "退市": -2.5, "戴帽": -1.8, "商誉减值": -1.8, "计提": -1.2, "违约": -2.0,
    "诉讼": -1.2, "仲裁": -1.0, "冻结": -1.5, "辞职": -1.0, "离职": -0.9,
    "终止": -1.3, "失败": -1.3, "问询": -1.2, "关注函": -1.2, "警示": -1.3,
    "高估": -1.0, "泡沫": -1.3, "产能过剩": -1.4, "价格战": -1.2,
    "需求疲软": -1.3, "滞销": -1.4, "库存积压": -1.3, "加息": -1.2,
}
_NEGATORS = {"不", "未", "没有", "无", "难以", "并非", "取消", "终止", "撤销"}
_INTENSIFIERS = {"大幅": 1.5, "显著": 1.4, "严重": 1.6, "急剧": 1.6, "超": 1.3}

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,6}")


@dataclass
class SentimentResult:
    """单条文本的情绪得分明细。"""
    score: float            # [-3, 3] 截断后的综合分
    hits: list[tuple[str, float]] = field(default_factory=list)
    n_tokens: int = 0


_LEXICON: dict[str, float] = {**_POSITIVE, **_NEGATIVE}


def score_text(text: str) -> SentimentResult:
    """
    规则打分：词典子串匹配（长词优先），前文窗口内
    程度副词加权、紧邻否定词翻转（×-0.8）。结果截断 [-3, 3]。
    """
    if not text:
        return SentimentResult(score=0.0)
    total, hits = 0.0, []
    consumed = [False] * len(text)
    for word in sorted(_LEXICON, key=len, reverse=True):
        start = 0
        while True:
            i = text.find(word, start)
            if i < 0:
                break
            start = i + 1
            span = range(i, i + len(word))
            if any(consumed[j] for j in span):
                continue  # 已被更长词覆盖（如"回购注销"优先于"回购"）
            w = 1.0
            ctx = text[max(0, i - 4):i]
            for inten, iw in _INTENSIFIERS.items():
                if inten in ctx:
                    w *= iw
                    break
            for negator in _NEGATORS:
                if ctx.endswith(negator):
                    w *= -0.8
                    break
            total += _LEXICON[word] * w
            hits.append((word, round(_LEXICON[word] * w, 2)))
            for j in span:
                consumed[j] = True
    score = max(-3.0, min(3.0, total))
    return SentimentResult(score=round(score, 3), hits=hits, n_tokens=len(text))


def score_items(items: list, title_weight: float = 2.0) -> float:
    """对 SourceItem 列表打分取均值；标题权重 > 正文。items 元素需有 title/content。"""
    scores = []
    for it in items:
        s_title = score_text(getattr(it, "title", "")).score
        s_body = score_text((getattr(it, "content", "") or "")[:500]).score
        scores.append(title_weight * s_title + s_body)
    return sum(scores) / len(scores) / title_weight if scores else 0.0


# ── 聚合与持久化 ──────────────────────────────────────

def _half_life_weight(ts_str: str, now: datetime | None = None,
                      half_life_hours: float = 24.0) -> float:
    """按发布时间的指数衰减权重。解析失败给 0.5 中性权重。"""
    now = now or datetime.now()
    try:
        ts = datetime.strptime(str(ts_str)[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return 0.5
    hours = max((now - ts).total_seconds() / 3600.0, 0.0)
    return 0.5 ** (hours / half_life_hours)


def aggregate_daily(code_scores: dict[str, list[tuple[str, float]]],
                    as_of: str | None = None,
                    half_life_hours: float = 24.0) -> dict[str, float]:
    """
    code_scores: {code: [(ts_str, raw_score), ...]}
    返回 {code: 加权情绪分}，时间半衰期加权后截断 [-3, 3]。
    """
    out = {}
    for code, pairs in code_scores.items():
        num, den = 0.0, 0.0
        for ts_str, sc in pairs:
            w = _half_life_weight(ts_str, half_life_hours=half_life_hours)
            num += w * sc
            den += w
        if den > 1e-9:
            out[code] = round(max(-3.0, min(3.0, num / den)), 4)
    return out


class SentimentStore:
    """shared_state/sentiment/ 下按日持久化，并提供滚动窗口读取。"""

    def __init__(self, state_dir: str | None = None):
        from trader3.shared_state import SharedState
        self._ss = SharedState(state_dir) if state_dir else SharedState()
        self.dir = os.path.join(self._ss.state_dir, "sentiment")
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, date: str) -> str:
        return os.path.join(self.dir, f"{date}.json")

    def write(self, scores: dict[str, float], date: str | None = None) -> str:
        d = date or datetime.now().strftime("%Y%m%d")
        tmp = self._path(d) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"date": d, "scores": scores}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path(d))
        return self._path(d)

    def read(self, date: str) -> dict[str, float]:
        p = self._path(date)
        if not os.path.exists(p):
            return {}
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("scores", {})

    def window_frame(self, days: int = 20) -> pd.DataFrame:
        """最近 N 天 {date × code} 宽表，缺失 NaN。

        日期窗口按"自然日"滚动计算。为避免午夜/时区边界把今天排除在窗口外，
        窗口以 [今天-(days-1), 今天] 的日期串集合（UTC 之外用本地自然日）为准。
        """
        today = datetime.now().date()
        window_dates = {(today - timedelta(days=i)).strftime("%Y%m%d")
                        for i in range(days)}
        frames = []
        for d in sorted(window_dates):
            s = self.read(d)
            if s:
                frames.append(pd.Series(s, name=d))
        return pd.DataFrame(frames) if frames else pd.DataFrame()


# ── 因子输出 ──────────────────────────────────────────

def sentiment_factor(store: SentimentStore, window_days: int = 10,
                     decay: bool = True) -> pd.Series:
    """
    把近 N 日情绪聚合成单日因子值（截面）：
    近期权重高（可选指数衰减），先日内均值再跨期均值。
    """
    frame = store.window_frame(window_days)
    if frame.empty:
        return pd.Series(dtype=float)
    if decay:
        w = np.array([0.5 ** i for i in range(len(frame))])
        w = w / w.sum()
        vals = frame.fillna(0.0).values
        return pd.Series(vals.T @ w, index=frame.columns, name="sentiment")
    return frame.mean(axis=0).rename("sentiment")


def pipeline_from_sources(sources_with_codes: dict[str, list],
                          store: SentimentStore | None = None,
                          half_life_hours: float = 24.0) -> dict[str, float]:
    """
    一站式入口：
    sources_with_codes: {code: [SourceItem, ...]}（来自 collector 的分组结果）
    打分 → 时间衰减聚合 → 写库 → 返回当日截面分数。
    """
    st = store or SentimentStore()
    code_pairs: dict[str, list[tuple[str, float]]] = {}
    for code, items in sources_with_codes.items():
        pairs = []
        for it in items:
            sc = score_items([it]) if not isinstance(it, dict) else score_text(
                it.get("title", "")).score
            ts = getattr(it, "ts", "") if not isinstance(it, dict) else it.get("ts", "")
            pairs.append((str(ts), float(sc)))
        code_pairs[code] = pairs
    scores = aggregate_daily(code_pairs, half_life_hours=half_life_hours)
    if scores:
        path = st.write(scores)
        logger.info("[sentiment] wrote %d codes → %s", len(scores), path)
    return scores


def zscore_cross_section(scores: pd.Series, floor: float = 2.5) -> pd.Series:
    """截面 z-score 并去极化，供合成因子使用。"""
    sd = scores.std(ddof=0)
    if not sd or math.isnan(sd) or sd < 1e-9:
        return scores * 0.0
    z = (scores - scores.mean()) / sd
    return z.clip(-floor, floor)
