"""多空辩论轻量版（P2-6，TradingAgents-CN 两段辩论思想落地）

设计要点（与 2026-09-03 差距报告 C2 对齐）：
- 触发引擎给出 TriggerResult 后，本模块组织一场"看多 vs 看空"结构化辩论，
  输出 DebateVerdict（confirmation / veto / abstain）+ 置信度 + 双方论点。
- LLM 不可用或辩论失败时 **静默放行（return None）**——辩论是增强层，
  绝不阻断主链路（触发→风控→纸面执行），与 interpret/sentiment 的降级哲学一致。
- 上下文注入全部来自系统已有真实数据：事件库（近7天正负面事件）、
  TriggerResult 三因子明细、持仓/价格信息——不引入新数据通道。
- 记忆沉淀：每场辩论落盘 shared_state/debate_log/<date>.json（lesson 可被后续检索）。

轻量版取舍（相对 TradingAgents 全量）：
- 只做 2 段（立论→驳论），不做多 Agent 角色链（analyst/researcher/trader/…）
- 单 LLM 双角色提示切换，不做多模型并行
- 结论只产出"维持/否决/弃权"三态，不产出仓位建议
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 数据结构 ─────────────────────────────────────────

VERDICTS = ("confirm", "veto", "abstain")


@dataclass
class DebateArgument:
    """单方论点：观点 + 依据（引用真实数据）"""
    side: str                    # bull / bear
    claim: str = ""              # 一句话主张
    evidence: list[str] = field(default_factory=list)  # 依据列表（引用事件/数字）
    conviction: float = 0.0      # 0~1


@dataclass
class DebateVerdict:
    """辩论结论：附加在 TriggerResult 之后的复核意见"""
    code: str
    verdict: str = "abstain"     # confirm / veto / abstain
    confidence: float = 0.0      # 0~1
    bull: DebateArgument = field(default_factory=lambda: DebateArgument(side="bull"))
    bear: DebateArgument = field(default_factory=lambda: DebateArgument(side="bear"))
    lesson: str = ""             # 一句话教训（沉淀用）
    llm_used: bool = False       # False=规则降级版辩论
    timestamp: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @property
    def is_veto(self) -> bool:
        return self.verdict == "veto"


# ── 上下文构建（全部来自已有真实数据） ─────────────────

def build_context(trigger_result, event_library=None, days: int = 7) -> dict:
    """汇总一只股票的辩论上下文：三因子明细 + 近期事件（正/负分组）。"""
    ctx = {
        "code": trigger_result.code,
        "direction": getattr(trigger_result, "direction", "buy"),
        "triggered": bool(getattr(trigger_result, "triggered", False)),
        "score": getattr(trigger_result, "score", 0.0),
        "catalyst_score": getattr(trigger_result, "catalyst_score", 0.0),
        "valuation_score": getattr(trigger_result, "valuation_score", 0.0),
        "tech_score": getattr(trigger_result, "tech_score", 0.0),
        "implied_return": getattr(trigger_result, "implied_return", 0.0),
        "fair_value": getattr(trigger_result, "fair_value", 0.0),
        "current_price": getattr(trigger_result, "current_price", 0.0),
        "reason": getattr(trigger_result, "reason", ""),
        "caveats": list(getattr(trigger_result, "caveats", []) or []),
        "positive_events": [],
        "negative_events": [],
    }
    lib = event_library
    if lib is None:
        try:
            from trader3.v2.events import get_event_library
            lib = get_event_library()
        except Exception as e:  # noqa: BLE001
            logger.warning("[debate] 事件库不可用: %s", e)
            lib = None
    if lib is not None:
        try:
            evs = lib.get_recent(trigger_result.code, days=days)
            for ev in evs:
                item = {
                    "title": getattr(ev, "title", ""),
                    "direction": getattr(ev, "direction", "neutral"),
                    "score": getattr(ev, "catalyst_score", 0.0),
                    "category": getattr(ev, "category", ""),
                }
                if item["direction"] == "negative":
                    ctx["negative_events"].append(item)
                elif item["direction"] == "positive":
                    ctx["positive_events"].append(item)
        except Exception as e:  # noqa: BLE001
            logger.warning("[debate] 事件读取失败: %s", e)
    return ctx


# ── LLM 辩论（可选） ──────────────────────────────────

_BULL_PROMPT = (
    "你是专注A股的多头研究员（bull researcher）。基于给定上下文，给出最强的看多论据。"
    "要求：引用上下文中的真实数字与事件，不得编造。输出 JSON："
    '{{"claim":"一句话主张","evidence":["依据1","依据2","依据3"],'
    '"conviction":0.0~1.0}}\n\n上下文：{ctx}'
)

_BEAR_PROMPT = (
    "你是专注A股的空头研究员（bear researcher）。基于给定上下文，给出最强的看空/风险论据，"
    "重点攻击多头论据的薄弱处。要求：引用上下文中的真实数字与事件，不得编造。输出 JSON："
    '{{"claim":"一句话主张","evidence":["依据1","依据2","依据3"],'
    '"conviction":0.0~1.0}}\n\n上下文：{ctx}'
)

_JUDGE_PROMPT = (
    "你是投资决策者。根据多头与空头论据，以及原始触发信号，做出裁决。"
    "输出 JSON："
    '{{"verdict":"confirm|veto|abstain","confidence":0.0~1.0,'
    '"lesson":"一句话教训（供未来决策沉淀）"}}\n'
    "裁决原则：空头论据显著强于多头（conviction 差 > 0.2 且证据扎实）才 veto；"
    "势均力敌 abstain；多头扎实才 confirm。\n\n"
    "原始信号：{trigger}\n多头：{bull}\n空头：{bear}"
)


def _parse_argument_json(text: str, side: str) -> DebateArgument:
    """从 LLM 输出解析论点 JSON；失败返回空论点（不抛异常）。"""
    arg = DebateArgument(side=side)
    try:
        m = re.search(r"\{.*\}", text, re.S)
        obj = json.loads(m.group(0)) if m else None
        if isinstance(obj, dict):
            arg.claim = str(obj.get("claim", ""))[:200]
            arg.evidence = [str(e)[:160] for e in (obj.get("evidence") or [])][:5]
            c = float(obj.get("conviction", 0) or 0)
            arg.conviction = max(0.0, min(1.0, c))
    except Exception:  # noqa: BLE001
        pass
    return arg


def _parse_verdict_json(text: str) -> tuple[str, float, str]:
    v, c, lesson = "abstain", 0.0, ""
    try:
        m = re.search(r"\{.*\}", text, re.S)
        obj = json.loads(m.group(0)) if m else None
        if isinstance(obj, dict):
            vv = str(obj.get("verdict", "abstain")).lower()
            v = vv if vv in VERDICTS else "abstain"
            c = max(0.0, min(1.0, float(obj.get("confidence", 0) or 0)))
            lesson = str(obj.get("lesson", ""))[:200]
    except Exception:  # noqa: BLE001
        pass
    return v, c, lesson


def _llm_chat(prompt: str, llm_fn=None) -> str | None:
    """统一 LLM 调用：优先注入的 llm_fn，其次 OpenRouter（llm_sentiment.LLMScorer 的
    _chat 端点/降级策略）。任何失败返回 None（调用方走规则辩论）。"""
    if llm_fn is not None:
        try:
            out = llm_fn(prompt)
            return str(out) if out else None
        except Exception as e:  # noqa: BLE001
            logger.warning("[debate] llm_fn 调用失败: %s", e)
            return None
    try:
        from trader3.v2.llm_sentiment import LLMScorer
        scorer = LLMScorer()
        if not scorer.api_key:
            return None
        # 依次试模型列表（_chat 已含限速/重试/降级日志）
        for model in scorer.models[:2]:
            text = scorer._chat(model, prompt)
            if text:
                return text
    except Exception as e:  # noqa: BLE001
        logger.warning("[debate] LLM 辩论不可用（走规则辩论）: %s", e)
    return None


# ── 规则辩论（LLM 降级版：确定性、无网络） ──────────────

def _rule_debate(ctx: dict) -> DebateVerdict:
    """无 LLM 时的确定性辩论：多空论据由规则从上下文抽取，裁决由三因子结构决定。

    裁决规则（保守偏 abstain，与触发引擎互补而非重复）：
    - 空头否决条件（满足其一即 veto）：负面事件存在且触发分<0.75；
      caveats 非空（风控链已亮黄牌）；implied_return <= 0（估值无安全边际）
    - 多头确认条件（全部满足才 confirm）：triggered 且 catalyst>=0.5
      且无负面事件 且 implied_return>0.05
    - 其余 abstain
    """
    v = DebateVerdict(code=ctx["code"], verdict="abstain", llm_used=False)
    # 多头论据：三因子中的强项
    if ctx["catalyst_score"] >= 0.5:
        v.bull.evidence.append(f"催化强度 {ctx['catalyst_score']:.2f}（事件驱动明确）")
    if ctx["valuation_score"] >= 0.6:
        v.bull.evidence.append(
            f"估值分位 {ctx['valuation_score']:.2f}，"
            f"估值锚 ¥{ctx['fair_value']:.1f} vs 现价 ¥{ctx['current_price']:.1f}，"
            f"隐含收益 {ctx['implied_return']*100:+.1f}%")
    if ctx["tech_score"] >= 0.7:
        v.bull.evidence.append(f"技术确认 {ctx['tech_score']:.2f}（{ctx['reason'][:40]}）")
    if ctx["positive_events"]:
        v.bull.evidence.append(
            f"近7天正面事件 {len(ctx['positive_events'])} 条"
            f"（最强：{ctx['positive_events'][0]['title'][:40]}）")
    v.bull.conviction = min(1.0, 0.2 * (ctx["catalyst_score"] >= 0.5)
                            + 0.2 * (ctx["valuation_score"] >= 0.6)
                            + 0.2 * (ctx["tech_score"] >= 0.7)
                            + 0.2 * min(1.0, ctx["score"]))
    v.bull.claim = "三因子与事件面支持该信号" if v.bull.evidence else "多头论据不足"
    # 空头论据：风险点
    if ctx["negative_events"]:
        v.bear.evidence.append(
            f"近7天负面事件 {len(ctx['negative_events'])} 条"
            f"（最强：{ctx['negative_events'][0]['title'][:40]}）")
        v.bear.conviction = 0.5
    if ctx["caveats"]:
        v.bear.evidence.append(f"风控链警示：{'; '.join(ctx['caveats'][:2])[:120]}")
        v.bear.conviction = max(v.bear.conviction, 0.6)
    if ctx["implied_return"] <= 0 and ctx["triggered"]:
        v.bear.evidence.append(
            f"隐含收益 {ctx['implied_return']*100:+.1f}% —— 估值锚不支撑买入")
        v.bear.conviction = max(v.bear.conviction, 0.55)
    if not ctx["triggered"]:
        v.bear.evidence.append(f"三因子未达触发线（score={ctx['score']:.2f}）")
        v.bear.conviction = max(v.bear.conviction, 0.7)
    v.bear.claim = "存在未被定价的风险" if v.bear.evidence else "空头论据不足"
    # 裁决
    if ctx["negative_events"] and ctx["score"] < 0.75:
        v.verdict, v.confidence, v.lesson = "veto", 0.6, "负面事件在场且信号未达强阈值——宁可错过"
    elif ctx["caveats"]:
        v.verdict, v.confidence, v.lesson = "veto", 0.65, "风控链已亮牌，辩论不越过风控"
    elif ctx["triggered"] and ctx["implied_return"] <= 0:
        v.verdict, v.confidence, v.lesson = "veto", 0.6, "触发但估值无安全边际——信号与估值矛盾"
    elif (ctx["triggered"] and ctx["catalyst_score"] >= 0.5
          and not ctx["negative_events"] and ctx["implied_return"] > 0.05):
        v.verdict, v.confidence, v.lesson = "confirm", 0.6, "三因子齐备且无负面事件对冲"
    else:
        v.verdict, v.confidence, v.lesson = "abstain", 0.5, "多空论据不形成一边倒，维持观察"
    v.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return v


# ── 主入口 ───────────────────────────────────────────

def debate(trigger_result, event_library=None, llm_fn=None,
           log_dir: str | Path | None = None) -> DebateVerdict | None:
    """对一条 TriggerResult 跑多空辩论。任何失败返回 None（不阻断主链路）。

    LLM 可用 → 两段辩论（立论→裁决）；
    LLM 不可用 → 规则辩论（确定性降级，仍产出论据与裁决）。
    """
    try:
        ctx = build_context(trigger_result, event_library)
        bull_text = _llm_chat(_BULL_PROMPT.format(ctx=json.dumps(ctx, ensure_ascii=False)[:1500]), llm_fn)
        v = None
        if bull_text:
            bear_text = _llm_chat(_BEAR_PROMPT.format(ctx=json.dumps(ctx, ensure_ascii=False)[:1500]), llm_fn)
            if bear_text:
                bull = _parse_argument_json(bull_text, "bull")
                bear = _parse_argument_json(bear_text, "bear")
                trig = {k: ctx[k] for k in ("triggered", "score", "implied_return")}
                judge_text = _llm_chat(_JUDGE_PROMPT.format(
                    trigger=json.dumps(trig, ensure_ascii=False),
                    bull=json.dumps(asdict(bull), ensure_ascii=False)[:400],
                    bear=json.dumps(asdict(bear), ensure_ascii=False)[:400]), llm_fn)
                if judge_text:
                    verdict, conf, lesson = _parse_verdict_json(judge_text)
                    v = DebateVerdict(code=ctx["code"], verdict=verdict,
                                      confidence=conf, bull=bull, bear=bear,
                                      lesson=lesson, llm_used=True,
                                      timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"))
        if v is None:
            v = _rule_debate(ctx)
        _log_debate(v, log_dir)
        return v
    except Exception as e:  # noqa: BLE001
        logger.warning("[debate] 辩论失败（放行不阻断）: %s", e)
        return None


def _log_debate(v: DebateVerdict, log_dir: str | Path | None) -> None:
    """辩论日志落盘：shared_state/debate_log/<date>.json（lesson 沉淀检索用）"""
    try:
        base = Path(log_dir or "shared_state/debate_log")
        base.mkdir(parents=True, exist_ok=True)
        day = datetime.now().strftime("%Y-%m-%d")
        p = base / f"{day}.json"
        entries = []
        if p.exists():
            try:
                entries = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                entries = []
        entries.append(v.to_dict())
        p.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.warning("[debate] 日志落盘失败（忽略）: %s", e)


def debate_batch(trigger_results: list, event_library=None, llm_fn=None,
                 only_triggered: bool = True) -> dict[str, DebateVerdict]:
    """批量辩论：默认只辩已触发的信号（成本控制）。返回 {code: verdict}。"""
    out: dict[str, DebateVerdict] = {}
    for r in trigger_results:
        if only_triggered and not getattr(r, "triggered", False):
            continue
        v = debate(r, event_library=event_library, llm_fn=llm_fn)
        if v is not None:
            out[getattr(r, "code", "?")] = v
    return out
