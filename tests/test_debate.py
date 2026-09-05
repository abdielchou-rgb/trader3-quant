"""多空辩论（debate.py）行为测试 —— 全部离线（无网络/无 LLM key）

覆盖：
- 上下文构建（事件分组/缺库降级）
- 规则辩论裁决矩阵（veto 三分支 / confirm / abstain）
- LLM 两段辩论（注入 llm_fn，验证解析与裁决）
- LLM 输出畸形 JSON 的解析降级
- debate 失败静默放行（返回 None 不抛）
- debate_batch 只辩已触发
- 日志落盘与 lesson 沉淀
"""

import json

from trader3.v2.debate import (
    DebateVerdict,
    _llm_chat,
    _parse_argument_json,
    _parse_verdict_json,
    _rule_debate,
    build_context,
    debate,
    debate_batch,
)
from trader3.v2.trigger import TriggerResult


def _tr(code="600519", triggered=True, score=0.78, catalyst=0.55,
        valuation=0.72, tech=0.8, implied=0.12, caveats=None):
    return TriggerResult(
        code=code, triggered=triggered, score=score,
        catalyst_score=catalyst, valuation_score=valuation, tech_score=tech,
        reason="估值分位0.72,技术0.80", implied_return=implied,
        fair_value=258.0, current_price=230.0, caveats=caveats or [],
    )


class _FakeEvent:
    def __init__(self, title, direction, score=0.5, category="业绩"):
        self.title, self.direction = title, direction
        self.catalyst_score, self.category = score, category


class _FakeLib:
    def __init__(self, events):
        self._evs = events

    def get_recent(self, code, days=7, source=None):
        return self._evs


# ── 上下文构建 ─────────────────────────────────────

def test_build_context_groups_events():
    lib = _FakeEvent.__new__(_FakeEvent)
    lib.get_recent = lambda code, days=7, source=None: [
        _FakeEvent("业绩预增", "positive", 0.8),
        _FakeEvent("股东减持", "negative", 0.6),
        _FakeEvent("调研", "neutral", 0.2),
    ]
    ctx = build_context(_tr(), event_library=lib)
    assert len(ctx["positive_events"]) == 1
    assert len(ctx["negative_events"]) == 1
    assert ctx["code"] == "600519"
    assert ctx["implied_return"] == 0.12


def test_build_context_event_lib_none_degrades():
    # 库不可用时不抛异常，事件字段为空
    class _Broken:
        def get_recent(self, *a, **k):
            raise RuntimeError("db locked")

    ctx = build_context(_tr(), event_library=_Broken())
    assert ctx["positive_events"] == [] and ctx["negative_events"] == []


# ── 规则辩论裁决矩阵 ───────────────────────────────

def test_rule_debate_confirm_strong_signal():
    v = _rule_debate(build_context(_tr(), event_library=_FakeLib([])))
    assert v.verdict == "confirm"
    assert v.llm_used is False
    assert v.bull.evidence  # 有论据
    assert "lesson" in v.verdict or v.lesson  # 有沉淀


def test_rule_debate_veto_negative_event():
    lib = _FakeLib([_FakeEvent("立案调查", "negative", 0.9)])
    v = _rule_debate(build_context(_tr(score=0.7), event_library=lib))
    assert v.verdict == "veto"
    assert any("负面事件" in e for e in v.bear.evidence)


def test_rule_debate_veto_caveats():
    v = _rule_debate(build_context(_tr(caveats=["负面事件否决：立案调查"])))
    assert v.verdict == "veto"
    assert "风控" in v.lesson or "风控链" in json.dumps(v.to_dict(), ensure_ascii=False)


def test_rule_debate_veto_no_margin_of_safety():
    # 触发但隐含收益 <= 0：信号与估值矛盾 → veto
    v = _rule_debate(build_context(_tr(implied=-0.05)))
    assert v.verdict == "veto"


def test_rule_debate_abstain_weak():
    # 未触发 + 无事件 → abstain
    v = _rule_debate(build_context(_tr(triggered=False, score=0.4),
                                   event_library=_FakeLib([])))
    assert v.verdict == "abstain"


def test_rule_debate_bull_bear_conviction_bounds():
    v = _rule_debate(build_context(_tr(), event_library=_FakeLib([])))
    assert 0.0 <= v.bull.conviction <= 1.0
    assert 0.0 <= v.bear.conviction <= 1.0
    assert v.verdict in ("confirm", "veto", "abstain")


# ── LLM 两段辩论 ──────────────────────────────────

def test_llm_debate_full_flow(tmp_path, monkeypatch):
    calls = []

    def llm_fn(prompt):
        calls.append(prompt)
        if "多头研究员" in prompt:
            return '{"claim":"盈利上修","evidence":["Q2预增80%","订单饱满","估值分位0.72"],"conviction":0.8}'
        if "空头研究员" in prompt:
            return '{"claim":"涨价周期见顶","evidence":["大宗价格回落","库存上行"],"conviction":0.6}'
        return '{"verdict":"confirm","confidence":0.7,"lesson":"正面事件扎实"}'

    v = debate(_tr(), llm_fn=llm_fn, log_dir=tmp_path)
    assert v is not None and v.llm_used is True
    assert v.verdict == "confirm" and abs(v.confidence - 0.7) < 1e-9
    assert v.bull.claim == "盈利上修" and len(v.bull.evidence) == 3
    assert len(calls) == 3  # bull + bear + judge
    # 日志落盘
    log_files = list(tmp_path.glob("*.json"))
    assert log_files and json.loads(log_files[0].read_text(encoding="utf-8"))[0]["code"] == "600519"


def test_llm_debate_llm_fails_falls_to_rule(tmp_path):
    def llm_fn(prompt):
        raise ConnectionError("network down")

    v = debate(_tr(), event_library=_FakeLib([]), llm_fn=llm_fn, log_dir=tmp_path)
    assert v is not None and v.llm_used is False  # 规则降级
    assert v.verdict == "confirm"  # 同规则辩论结果


def test_parse_argument_malformed_json():
    arg = _parse_argument_json("not json at all", "bull")
    assert arg.claim == "" and arg.evidence == []
    arg2 = _parse_argument_json('{"claim":"ok","conviction":9.9}', "bull")
    assert arg2.claim == "ok" and arg2.conviction == 1.0  # clamp


def test_parse_verdict_invalid_value():
    v, c, lesson = _parse_verdict_json('{"verdict":"maybe","confidence":0.5}')
    assert v == "abstain"  # 非法值降级
    v2, _, _ = _parse_verdict_json('{"verdict":"veto"}')
    assert v2 == "veto"


# ── 静默放行 ───────────────────────────────────────

def test_debate_exception_returns_none(tmp_path):
    class _Boom:
        code = "600519"

        @property
        def triggered(self):
            raise RuntimeError("boom")

    v = debate(_Boom(), log_dir=tmp_path)
    assert v is None  # 不抛异常、不阻断


def test_llm_chat_none_fn_no_key(monkeypatch):
    # llm_fn=None 且无 OPENROUTER key → None（不发请求）
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert _llm_chat("prompt") is None


# ── 批量 ───────────────────────────────────────────

def test_debate_batch_only_triggered(tmp_path):
    rs = [_tr("600519", triggered=True), _tr("000858", triggered=False)]
    out = debate_batch(rs, event_library=_FakeLib([]), llm_fn=None, )
    # llm_fn=None → _llm_chat 走 LLMScorer 无 key → None → 规则辩论
    assert "600519" in out and "000858" not in out


def test_verdict_is_veto_property():
    v = DebateVerdict(code="x", verdict="veto")
    assert v.is_veto
    v2 = DebateVerdict(code="x", verdict="confirm")
    assert not v2.is_veto
