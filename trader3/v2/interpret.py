"""
3号交易员 v2.0 — 大模型解读层 (interpret)

把采集到的原始事件/文本交给大模型做语义解读，判断：
- 催化方向（正面/负面/中性）
- 催化强度（0~1）
- 事件类别（业绩/订单/政策/重组/资金/情绪）
- 关键实体（影响哪些公司/行业）

设计：
- LLM 解读优先（准确性），关键词规则 CatalystScorer 作兜底（无 LLM 时）
- 用 trader3 配置的 LLM 提供者（与 2hao agent_provider 对齐）或 Claude 自身
- 若 LLM 不可用 → 回退 CatalystScorer 规则评分（不阻断流程）
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from trader3.v2.events import CatalystScorer

logger = logging.getLogger("trader3.v2.interpret")

LLM_PROVIDER_ENV = "TRADER3_LLM_PROVIDER"   # 可用环境变量显式指定 provider

# provider → 所需环境配置（key/端点/模型各归其位，绝不串用）
_PROVIDER_CONF: Dict[str, dict] = {
    "deepseek": {
        "key_env": "DEEPSEEK_API_KEY",
        "base_env": "DEEPSEEK_BASE_URL",
        "base_default": "https://api.deepseek.com/v1",
        "model_env": "DEEPSEEK_MODEL",
        "model_default": "deepseek-chat",
    },
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "base_env": "OPENAI_BASE_URL",
        "base_default": "https://api.openai.com/v1",
        "model_env": "OPENAI_MODEL",
        "model_default": "gpt-4o-mini",
    },
}


@dataclass
class Interpretation:
    """单条文本的解读结果"""
    text: str = ""
    direction: str = "neutral"      # positive / negative / neutral
    catalyst_score: float = 0.5     # 0~1
    category: str = "other"         # 业绩/订单/政策/重组/资金/情绪/其他
    affected_codes: List[str] = field(default_factory=list)   # 受影响标的
    reasoning: str = ""             # 一句话解读
    model: str = "rules"            # llm / rules

    def to_dict(self) -> dict:
        return {
            "direction": self.direction, "catalyst_score": self.catalyst_score,
            "category": self.category, "affected_codes": self.affected_codes,
            "reasoning": self.reasoning, "model": self.model,
        }


# ── LLM 提供者（优先用 Claude 自身能力，退化到规则） ──

def resolve_provider(explicit: Optional[str] = None) -> str:
    """解析 LLM provider：显式参数 > 环境变量 > 按 key 自动选择。

    返回 "deepseek" / "openai"；无法确定时返回 ""。
    """
    p = (explicit or os.environ.get(LLM_PROVIDER_ENV, "") or "").strip().lower()
    if p in _PROVIDER_CONF:
        return p
    if p:
        return ""   # 显式指定了未知/不可用 provider
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return ""


def _extract_first_json(text: str) -> Optional[dict]:
    """从首个 { 起做 brace-counting，提取第一个完整 JSON 对象。

    修复原 re.search(r"\\{.*\\}", DOTALL) 贪婪匹配会把文本中
    多个 JSON 块/杂讯拼在一起的问题；字符串内的花括号不计数。
    """
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start:i + 1])
                            return obj if isinstance(obj, dict) else None
                        except Exception:
                            break   # 该起点不是合法 JSON，尝试下一个 {
        start = text.find("{", start + 1)
    return None


def _try_llm(text: str, provider: Optional[str] = None,
             timeout: float = 15.0) -> Optional[dict]:
    """
    尝试用 LLM 解读。返回 {direction, score, category, codes, reasoning} 或 None。

    实现方式：
    1. 显式 provider（deepseek/openai）→ 校验对应 key 存在，缺 key 直接降级
       （绝不把 OpenAI key 发往 deepseek 端点等串用行为）
    2. 未显式指定时按 DEEPSEEK_API_KEY / OPENAI_API_KEY 自动选择
    3. 都不可用返回 None → 上层回退规则评分
    """
    prov = resolve_provider(provider)
    conf = _PROVIDER_CONF.get(prov)
    api_key = os.environ.get(conf["key_env"], "") if conf else ""
    if not (conf and api_key):
        reason = (f"provider='{provider}' 未知或不可用"
                  if provider and not conf else
                  f"provider='{prov}' 未配置 {conf['key_env']}" if conf
                  else "未配置任何 LLM API key")
        logger.warning("[interpret] %s，不发请求，降级规则评分", reason)
        return None
    try:
        import requests
        base = os.environ.get(conf["base_env"], conf["base_default"]).rstrip("/")
        model = os.environ.get(conf["model_env"], conf["model_default"])
        prompt = (
            "你是投研助理。分析这条新闻/事件，输出 JSON："
            "{\"direction\":\"positive|negative|neutral\",\"score\":0.0~1.0,"
            "\"category\":\"业绩|订单|政策|重组|资金|情绪|其他\","
            "\"codes\":[\"受影响的股票代码\"],\"reasoning\":\"一句话解读\"}\n\n"
            f"事件：{text[:600]}"
        )
        r = requests.post(f"{base}/chat/completions", headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 300,
        }, timeout=timeout)
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return _extract_first_json(content)
    except Exception as e:
        logger.debug("[interpret] LLM API 解读失败: %s", str(e)[:80])
    return None


class Interpreter:
    """事件解读器：LLM 优先，规则兜底"""

    def __init__(self, use_rules_fallback: bool = True,
                 llm_provider: Optional[str] = None):
        self.scorer = CatalystScorer()
        self.use_rules_fallback = use_rules_fallback
        self.llm_provider = llm_provider   # None = 按环境自动选择

    def interpret(self, text: str) -> Interpretation:
        """解读单条文本"""
        if not text:
            return Interpretation(text=text)

        llm = _try_llm(text, provider=self.llm_provider)
        if llm:
            try:
                direction = llm.get("direction", "neutral")
                score = float(llm.get("score", 0.5))
                category = llm.get("category", "其他")
                codes = llm.get("codes") or []
                reasoning = llm.get("reasoning", "")
                return Interpretation(
                    text=text, direction=direction,
                    catalyst_score=max(0.0, min(score, 1.0)),
                    category=category, affected_codes=codes,
                    reasoning=reasoning, model="llm",
                )
            except Exception as e:
                logger.debug("[interpret] LLM 结果解析失败: %s", str(e)[:60])

        # 规则兜底
        if self.use_rules_fallback:
            r = self.scorer.score(text)
            return Interpretation(
                text=text, direction=r["direction"],
                catalyst_score=r["score"], category=r["category"],
                reasoning=f"规则匹配: {', '.join(r['keywords_hit'][:3])}",
                model="rules",
            )
        return Interpretation(text=text)


def get_interpreter() -> Interpreter:
    return Interpreter()