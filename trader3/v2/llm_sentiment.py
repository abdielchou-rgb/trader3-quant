"""
LLM 情绪打分（OpenRouter 免费模型）。

设计：
- OpenRouter /chat/completions，默认轮询一组 :free 模型，失败自动降级到下一个
- 批量打分：一次请求打 N 条标题，严格 JSON 输出，鲁棒解析（剥码栏/正则截取）
- 三层降级：LLM → 缓存命中 → 规则词典（sentiment.score_text），永不抛出
- 磁盘缓存 sha1(title)→score，避免重复计费/限流
- 免费档限流保护：请求间最小间隔 + 429/5xx 指数退避重试

环境变量：
    OPENROUTER_API_KEY   必需；缺失时直接走词典降级
    TRADER3_LLM_MODELS   可选；逗号分隔覆盖默认模型列表

用法：
    from trader3.v2.llm_sentiment import LLMScorer
    scorer = LLMScorer()                      # 无 key 时自动降级为词典
    texts = ["业绩超预期 中标大单", "立案调查 预亏"]
    scores = scorer.score_batch(texts)        # [-3, 3] 与词典同刻度
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable

logger = logging.getLogger("trader3.v2.llm_sentiment")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_FREE_MODELS = [
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen3-8b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

SYSTEM_PROMPT = (
    "你是A股新闻情绪标注器。对每条新闻标题输出 -3..3 的情绪分："
    "强利空-3..-2，利空-2..-1，中性-1..1，利好1..2，强利好2..3。"
    '只输出JSON数组，格式：[{"i":0,"s":1.5},...]，不要解释。'
)

_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_OBJ_RE = re.compile(r"\{[^{}]*\"i\"\s*:\s*(\d+)[^{}]*\}")


class LLMScorer:
    """OpenRouter 批量情绪打分器，带缓存与词典降级。"""

    def __init__(
        self,
        api_key: str | None = None,
        models: list[str] | None = None,
        cache_dir: str | None = None,
        min_interval_s: float = 2.0,
        max_retries: int = 2,
        timeout_s: float = 30.0,
        batch_size: int = 20,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        env_models = os.environ.get("TRADER3_LLM_MODELS")
        self.models = models or (
            [m.strip() for m in env_models.split(",") if m.strip()]
            if env_models else list(DEFAULT_FREE_MODELS)
        )
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.batch_size = batch_size
        self._last_call_ts = 0.0
        self.model_used: str | None = None
        self.fallback_count = 0

        import tempfile
        self.cache_dir = cache_dir or os.path.join(
            tempfile.gettempdir(), "trader3_llm_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    # ── 缓存 ──────────────────────────────────────────
    def _cache_path(self, text: str) -> str:
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.json")

    def _cache_get(self, text: str) -> float | None:
        p = self._cache_path(text)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return float(json.load(f)["s"])
            except Exception:
                pass
        return None

    def _cache_put(self, text: str, score: float) -> None:
        try:
            with open(self._cache_path(text), "w", encoding="utf-8") as f:
                json.dump({"s": score}, f)
        except Exception:
            pass

    # ── HTTP ──────────────────────────────────────────
    def _throttle(self) -> None:
        wait = self.min_interval_s - (time.time() - self._last_call_ts)
        if wait > 0:
            time.sleep(wait)

    def _chat(self, model: str, user_prompt: str) -> str | None:
        import httpx
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 2048,
        }
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self._last_call_ts = time.time()
            try:
                resp = httpx.post(OPENROUTER_URL, headers=headers, json=payload,
                                  timeout=self.timeout_s)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                if resp.status_code == 429 or resp.status_code >= 500:
                    logger.warning("[llm] %s HTTP %d, retry %.0fs",
                                   model, resp.status_code, delay)
                    time.sleep(delay)
                    delay *= 2
                    continue
                logger.warning("[llm] %s HTTP %d: %s", model, resp.status_code,
                               resp.text[:200])
                return None
            except Exception as e:  # noqa: BLE001
                logger.warning("[llm] %s error: %s", model, e)
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
        return None

    @staticmethod
    def parse_scores(content: str | None, n: int) -> list[float] | None:
        """解析模型输出为长度 n 的分数列表；结构不符返回 None。"""
        if not content:
            return None
        text = re.sub(r"```(?:json)?|```", "", content).strip()
        out: dict[int, float] = {}
        # 首选：整体 JSON 数组
        m = _ARRAY_RE.search(text)
        if m:
            try:
                arr = json.loads(m.group(0))
                if isinstance(arr, list):
                    for obj in arr:
                        if isinstance(obj, dict) and "i" in obj and "s" in obj:
                            idx, sc = int(obj["i"]), float(obj["s"])
                            if 0 <= idx < n:
                                out[idx] = max(-3.0, min(3.0, sc))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        if len(out) < n:
            # 兜底：逐对象正则提取（容忍模型输出前后杂讯）
            for om in re.finditer(r'\{\s*"i"\s*:\s*(\d+)\s*,\s*"s"\s*:\s*'
                                  r'(-?\d+(?:\.\d+)?)\s*\}', text):
                idx, sc = int(om.group(1)), float(om.group(2))
                if 0 <= idx < n:
                    out.setdefault(idx, max(-3.0, min(3.0, sc)))
        if len(out) < n:
            return None
        return [out[i] for i in range(n)]

    def _call_models(self, texts: list[str]) -> tuple[list[float] | None, str]:
        """按模型顺序尝试，直到一个成功解析。"""
        user_prompt = "\n".join(f'{{"i":{i},"t":"{t[:120]}"}}'
                                for i, t in enumerate(texts))
        for model in self.models:
            content = self._chat(model, user_prompt)
            scores = self.parse_scores(content, len(texts))
            if scores is not None:
                return scores, model
            logger.info("[llm] model %s 解析失败/无内容，换下一个", model)
        return None, ""

    # ── 公共入口 ──────────────────────────────────────
    def score_batch(self, texts: list[str]) -> list[float]:
        """
        打分入口。三层降级：缓存 → LLM → 词典。
        返回 [-3, 3] 刻度分数，永不抛出。
        """
        n = len(texts)
        results = [None] * n
        missing_idx: list[int] = []

        for i, t in enumerate(texts):
            cached = self._cache_get(t)
            if cached is not None:
                results[i] = cached
            else:
                missing_idx.append(i)

        if missing_idx and self.api_key:
            try:
                sub_texts = [texts[i] for i in missing_idx]
                got: list[float] = []
                for start in range(0, len(sub_texts), self.batch_size):
                    chunk = sub_texts[start:start + self.batch_size]
                    scores, model = self._call_models(chunk)
                    if scores is None:
                        break
                    self.model_used = model
                    got.extend(scores)
                if len(got) == len(sub_texts):
                    for i, sc in zip(missing_idx, got, strict=True):
                        results[i] = sc
                        self._cache_put(texts[i], sc)
            except Exception as e:  # noqa: BLE001
                logger.warning("[llm] batch failed: %s", e)

        # 词典兜底
        from trader3.v2.sentiment import score_text
        for i in range(n):
            if results[i] is None:
                results[i] = score_text(texts[i]).score
                self.fallback_count += 1
        return [float(x) for x in results]

    def score_items(self, items: list, title_weight: float = 2.0) -> float:
        """SourceItem 列表均分（标题权重>正文），与 sentiment.score_items 同口径。"""
        if not items:
            return 0.0
        texts = [getattr(it, "title", "") or "" for it in items]
        scores = self.score_batch(texts)
        return sum(scores) / len(scores)


def make_llm_or_lexicon_scorer() -> Callable:
    """工厂：有 key 返回 LLM 打分器，否则返回纯词典函数（接口一致）。"""
    scorer = LLMScorer()
    if scorer.api_key:
        return scorer.score_items
    from trader3.v2.sentiment import score_items as lexicon_items
    return lexicon_items
