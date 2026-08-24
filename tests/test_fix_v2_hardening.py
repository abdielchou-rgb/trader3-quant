"""
v2 加固包修复验证（TDD）——零联网测试

覆盖审计确认项：
(1) collector 限速+退避+UA 轮换
(2) interpret LLM provider key 守卫 + brace-counting JSON 提取
(3) search.py DDG query urlencode
(4) sources.py 新浪行情短行不越界
(5) announcement_calendar 无 import 副作用（不 basicConfig）
(6) 纸面账户支持卖出信号（整单市价卖出 + no_position 跳过记录）
"""

from __future__ import annotations

import json
import logging
import types
from datetime import datetime

# ──────────────────────────────────────────────
# (1) collector 限速 + 指数退避
# ──────────────────────────────────────────────

class TestCollectorBackoff:
    def test_collector_backoff_after_failures(self, monkeypatch):
        from trader3.v2 import collector as col_mod

        sleeps: list = []

        class _FakeTime:
            def sleep(self, s):
                sleeps.append(float(s))

        monkeypatch.setattr(col_mod, "time", _FakeTime())

        class _BoomSession:
            def __init__(self):
                self.calls = 0

            def get(self, *a, **k):
                self.calls += 1
                raise RuntimeError("network down")

        sess = _BoomSession()
        c = col_mod.DataCollector(skip_akshare=True, use_llm_interpret=False)
        # 库不可用时 _collect_news 直接返回，需注入假库以走到网络分支
        if not c._library:
            c._library = types.SimpleNamespace()
        monkeypatch.setattr(c, "_http", lambda: sess)
        monkeypatch.setattr(c, "_save_event", lambda *a, **k: False)

        # 前 3 次：正常节流间隔（1.5±0.5s），无退避
        for _ in range(3):
            c._collect_news_direct("600519")
        pacing_only = [s for s in sleeps if s < 5]
        backoffs = [s for s in sleeps if s >= 5]
        assert len(pacing_only) >= 1, "每源请求前应有 SOURCE_DELAY_SECONDS 节流 sleep"
        assert all(abs(s - col_mod.SOURCE_DELAY_SECONDS) <= 0.51 for s in pacing_only)
        assert c._fail_counts.get("sina_news") == 3
        assert backoffs == [], "连续失败未达阈值不应触发退避"

        # 第 4 轮：连续失败≥3 → 指数退避 2^3=8s 并跳过本轮（不再发请求）
        calls_before = sess.calls
        got = c._collect_news_direct("600519")
        assert any(s >= 8 for s in sleeps), f"应记录 8s 退避 sleep, sleeps={sleeps}"
        assert sess.calls == calls_before, "退避轮次应跳过该源不发请求"
        assert got == 0
        assert c._fail_counts.get("sina_news") == 3  # 跳过轮不计新失败

    def test_collector_ua_rotation(self, monkeypatch):
        from trader3.v2 import collector as col_mod

        c = col_mod.DataCollector(skip_akshare=True, use_llm_interpret=False)
        uas = {c._next_ua() for _ in range(len(col_mod.USER_AGENTS))}
        assert len(uas) > 1, "UA 应为多元素列表轮换"
        assert set(col_mod.USER_AGENTS) == uas


# ──────────────────────────────────────────────
# (2) interpret provider 守卫 + JSON 提取
# ──────────────────────────────────────────────

class TestInterpretProviderGuard:
    def test_interpret_provider_key_guard(self, monkeypatch, caplog):
        from trader3.v2 import interpret as interp_mod

        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        posted: list = []

        def fake_post(url, **kw):
            posted.append(url)
            raise AssertionError(f"不应发出请求: {url}")

        monkeypatch.setattr("requests.post", fake_post)

        with caplog.at_level(logging.WARNING, logger="trader3.v2.interpret"):
            out = interp_mod._try_llm("利好消息", provider="deepseek")

        assert out is None, "deepseek 缺 key 应直接降级而非发错端点"
        assert posted == []
        assert any("DEEPSEEK_API_KEY" in r.message for r in caplog.records), \
            f"应有 warning 日志说明缺 key: {[r.message for r in caplog.records]}"

    def test_openai_key_goes_to_openai_endpoint(self, monkeypatch):
        """仅设 OPENAI_API_KEY 时不得把 key 发往 deepseek 端点"""
        from trader3.v2 import interpret as interp_mod

        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
        urls: list = []

        class _Resp:
            def json(self):
                payload = {"direction": "positive", "score": 0.9,
                           "category": "订单", "codes": ["600519"], "reasoning": "大单"}
                return {"choices": [{"message": {"content": json.dumps(payload,
                                                                      ensure_ascii=False)}}]}

        def fake_post(url, **kw):
            urls.append(url)
            return _Resp()

        monkeypatch.setattr("requests.post", fake_post)
        out = interp_mod._try_llm("签大额订单", provider=None)
        assert out and out["direction"] == "positive"
        assert urls and "api.openai.com" in urls[0], f"应走 openai 端点: {urls}"
        assert not any("deepseek" in u for u in urls)

    def test_extract_first_json_brace_counting(self):
        from trader3.v2.interpret import _extract_first_json

        text = ('前置说明 {"direction":"positive","score":0.9,'
                '"note":"含 } 花括号"} 尾部第二个 {"bad":1}')
        obj = _extract_first_json(text)
        assert obj is not None
        assert obj["direction"] == "positive"
        assert obj["note"] == "含 } 花括号"
        assert "bad" not in obj, "应取第一个完整 JSON 对象"
        assert _extract_first_json("没有任何花括号") is None
        assert _extract_first_json("{broken json") is None


# ──────────────────────────────────────────────
# (3) search.py DDG 编码
# ──────────────────────────────────────────────

class TestDDGEncoding:
    def test_ddg_query_encoded(self, monkeypatch):
        from trader3.v2.search import ddg_websearch

        captured: dict = {}

        class _Resp:
            status_code = 200
            text = '<a class="result__a" href="https://example.com/a">测试<b>标题</b>一</a>'

        def fake_get(url, **kw):
            captured["url"] = url
            return _Resp()

        monkeypatch.setattr("requests.get", fake_get)
        out = ddg_websearch("茅台 & 五粮液 库存", limit=5)

        url = captured["url"]
        assert url.startswith("https://html.duckduckgo.com/html/?")
        assert "%26" in url, "& 应被编码为 %26"
        assert "+" in url or "%20" in url, "空格应被编码"
        assert "茅台" not in url.split("?")[1], "中文应经 URL 编码"
        assert out == [("测试标题一", "https://example.com/a")]

    def test_ddg_failure_returns_empty_list(self, monkeypatch):
        from trader3.v2.search import ddg_websearch

        def boom(*a, **k):
            raise RuntimeError("conn reset")

        monkeypatch.setattr("requests.get", boom)
        assert ddg_websearch("任意 query & 带特殊字符") == []


# ──────────────────────────────────────────────
# (4) sources.py 新浪行情短行守卫
# ──────────────────────────────────────────────

class TestSinaShortLine:
    def test_sina_short_line_no_crash(self, monkeypatch):
        from trader3.v2.sources import SinaQuoteSource

        class _Resp:
            text = 'var hq_str_sh600000="测试股,3.50,3.40,3.55";'

        monkeypatch.setattr("requests.get", lambda url, **k: _Resp())
        src = SinaQuoteSource()
        items = src.fetch(["sh600000"])
        assert len(items) == 1
        assert "测试股" in items[0].title
        assert items[0].ts  # 时间戳字段仍填充

    def test_sina_full_line_keeps_high_low_volume(self, monkeypatch):
        from trader3.v2.sources import SinaQuoteSource

        full = "浦发银行,10.00,9.90,9.95,10.05,9.85,88888888,6666,123456789,9999"
        empty_line = 'var hq_str_sh600000="' + full + '";'

        class _Resp:
            text = empty_line

        monkeypatch.setattr("requests.get", lambda url, **k: _Resp())
        items = SinaQuoteSource().fetch(["sh600000"])
        assert len(items) == 1
        assert "高10.05" in items[0].content
        assert "低9.85" in items[0].content
        assert "量123456789" in items[0].content


# ──────────────────────────────────────────────
# (5) announcement_calendar 无 import 副作用
# ──────────────────────────────────────────────

class TestCalendarNoBasicConfig:
    def test_calendar_no_basicconfig(self, monkeypatch):
        logging.getLogger().setLevel(logging.WARNING)

        import importlib

        from trader3.v2 import announcement_calendar as cal_mod

        called: dict = {}

        def spy_basic_config(**kwargs):
            called.update(kwargs)

        monkeypatch.setattr(logging, "basicConfig", spy_basic_config)
        cal = importlib.reload(cal_mod)

        assert called == {}, \
            f"库模块顶层禁止 logging.basicConfig, got {called}"
        assert logging.getLogger().level == logging.WARNING, \
            "import 不应改动 root logger level"
        assert cal.logger.name == "trader3.v2.announcement_calendar"


# ──────────────────────────────────────────────
# (6) 纸面账户卖出信号
# ──────────────────────────────────────────────

class TestPaperSellFlow:
    def test_paper_sell_flow(self, monkeypatch, tmp_path):
        from trader3.shared_state import SharedState
        from trader3.v2 import daily_pipeline as dp
        from trader3.v2 import qa_accessor
        from trader3.v2.trigger import TriggerResult

        monkeypatch.setattr(dp, "PAPER_STATE_DIR", str(tmp_path))
        monkeypatch.setattr(qa_accessor, "get_quote_snapshot",
                            lambda code: {"code": code, "name": "模拟股", "price": 12.0})

        today = datetime.now().strftime("%Y-%m-%d")
        preset = {
            "as_of": today, "date": today,
            "cash": 100000.0,
            "positions": [{"code": "600519", "his": 200.0, "today": 0.0,
                            "avg_cost": 10.0}],
            "trades_today": [],
        }
        SharedState(str(tmp_path)).write_json("account", preset)

        sell_sig = TriggerResult(code="600519", triggered=True, score=0.7,
                                 reason="破位止损", direction="sell",
                                 current_price=12.0)
        nopos_sig = TriggerResult(code="000858", triggered=True,
                                  reason="无持仓标的的卖出信号", direction="sell")
        state = dp.run_paper_trades([sell_sig, nopos_sig])

        sells = [t for t in state["trades_today"] if t.get("side") == "sell"]
        filled = [t for t in sells if t.get("code") == "600519"]
        assert filled, f"应有 side=sell 的成交记录: {state['trades_today']}"
        assert filled[0]["volume"] == 200
        assert filled[0]["ok"] is True

        skipped = [t for t in state["trades_today"]
                   if t.get("skipped_trade") == "no_position"]
        assert len(skipped) == 1 and skipped[0]["code"] == "000858"

        assert state["cash"] > 100000.0, "卖出后现金应增加（扣费后净入）"
        pos_codes = {p["code"] for p in state["positions"]}
        assert "600519" not in pos_codes, "整单卖出后仓位应清零"

        reloaded = SharedState(str(tmp_path)).read_json("account")
        assert any(t.get("side") == "sell" for t in reloaded["trades_today"]), \
            "account.json 应持久化卖出流水"
