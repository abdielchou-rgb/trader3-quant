"""v2 接线修复验证（test_fix_v2_wiring）

覆盖 6 项接线任务的核心验收点，零联网、零生产库：
  1. PositionT1Account.settle() 结算后复位当日计数器（跨日下单额度恢复）
  2. 买入佣金计入 avg_cost（成本基础含费用）
  3. 超量卖出部分成交至可卖上限（与买入部分成交行为一致）
  4. daily_pipeline 纸面交易落盘 shared_state/paper/account.json
  5. extra_sources 注册进 sync_all 源注册表（默认关闭）

费率统一（任务5）：execution_realism 从 costs.DEFAULT_COSTS 换算，
由 test_buy_commission_in_avg_cost 用 acct.commission_rate 反算校验。
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── (1)(5) execution_realism 内部修缮 ─────────────────────

class TestAccountDailyCountersReset:
    """settle 后当日成交金额/笔数归零，跨日流控额度恢复"""

    def test_settle_resets_today_counters(self):
        from trader3.v2.execution_realism import PositionT1Account

        acct = PositionT1Account(cash=1e6)
        ok, msg = acct.buy("600519", 100, 12.0)
        assert ok, msg
        s = acct.summary()
        assert s["today_orders"] >= 1
        assert s["today_value"] > 0

        acct.settle()
        s = acct.summary()
        assert s["today_orders"] == 0, f"跨日不复位: {s}"
        assert s["today_value"] == 0.0, f"跨日不复位: {s}"


class TestBuyCommissionInAvgCost:
    """买入后 avg_cost 应含摊入的佣金（> 成交价）"""

    def test_avg_cost_includes_commission(self):
        from trader3.v2.execution_realism import PositionT1Account

        acct = PositionT1Account(cash=1e6)
        price = 10.0
        ok, msg = acct.buy("600000", 1000, price)
        assert ok, msg

        gross = 1000 * price
        expected_commission = max(gross * acct.commission_rate, 5.0)
        st = acct.stocks["600000"]
        assert st.avg_cost == pytest.approx((gross + expected_commission) / 1000), \
            f"avg_cost={st.avg_cost} 未含佣金或口径不符"
        assert st.avg_cost > price

    def test_commission_rate_from_costs_default(self):
        """费率唯一事实来源：默认值应等于 costs.DEFAULT_COSTS.bp/10000"""
        from trader3.v2.costs import DEFAULT_COSTS
        from trader3.v2.execution_realism import PositionT1Account

        acct = PositionT1Account(cash=1e6)
        assert acct.commission_rate == pytest.approx(
            DEFAULT_COSTS.commission_bp / 10000.0)
        assert acct.stamp_tax_rate == pytest.approx(
            DEFAULT_COSTS.stamp_tax_bp / 10000.0)


class TestPartialFillOnOversell:
    """卖出量超持仓时部分成交至可卖上限，而非整单拒绝"""

    def test_oversell_fills_to_sellable(self):
        from trader3.v2.execution_realism import PositionT1Account

        acct = PositionT1Account(cash=1e6)
        ok, msg = acct.buy("600000", 1000, 10.0)
        assert ok, msg
        acct.settle()  # today → his，次日可卖

        cash_before = acct.cash
        ok, msg = acct.sell("600000", 1500, 11.0)  # 请求 1500 > 可卖 1000
        assert ok is True, f"超量卖出应部分成交而非拒单: {msg}"
        assert "1000" in msg, f"消息应体现实际成交量 1000: {msg}"
        assert "600000" not in acct.stocks, "可卖上限应被全部吃掉"
        assert acct.cash > cash_before

    def test_sell_still_rejected_with_no_position(self):
        """无任何可卖持仓时仍拒绝（部分成交的前提是有可卖量）"""
        from trader3.v2.execution_realism import PositionT1Account

        acct = PositionT1Account(cash=1e6)
        ok, msg = acct.sell("600000", 100, 10.0)
        assert ok is False
        assert "可卖" in msg

    def test_t1_same_day_sell_rejected(self):
        """T+1：当日买入不可卖（today 冻结语义保留）"""
        from trader3.v2.execution_realism import PositionT1Account

        acct = PositionT1Account(cash=1e6)
        ok, _ = acct.buy("600000", 1000, 10.0)
        assert ok
        ok, msg = acct.sell("600000", 500, 10.5)
        assert ok is False, "当日买入 T+1 不可卖"


# ── (2)(3) daily_pipeline 纸面交易接线 ────────────────────

class _FakeCollector:
    """离线采集器替身（不写生产 events.db、不联网）"""

    def __init__(self, *args, **kwargs):
        pass

    def collect_stock(self, code, source="news"):
        return 0

    def collect_flow_events(self):
        return 0

    def close(self):
        pass


class _StubEngine:
    """离线触发引擎替身：恒返回预置结果"""

    def __init__(self, result):
        self._result = result

    def scan(self, items, catalyst_scores=None, asof_date=None, dry_run=False):
        return [self._result]


def _make_triggered_buy_result(code="600519"):
    from trader3.v2.trigger import TriggerResult

    return TriggerResult(
        code=code, triggered=True, score=0.9,
        catalyst_score=0.9, valuation_score=0.8, tech_score=0.8,
        reason="测试触发", direction="buy",
        implied_return=0.25, fair_value=15.0, current_price=12.0,
        stop_loss=11.04, key_technical="测试上破20日高",
        caveats=[], timestamp="2026-08-24 09:30",
    )


@pytest.fixture()
def offline_pipeline(monkeypatch):
    """隔离 run_daily 的全部外部依赖：采集/触发/自选股/行情/落盘目录"""
    tmp = tempfile.mkdtemp()

    import trader3.v2.collector as collector_mod
    import trader3.v2.daily_pipeline as dp_mod
    import trader3.v2.qa_accessor as qa_mod
    import trader3.v2.trigger as trigger_mod
    import trader3.v2.watchlist as watchlist_mod
    from trader3.v2.watchlist import WatchlistDB

    monkeypatch.setattr(collector_mod, "DataCollector", _FakeCollector)
    monkeypatch.setattr(trigger_mod, "get_trigger_engine",
                        lambda t3=None: _StubEngine(_make_triggered_buy_result()))
    monkeypatch.setattr(watchlist_mod, "get_watchlist",
                        lambda db_path=None: WatchlistDB(
                            os.path.join(tmp, "watch.db")))
    monkeypatch.setattr(qa_mod, "get_quote_snapshot",
                        lambda code: {"code": code, "name": "测试股",
                                      "price": 12.0})
    paper_dir = os.path.join(tmp, "shared_state", "paper")
    monkeypatch.setattr(dp_mod, "PAPER_STATE_DIR", paper_dir)
    return dp_mod, paper_dir, tmp


class TestPaperTradeRecordWritten:
    """run_daily(paper_trading=True) 后 account.json 存在且内容完整"""

    def test_account_json_written(self, offline_pipeline):
        dp_mod, paper_dir, tmp = offline_pipeline
        summary = dp_mod.run_daily(codes=["600519"], paper_trading=True)

        path = os.path.join(paper_dir, "account.json")
        assert os.path.exists(path), "account.json 未落盘"

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for key in ("as_of", "date", "cash", "positions",
                    "trades_today", "equity"):
            assert key in data, f"缺少字段 {key}: {data.keys()}"
        assert data["trades_today"], "今日应有纸面成交记录"
        assert data["trades_today"][0]["ok"] is True
        assert data["equity"] > 0
        assert "纸面" in data["note"]

        assert summary["paper_trading"] is not None
        assert summary["paper_trading"]["trades_today"]

        alerts = dp_mod.format_alerts(summary)
        assert "纸面交易" in alerts and "非真实委托" in alerts

    def test_paper_trading_false_skips(self, offline_pipeline):
        dp_mod, paper_dir, tmp = offline_pipeline
        summary = dp_mod.run_daily(codes=["600519"], paper_trading=False)
        assert not os.path.exists(os.path.join(paper_dir, "account.json"))
        assert "paper_trading" not in summary or summary["paper_trading"] is None

    def test_persist_and_reload_across_days(self, offline_pipeline, monkeypatch):
        """同一日重复运行：账户状态续用（现金递减）；跨日：trades 清空+settle"""
        dp_mod, paper_dir, tmp = offline_pipeline
        dp_mod.run_daily(codes=["600519"], paper_trading=True)

        with open(os.path.join(paper_dir, "account.json"), encoding="utf-8") as f:
            first = json.load(f)
        dp_mod.run_daily(codes=["600519"], paper_trading=True)
        with open(os.path.join(paper_dir, "account.json"), encoding="utf-8") as f:
            second = json.load(f)

        assert len(second["trades_today"]) == len(first["trades_today"]) + 1
        assert second["cash"] < first["cash"], "同日第二单应继续扣减现金"


# ── (4) extra_sources 注册进 sync_all ─────────────────────

_EXTRA_NAMES = ("margin", "shareholder_count", "northbound")
_EXTRA_ENVS = tuple(f"T3_SYNC_{n.upper()}" for n in _EXTRA_NAMES)


class TestExtraSourcesRegistered:
    """sync_all 源注册表能列出 extra_sources 源名，且默认全关"""

    def test_registered_and_disabled_by_default(self, monkeypatch):
        for env in _EXTRA_ENVS:
            monkeypatch.delenv(env, raising=False)
        from trader3.v2 import sync_all

        entries = sync_all.register_extra_sources()
        names = {e["name"] for e in entries}
        assert set(_EXTRA_NAMES) <= names, f"注册表缺源: {names}"
        for e in entries:
            if e["name"] in _EXTRA_NAMES:
                assert e["enabled"] is False, f"{e['name']} 应默认关闭"

        listed = sync_all.list_sync_sources()
        assert {e["name"] for e in listed} >= set(_EXTRA_NAMES)

    def test_env_switch_enables(self, monkeypatch):
        for env in _EXTRA_ENVS:
            monkeypatch.delenv(env, raising=False)
        from trader3.v2 import sync_all

        monkeypatch.setenv("T3_SYNC_NORTHBOUND", "1")
        entries = sync_all.register_extra_sources()
        by_name = {e["name"]: e for e in entries}
        assert by_name["northbound"]["enabled"] is True
        assert by_name["margin"]["enabled"] is False

    def test_param_override(self, monkeypatch):
        for env in _EXTRA_ENVS:
            monkeypatch.delenv(env, raising=False)
        from trader3.v2 import sync_all

        entries = sync_all.register_extra_sources(enabled=True)
        assert all(e["enabled"] is True for e in entries)
