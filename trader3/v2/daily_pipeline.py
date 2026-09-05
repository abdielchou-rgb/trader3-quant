"""
3号交易员 v2.0 — 每日扫描管线（daily_pipeline）

串联：数据采集 → 事件入库 → 三因子扫描 → 状态迁移 → 纸面执行 → 提醒输出

闭环链路：触发（trigger）→ 风控否决（risk_rules/NegativeEventRule）
        → 纸面下单（execution_realism 账户 + costs 费用模型）
纸面成交与每日净值落盘 shared_state/paper/account.json（SharedState 原子写）。

用法：
    python -m trader3.v2.daily_pipeline                         # 跑当日全量
    python -m trader3.v2.daily_pipeline --codes 600519,000858   # 指定自选股
    python -m trader3.v2.daily_pipeline --collect-only          # 只采集不触发
    python -m trader3.v2.daily_pipeline --no-paper              # 跳过纸面交易
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime

from trader3.v2.costs import DEFAULT_COSTS

logger = logging.getLogger("trader3.v2.daily")

# ── 纸面交易配置（模块顶部常量，可调） ──────────────────
PAPER_TRADE_PCT = 0.10            # 每单资金 = 账户总资产 × 10%
PAPER_INITIAL_CASH = 1_000_000.0  # 初始虚拟现金（首次运行建账）
PAPER_LOT_SIZE = 100              # A股一手股数，按整手下单
PAPER_STATE_DIR = os.path.join(   # 账户状态目录：shared_state/paper/
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "shared_state", "paper",
)
PAPER_NOTE = "纸面交易，非真实委托"


def run_daily(
    codes: list[str] = None,
    collect_only: bool = False,
    collect_flow: bool = True,
    dry_run: bool = False,
    paper_trading: bool = True,
    overlay=None,
    debate_review: bool = False,
    debate_llm_fn=None,
) -> dict:
    """当日全量扫描。返回 {collected, results, alerts, paper_trading}。

    dry_run=True：采集照常，扫描只报告，不迁移状态不写库（含纸面账户）。
    paper_trading=True：对触发买入信号执行纸面下单并写 account.json；
        全程为模拟撮合——纸面交易，非真实委托。
    overlay：可选 trader3.v2.risk_overlay.OverlayResult——组合级风险覆盖层，
        调节每单资金比例（size_multiplier）或暂停新开仓（halt_new_buys）。
        由调用方经 compute_risk_overlay 预先计算；缺省 None 行为不变。
    debate_review：P2-6 多空辩论复核（默认关）。开启后对已触发信号逐只
        组织看多/看空辩论（LLM 缺省走规则版），verdict=veto 的信号
        从 triggered 中剔除并写入 caveats（"辩论否决：..."）；
        任何辩论失败静默放行，不阻断主链路。结果落 shared_state/debate_log/。
    """
    from trader3.v2.collector import DataCollector
    from trader3.v2.trigger import apply_trigger_transition, get_trigger_engine
    from trader3.v2.watchlist import get_watchlist

    summary = {}

    # 1. 数据采集（事件入库）
    collector = DataCollector()
    added_stock = 0
    wl = get_watchlist()
    target_codes = codes or [c.code for c in wl.list()]
    if not target_codes:
        target_codes = ["600519", "000858"]  # 演示兜底
    for code in target_codes:
        added_stock += collector.collect_stock(code, source="news")
    added_flow = 0
    if collect_flow:
        added_flow = collector.collect_flow_events()
    collector.close()
    wl.close()
    summary["collected_stock_events"] = added_stock
    summary["collected_flow_events"] = added_flow

    if collect_only:
        summary["mode"] = "collect_only"
        return summary

    # 2. 三因子扫描 + 事前风控链（公共判定 TriggerEngine._evaluate，
    #    与 trigger._scan_one 共用同一实现；含负面事件否决）
    engine = get_trigger_engine()
    wl = get_watchlist()
    items = wl.list()
    results = engine.scan(items)

    # 3. 状态迁移（两段式白名单：观察中→关注→买入区间；dry_run 跳过）
    triggered = [r for r in results if r.triggered]

    # 3.5 多空辩论复核（P2-6，默认关）：veto 的信号退出 triggered 并记 caveat
    debate_verdicts: dict[str, dict] = {}
    if debate_review and triggered:
        try:
            from trader3.v2.debate import debate_batch
            verdicts = debate_batch(triggered, llm_fn=debate_llm_fn)
            vetoed = []
            for r in triggered:
                v = verdicts.get(r.code)
                if v is not None:
                    debate_verdicts[r.code] = v.to_dict()
                    if v.is_veto:
                        r.caveats.append(f"辩论否决：{v.bear.claim[:60]}")
                        vetoed.append(r)
            if vetoed:
                logger.info("[daily] 辩论否决 %d 个信号: %s", len(vetoed),
                            [r.code for r in vetoed])
                triggered = [r for r in triggered if r not in vetoed]
        except Exception as e:  # noqa: BLE001
            logger.warning("[daily] 辩论复核失败（放行全部信号）: %s", e)
    if debate_verdicts:
        summary["debate"] = debate_verdicts

    if not dry_run:
        for r in results:
            apply_trigger_transition(wl, r)
    wl.close()

    # 4. 纸面执行闭环：触发 → 已过风控的买入信号 → 模拟撮合入账
    #    （dry_run 不动账户；纸面交易，非真实委托）
    paper_state = None
    if paper_trading and not dry_run:
        try:
            paper_state = run_paper_trades(triggered, overlay=overlay)
        except Exception as e:
            logger.warning("[daily] 纸面交易执行失败（跳过不影响主链路）: %s", e)
    summary["paper_trading"] = paper_state
    if overlay is not None:
        try:
            from trader3.v2.risk_overlay import summarize as _ov_sum
            summary["risk_overlay"] = _ov_sum(overlay)
        except Exception:
            pass

    summary["mode"] = "dry_run" if dry_run else "full"
    summary["scanned"] = len(results)
    summary["triggered"] = triggered
    summary["all_results"] = results
    return summary


def run_paper_trades(triggered_results: list, overlay=None) -> dict:
    """对触发的买卖信号逐只纸面下单，并把账户状态原子落盘。

    - 账户持久化于 shared_state/paper/account.json（SharedState.write_json）：
      {as_of, date, cash, positions, trades_today, equity, ...}
    - 每单资金 = 总资产 × PAPER_TRADE_PCT，按一手（100股）整数取整
    - overlay：可选风险覆盖层——halt_new_buys 时跳过全部买入；
      否则每单比例 = PAPER_TRADE_PCT × size_multiplier
    - 卖出信号：对纸面持仓该标的可卖量整单市价卖出
      （Fillers 模拟成交，佣金/印花税按 costs 费率；无持仓记 no_position）
    - 行情快照统一经 qa_accessor.get_quote_snapshot（换源只改一处）
    - 同日续用账户状态；跨日先 settle()（T+1 解冻 + 当日计数复位）
    - 全部为模拟撮合：纸面交易，非真实委托
    """
    from trader3.shared_state import SharedState
    from trader3.v2.execution_realism import PositionT1Account
    from trader3.v2.qa_accessor import get_quote_snapshot
    from trader3.v2.risk_overlay import apply_to_trade_pct

    ss = SharedState(PAPER_STATE_DIR)
    prev = ss.read_json("account") or {}
    today = datetime.now().strftime("%Y-%m-%d")

    acct = PositionT1Account.from_state(prev) if prev \
        else PositionT1Account(cash=PAPER_INITIAL_CASH)
    trades_today: list[dict] = list(prev.get("trades_today") or [])
    if prev.get("as_of") != today:      # 跨日：解冻 + 计数复位，重开交易日流水
        acct.settle()
        trades_today = []

    effective_pct = apply_to_trade_pct(PAPER_TRADE_PCT, overlay) \
        if overlay is not None else PAPER_TRADE_PCT
    halt_buys = bool(overlay is not None and getattr(overlay, "halt_new_buys", False))
    if halt_buys:
        logger.warning("[overlay] 风险覆盖层熔断：今日暂停全部新开仓")

    latest_prices: dict[str, float] = {}
    _daily_pnl_start_equity = acct.total_assets()
    _max_daily_loss_pct = 0.05     # 日内亏损超5%熔断停止开新仓
    _max_positions = 20            # 最大持仓只数
    seen_codes_in_batch: set[str] = set()

    for r in triggered_results:
        code = getattr(r, "code", "")
        direction = getattr(r, "direction", "buy")

        # ── 逐笔前置风控（order-level pre-trade checks）──
        if code in seen_codes_in_batch:
            logger.info("[风控-前置] %s 同批次重复信号，跳过", code)
            continue
        seen_codes_in_batch.add(code)

        equity_now = acct.total_assets()
        daily_pnl_pct = (equity_now / _daily_pnl_start_equity - 1.0
                         ) if _daily_pnl_start_equity > 0 else 0.0
        if (direction == "buy"
                and daily_pnl_pct < -_max_daily_loss_pct):
            logger.warning("[风控-熔断] 日内亏损 %.1f%% 超限 %.0f%%，暂停开新仓",
                           daily_pnl_pct * 100, _max_daily_loss_pct * 100)
            break
        if (direction == "buy" and len(acct.stocks) >= _max_positions):
            logger.warning("[风控-持仓上限] 已达 %d 只，跳过买入", _max_positions)
            continue

        rec = {"code": code,
               "name": str(getattr(r, "reason", ""))[:40],
               "action": direction,
               "ts": datetime.now().strftime("%Y-%m-%d %H:%M")}
        if direction in ("sell", "negative"):
            # ── 卖出分支：纸面持仓整单市价卖出 ──
            snap = get_quote_snapshot(code)
            price = float(snap.get("price") or 0.0)
            if price <= 0:
                rec.update({"ok": False, "side": "sell",
                            "msg": "无有效行情价，跳过"})
                trades_today.append(rec)
                logger.info("[纸面交易，非真实委托] %s 无行情价，卖出跳过", code)
                continue
            st = acct.stocks.get(code)
            sellable = float(st.sellable) if st else 0.0
            if sellable <= 0:
                rec.update({"ok": False, "side": "sell",
                            "skipped_trade": "no_position",
                            "msg": "纸面无持仓（或当日买入 T+1 未到期），跳过"})
                trades_today.append(rec)
                logger.info("[纸面交易，非真实委托] %s 无持仓可卖，跳过", code)
                continue
            volume = int(sellable)
            ok, msg = acct.sell(code, volume, price)
            logger.info("[纸面交易，非真实委托] %s", msg)
            rec.update({"side": "sell", "volume": volume, "price": price,
                        "ok": bool(ok), "msg": msg})
            trades_today.append(rec)
            if not ok and code in acct.stocks:   # 成交失败时仍用最新价估值
                latest_prices.setdefault(code, price)
            elif ok:
                latest_prices[code] = price   # 部分清仓场景按成交价估值余仓
            continue
        if direction != "buy":
            continue  # 纸面段只接已过风控的买入信号
        if halt_buys:
            rec.update({"ok": False, "side": "buy",
                        "msg": "风险覆盖层熔断：暂停新开仓"})
            trades_today.append(rec)
            logger.warning("[纸面交易，非真实委托] %s 覆盖层熔断，买入跳过", code)
            continue
        snap = get_quote_snapshot(code)
        price = float(snap.get("price") or 0.0)
        if price <= 0:
            rec.update({"ok": False, "side": "buy", "msg": "无有效行情价，跳过"})
            trades_today.append(rec)
            logger.info("[纸面交易，非真实委托] %s 无行情价，跳过", code)
            continue
        latest_prices[code] = price
        budget = acct.total_assets() * effective_pct
        volume = int(budget // (price * PAPER_LOT_SIZE)) * PAPER_LOT_SIZE
        if volume <= 0:
            rec.update({"ok": False, "side": "buy",
                        "msg": f"资金不足一手: 预算 {budget:.0f} < 一手约 {price * PAPER_LOT_SIZE:.0f}"})
            trades_today.append(rec)
            logger.info("[纸面交易，非真实委托] %s %s", code, rec["msg"])
            continue
        ok, msg = acct.buy(code, volume, price)
        logger.info("[纸面交易，非真实委托] %s", msg)
        rec.update({"side": "buy", "volume": volume, "price": price,
                    "ok": bool(ok), "msg": msg})
        trades_today.append(rec)

    # 净值：现金 + Σ持仓×最新价（本轮无行情的持仓按成本计）
    equity = acct.cash
    for code_, st in acct.stocks.items():
        equity += st.total * latest_prices.get(code_, st.avg_cost)

    state = {
        "as_of": today,
        "date": today,
        "cash": round(acct.cash, 2),
        "positions": acct.to_state()["positions"],
        "trades_today": trades_today,
        "equity": round(equity, 2),
        "pct_per_trade": PAPER_TRADE_PCT,
        "initial_cash": PAPER_INITIAL_CASH,
        "note": PAPER_NOTE,
    }
    ss.write_json("account", state)   # 原子写（临时文件+os.replace+跨进程锁）
    _append_fills_log(today, trades_today)
    return state


FILLS_LOG_HEADER = "date,code,side,volume,price,cost_cny,ok,msg\n"


def _append_fills_log(today: str, trades: list[dict]) -> None:
    """
    追加式成交流水（成本模型校准的数据来源）。
    只记录 ok=True 的实际成交；CSV 追加写，文件不存在时先写表头。
    """
    import csv as _csv
    import os

    from trader3.shared_state import SharedState

    path = os.path.join(SharedState().state_dir, "paper", "fills.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    need_header = not os.path.exists(path)
    rows = []
    stamp_bp = 1.0  # 印花税近似（卖出单边，bp）——与 costs 模型同口径的流水级估算
    for t in trades:
        if not (t.get("ok") and t.get("side") in ("buy", "sell")):
            continue
        vol = float(t.get("volume", 0) or 0)
        price = float(t.get("price", 0) or 0)
        gross = vol * price
        commission = max(gross * (DEFAULT_COSTS.commission_bp / 10000.0),
                         DEFAULT_COSTS.min_commission or 0.0)
        cost = commission + (gross * stamp_bp / 10000.0 if t["side"] == "sell" else 0.0)
        rows.append({
            "date": today,
            "code": t.get("code", ""),
            "side": t.get("side", ""),
            "volume": int(vol),
            "price": price,
            "cost_cny": round(cost, 2),
            "ok": True,
            "msg": str(t.get("msg", ""))[:60],
        })
    rows_out = rows
    if not rows_out and need_header:
        return  # 无成交且无文件 → 不产生空壳
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=FILLS_LOG_HEADER.strip().split(","))
        if need_header:
            w.writeheader()
        w.writerows(rows_out)


def format_alerts(summary: dict) -> str:
    """把触发结果格式化为提醒文本（写作规范三道关前置）"""
    lines = [f"# 3号交易员 v2.0 · 每日扫描 {datetime.now().strftime('%Y-%m-%d')}"]
    lines.append("")
    lines.append(f"采集：个股事件 {summary.get('collected_stock_events', 0)} 条，"
                 f"市场事件 {summary.get('collected_flow_events', 0)} 条")
    lines.append("")
    triggered = summary.get("triggered", [])
    if not triggered:
        lines.append("**今日无触发** —— 自选股均未同时满足 催化×预期差×技术 三因子。")
    else:
        lines.append("## 触发提醒")
        for r in triggered:
            lines.append("")
            lines.append(f"### 【买卖点·{r.direction}】{r.code}")
            lines.append(f"- 信号强度：{r.score:.2f}（催化 {r.catalyst_score:.2f} × "
                         f"估值 {r.valuation_score:.2f} × 技术 {r.tech_score:.2f}）")
            lines.append(f"- {r.reason}")
            if r.direction == "buy":
                lines.append(f"- 估值锚 ¥{r.fair_value}，现价 ¥{r.current_price}，"
                             f"隐含收益 {r.implied_return:+.1%}")
                lines.append(f"- 止损价 ¥{r.stop_loss}")
            lines.append(f"- 技术位：{r.key_technical}")
        vetoed = [r for r in summary.get("all_results", []) if getattr(r, "caveats", None)]
        if vetoed:
            lines.append("")
            lines.append("## 风控否决")
            for r in vetoed:
                lines.append(f"- {r.code}：{'；'.join(r.caveats)}")
    paper = summary.get("paper_trading")
    if paper:
        lines.append("")
        lines.append("## 纸面交易（非真实委托）")
        lines.append(f"- 今日成交 {len(paper.get('trades_today', []))} 笔，"
                     f"账户净值 ¥{paper.get('equity', 0):,.2f}，现金 ¥{paper.get('cash', 0):,.2f}")
        lines.append("> 模拟撮合，非真实委托；成交明细见 shared_state/paper/account.json。")
    lines.append("")
    lines.append("> 候选信号，非投资建议。写作规范：作者姿态 + 数字来源 + 反向批判前置。")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="3号交易员 v2.0 每日扫描")
    parser.add_argument("--codes", default="", help="自选股代码，逗号分隔")
    parser.add_argument("--collect-only", action="store_true", help="仅采集不扫描")
    parser.add_argument("--no-flow", action="store_true", help="跳过市场事件采集")
    parser.add_argument("--no-paper", action="store_true", help="跳过纸面交易（默认开启）")
    parser.add_argument("--debate", action="store_true",
                        help="开启多空辩论复核（默认关；LLM 缺省走规则辩论）")
    flags = parser.parse_args()

    codes = [c.strip() for c in flags.codes.split(",") if c.strip()] if flags.codes else None
    summary = run_daily(
        codes=codes,
        collect_only=flags.collect_only,
        collect_flow=not flags.no_flow,
        paper_trading=not flags.no_paper,
        debate_review=flags.debate,
    )
    print(format_alerts(summary))


if __name__ == "__main__":
    main()
