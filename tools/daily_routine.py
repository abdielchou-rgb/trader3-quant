r"""
P1-② 每日例行任务
用法：python daily_routine.py [--universe csi300,csi500,csi1000,etf] [--output D:\Marvis\output\]
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVOLVE_SCRIPT = os.path.join(PROJECT, "evolve", "run_evolution.py")
AKSHARE_SCRIPT = os.path.join(PROJECT, "evolve", "scripts", "fetch_etf_data.py")
STATE_FILE = os.path.join(PROJECT, "evolve", "last_daily_run.json")

TASKS = {
    "market_data_update": "qlib_bin 增量更新（指数重建+日历扩展）",
    "etf_data": "拉取 ETF 日线数据（akshare）",
    "evolve_csi300": "CSI300 策略进化",
    "evolve_csi500": "CSI500 策略进化",
    "evolve_csi1000": "CSI1000 策略进化",
    "evolve_etf": "ETF 策略进化",
}

def run(cmd, desc):
    print(f"[{datetime.now():%H:%M:%S}] {desc}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=PROJECT)
    if result.returncode != 0:
        print(f"  ✗ FAILED: {(result.stderr or result.stdout)[:200]}")
        return False, (result.stderr or result.stdout)[:500]
    print("  ✓ OK")
    return True, result.stdout[-300:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="csi300,csi500,csi1000,etf")
    parser.add_argument("--output", default=r"D:\Marvis\output")
    parser.add_argument("--skip-etf-data", action="store_true")
    args = parser.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    universes = [u.strip() for u in args.universe.split(",")]
    results = {"date": today, "tasks": {}}

    # Step 0: qlib_bin 增量更新（指数重建 + data_version 盖章，失败不阻断后续）
    update_script = os.path.join(PROJECT, "scripts", "update_market_data.py")
    if os.path.exists(update_script):
        ok, out = run(f'python "{update_script}" --apply', TASKS["market_data_update"])
        results["tasks"]["market_data_update"] = {"ok": ok, "output": (out or "")[-500:]}

    # Step 1: ETF 数据拉取（每天拉最新）
    if not args.skip_etf_data:
        if os.path.exists(AKSHARE_SCRIPT):
            ok, out = run(f'python "{AKSHARE_SCRIPT}"', TASKS["etf_data"])
            results["tasks"]["etf_data"] = {"ok": ok, "output": (out or "")[-500:]}
        else:
            results["tasks"]["etf_data"] = {"ok": False, "skipped": True, "reason": f"脚本不存在: {AKSHARE_SCRIPT}"}

    # Step 2: 策略进化（按 universe）
    for u in universes:
        task_key = f"evolve_{u}"
        if task_key not in TASKS:
            continue
        if not os.path.exists(EVOLVE_SCRIPT):
            results["tasks"][task_key] = {"ok": False, "skipped": True, "reason": f"脚本不存在: {EVOLVE_SCRIPT}"}
            continue
        cmd = f'python "{EVOLVE_SCRIPT}" --source etf --universe {u} --gen 60 --pop 100' if u == "etf" else \
              f'python "{EVOLVE_SCRIPT}" --universe {u} --gen 60 --pop 100'
        ok, out = run(cmd, TASKS[task_key])
        results["tasks"][task_key] = {"ok": ok, "output": (out or "")[-500:]}

    # Step 3: 收集 selected 策略汇总
    by_universe = os.path.join(PROJECT, "evolve", "strategies", "by_universe")
    strategies_today = {}
    for u in universes:
        f = os.path.join(by_universe, f"{u}_selected.json")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            if not data:
                continue
            strategies_today[u] = {
                "count": len(data),
                "top_expr": str(data[0].get("expr", ""))[:80],
                "top_score": data[0].get("score"),
                "top_ic": data[0].get("gates", {}).get("ic", {}).get("value"),
            }

    results["strategies"] = strategies_today

    # Step 4: 写日报
    os.makedirs(args.output, exist_ok=True)
    daily_path = os.path.join(args.output, f"daily_{today.replace('-', '')}.json")
    with open(daily_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 更新状态
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_run": today, "results": strategies_today}, f, ensure_ascii=False, indent=2)

    print(f"\n📄 日报: {daily_path}")
    print(f"📊 今日策略: {json.dumps(strategies_today, ensure_ascii=False)}")

    # Step 5: 因子衰减监控（F1）+ 纸面 vs 基线对照；失败不阻断
    watch_lines = []
    try:
        sys.path.insert(0, PROJECT)
        from trader3.v2.factor_watch import append_ic_history, compute_ic_series

        ic = compute_ic_series(lookback_days=260)
        last_date = ic["dates"][-1] if ic["dates"] else today
        rec = {"last_date": last_date,
               "ic_last": ic["ic_series"][-1] if ic["ic_series"] else None}
        st = append_ic_history(os.path.join(PROJECT, "shared_state"),
                               "f1", rec, decay_window=20)
        results["factor_watch"] = {"ic_mean_recent": st["ic_mean_recent"],
                                   "alert": st["alert"]}
        watch_lines.append(
            f"F1 近20日IC均值 {st['ic_mean_recent']}"
            + (" ⚠衰减告警" if st["alert"] else "")
        )
    except Exception as e:
        print(f"  ⚠ 因子监控失败(不阻断): {e}")

    anchor_line = ""
    try:
        acct_path = os.path.join(PROJECT, "shared_state", "paper", "account.json")
        base_path = os.path.join(PROJECT, "docs", "baseline", "baseline_results.json")
        if os.path.exists(acct_path) and os.path.exists(base_path):
            with open(acct_path, encoding="utf-8") as f:
                acct = json.load(f)
            with open(base_path, encoding="utf-8") as f:
                base = json.load(f)
            init = acct.get("initial_cash") or 0
            eq = acct.get("equity") or 0
            if init > 0 and eq > 0:
                cum = eq / init - 1.0
                f1_ann = ((base.get("strategies", {}).get("S2_F1_vwap_gap", {})
                           .get("periods", {}).get("OOS", {}).get("metrics", {})
                           .get("年化收益")) or 0.234)
                daily_anchor = (1 + f1_ann) ** (1 / 244) - 1
                watch_lines.append(
                    f"纸面累计 {cum:+.2%} vs F1基线日均锚 {daily_anchor:+.3%}"
                )
                anchor_line = f"\n对照: 纸面累计 {cum:+.2%}（F1 日均锚 {daily_anchor:+.3%}）"
    except Exception as e:
        print(f"  ⚠ 基线对照失败(不阻断): {e}")

    # Step 6: 推送通知（未配置通道时仅日志，不阻断）
    try:
        sys.path.insert(0, PROJECT)
        from trader3.notify import send_notification

        n_ok = sum(1 for t in results["tasks"].values() if t.get("ok"))
        brief = "\n".join(
            f"- {k}: {'✓' if v.get('ok') else '✗'} {str(v.get('output', ''))[:80]}"
            for k, v in results["tasks"].items()
        )
        extra = ("\n\n" + "\n".join(watch_lines)) if watch_lines else ""
        send_notification(
            f"3号交易员日报 {today}（{n_ok}/{len(results['tasks'])} 任务成功）",
            f"{brief}{extra}{anchor_line}\n\n策略: "
            f"{json.dumps(strategies_today, ensure_ascii=False)[:600]}",
        )
    except Exception as e:
        print(f"  ⚠ 通知推送失败(不阻断): {e}")

if __name__ == "__main__":
    main()
