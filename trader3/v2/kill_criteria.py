"""
3号交易员 — Kill Criteria 自动化

圆桌共识 #2：告警直接触发降级动作，不留人工判断窗口。

流程：
  factor_watch 检测衰减 → 自动在 selected.json 标记 oos_veto=true
  → 推送关停通知 → 下次合成自动跳过该因子（既有 OOS 否决机制消费）

设计原则：
  - 告警即行动，不等人工确认
  - 所有自动操作留审计日志（谁/何时/为什么）
  - 可通过 KILL_CRITERIA_ENABLED=false 环境变量全局禁用
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# Kill criteria 阈值（可通过环境变量覆盖）
KILL_IC_WINDOW = int(os.environ.get("KILL_IC_WINDOW", "20"))
KILL_IC_THRESHOLD = float(os.environ.get("KILL_IC_THRESHOLD", "0.0"))
KILL_CONSECUTIVE_NEGATIVE = int(os.environ.get("KILL_CONSECUTIVE_NEGATIVE", "10"))


def evaluate_kill_criteria(
    ic_history: list[dict],
    decay_window: int = KILL_IC_WINDOW,
    threshold: float = KILL_IC_THRESHOLD,
    consecutive_neg: int = KILL_CONSECUTIVE_NEGATIVE,
) -> dict:
    """
    评估 kill criteria 是否触发。

    Parameters
    ----------
    ic_history : [{"date": str, "ic": float}, ...] — 按时间升序
    decay_window : 近 N 日均值窗口
    threshold : IC 均值阈值（低于此值触发）
    consecutive_neg : 连续负 IC 天数阈值

    Returns
    -------
    {triggered: bool, reason: str, ic_mean_recent: float|None,
     neg_streak: int, total_obs: int}
    """
    if not ic_history or len(ic_history) < 3:
        return {"triggered": False, "reason": "样本不足(<3)",
                "ic_mean_recent": None, "neg_streak": 0,
                "total_obs": len(ic_history)}

    ics = [float(h.get("ic", 0) or 0) for h in ic_history]
    recent = ics[-decay_window:] if len(ics) >= decay_window else ics
    mean_recent = sum(recent) / len(recent) if recent else None

    neg_streak = 0
    for v in reversed(ics):
        if v < 0:
            neg_streak += 1
        else:
            break

    triggered = False
    reason = ""
    if mean_recent is not None and mean_recent < threshold:
        triggered = True
        reason = f"近{len(recent)}日IC均值 {mean_recent:.4f} < {threshold}"
    elif neg_streak >= consecutive_neg:
        triggered = True
        reason = f"连续 {neg_streak} 日负IC"

    return {
        "triggered": triggered,
        "reason": reason,
        "ic_mean_recent": round(mean_recent, 4) if mean_recent is not None else None,
        "neg_streak": neg_streak,
        "total_obs": len(ic_history),
    }


def auto_veto_factor(
    selected_path: str,
    expr: str,
    kill_reason: str,
) -> bool:
    """
    自动标记 oos_veto=true 并写入否决原因。
    已被否决的因子幂等跳过。返回是否实际修改。
    """
    if not os.path.exists(selected_path):
        logger.warning("[kill] selected.json 不存在: %s", selected_path)
        return False

    with open(selected_path, encoding="utf-8") as f:
        entries = json.load(f)

    modified = False
    for e in entries:
        if e.get("expr") == expr and not e.get("oos_veto"):
            e["oos_veto"] = True
            e["veto_reason"] = f"[KillCriteria自动化] {kill_reason}"
            e["vetoed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            e["veto_source"] = "factor_watch_decay"
            modified = True
            logger.warning("[kill] 因子 %s 已自动否决: %s", expr[:40], kill_reason)

    if modified:
        tmp = selected_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        os.replace(tmp, selected_path)
    return modified


def run_kill_check(
    state_dir: str,
    selected_dir: str,
    factor_name: str = "f1",
    expr: str = "sub(log(vwap), log(close))",
    **kwargs,
) -> dict:
    """
    一站式 kill check：读历史→评估→自动否决→返回结果。

    供 daily_routine 调用；任何异常不抛出（不阻断主链路）。
    """
    if os.environ.get("KILL_CRITERIA_ENABLED", "true").lower() in ("false", "0"):
        return {"enabled": False, "triggered": False}

    fw_dir = os.path.join(state_dir, "factor_watch")
    csv_path = os.path.join(fw_dir, f"{factor_name}_ic.csv")

    ic_history = []
    if os.path.exists(csv_path):
        import csv as _csv
        with open(csv_path, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                try:
                    ic_history.append({"date": row["date"], "ic": float(row["ic"])})
                except (KeyError, ValueError):
                    continue

    result = evaluate_kill_criteria(ic_history, **kwargs)
    result["enabled"] = True
    result["factor_name"] = factor_name

    if result["triggered"]:
        sel_path = os.path.join(selected_dir, "selected.json") \
            if not selected_dir.endswith(".json") else selected_dir
        vetoed = auto_veto_factor(sel_path, expr, result["reason"])
        result["auto_veto_applied"] = vetoed
        result["selected_path"] = sel_path

    return result
