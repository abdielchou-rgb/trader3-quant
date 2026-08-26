"""
滚动再训练编排器（Retrain Loop）。

闭环：factor_watch IC 衰减 → 触发 ensemble 重训练 → 回归门禁 → 注册表切换。

流程：
1. should_retrain(): 读 shared_state/factor_watch/<factor>_ic.csv，
   近 decay_window 日均值 < 阈值（复用 kill_criteria 口径）→ 触发
2. run_retrain(): 调用注入的特征/收益构造函数 → build_ensemble 训练新模型
3. gate: 新模型 OOS RankIC ≥ 旧模型 − tolerance，否则拒绝切换（保留旧模型）
4. 通过则写注册表 registry.json（原子替换）+ 分数 npz 归档

设计原则：
- 特征构造完全由调用方注入（可测、可换数据源），本模块只管编排与门禁
- 任何一步失败不抛出到主链路，返回带 ok=False 的报告
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("trader3.v2.retrain")


@dataclass
class RetrainConfig:
    factor_name: str = "f1"
    ic_csv: str = ""                      # 缺省: <state_dir>/factor_watch/<name>_ic.csv
    state_dir: str = ""                   # 缺省: <repo>/shared_state
    registry_dir: str = ""                # 缺省: <state_dir>/model_registry
    decay_window: int = 20
    ic_threshold: float = 0.0             # 近窗均值低于此值触发
    min_ic_gap_days: int = 5              # 两次重训练最小间隔（天）
    gate_tolerance: float = 0.005         # 新模型允许比旧模型差的 IC 余量
    cooldown_hours: float = 20.0          # 同因子的时间冷却
    ensemble_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrainReport:
    triggered: bool
    trained: bool
    promoted: bool            # 是否通过门禁写入注册表
    reason: str
    old_rank_ic: float | None = None
    new_rank_ic: float | None = None
    model_used: str | None = None
    artifact_path: str | None = None


def _default_state_dir() -> str:
    import trader3
    root = os.path.dirname(os.path.abspath(trader3.__file__))
    return os.path.join(os.path.dirname(root), "shared_state")


def read_ic_history(path: str) -> pd.DataFrame:
    """读 factor_watch 的 {date,ic} CSV → DataFrame(date index, ic)。"""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ic"])
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({"date": row["date"], "ic": float(row["ic"])})
            except (KeyError, ValueError):
                continue
    if not rows:
        return pd.DataFrame(columns=["ic"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def should_retrain(config: RetrainConfig,
                   ic_df: pd.DataFrame | None = None) -> tuple[bool, str]:
    """
    判定是否触发重训练。返回 (triggered, reason)。
    条件：样本充足 + 近窗均值 < 阈值 + 距上次成功重训练超冷却。
    """
    state_dir = config.state_dir or _default_state_dir()
    csv_path = config.ic_csv or os.path.join(
        state_dir, "factor_watch", f"{config.factor_name}_ic.csv")
    df = ic_df if ic_df is not None else read_ic_history(csv_path)

    if len(df) < config.decay_window:
        return False, f"样本不足({len(df)}<{config.decay_window})，不触发"

    recent_mean = float(df["ic"].tail(config.decay_window).mean())
    if recent_mean >= config.ic_threshold:
        return False, f"近{config.decay_window}日IC={recent_mean:.4f} ≥ 阈值，健康"

    # 时间冷却：查注册表最近一次 promoted 记录
    registry_dir = config.registry_dir or os.path.join(state_dir, "model_registry")
    reg_path = os.path.join(registry_dir, "registry.json")
    if os.path.exists(reg_path):
        try:
            with open(reg_path, encoding="utf-8") as f:
                reg = json.load(f)
            last_ts = (reg.get("entries") or {}).get(config.factor_name, {}).get(
                "promoted_at_epoch", 0)
            hours = (time.time() - last_ts) / 3600.0
            if hours < config.cooldown_hours:
                return False, f"冷却中(距上次{hours:.1f}h<{config.cooldown_hours}h)"
        except Exception:
            pass

    return True, (f"衰减触发: 近{config.decay_window}日IC={recent_mean:.4f} "
                  f"< {config.ic_threshold}")


def _load_registry(registry_dir: str) -> dict:
    path = os.path.join(registry_dir, "registry.json")
    if not os.path.exists(path):
        return {"version": 1, "entries": {}}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"version": 1, "entries": {}}


def _save_registry_atomic(registry_dir: str, reg: dict) -> str:
    os.makedirs(registry_dir, exist_ok=True)
    path = os.path.join(registry_dir, "registry.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


def run_retrain(
    build_data: Callable[[], tuple[dict[str, pd.DataFrame], pd.DataFrame]],
    config: RetrainConfig | None = None,
    force: bool = False,
) -> RetrainReport:
    """
    完整一次「触发→训练→门禁→晋升」。

    build_data: () -> (features_wide, forward_returns)
        features_wide:  {factor_name -> DataFrame(date × asset)}
        forward_returns: DataFrame(date × asset)，与因子日期对齐
    force=True: 跳过衰减判定直接训练（仍过门禁）。

    返回 RetrainReport；全链路异常被捕获并写入 report.reason。
    """
    from trader3.v2.ensemble import EnsembleConfig, build_ensemble

    cfg = config or RetrainConfig()
    state_dir = cfg.state_dir or _default_state_dir()
    registry_dir = cfg.registry_dir or os.path.join(state_dir, "model_registry")

    # 1. 触发判定
    try:
        trig, why = (True, "force=True 跳过判定") if force else should_retrain(cfg)
    except Exception as e:  # noqa: BLE001
        return RetrainReport(False, False, False, f"触发判定异常: {e}")
    if not trig:
        return RetrainReport(False, False, False, why)

    # 2. 构造数据 + 训练
    try:
        features_wide, fwd = build_data()
        ens_cfg = EnsembleConfig(**cfg.ensemble_kwargs)
        result = build_ensemble(features_wide, fwd, config=ens_cfg)
    except Exception as e:  # noqa: BLE001
        return RetrainReport(True, False, False, f"训练失败: {e}")

    # 3. 回归门禁：对比注册表中的旧模型
    reg = _load_registry(registry_dir)
    entry = (reg.get("entries") or {}).get(cfg.factor_name, {})
    old_ic = entry.get("oos_rank_ic")

    promoted = False
    if old_ic is None:
        promoted = True
        gate_msg = "首版模型，直接晋升"
    elif result.oos_rank_ic >= old_ic - cfg.gate_tolerance:
        promoted = True
        gate_msg = (f"门禁通过: 新{result.oos_rank_ic:.4f} ≥ "
                    f"旧{old_ic:.4f}−tol({cfg.gate_tolerance})")
    else:
        gate_msg = (f"门禁拒绝: 新{result.oos_rank_ic:.4f} < "
                    f"旧{old_ic:.4f}−tol({cfg.gate_tolerance})，保留旧模型")

    artifact_path = None
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    if promoted:
        # 4. 晋升：归档分数 npz + 更新注册表（原子）
        try:
            os.makedirs(registry_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            artifact_path = os.path.join(
                registry_dir, f"{cfg.factor_name}_{stamp}.npz")
            scores_wide = result.scores.unstack(level=-1) \
                if isinstance(result.scores.index, pd.MultiIndex) \
                else result.scores.to_frame()
            np.savez_compressed(
                artifact_path,
                scores=scores_wide.values.astype(np.float32),
                index=np.array([str(x) for x in scores_wide.index]),
                columns=np.array([str(x) for x in scores_wide.columns]),
                name=result.scores.name or "score",
            )
            reg.setdefault("version", 1)
            reg.setdefault("entries", {})
            reg["entries"][cfg.factor_name] = {
                "oos_rank_ic": result.oos_rank_ic,
                "oos_ic_ir": result.oos_ic_ir,
                "baseline_rank_ic": result.baseline_rank_ic,
                "model_used": result.model_used,
                "used_ml": bool(result.used_ml),
                "feature_importance": result.feature_importance,
                "artifact": os.path.basename(artifact_path),
                "promoted_at": now_str,
                "promoted_at_epoch": time.time(),
                "gate": gate_msg,
            }
            _save_registry_atomic(registry_dir, reg)
        except Exception as e:  # noqa: BLE001
            return RetrainReport(True, True, False,
                                 f"晋升落盘失败(模型未切换): {e}",
                                 old_rank_ic=old_ic,
                                 new_rank_ic=result.oos_rank_ic,
                                 model_used=result.model_used)

    logger.info("[retrain] %s triggered=%s trained=%s promoted=%s :: %s",
                cfg.factor_name, True, True, promoted, gate_msg)
    return RetrainReport(
        triggered=True, trained=True, promoted=promoted,
        reason=gate_msg, old_rank_ic=old_ic,
        new_rank_ic=float(result.oos_rank_ic), model_used=result.model_used,
        artifact_path=artifact_path,
    )


def load_current_scores(factor_name: str, config: RetrainConfig | None = None
                        ) -> pd.DataFrame | None:
    """从注册表加载当前在役模型的分数宽表（供管线消费）。"""
    cfg = config or RetrainConfig(factor_name=factor_name)
    registry_dir = cfg.registry_dir or os.path.join(_default_state_dir(),
                                                    "model_registry")
    reg = _load_registry(registry_dir)
    entry = (reg.get("entries") or {}).get(factor_name)
    if not entry or "artifact" not in entry:
        return None
    path = os.path.join(registry_dir, entry["artifact"])
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    return pd.DataFrame(z["scores"], index=[str(x) for x in z["index"]],
                        columns=[str(x) for x in z["columns"]])


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="手动触发重训练检查")
    p.add_argument("--force", action="store_true", help="跳过衰减判定")
    p.add_argument("--check-only", action="store_true", help="只做触发判定")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = RetrainConfig()
    if args.check_only:
        t, r = should_retrain(cfg)
        print(f"triggered={t}: {r}")
    else:
        raise SystemExit("run_retrain 需要注入 build_data，请用代码调用而非 CLI 直接跑")
