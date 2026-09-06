"""
轻量实验档案层（R1）—— 补 trader3 缺的"死的、可检索的研究记录"。

与运行时遥测（obs/telemetry.py，活的仪表盘）正交：
  - telemetry：下单/风控/权益/延迟 实时指标 → Prometheus
  - experiment：因子挖掘/回测 run 的可复现档案 → jsonl，可 search/best/top 横向对比

设计（对齐 qlib workflow 教训但零依赖）：
  - 追加式 jsonl（每 run 一行），崩溃安全；坏行跳过不炸
  - run_id 幂等：重复 id 覆盖（同参数重跑记录最终结果）
  - Experiment.search(tags=, params=) 过滤 + best/top(metric) 对比
  - 自动记录 ts / git_sha（可关）/ params / metrics / tags / artifacts
"""

from __future__ import annotations

import json
import os
import subprocess
import time


def _git_sha(cwd: str | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return out or "nogit"
    except Exception:  # noqa: BLE001
        return "nogit"


class Experiment:
    """一个研究主题（如某因子族/某策略集）的 run 档案。"""

    def __init__(self, log_dir: str, name: str = "default",
                 record_git: bool = True):
        self.dir = os.path.abspath(log_dir)
        self.name = name
        self.record_git = record_git
        self._path = os.path.join(self.dir, f"{name}.jsonl")
        os.makedirs(self.dir, exist_ok=True)

    # ── 写 ──────────────────────────────────────

    def run(self, run_id: str, *, params: dict | None = None,
            metrics: dict | None = None, tags: dict | None = None,
            artifacts: dict | None = None, commit: str | None = None) -> dict:
        """记录一次 run。run_id 重复 → 覆盖（先删旧行再追加）。"""
        record = {
            "run_id": run_id,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_sha": commit if commit is not None else
                       (self._git_sha_now() if self.record_git else "off"),
            "params": params or {},
            "metrics": metrics or {},
            "tags": tags or {},
            "artifacts": artifacts or {},
        }
        # 覆盖语义：移除同 run_id 旧行
        self._rewrite_without(run_id)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def _git_sha_now(self) -> str:
        try:
            return _git_sha(os.path.dirname(self.dir))
        except Exception:  # noqa: BLE001
            return "nogit"

    def _rewrite_without(self, run_id: str) -> None:
        if not os.path.exists(self._path):
            return
        kept = [rec for rec in self._read_raw() if rec.get("run_id") != run_id]
        with open(self._path, "w", encoding="utf-8") as f:
            for rec in kept:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _read_raw(self) -> list[dict]:
        if not os.path.exists(self._path):
            return []
        out: list[dict] = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue  # 坏行跳过
                if isinstance(rec, dict):
                    out.append(rec)
        return out

    # ── 读 ──────────────────────────────────────

    def list_runs(self) -> list[dict]:
        """全部 run（按落盘顺序）。"""
        return self._read_raw()

    def search(self, *, tags: dict | None = None,
               params: dict | None = None) -> list[dict]:
        """过滤检索：tags/params 为子集匹配（record 含全部给定键值）。"""
        out = []
        for rec in self._read_raw():
            if tags and not all(rec["tags"].get(k) == v
                                for k, v in tags.items()):
                continue
            if params and not all(rec["params"].get(k) == v
                                  for k, v in params.items()):
                continue
            out.append(rec)
        return out

    def best(self, metric: str, *, higher_is_better: bool = True) -> dict | None:
        top = self.top(metric, k=1, higher_is_better=higher_is_better)
        return top[0] if top else None

    def top(self, metric: str, k: int = 5, *,
            higher_is_better: bool = True) -> list[dict]:
        """按指标取 top-k。缺失该指标的 run 排除。"""
        scored = [r for r in self._read_raw() if metric in r["metrics"]]
        scored.sort(key=lambda r: r["metrics"][metric],
                    reverse=higher_is_better)
        return scored[:k]
