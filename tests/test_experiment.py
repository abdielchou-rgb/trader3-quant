"""
R1 实验档案层 — 回归测试

痛点（qlib 解剖结论 #1）：trader3 有活的运行时遥测（telemetry），缺"死的、
可检索的研究档案"——因子挖掘/回测 run 结果散落 json，不可横向对比、无血缘。

契约：
  1. Experiment.run(params, metrics, artifacts) → 追加式 jsonl 落盘（可崩溃恢复）
  2. 每次 run 自动记录：时间戳、git_sha（可选）、params、metrics、tags、artifacts 路径
  3. Experiment.search(**metric_filters) → 按指标横向对比（如 ic 降序）
  4. 同一 experiment 多次 run 可对比；幂等追加（重复 run_id 覆盖）
  5. 无第三方依赖（jsonl + 标准库）
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.research.experiment import Experiment  # noqa: E402


def _exp(tmp_path: Path) -> Experiment:
    return Experiment(log_dir=str(tmp_path / "exp"), name="factor_test")


def test_run_persists_and_reloadable(tmp_path):
    exp = _exp(tmp_path)
    exp.run(run_id="r1", params={"epochs": 10, "loss": "mse"},
            metrics={"ic": 0.05, "icir": 0.2}, tags={"factor": "mom20"})
    # 新实例同目录 → reload 可检索
    exp2 = Experiment(log_dir=str(tmp_path / "exp"), name="factor_test")
    runs = exp2.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "r1"
    assert runs[0]["metrics"]["ic"] == pytest.approx(0.05)
    assert runs[0]["params"]["epochs"] == 10


def test_multiple_runs_searchable_by_metric(tmp_path):
    exp = _exp(tmp_path)
    exp.run(run_id="a", params={"seed": 1}, metrics={"ic": 0.03})
    exp.run(run_id="b", params={"seed": 2}, metrics={"ic": 0.09})
    exp.run(run_id="c", params={"seed": 3}, metrics={"ic": 0.06})
    best = exp.best(metric="ic")
    assert best["run_id"] == "b"
    top = exp.top(metric="ic", k=2)
    assert [r["run_id"] for r in top] == ["b", "c"]


def test_same_run_id_overwrites(tmp_path):
    exp = _exp(tmp_path)
    exp.run(run_id="r1", params={"seed": 1}, metrics={"ic": 0.03})
    exp.run(run_id="r1", params={"seed": 1}, metrics={"ic": 0.11})
    runs = exp.list_runs()
    assert len(runs) == 1  # 覆盖非追加
    assert runs[0]["metrics"]["ic"] == pytest.approx(0.11)


def test_filter_by_tag(tmp_path):
    exp = _exp(tmp_path)
    exp.run(run_id="a", params={}, metrics={"ic": 0.05}, tags={"family": "gp"})
    exp.run(run_id="b", params={}, metrics={"ic": 0.08}, tags={"family": "lstm"})
    gp = exp.search(tags={"family": "gp"})
    assert [r["run_id"] for r in gp] == ["a"]
    all_runs = exp.search()
    assert len(all_runs) == 2


def test_corrupt_line_skipped_not_fatal(tmp_path):
    exp = _exp(tmp_path)
    exp.run(run_id="ok", params={}, metrics={"ic": 0.05})
    # 手动注入一行坏 json
    logfile = Path(tmp_path) / "exp" / "runs.jsonl"
    with open(logfile, "a", encoding="utf-8") as f:
        f.write("{not valid json}\n")
    exp2 = Experiment(log_dir=str(tmp_path / "exp"), name="factor_test")
    runs = exp2.list_runs()
    assert len(runs) == 1  # 坏行跳过，不炸
