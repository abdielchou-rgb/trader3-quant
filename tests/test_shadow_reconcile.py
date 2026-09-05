"""
影子链路接入每日管线 — 回归测试
契约：
  1. run_shadow_reconcile 用 ShadowBroker 包 QMTBroker(SIM)，跑一次目标组合对账，
     产出 shadow_run.json（含订单、目标权重、目标与影子持仓的缺口）
  2. 影子订单全部带 shadow 标记且不下发真实通道
  3. dry_run=True 只写计划不落盘
  4. 影子状态文件含 mode="shadow" 与时间戳，供后续 3-6 个月对照积累
"""
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.v2.shadow_reconcile import (  # noqa: E402
    run_shadow_reconcile,
)


def _write_targets(tmp_path: Path, weights: dict) -> str:
    p = tmp_path / "targets.json"
    p.write_text(json.dumps({"weights": weights, "cash_pct": 0.0}), encoding="utf-8")
    return str(p)


def test_shadow_reconcile_produces_state(tmp_path):
    targets = _write_targets(tmp_path, {"600519.SH": 0.4, "000001.SZ": 0.6})
    # 显式重定向状态目录到临时区（不污染真实 shared_state）
    state_file = tmp_path / "shadow_run.json"
    result = run_shadow_reconcile(
        targets_path=targets,
        qmt_path=r"C:\qmt\userdata_mini",
        account_id="12345678",
        simulated=True,
        state_file=str(state_file),
    )
    assert result["mode"] == "shadow"
    assert result["n_orders"] >= 2
    assert all(o["shadow"] for o in result["orders"])
    # 落盘文件与返回值一致
    disk = json.loads(state_file.read_text(encoding="utf-8"))
    assert disk["mode"] == "shadow"
    assert disk["ts"]
    assert "gaps" in disk


def test_shadow_reconcile_dry_run_no_file(tmp_path):
    targets = _write_targets(tmp_path, {"600519.SH": 1.0})
    state_file = tmp_path / "shadow_run.json"
    result = run_shadow_reconcile(
        targets_path=targets,
        qmt_path=r"C:\qmt\userdata_mini",
        account_id="12345678",
        simulated=True,
        dry_run=True,
        state_file=str(state_file),
    )
    assert result["mode"] == "shadow-dry"
    assert not state_file.exists()


def test_shadow_reconcile_rejects_bad_targets(tmp_path):
    bad = tmp_path / "targets.json"
    bad.write_text('{"weights": {"600519.SH": 0.3}}', encoding="utf-8")  # 权重和≠1
    with pytest.raises(ValueError, match="权重和"):
        run_shadow_reconcile(
            targets_path=str(bad),
            qmt_path=r"C:\qmt\userdata_mini",
            account_id="12345678",
            simulated=True,
            state_file=str(tmp_path / "s.json"),
        )
