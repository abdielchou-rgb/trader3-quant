#!/usr/bin/env python3
"""
kill_criteria 模块测试
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tests"))

from trader3.v2.kill_criteria import (  # noqa: E402
    auto_veto_factor,
    evaluate_kill_criteria,
    run_kill_check,
)


def _make_history(n=25, mean_ic=0.0, seed=7):
    dates = [f"d{i}" for i in range(n)]
    return [
        {"date": d, "ic": float(ic)}
        for d, ic in zip(dates, np.random.normal(mean_ic, 0.02, size=n), strict=True)
    ]


def _make_history_neg(n=25, mean_ic=-0.05, seed=7):
    dates = [f"d{i}" for i in range(n)]
    return [
        {"date": d, "ic": float(ic)}
        for d, ic in zip(dates, np.random.normal(mean_ic, 0.02, size=n), strict=True)
    ]


class TestEvaluate:
    def test_decayed_signal_triggers(self):
        hist = _make_history_neg(n=25, mean_ic=-0.05)
        r = evaluate_kill_criteria(hist)
        assert r["triggered"]
        assert "均值" in r["reason"] or "IC" in r["reason"] or "连续" in r["reason"]

    def test_consecutive_negative_triggers(self):
        hist = [{"date": f"d{i}", "ic": -0.01} for i in range(15)]
        r = evaluate_kill_criteria(hist)
        assert r["triggered"]
        assert r["neg_streak"] >= 10

    def test_insufficient_samples_no_trigger(self):
        r = evaluate_kill_criteria([{"date": "d0", "ic": -0.5}])
        assert not r["triggered"]

    def test_empty_history(self):
        r = evaluate_kill_criteria([])
        assert not r["triggered"]

    def test_mixed_recent_decay(self):
        hist = _make_history(15, 0.04) + _make_history_neg(15, -0.06, seed=99)
        combined = hist[:15] + hist[15:]
        r = evaluate_kill_criteria(combined, decay_window=10)
        assert r["triggered"], "近期衰减应触发"


class TestAutoVeto:
    def test_marks_and_preserves_others(self, tmp_path):
        sel = tmp_path / "selected.json"
        entries = [
            {"expr": "good_factor", "score": 1.0},
            {"expr": "bad_factor", "score": 0.5},
            {"expr": "another", "score": 0.3},
        ]
        sel.write_text(json.dumps(entries), encoding="utf-8")
        modified = auto_veto_factor(str(sel), "bad_factor", "test reason")
        assert modified is True
        data = json.loads(sel.read_text(encoding="utf-8"))
        assert data[1]["oos_veto"] is True
        assert data[1]["veto_source"] == "factor_watch_decay"
        assert "oos_veto" not in data[0]
        assert "oos_veto" not in data[2]

    def test_idempotent_already_vetoed(self, tmp_path):
        sel = tmp_path / "selected.json"
        entries = [{"expr": "f", "oos_veto": True}]
        sel.write_text(json.dumps(entries), encoding="utf-8")
        modified = auto_veto_factor(str(sel), "f", "test reason")
        assert modified is False

    def test_missing_file_returns_false(self, tmp_path):
        assert auto_veto_factor(str(tmp_path / "nonexistent.json"), "x", "y") is False


class TestRunKillCheck:
    def test_disabled_by_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KILL_CRITERIA_ENABLED", "false")
        r = run_kill_check(str(tmp_path), str(tmp_path))
        assert r["enabled"] is False

    def test_end_to_end_decay_triggers_veto(self, tmp_path, monkeypatch):
        monkeypatch.delenv("KILL_CRITERIA_ENABLED", raising=False)
        os.makedirs(os.path.join(tmp_path, "state", "factor_watch"), exist_ok=True)
        ic_path = os.path.join(tmp_path, "state", "factor_watch", "f1_ic.csv")
        os.makedirs(os.path.dirname(ic_path), exist_ok=True)
        with open(ic_path, "w") as f:
            f.write("date,ic\n")
            for i in range(20):
                f.write(f"2026-08-{i+1:02d},{-0.05 - i * 0.001}\n")

        monkeypatch.setattr(
            "trader3.v2.kill_criteria.evaluate_kill_criteria",
            lambda *a, **k: {
                "enabled": True,
                "triggered": True,
                "factor_name": "f1",
                "reason": "测试衰减触发",
                "ic_mean_recent": -0.05,
                "neg_streak": 20,
                "total_obs": 20,
            },
        )

        sel_path = os.path.join(tmp_path, "selected.json")
        with open(sel_path, "w", encoding="utf-8") as f:
            json.dump([{"expr": "sub(log(vwap), log(close))"}], f, ensure_ascii=False, indent=2)

        import trader3.v2.kill_criteria as kc

        r = kc.run_kill_check(
            str(tmp_path / "state"), str(tmp_path), factor_name="f1", expr="sub(log(vwap), log(close))"
        )
        assert r["triggered"] is True
        assert r.get("auto_veto_applied") is True
        updated = json.load(open(sel_path, encoding="utf-8"))
        assert updated[0]["oos_veto"] is True


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
