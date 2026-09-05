#!/usr/bin/env python3
"""llm_sentiment / retrain_loop 模块测试（HTTP 全 mock，不联网）"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader3.v2.llm_sentiment import LLMScorer
from trader3.v2.retrain_loop import (
    RetrainConfig,
    _load_registry,
    load_current_scores,
    run_retrain,
    should_retrain,
)
from trader3.v2.sentiment import score_text

# ================= llm_sentiment =================

class TestParseScores:
    def test_plain_json_array(self):
        out = LLMScorer.parse_scores('[{"i":0,"s":1.5},{"i":1,"s":-2.0}]', 2)
        assert out == [1.5, -2.0]

    def test_code_fenced(self):
        content = '```json\n[{"i":0,"s":0.5},{"i":1,"s":-1.5}]\n```'
        assert LLMScorer.parse_scores(content, 2) == [0.5, -1.5]

    def test_noisy_prefix_suffix(self):
        content = '好的，结果如下：[{"i":0,"s":2}] 希望有帮助'
        assert LLMScorer.parse_scores(content, 1) == [2.0]

    def test_object_regex_fallback(self):
        content = '前置杂讯 {"i":0,"s":-0.7} {"i":1,"s":1.2} 后置'
        assert LLMScorer.parse_scores(content, 2) == [-0.7, 1.2]

    def test_clamped(self):
        assert LLMScorer.parse_scores('[{"i":0,"s":99.0}]', 1) == [3.0]
        assert LLMScorer.parse_scores('[{"i":0,"s":-99.0}]', 1) == [-3.0]

    def test_missing_index_returns_none(self):
        assert LLMScorer.parse_scores('[{"i":0,"s":1.0}]', 3) is None
        assert LLMScorer.parse_scores(None, 2) is None
        assert LLMScorer.parse_scores("完全不是json", 1) is None


class TestLLMScorerDegradation:
    def test_no_key_pure_lexicon(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        s = LLMScorer(cache_dir=str(tmp_path))
        scores = s.score_batch(["业绩超预期 中标", "立案调查 预亏"])
        assert len(scores) == 2
        assert scores[0] > 0 > scores[1]
        assert s.fallback_count == 2 and s.model_used is None

    def test_cache_hit_skips_http(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        s = LLMScorer(cache_dir=str(tmp_path))
        t = "回购增持利好"
        s._cache_put(t, 2.4)
        called = {"http": False}

        class Boom:
            def post(self, *a, **k):  # pragma: no cover — 不应被调用
                called["http"] = True
                raise AssertionError("cache hit 不应发 HTTP")
        import httpx
        monkeypatch.setattr(httpx, "post", Boom().post)
        assert s.score_batch([t]) == [2.4]
        assert not called["http"]

    def test_http_success_flow(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        s = LLMScorer(cache_dir=str(tmp_path), min_interval_s=0.0)

        class FakeResp:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {
                    "content": '[{"i":0,"s":2.5},{"i":1,"s":-1.0}]'}}]}

        import httpx
        calls = {"n": 0}

        def fake_post(url, headers=None, json=None, timeout=None):
            calls["n"] += 1
            assert "openrouter.ai" in url
            assert headers["Authorization"] == "Bearer test-key"
            return FakeResp()

        monkeypatch.setattr(httpx, "post", fake_post)
        got = s.score_batch(["利好A", "利空B"])
        assert got == [2.5, -1.0]
        assert s.model_used == s.models[0]
        assert calls["n"] == 1
        # 二次调用走缓存
        assert s.score_batch(["利好A", "利空B"]) == [2.5, -1.0]
        assert calls["n"] == 1

    def test_429_retry_then_next_model(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        s = LLMScorer(cache_dir=str(tmp_path), min_interval_s=0.0,
                      max_retries=1, models=["bad/model:free", "good/model:free"])

        class R429:
            status_code = 429
            text = "rate limited"

            def json(self):
                return {}

        class R200:
            status_code = 200

            def json(self):
                return {"choices": [{"message":
                                     {"content": '[{"i":0,"s":1.0}]'}}]}

        import httpx
        seq = []

        def fake_post(url, headers=None, json=None, timeout=None):
            seq.append(json["model"])
            return R429() if json["model"].startswith("bad") else R200()

        monkeypatch.setattr(httpx, "post", fake_post)
        monkeypatch.setattr(time, "sleep", lambda *_: None)
        assert s.score_batch(["标题"]) == [1.0]
        assert seq[0].startswith("bad") and s.model_used == "good/model:free"

    def test_total_failure_falls_back_to_lexicon(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        s = LLMScorer(cache_dir=str(tmp_path), min_interval_s=0.0,
                      max_retries=0, models=["x/y:free"])

        import httpx

        def fail_post(*a, **k):
            raise ConnectionError("network down")

        monkeypatch.setattr(httpx, "post", fail_post)
        monkeypatch.setattr(time, "sleep", lambda *_: None)
        got = s.score_batch(["重大利好：中标"])
        assert got[0] == pytest.approx(score_text("重大利好：中标").score)

    def test_score_items_interface(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        class Item:
            title = "涨停 利好"
        s = LLMScorer(cache_dir=str(tmp_path))
        assert s.score_items([Item(), Item()]) > 0


# ================= retrain_loop =================

def make_ic_csv(tmp_path, values, name="f1"):
    fw = tmp_path / "factor_watch"
    fw.mkdir(parents=True, exist_ok=True)
    p = fw / f"{name}_ic.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write("date,ic\n")
        for i, v in enumerate(values):
            f.write(f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d},{v}\n")
    return str(p)


def make_build_data(n_days=420, n_assets=25, seed=21):
    rng = np.random.default_rng(seed)
    assets = [f"A{i:02d}" for i in range(n_assets)]
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    f1 = pd.DataFrame(rng.normal(size=(n_days, n_assets)), index=dates, columns=assets)
    fwd = f1.shift(-1) * 0.01
    fwd = fwd.fillna(pd.DataFrame(rng.normal(0, 0.01, (n_days, n_assets)),
                                  index=dates, columns=assets))
    return lambda: ({"sig": f1}, fwd)


class TestShouldRetrain:
    def _cfg(self, tmp_path, **kw):
        kw.setdefault("cooldown_hours", 0.0)
        return RetrainConfig(
            state_dir=str(tmp_path),
            registry_dir=str(tmp_path / "model_registry"), **kw)

    def test_decay_triggers(self, tmp_path):
        vals = [0.05] * 20 + [-0.04] * 20          # 近窗均值 -0.04 < 0
        cfg = self._cfg(tmp_path, ic_csv=make_ic_csv(tmp_path, vals))
        trig, why = should_retrain(cfg)
        assert trig and "衰减触发" in why

    def test_healthy_no_trigger(self, tmp_path):
        vals = [0.03] * 40
        cfg = self._cfg(tmp_path, ic_csv=make_ic_csv(tmp_path, vals))
        trig, why = should_retrain(cfg)
        assert not trig and "健康" in why

    def test_insufficient_samples(self, tmp_path):
        cfg = self._cfg(tmp_path, ic_csv=make_ic_csv(tmp_path, [0.01] * 5),
                        decay_window=20)
        trig, why = should_retrain(cfg)
        assert not trig and "样本不足" in why

    def test_cooldown_blocks(self, tmp_path):
        vals = [-0.05] * 40
        csv = make_ic_csv(tmp_path, vals)
        cfg = self._cfg(tmp_path, ic_csv=csv, cooldown_hours=24)
        # 先写一条"刚刚晋升过"的注册表
        reg_dir = str(tmp_path / "model_registry")
        os_mkdir = __import__("os").makedirs(reg_dir, exist_ok=True)
        del os_mkdir
        reg = {"version": 1, "entries": {"f1": {"promoted_at_epoch": __import__("time").time()}}}
        with open(f"{reg_dir}/registry.json", "w", encoding="utf-8") as f:
            json.dump(reg, f)
        trig, why = should_retrain(cfg)
        assert not trig and "冷却" in why


class TestRunRetrain:
    def _cfg(self, tmp_path, **kw):
        kw.setdefault("cooldown_hours", 0.0)
        return RetrainConfig(
            state_dir=str(tmp_path),
            registry_dir=str(tmp_path / "model_registry"),
            ensemble_kwargs={"min_train": 150, "retrain_every": 90}, **kw)

    def test_first_promotion(self, tmp_path):
        cfg = self._cfg(tmp_path, factor_name="fx")
        report = run_retrain(make_build_data(), config=cfg, force=True)
        assert report.triggered and report.trained and report.promoted
        assert report.old_rank_ic is None
        assert report.artifact_path and report.artifact_path.endswith(".npz")
        reg = _load_registry(str(tmp_path / "model_registry"))
        e = reg["entries"]["fx"]
        assert e["oos_rank_ic"] == pytest.approx(report.new_rank_ic)
        assert "artifact" in e and "promoted_at" in e

    def test_gate_pass_and_reject(self, tmp_path):
        cfg = self._cfg(tmp_path, factor_name="fy", gate_tolerance=0.10)
        r1 = run_retrain(make_build_data(seed=5), config=cfg, force=True)
        assert r1.promoted and r1.old_rank_ic is None
        # 第二次训练不同数据 → 新IC若 ≥ 旧−tol 则晋升；否则拒绝。两种都必须有明确 reason。
        r2 = run_retrain(make_build_data(seed=99), config=cfg, force=True)
        if r2.promoted:
            assert r2.new_rank_ic >= r2.old_rank_ic - 0.10
        else:
            assert "门禁拒绝" in r2.reason
            # 注册表未被覆盖
            reg = _load_registry(str(tmp_path / "model_registry"))
            assert reg["entries"]["fy"]["oos_rank_ic"] == pytest.approx(r1.new_rank_ic)

    def test_gate_strict_reject(self, tmp_path):
        # tolerance=负无穷等价：把旧IC设为极高值，新模型必被拒
        cfg = self._cfg(tmp_path, factor_name="fz", gate_tolerance=-1e9)
        reg_dir = str(tmp_path / "model_registry")
        __import__("os").makedirs(reg_dir, exist_ok=True)
        reg = {"version": 1, "entries": {"fz": {"oos_rank_ic": 9.9}}}
        with open(f"{reg_dir}/registry.json", "w", encoding="utf-8") as f:
            json.dump(reg, f)
        report = run_retrain(make_build_data(seed=3), config=cfg, force=True)
        assert report.trained and not report.promoted
        assert "门禁拒绝" in report.reason
        reg2 = _load_registry(reg_dir)
        assert reg2["entries"]["fz"]["oos_rank_ic"] == 9.9, "旧模型必须原样保留"

    def test_not_triggered_short_circuits(self, tmp_path):
        vals = [0.03] * 40   # 健康
        cfg = self._cfg(tmp_path, factor_name="fh",
                        ic_csv=make_ic_csv(tmp_path, vals, name="fh"))
        report = run_retrain(lambda: ({}, pd.DataFrame()), config=cfg)
        assert not report.triggered and not report.trained
        assert "健康" in report.reason

    def test_load_current_scores_roundtrip(self, tmp_path):
        cfg = self._cfg(tmp_path, factor_name="fs")
        run_retrain(make_build_data(seed=8), config=cfg, force=True)
        df = load_current_scores("fs",
                                 RetrainConfig(registry_dir=str(tmp_path / "model_registry")))
        assert df is not None and not df.empty
        assert isinstance(df.index[0], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
