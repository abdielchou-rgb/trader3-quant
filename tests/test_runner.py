"""顶层编排器/CLI 集成测试（离线：SIMULATED 券商 + 合成面板；真实 Qlib 数据可选）。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from trader3.v2.config import Settings
from trader3.v2.live.broker_base import ShadowBroker
from trader3.v2.runner import (
    build_broker,
    run_once,
    summarize,
)

_ROOT = Path(__file__).resolve().parent.parent
QLIB_BIN = os.environ.get(
    "QLIB_BIN", str(_ROOT.parent / "2hao-analyst" / "data" / "qlib_bin")
)


def _settings(**kw) -> Settings:
    base = {
        "OPENROUTER_API_KEY": "",
        "PAPER_TRADE": "true",
        "UNIVERSE": "A0,A1,A2,A3,A4,A5",
        "LOOKBACK_DAYS": "140",
        "NESTED_EXECUTION": "false",
        "FACTORY_REFRESH": "false",
        "LEGACY_DAILY": "false",
        "SHADOW_MODE": "false",
    }
    base.update(kw)
    return Settings.load(base)


def test_build_broker_selection():
    s = _settings()
    brk = build_broker(s)
    assert isinstance(brk, ShadowBroker) is False  # paper_trade → SIMULATED CTPBroker
    sh = build_broker(s, shadow=True)
    assert isinstance(sh, ShadowBroker)


def test_run_once_offline_auto_panel():
    s = _settings(NESTED_EXECUTION="true", FACTORY_REFRESH="false")
    run = asyncio.run(run_once(s, universe=["A0", "A1", "A2", "A3", "A4", "A5"],
                               kind="manual"))
    assert not run.quant["weights"].empty
    assert run.quant["meta"]["n_orders"] >= 0
    # 嵌套执行开启 → 应有 execution_plan
    assert "execution_plan" in run.quant["meta"]


def test_run_once_shadow_does_not_fill_real():
    s = _settings(SHADOW_MODE="true")
    run = asyncio.run(run_once(s, universe=[f"A{i}" for i in range(6)], kind="manual"))
    # 影子模式：所有订单带 shadow 标记，且未产生真实成交
    assert run.quant["meta"]["n_orders"] >= 0
    assert all(o.metadata.get("shadow") for o in run.quant["orders"])


def test_summarize_runs():
    s = _settings(NESTED_EXECUTION="true")
    run = asyncio.run(run_once(s, universe=[f"A{i}" for i in range(6)], kind="manual"))
    txt = summarize(run)
    assert "量化主链路运行" in txt
    assert "嵌套执行" in txt


def test_run_once_with_qlib_real_data():
    if not os.path.isdir(QLIB_BIN):
        import pytest

        pytest.skip("QLIB_BIN 不存在")
    s = _settings(
        DATA_SOURCE="qlib", QLIB_URI=QLIB_BIN, LOOKBACK_DAYS="200",
        NESTED_EXECUTION="true",
        UNIVERSE="bj430017,bj430047,bj430090,bj430139,bj430198,bj430300",
    )
    run = asyncio.run(run_once(s, kind="manual"))
    assert not run.quant["weights"].empty
    assert set(run.quant["weights"].index).issubset(
        {"bj430017", "bj430047", "bj430090", "bj430139", "bj430198", "bj430300"}
    )
    assert "execution_plan" in run.quant["meta"]
