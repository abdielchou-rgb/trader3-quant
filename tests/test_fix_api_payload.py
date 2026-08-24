"""
API 请求白名单测试 (M6 hardening):

  - payload 模型 extra="forbid" → 未知字段 422
  - model_dump(exclude_unset=True) → 未传字段不出现在透传 kwargs（保持工具默认语义）
  - 数值边界轻量校验 → 越界 422
  - 配置 TRADER3_API_KEY 后 X-API-Key 鉴权仍生效
  - /regime 允许空请求体，透传 {}

用 StubTrader3 记录 kwargs，避免真实回测耗时。
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi 未安装，跳过")
pytest.importorskip("httpx", reason="httpx 未安装，跳过")

from fastapi.testclient import TestClient  # noqa: E402

from trader3.base_tool import Trader3Response  # noqa: E402


class _StubGates:
    enabled = False

    def summary(self) -> dict:
        return {}


class StubTrader3:
    """记录 (tool_name, kwargs) 的桩引擎"""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.gates = _StubGates()

    def __getattr__(self, name):
        def _call(**kwargs):
            self.calls.append((name, dict(kwargs)))
            return Trader3Response(success=True, summary="stub")

        return _call

    def list_tools(self) -> list:
        return []


@pytest.fixture()
def client(monkeypatch):
    from trader3.api import server

    stub = StubTrader3()
    monkeypatch.setattr(server, "_t3", stub)
    yield TestClient(server.app), stub


def test_unknown_field_rejected_422(client):
    c, _ = client
    resp = c.post("/backtest", json={"hacker": 1})
    assert resp.status_code == 422


def test_known_fields_forwarded(client):
    c, stub = client
    body = {
        "start_date": "2021-06-01",
        "factor_from_selected": 2,
        "universe": ["SH600000"],
    }
    resp = c.post("/backtest", json=body)
    assert resp.status_code == 200
    assert len(stub.calls) == 1
    tool_name, kwargs = stub.calls[0]
    assert tool_name == "run_backtest"
    # 收到的 kwargs 与请求等价
    assert kwargs == body
    # exclude_unset 语义：未传字段不在 kwargs 中（工具默认生效）
    assert "end_date" not in kwargs
    assert "benchmark" not in kwargs


def test_validation_bounds(client):
    c, _ = client
    resp = c.post("/wfa", json={"train_window": 5})
    assert resp.status_code == 422


def test_auth_still_enforced(client, monkeypatch):
    c, _ = client
    from trader3.api import server

    # 直接改模块属性 _API_KEY，等价于在 import 前设置 TRADER3_API_KEY
    monkeypatch.setattr(server, "_API_KEY", "secret-123")
    body = {"start_date": "2021-01-01"}
    assert c.post("/backtest", json=body).status_code == 401
    wrong = c.post(
        "/backtest", json=body, headers={"X-API-Key": "wrong-key"}
    )
    assert wrong.status_code == 401
    ok = c.post(
        "/backtest", json=body, headers={"X-API-Key": "secret-123"}
    )
    assert ok.status_code == 200


def test_regime_empty_body_ok(client):
    c, stub = client
    resp = c.post("/regime", json={})
    assert resp.status_code == 200
    tool_name, kwargs = stub.calls[-1]
    assert tool_name == "diagnose_market_regime"
    assert kwargs == {}
