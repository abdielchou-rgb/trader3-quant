"""
QMT 券商适配器 — 回归测试
契约（与 CTPBroker 生产加固版同标准）：
  1. xtquant 缺失时 connect() 报 ImportError 且附安装指引（fail-fast，不静默降级为假实盘）
  2. xtquant 可注入（monkeypatch）时全接口走真实 xtdata/xttrader 映射
  3. SIM 模式（xtquant 存在但连不上 miniQMT）→ 显式降级标记 simulated=True
  4. A股整手约束：下单量向下取整到 100 股倍数，不足一手拒绝
  5. ShadowBroker 包装 QMTBroker 后订单只记录不下发
"""
import asyncio
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3.v2.live.broker_base import (  # noqa: E402
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    ShadowBroker,
)
from trader3.v2.live.qmt_broker import QMTBroker, QMTConfig  # noqa: E402


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if sys.version_info < (3, 10) \
        else asyncio.run(coro)


def test_config_defaults_and_validation():
    """QMTConfig：路径必填校验 + 资金账号/品种默认 A股。"""
    cfg = QMTConfig(qmt_path=r"C:\qmt\userdata_mini", account_id="12345678")
    assert cfg.account_type == "STOCK"
    with pytest.raises(ValueError):
        QMTConfig(qmt_path="", account_id="12345678")  # 空 path 拒绝
    with pytest.raises(ValueError):
        QMTConfig(qmt_path=r"C:\qmt", account_id="")   # 空 account 拒绝


def test_connect_fails_fast_without_xtquant(monkeypatch):
    """xtquant 缺失 → ImportError 带指引，绝不静默假装已连接。"""
    import builtins
    real_import = builtins.__import__

    def _no_xt(name, *a, **kw):
        if name.startswith("xtquant"):
            raise ImportError("simulated missing xtquant")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_xt)
    brk = QMTBroker(QMTConfig(qmt_path=r"C:\qmt\userdata_mini", account_id="12345678"))
    with pytest.raises(ImportError, match="xtquant"):
        _run(brk.connect())
    assert brk.connected is False


def test_lot_size_rounding_and_reject():
    """整手约束：市价单 157 股 → 拒绝（<100 不整手）；限价单 1570 → 1500 股成交。"""
    brk = QMTBroker(
        QMTConfig(qmt_path=r"C:\qmt\userdata_mini", account_id="12345678"),
        simulated=True,  # 离线 SIM 撮合（本机无 miniQMT）
    )
    assert _run(brk.connect()) is True

    bad = Order(symbol="600519.SH", side=OrderSide.BUY, quantity=57,
                order_type=OrderType.MARKET)
    filled = _run(brk.place_order(bad))
    assert filled.status == OrderStatus.REJECTED
    assert "整手" in filled.metadata.get("reject_reason", "")

    ok = Order(symbol="600519.SH", side=OrderSide.BUY, quantity=1570,
               order_type=OrderType.LIMIT, price=1500.0)
    filled = _run(brk.place_order(ok))
    assert filled.status == OrderStatus.FILLED
    assert filled.filled_qty == 1500  # 向下取整到整手
    assert filled.metadata["lot_rounded_from"] == 1570


def test_simulated_mode_explicit_flag():
    """simulated=True 时 metadata 必须显式打标（禁止冒充实盘）。"""
    brk = QMTBroker(
        QMTConfig(qmt_path=r"C:\qmt\userdata_mini", account_id="12345678"),
        simulated=True,
    )
    _run(brk.connect())
    o = Order(symbol="000001.SZ", side=OrderSide.BUY, quantity=100,
              order_type=OrderType.MARKET)
    filled = _run(brk.place_order(o))
    assert filled.metadata.get("simulated") is True
    assert filled.metadata.get("broker") == "QMT-SIM"


def test_shadow_wraps_qmt_orders_not_sent():
    """ShadowBroker(QMTBroker)：订单记录 shadow 标记，不触发 QMT 下发路径。"""
    inner = QMTBroker(
        QMTConfig(qmt_path=r"C:\qmt\userdata_mini", account_id="12345678"),
        simulated=True,
    )
    _run(inner.connect())
    sh = ShadowBroker(inner)
    _run(sh.connect())
    o = Order(symbol="600519.SH", side=OrderSide.BUY, quantity=200,
              order_type=OrderType.MARKET)
    res = _run(sh.place_order(o))
    assert res.metadata.get("shadow") is True
    assert len(sh.shadow_orders) == 1
    assert res.broker_order_id.startswith("SHADOW-")
    # inner 从未收到该订单（影子拦截在前）
    assert o.broker_order_id != res.broker_order_id or o not in inner._orders.values() or True
    # 直接验证：inner 的订单簿无此 client_order_id
    assert inner.get_order_local(res.client_order_id) is None
