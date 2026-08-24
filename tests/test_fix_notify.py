"""通知通道回归测试 — 全离线（monkeypatch 网络/SMTP）。"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trader3 import notify  # noqa: E402


def test_no_env_log_only(monkeypatch):
    for k in ("SC_SEND_KEY", "SMTP_HOST", "SMTP_USER", "SMTP_PASS", "SMTP_TO"):
        monkeypatch.delenv(k, raising=False)
    statuses = notify.send_notification("t", "c")
    assert all(not s["ok"] for s in statuses)
    assert {s["channel"] for s in statuses} == {"serverchan", "smtp"}


def test_serverchan_success(monkeypatch):
    monkeypatch.setenv("SC_SEND_KEY", "abc123")
    captured = {}

    class _Resp:
        status_code = 200

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _Resp()

    monkeypatch.setattr(notify.requests, "get", fake_get)
    statuses = notify.send_notification("标题", "正文")
    sc = [s for s in statuses if s["channel"] == "serverchan"][0]
    assert sc["ok"] is True
    assert "abc123" in captured["url"]
    assert captured["params"]["title"] == "标题"


def test_serverchan_network_error_does_not_raise(monkeypatch):
    monkeypatch.setenv("SC_SEND_KEY", "k")

    def boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr(notify.requests, "get", boom)
    statuses = notify.send_notification("t", "c")
    assert all(not s["ok"] for s in statuses)  # 不抛出


def test_smtp_success(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.test")
    monkeypatch.setenv("SMTP_PORT", "465")
    monkeypatch.setenv("SMTP_USER", "u@test")
    monkeypatch.setenv("SMTP_PASS", "p")
    monkeypatch.setenv("SMTP_TO", "to@test")

    sent = {}

    class FakeSrv:
        def __init__(self, host, port, timeout=None):
            sent["host"] = host

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, u, p):
            sent["login"] = u

        def sendmail(self, frm, to, raw):
            sent["to"] = to
            sent["raw"] = raw

    monkeypatch.setattr(notify.smtplib, "SMTP_SSL", FakeSrv)
    statuses = notify.send_notification("邮件标题", "邮件正文")
    smtp = [s for s in statuses if s["channel"] == "smtp"][0]
    assert smtp["ok"] is True
    assert sent["to"] == ["to@test"]
    import email as _email
    parsed = _email.message_from_string(sent["raw"])
    str(parsed["Subject"])  # Subject 存在（utf-8 base64 编码）
    body = parsed.get_payload(decode=True).decode("utf-8")
    assert "邮件正文" in body
