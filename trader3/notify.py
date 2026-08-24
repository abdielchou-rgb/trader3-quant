"""
3号交易员 — 通知推送通道

按可用性依次尝试多通道，全部失败仅记录日志（不抛出，保证日报流程不中断）：
  1. Server酱（环境变量 SC_SEND_KEY）——微信推送
  2. SMTP 邮件（SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASS/SMTP_TO）
  3. 兜底：仅日志

用法:
    from trader3.notify import send_notification
    send_notification("3号交易员日报 2026-08-24", "内容正文...")
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.text import MIMEText

import requests

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 15


def _send_serverchan(title: str, content: str) -> dict[str, str]:
    key = os.environ.get("SC_SEND_KEY", "").strip()
    if not key:
        return {"channel": "serverchan", "ok": False, "detail": "SC_SEND_KEY 未设置"}
    resp = requests.get(
        f"https://sctapi.ftqq.com/{key}.send",
        params={"title": title[:64], "desp": content},
        timeout=_TIMEOUT_SECONDS,
    )
    ok = resp.status_code == 200
    return {"channel": "serverchan", "ok": ok, "detail": f"http {resp.status_code}"}


def _send_smtp(title: str, content: str) -> dict[str, str]:
    host = os.environ.get("SMTP_HOST", "").strip()
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASS", "").strip()
    to_addr = os.environ.get("SMTP_TO", "").strip()
    port = int(os.environ.get("SMTP_PORT", "465") or 465)
    missing = [k for k, v in (("SMTP_HOST", host), ("SMTP_USER", user),
                              ("SMTP_PASS", password), ("SMTP_TO", to_addr)) if not v]
    if missing:
        return {"channel": "smtp", "ok": False,
                "detail": f"缺少环境变量: {','.join(missing)}"}

    msg = MIMEText(content, "plain", "utf-8")
    msg["Subject"] = title
    msg["From"] = user
    msg["To"] = to_addr
    with smtplib.SMTP_SSL(host, port, timeout=_TIMEOUT_SECONDS) as srv:
        srv.login(user, password)
        srv.sendmail(user, [to_addr], msg.as_string())
    return {"channel": "smtp", "ok": True, "detail": f"sent to {to_addr}"}


def send_notification(title: str, content: str) -> list[dict]:
    """尝试所有已配置通道；未配置/失败仅告警。返回各通道状态列表。"""
    statuses: list[dict] = []
    for sender in (_send_serverchan, _send_smtp):
        try:
            st = sender(title, content)
        except Exception as e:  # 网络异常不中断主流程
            st = {"channel": sender.__name__, "ok": False, "detail": f"{type(e).__name__}: {e}"}
        statuses.append(st)
        if st["ok"]:
            logger.info("[notify] %s 推送成功: %s", st["channel"], title)
        else:
            logger.warning("[notify] %s 未送达: %s", st["channel"], st["detail"])

    if not any(s["ok"] for s in statuses):
        logger.warning("[notify] 所有通道未配置/失败，标题仅记日志: %s", title)
    return statuses
