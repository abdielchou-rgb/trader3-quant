"""
全局配置（生产加固）：从环境变量加载密钥与连接参数，集中管理。

用法：
    from trader3.v2.config import Settings
    settings = Settings.load()            # 读 os.environ
    settings.llm_client(...)              # 构造 OpenRouter 客户端（有 Key 时）
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    openrouter_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-4o-mini"
    ctp_front: str = "tcp://180.168.146.187:10130"
    ctp_md_front: str = "tcp://180.168.146.187:10131"
    ctp_broker_id: str = "9999"
    ctp_user_id: str = ""
    ctp_password: str = ""
    ctp_auth_code: str = "0000000000000000"
    ctp_investor_id: str = ""
    ctp_app_id: str = "simnow_client_test"
    paper_trade: bool = True
    log_level: str = "INFO"
    # 量化主链路
    quant_method: str = "ic_weighted"
    kill_switch_dd: float = 0.05
    nested_execution: bool = False
    factory_refresh: bool = False
    factory_every_n: int = 5
    legacy_daily: bool = False
    lookback_days: int = 250
    universe: str = ""          # 逗号分隔标的；非空则启用自主面板
    shadow_mode: bool = False   # 影子模式（真实券商仅空跑）

    @classmethod
    def load(cls, environ: dict | None = None) -> Settings:
        e = environ if environ is not None else dict(os.environ)

        def g(key: str, default: str) -> str:
            v = e.get(key)
            return v if v is not None else default

        return cls(
            openrouter_api_key=g("OPENROUTER_API_KEY", ""),
            llm_base_url=g("LLM_BASE_URL", cls.llm_base_url),
            llm_model=g("LLM_MODEL", cls.llm_model),
            ctp_front=g("CTP_FRONT", cls.ctp_front),
            ctp_md_front=g("CTP_MD_FRONT", cls.ctp_md_front),
            ctp_broker_id=g("CTP_BROKER_ID", cls.ctp_broker_id),
            ctp_user_id=g("CTP_USER_ID", ""),
            ctp_password=g("CTP_PASSWORD", ""),
            ctp_auth_code=g("CTP_AUTH_CODE", cls.ctp_auth_code),
            ctp_investor_id=g("CTP_INVESTOR_ID", ""),
            ctp_app_id=g("CTP_APP_ID", cls.ctp_app_id),
            paper_trade=g("PAPER_TRADE", "true").strip().lower() in ("1", "true", "yes"),
            log_level=g("LOG_LEVEL", cls.log_level),
            quant_method=g("QUANT_METHOD", cls.quant_method),
            kill_switch_dd=float(g("KILL_SWITCH_DD", str(cls.kill_switch_dd))),
            nested_execution=g("NESTED_EXECUTION", "false").strip().lower()
            in ("1", "true", "yes"),
            factory_refresh=g("FACTORY_REFRESH", "false").strip().lower()
            in ("1", "true", "yes"),
            factory_every_n=int(g("FACTORY_EVERY_N", str(cls.factory_every_n))),
            legacy_daily=g("LEGACY_DAILY", "false").strip().lower() in ("1", "true", "yes"),
            lookback_days=int(g("LOOKBACK_DAYS", str(cls.lookback_days))),
            universe=g("UNIVERSE", cls.universe),
            shadow_mode=g("SHADOW_MODE", "false").strip().lower() in ("1", "true", "yes"),
        )

    def ctp_config(self) -> dict:
        return {
            "ctp_front": self.ctp_front,
            "ctp_md_front": self.ctp_md_front,
            "ctp_broker_id": self.ctp_broker_id,
            "ctp_user_id": self.ctp_user_id,
            "ctp_password": self.ctp_password,
            "ctp_auth_code": self.ctp_auth_code,
            "ctp_investor_id": self.ctp_investor_id,
            "ctp_app_id": self.ctp_app_id,
        }

    def universe_list(self) -> list[str]:
        return [u.strip() for u in self.universe.split(",") if u.strip()]

    def has_llm(self) -> bool:
        return bool(self.openrouter_api_key)

    def make_llm_fn(self):
        """构造 factor_factory 可用的 llm_fn(prompt)->str；无 Key 返回 None。"""
        if not self.openrouter_api_key:
            return None
        try:
            from openai import OpenAI
        except ImportError:
            return None
        client = OpenAI(base_url=self.llm_base_url, api_key=self.openrouter_api_key)

        def _fn(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return resp.choices[0].message.content or ""
        return _fn
