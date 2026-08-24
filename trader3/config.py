"""
3号交易员 — 配置管理
"""

from __future__ import annotations

import os
from typing import Any

import yaml

# 默认配置路径（相对于项目根目录）
_DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")


def load_yaml(path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(relative_path: str, base_dir: str | None = None) -> str:
    """解析配置路径（支持相对路径和绝对路径）"""
    if os.path.isabs(relative_path):
        return relative_path
    base = base_dir or _DEFAULT_CONFIG_DIR
    resolved = os.path.abspath(os.path.join(base, relative_path))
    if not os.path.exists(resolved):
        # 再尝试相对于项目根目录
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        resolved = os.path.join(project_root, relative_path)
    return resolved


class ConfigManager:
    """配置管理器（YAML + 环境变量覆盖）"""

    def __init__(self, config_dir: str | None = None):
        self.config_dir = config_dir or _DEFAULT_CONFIG_DIR
        self._cache: dict[str, dict] = {}

    def load(self, name: str, env_prefix: str = "T3_") -> dict:
        """
        加载配置（支持缓存和环境变量覆盖）
        环境变量覆盖规则：T3_{FLATTENED_KEY}，如 T3_MAX_SINGLE_WEIGHT
        """
        if name in self._cache:
            return self._cache[name]

        path = resolve_path(name if name.endswith(".yaml") else f"{name}.yaml", self.config_dir)
        if not os.path.exists(path):
            return {}

        config = load_yaml(path)
        self._apply_env_overrides(config, env_prefix)
        self._cache[name] = config
        return config

    def merge(
        self,
        base: dict,
        override: dict,
        strategy: str = "deep",
    ) -> dict:
        """合并配置（base 被 override 覆盖）"""
        if strategy == "deep":
            return self._deep_merge(base, override)
        return {**base, **override}

    def _apply_env_overrides(self, config: dict, prefix: str, parent_key: str = "") -> None:
        """递归应用环境变量覆盖"""
        for key, value in list(config.items()):
            full_key = f"{parent_key}_{key}" if parent_key else key
            env_key = f"{prefix}{full_key.upper()}"
            env_value = os.environ.get(env_key)
            if env_value is not None:
                config[key] = self._parse_env_value(env_value)
            if isinstance(value, dict):
                self._apply_env_overrides(value, prefix, full_key)

    def _parse_env_value(self, value: str) -> Any:
        """解析环境变量值（支持数字/bool）"""
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """递归合并字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
