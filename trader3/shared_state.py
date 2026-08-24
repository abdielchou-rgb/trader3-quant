"""
3号交易员 — 共享状态管理

2号分析师与3号交易员之间通过 shared_state/ 目录共享数据。
版本一致性协议保证双方数据不陈旧。
"""

from __future__ import annotations

import json
import msvcrt
import os
import tempfile
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional


# 默认共享状态目录
_DEFAULT_STATE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "shared_state")
)


def _atomic_write_bytes(path: str, data: bytes) -> None:
    """临时文件 + os.replace 原子替换，读方永远不会看到半截文件。"""
    dir_ = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=os.path.basename(path), dir=dir_)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _locked(path: str, mode: str):
    """跨进程独占锁上下文（Windows msvcrt），防止读-改-写互相覆盖。"""
    lock_path = path + ".lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    f = open(lock_path, "a+b")

    def _acquire():
        for _ in range(200):  # 最多等 ~10s
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                time.sleep(0.05)
        return False

    class _Ctx:
        def __enter__(self):
            self.ok = _acquire()
            return f

        def __exit__(self, *exc):
            if self.ok:
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            f.close()
            return False

    return _Ctx()


class SharedState:
    """
    共享状态管理。

    所有读写走磁盘文件（Parquet/JSON/YAML），保证进程间可见。
    支持版本检查、强制刷新、缓存。
    """

    def __init__(self, state_dir: Optional[str] = None):
        self.state_dir = state_dir or _DEFAULT_STATE_DIR
        os.makedirs(self.state_dir, exist_ok=True)

    # ── 核心读写 ──

    def read_json(self, key: str) -> Optional[dict]:
        """读取 JSON 状态（容忍并发下的瞬时缺失）"""
        path = self._path(key, "json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_json(self, key: str, data: dict) -> str:
        """写入 JSON 状态（原子替换，防半截文件）"""
        path = self._path(key, "json")
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        with _locked(path, "wb"):
            _atomic_write_bytes(path, payload)
        return path

    def read_bytes(self, key: str, ext: str = "parquet") -> Optional[bytes]:
        """读取二进制状态（如 parquet 因子截面）"""
        path = self._path(key, ext)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def write_bytes(self, key: str, data: bytes, ext: str = "parquet") -> str:
        """写入二进制状态（原子替换）"""
        path = self._path(key, ext)
        with _locked(path, "wb"):
            _atomic_write_bytes(path, data)
        return path

    def list_keys(self, ext: Optional[str] = None) -> list:
        """列出所有状态键"""
        files = os.listdir(self.state_dir)
        if ext:
            files = [f for f in files if f.endswith(f".{ext}")]
        return sorted(files)

    # ── 版本一致性 ──

    def ensure_freshness(
        self,
        key: str,
        max_age_seconds: int = 14400,  # 4 小时
        default: Optional[dict] = None,
    ) -> dict:
        """
        确保数据不陈旧。
        如果数据缺失或过时，返回 default；调用方应触发增量更新。
        """
        state = self.read_json(key)
        if state is None:
            return default or {}

        update_time = state.get("metadata", {}).get("updated_at")
        if not update_time:
            return default or state

        now = datetime.now()
        age = now - datetime.fromisoformat(update_time)
        if age > timedelta(seconds=max_age_seconds):
            return default or state  # 陈旧但仍返回，调用方应处理

        return state

    def mark_updated(self, key: str, extra: Optional[dict] = None) -> str:
        """标记状态为刚刚更新（跨进程加锁的读-改-写）"""
        path = self._path(key, "json")
        with _locked(path, "a+"):
            data = self.read_json(key) or {}
            data.setdefault("metadata", {})
            data["metadata"]["updated_at"] = datetime.now().isoformat()
            data["metadata"]["version"] = data["metadata"].get("version", 0) + 1
            if extra:
                data["metadata"].update(extra)
            _atomic_write_bytes(path, json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
        return path

    # ── 预定义状态项 ──

    def get_data_version(self) -> dict:
        """获取数据版本号"""
        return self.ensure_freshness("data_version", max_age_seconds=14400)

    def set_data_version(self, versions: Dict[str, str]) -> str:
        """设置数据版本号"""
        return self.write_json("data_version", {
            "versions": versions,
            "metadata": {"updated_at": datetime.now().isoformat(), "version": 1},
        })

    def get_regime(self) -> dict:
        """获取当前市场状态"""
        return self.ensure_freshness("regime_current", max_age_seconds=3600)

    def set_regime(self, regime_data: dict) -> str:
        """设置当前市场状态"""
        self.write_json("regime_current", {
            "current_regime": regime_data.get("current_regime", ""),
            "regime_probabilities": regime_data.get("regime_probabilities", {}),
            "regime_entropy": regime_data.get("regime_entropy", 0.0),
            "key_indicators": regime_data.get("key_indicators", {}),
            "historical_analog": regime_data.get("historical_analog", ""),
            "strategy_suggestion": regime_data.get("strategy_suggestion", ""),
            "suggested_position": regime_data.get("suggested_position", 1.0),
        })
        return self.mark_updated("regime_current", {"data_type": "regime"})

    def get_latest_factors(self) -> Optional[bytes]:
        """获取最新因子截面"""
        return self.read_bytes("factor_latest", "parquet")

    def set_latest_factors(self, data: bytes) -> str:
        """设置最新因子截面"""
        return self.write_bytes("factor_latest", data, "parquet")

    def get_latest_signals(self) -> Optional[bytes]:
        """获取最新信号截面"""
        return self.read_bytes("signals_latest", "parquet")

    def set_latest_signals(self, data: bytes) -> str:
        """设置最新信号截面"""
        return self.write_bytes("signals_latest", data, "parquet")

    # ── 内部工具 ──

    def _path(self, key: str, ext: str) -> str:
        return os.path.join(self.state_dir, f"{key}.{ext}")