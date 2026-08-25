"""
FactorStore — 因子矩阵磁盘缓存。

按 (expr, universe, start, end, data_version) 五元组缓存因子截面矩阵，
避免进化/回测过程中重复计算同一表达式。

布局:
    shared_state/factor_store/<sha1>.npz        matrix(T,N) + dates(T) + codes(N)
    shared_state/factor_store/<sha1>.meta.json  expr/universe/period/data_version/created_at/sha1

data_version 取自 SharedState 的 data_version.json（versions.qlib_bin 键），
读不到时视为 "unknown"。data_version 变化后旧缓存自然失配（key 不同 + meta 校验双重保险）。
所有写入均为 原子写（tmp + os.replace），读方永远看不到半截文件。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from datetime import datetime

import numpy as np

from trader3.shared_state import SharedState

# 项目根 shared_state/（与 trader3.shared_state._DEFAULT_STATE_DIR 同一定位）
_DEFAULT_STATE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "shared_state")
)

_UNKNOWN_DATA_VERSION = "unknown"


def _atomic_write_bytes(path: str, data: bytes) -> None:
    """临时文件 + os.replace 原子替换（与 shared_state 同一策略）。"""
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


class FactorStore:
    """因子矩阵缓存仓库。线程安全不保证，跨进程安全（原子写 + 只读命中）。"""

    def __init__(self, root: str | None = None, state_dir: str | None = None):
        """
        root: 缓存目录，默认 <shared_state>/factor_store。
        state_dir: SharedState 目录（读取 data_version.json），默认取 root 的上级目录。
        """
        if root is None:
            root = os.path.join(state_dir or _DEFAULT_STATE_DIR, "factor_store")
        self.root = os.path.abspath(str(root))
        self.state_dir = os.path.abspath(state_dir) if state_dir else os.path.dirname(self.root)
        os.makedirs(self.root, exist_ok=True)

    # ── data_version ──

    def data_version(self) -> str:
        """读 SharedState 的 data_version.json（qlib_bin 键），读不到用 'unknown'。"""
        try:
            state = SharedState(self.state_dir).read_json("data_version")
        except (OSError, ValueError):
            return _UNKNOWN_DATA_VERSION
        if isinstance(state, dict):
            versions = state.get("versions")
            if isinstance(versions, dict):
                v = versions.get("qlib_bin")
                if v:
                    return str(v)
        return _UNKNOWN_DATA_VERSION

    def key_for(
        self,
        expr: str,
        universe: str,
        start: str,
        end: str,
        data_version: str | None = None,
    ) -> str:
        """缓存键 = sha1(expr|universe|start|end|data_version)。"""
        dv = self.data_version() if data_version is None else data_version
        raw = f"{expr}|{universe}|{start}|{end}|{dv}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    # ── 路径 ──

    def _paths(self, sha1: str) -> tuple[str, str]:
        return (
            os.path.join(self.root, f"{sha1}.npz"),
            os.path.join(self.root, f"{sha1}.meta.json"),
        )

    # ── 核心读写 ──

    def save(
        self,
        expr: str,
        universe: str,
        start: str,
        end: str,
        matrix: np.ndarray,
        dates,
        codes,
    ) -> dict:
        """写入因子矩阵与元信息，返回 meta dict。"""
        matrix = np.asarray(matrix)
        dates_arr = np.asarray(dates)
        codes_arr = np.asarray(codes)
        if matrix.ndim != 2 or matrix.shape != (len(dates_arr), len(codes_arr)):
            raise ValueError(
                f"matrix 形状 {matrix.shape} 与 dates({len(dates_arr)})×codes({len(codes_arr)}) 不符"
            )

        dv = self.data_version()
        sha1 = self.key_for(expr, universe, start, end, data_version=dv)
        npz_path, meta_path = self._paths(sha1)

        buf = io.BytesIO()
        np.savez(buf, matrix=matrix, dates=dates_arr, codes=codes_arr)
        _atomic_write_bytes(npz_path, buf.getvalue())

        meta = {
            "expr": expr,
            "universe": universe,
            "period": [str(start), str(end)],
            "data_version": dv,
            "created_at": datetime.now().isoformat(),
            "sha1": sha1,
        }
        _atomic_write_bytes(
            meta_path, json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
        )
        return meta

    def load(self, expr: str, universe: str, start: str, end: str) -> dict | None:
        """
        命中返回 {matrix, dates, codes, meta}；未命中或 data_version 不一致返回 None。
        损坏的 npz/meta 一律安全返回 None，不抛异常。
        """
        sha1 = self.key_for(expr, universe, start, end)
        npz_path, meta_path = self._paths(sha1)
        if not os.path.exists(npz_path):
            return None

        try:
            with np.load(npz_path, allow_pickle=False) as z:
                matrix = np.array(z["matrix"])
                dates = [str(x) for x in z["dates"].tolist()]
                codes = [str(x) for x in z["codes"].tolist()]
        except Exception:
            return None

        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            return None

        # 双重校验：key 已含 data_version，这里再对 meta 防御性核对
        if meta.get("data_version") != self.data_version() or meta.get("sha1") != sha1:
            return None

        return {"matrix": matrix, "dates": dates, "codes": codes, "meta": meta}

    def load_or_compute(
        self,
        expr: str,
        universe: str,
        start: str,
        end: str,
        compute_fn,
    ) -> dict:
        """
        命中直接返回缓存；miss 时调用 compute_fn() 并 save 后返回同构 dict。
        compute_fn() 返回 dict(matrix=..., dates=..., codes=...) 或 (matrix, dates, codes)。
        """
        hit = self.load(expr, universe, start, end)
        if hit is not None:
            return hit

        result = compute_fn()
        if isinstance(result, dict):
            matrix = result["matrix"]
            dates = result.get("dates", [])
            codes = result.get("codes", [])
        else:
            matrix, dates, codes = result

        meta = self.save(expr, universe, start, end, matrix, dates, codes)
        matrix_arr = np.asarray(matrix)
        return {
            "matrix": matrix_arr,
            "dates": [str(x) for x in np.asarray(dates).tolist()],
            "codes": [str(x) for x in np.asarray(codes).tolist()],
            "meta": meta,
        }

    # ── 管理 ──

    def invalidate(self) -> int:
        """清空缓存目录内所有文件，返回删除数量。目录本身保留。"""
        removed = 0
        for name in os.listdir(self.root):
            path = os.path.join(self.root, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass
        return removed

    def list_entries(self) -> list[dict]:
        """返回全部缓存条目的元信息列表（按 created_at 倒序）。"""
        entries: list[dict] = []
        for name in os.listdir(self.root):
            if not name.endswith(".meta.json"):
                continue
            try:
                with open(os.path.join(self.root, name), encoding="utf-8") as f:
                    entries.append(json.load(f))
            except (OSError, ValueError):
                continue
        entries.sort(key=lambda m: str(m.get("created_at", "")), reverse=True)
        return entries
