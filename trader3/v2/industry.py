"""3号交易员 v2.0 — 行业分类基建 (industry)

东财行业板块成分 → {code: industry} 映射，落盘 data/industry_map.json。

接口列名探查结论（akshare 1.18.81，源码静态确认；实网探测受沙箱出站限制未完成）:
- ak.stock_board_industry_name_em() → 列 ['排名','板块名称','板块代码',...]，
  板块代码形如 'BK1036'，可直接作为 cons 接口的 symbol 以跳过其内部二次拉取
- ak.stock_board_industry_cons_em(symbol=板块代码或名称) → 列 ['序号','代码','名称',...]，
  '代码' 为 6 位裸码（历史版本曾带 sh/sz 或 .SH/.SZ 后缀，统一归一化兜底）

缓存结构: {"updated_at": ISO时间, "map": {code: 行业}}，
环境变量 TRADER3_INDUSTRY_MAP 可重定向缓存路径（测试隔离）。
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_PATH_ENV = "TRADER3_INDUSTRY_MAP"
_CACHE_FILENAME = "industry_map.json"
_BOARD_NAME_COLS = ("板块名称", "名称")
_BOARD_CODE_COLS = ("板块代码", "代码")
_CONS_CODE_COL = "代码"


def default_cache_path() -> Path:
    env = os.environ.get(_PATH_ENV)
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data" / _CACHE_FILENAME


def normalize_code(value) -> str:
    """任意形态代码 → 6 位裸码: '600519.SH'/'sh600519'/'600519' → '600519'。"""
    digits = re.sub(r"\D", "", str(value))
    if len(digits) >= 6:
        return digits[-6:]
    return digits.zfill(6)


def fetch_sw_industry_map() -> dict[str, str]:
    """拉取东财行业板块成分，聚合为 {code: 行业}。

    单个板块失败降级跳过（保留其余结果）；整体失败抛 RuntimeError。
    """
    try:
        import akshare as ak

        boards_df = ak.stock_board_industry_name_em()
        if boards_df is None or boards_df.empty:
            raise RuntimeError("行业板块列表为空")
        name_col = next((c for c in _BOARD_NAME_COLS if c in boards_df.columns), None)
        if name_col is None:
            raise RuntimeError(
                f"板块列表缺少名称列（实际列: {list(boards_df.columns)}）"
            )
        code_col = next((c for c in _BOARD_CODE_COLS if c in boards_df.columns), None)

        result: dict[str, str] = {}
        for _, row in boards_df.iterrows():
            board = str(row[name_col]).strip()
            symbol = str(row[code_col]).strip() if code_col else board
            try:
                cons_df = ak.stock_board_industry_cons_em(symbol=symbol)
            except Exception as exc:
                logger.warning("行业板块 %s 成分抓取失败: %s", board, exc)
                continue
            if (
                cons_df is None
                or cons_df.empty
                or _CONS_CODE_COL not in cons_df.columns
            ):
                continue
            for raw in cons_df[_CONS_CODE_COL].astype(str):
                code = normalize_code(raw)
                if code:
                    result[code] = board

        if not result:
            raise RuntimeError("全部行业板块成分聚合为空")
        return result
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"行业分类抓取失败: {exc}") from exc


def _load_cache(path: Path) -> tuple[datetime | None, dict[str, str]]:
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        mapping_raw = payload.get("map")
        if not isinstance(mapping_raw, dict):
            return None, {}
        updated_raw = payload.get("updated_at")
        updated = datetime.fromisoformat(str(updated_raw)) if updated_raw else None
        mapping = {
            normalize_code(k): str(v) for k, v in mapping_raw.items() if normalize_code(k)
        }
        return updated, mapping
    except FileNotFoundError:
        return None, {}
    except (OSError, ValueError):
        logger.warning("行业映射缓存损坏，忽略: %s", path)
        return None, {}


def _write_cache(
    path: Path,
    mapping: dict[str, str],
    updated_at: datetime | None = None,
) -> None:
    """原子写: 同目录临时文件 + os.replace。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": (updated_at or datetime.now()).isoformat(),
        "map": dict(mapping),
    }
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, str(path))
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def load_industry_map(max_age_days: int = 30) -> dict[str, str]:
    """读取本地行业映射；新鲜直接返回，过期/缺失则触发 fetch 并原子刷新缓存。"""
    path = default_cache_path()
    updated, cached = _load_cache(path)
    is_fresh = updated is not None and datetime.now() - updated < timedelta(
        days=max_age_days
    )
    if is_fresh:
        return cached
    mapping = fetch_sw_industry_map()
    _write_cache(path, mapping)
    return mapping


def get_industry(code) -> str | None:
    """只读查询单票行业；无缓存或未收录返回 None（不触发联网）。"""
    _, cached = _load_cache(default_cache_path())
    if not cached:
        return None
    return cached.get(normalize_code(code))
