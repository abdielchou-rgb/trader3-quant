"""
3号交易员 v2 — 5分钟线采集试点 (minute_data)

为事件驱动回测储备 5 分钟 K 线：主源 akshare（东财）、备源 baostock，双源降级；
落盘 data/minute_5/{code}.parquet（pyarrow 可用时）否则退 {code}.csv.gz，
整文件原子替换（tmp + os.replace），schema 固定六列：
    datetime("YYYY-MM-DD HH:MM") / open / high / low / close / volume

接口探针实测结论（2026-08-25，本沙箱环境）：
  akshare 1.18.81 stock_zh_a_hist_min_em(period="5", adjust="qfq")：
      东财远端拒绝连接（ConnectionError ×3，稳定复现）——主源当前环境不可达。
      列名按 akshare 源码契约 ['时间','开盘','最高','最低','收盘','成交量',...]
      归一化并由单测固化；上游改名时运行期抛 KeyError 自动降级备源。
  baostock query_history_k_data_plus(fields="date,time,code,close,volume",
      frequency="5", adjustflag="2")：login/query/logout 全通，
      time 格式 YYYYMMDDHHMMSSsss，48 根/交易日，20 天窗口实测返回 720 行。
      备源字段仅含 close/volume —— open/high/low 以 NaN 占位（不伪造数据）。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger("trader3.v2.minute")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MINUTE_DIR = os.path.join(PROJECT_ROOT, "data", "minute_5")

SCHEMA_COLS = ["datetime", "open", "high", "low", "close", "volume"]
NUM_COLS = ["open", "high", "low", "close", "volume"]
JUMP_THRESHOLD = 0.11       # 单根 5 分钟 |ret| 上限：A股日内涨跌停约 ±10%，超即数据异常
JUMP_OFFENDER_CAP = 10
BACKENDS = (("parquet", ".parquet"), ("csv.gz", ".csv.gz"))

try:
    import pyarrow  # noqa: F401

    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False


# ── 代码与路径 ──────────────────────────────────────────

def bare_code(code: str) -> str:
    """'SH600519' / '600519.SH' / '600519' → '600519'"""
    c = code.upper().replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
    if c.startswith(("SH", "SZ", "BJ")):
        c = c[2:]
    return c.zfill(6)


def _bs_symbol(bare: str) -> str:
    """裸码 → baostock 带前缀代码。沪(6/9)→sh. 北(4/8)→bj. 其余→sz."""
    if bare.startswith(("6", "9")):
        return "sh." + bare
    if bare.startswith(("4", "8")):
        return "bj." + bare
    return "sz." + bare


def default_dir() -> str:
    return MINUTE_DIR


def active_backend() -> str:
    return "parquet" if HAVE_PYARROW else "csv.gz"


def backend_ext(backend: str | None = None) -> str:
    name = backend or active_backend()
    return dict(BACKENDS)[name]


# ── 源数据 → 统一行结构 ─────────────────────────────────

def _rows_from_ak(df: pd.DataFrame) -> list[dict]:
    """东财中文列 → 六列统一行。缺列抛 KeyError 由上层降级备源。"""
    out = []
    for t, o, h, low, c, v in zip(
        df["时间"], df["开盘"], df["最高"], df["最低"], df["收盘"], df["成交量"], strict=False
    ):
        out.append({
            "datetime": str(t)[:16],
            "open": float(o), "high": float(h), "low": float(low),
            "close": float(c), "volume": float(v),
        })
    return out


def _rows_from_bs(raw_rows: list[list[str]]) -> list[dict]:
    """baostock 行(date,time,code,close,volume 全字符串) → 六列统一行。

    备源无 OHLC：open/high/low 置 NaN 占位，绝不以 close 冒充。
    """
    out = []
    for date, t, _code, close, vol in (r[:5] for r in raw_rows if len(r) >= 5):
        if len(t) < 12:
            continue
        hh, mm = t[8:10], t[10:12]
        out.append({
            "datetime": f"{date} {hh}:{mm}",
            "open": np.nan, "high": np.nan, "low": np.nan,
            "close": float(close), "volume": float(vol),
        })
    return out


def _dedup_sort(rows: list[dict]) -> list[dict]:
    """按时间升序去重（同时刻保首条），数值统一 float。"""
    merged: dict[str, dict] = {}
    for r in rows:
        dt = str(r["datetime"])[:16]
        if dt in merged:
            continue
        merged[dt] = {
            "datetime": dt,
            **{c: float(r.get(c)) if r.get(c) is not None else np.nan for c in NUM_COLS},
        }
    return [merged[k] for k in sorted(merged)]


# ── 抓取（模块级函数便于测试 monkeypatch） ───────────────

def fetch_minute_ak(code: str, start_date: str) -> list[dict]:
    """主源：akshare 东财 5 分钟线（qfq）。start_date 形如 '2026-08-05 09:30:00'。"""
    import akshare as ak

    df = ak.stock_zh_a_hist_min_em(symbol=bare_code(code), period="5",
                                   adjust="qfq", start_date=start_date)
    return _rows_from_ak(df)


def fetch_minute_bs(code: str, start_date: str) -> list[dict]:
    """备源：baostock 5 分钟线（adjustflag=2 前复权）。start_date 形如 '2026-08-05'。"""
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login fail: {lg.error_code} {lg.error_msg}")
    try:
        rs = bs.query_history_k_data_plus(
            _bs_symbol(bare_code(code)), "date,time,code,close,volume",
            start_date=start_date, end_date=datetime.now().strftime("%Y-%m-%d"),
            frequency="5", adjustflag="2")
        if rs.error_code != "0":
            raise RuntimeError(f"baostock query fail: {rs.error_code} {rs.error_msg}")
        raw: list[list[str]] = []
        while rs.next():
            raw.append(rs.get_row_data())
    finally:
        bs.logout()
    return _rows_from_bs(raw)


def fetch_minute_verbose(code: str, days: int = 20) -> tuple[list[dict], str]:
    """双源降级抓取，返回 (升序去重行, 实际使用源)。双源均失败抛 RuntimeError。"""
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        rows = fetch_minute_ak(code, start)
        return _dedup_sort(rows), "akshare"
    except Exception as e1:
        logger.warning("[minute] %s 主源失败，降级 baostock: %s", code, str(e1)[:80])
        try:
            rows = fetch_minute_bs(code, start[:10])
            return _dedup_sort(rows), "baostock"
        except Exception as e2:
            raise RuntimeError(f"5分钟线双源失败({code}): {e1} / {e2}") from e2


def fetch_minute(code: str, days: int = 20) -> list[dict]:
    """抓取个股近 N 日 5 分钟线：[{datetime, open, high, low, close, volume}] 升序去重。"""
    rows, _source = fetch_minute_verbose(code, days=days)
    return rows


# ── 存储与加载 ──────────────────────────────────────────

def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """固定六列契约：datetime=str('YYYY-MM-DD HH:MM')，数值列=float64。"""
    df = df.reset_index(drop=True)
    if df.empty:
        return df.reindex(columns=SCHEMA_COLS).astype({c: "float64" for c in NUM_COLS})
    df["datetime"] = df["datetime"].astype(str)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df[SCHEMA_COLS]


def to_frame(rows: list[dict]) -> pd.DataFrame:
    return _coerce(pd.DataFrame(list(rows), columns=SCHEMA_COLS))


def save_minute_atomic(code: str, rows: list[dict], base_dir: str | None = None,
                       backend: str | None = None) -> str:
    """整文件原子替换写入（tmp + os.replace）；写失败清理 .tmp 防残留。返回落盘路径。"""
    backend = backend or active_backend()
    df = to_frame(rows)
    base = base_dir or MINUTE_DIR
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, bare_code(code) + backend_ext(backend))
    tmp = path + ".tmp"
    try:
        if backend == "parquet":
            df.to_parquet(tmp, engine="pyarrow", index=False)
        else:
            df.to_csv(tmp, index=False, encoding="utf-8", compression="gzip")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def load_minute(code: str, base_dir: str | None = None) -> pd.DataFrame | None:
    """读取已落盘 5 分钟线（自动探测 parquet / csv.gz），不存在返回 None。"""
    base = base_dir or MINUTE_DIR
    bare = bare_code(code)
    for backend, ext in BACKENDS:
        path = os.path.join(base, bare + ext)
        if not os.path.exists(path):
            continue
        if backend == "parquet":
            return _coerce(pd.read_parquet(path))
        return _coerce(pd.read_csv(path))
    return None


# ── QC ──────────────────────────────────────────────────

def qc_minute(df: pd.DataFrame) -> dict:
    """
    5 分钟线质检：单根 |ret|>11% 记 offender（A股日内不可能，含涨跌停）；
    时间倒序（相邻非严格递增对数）与重复时刻计数。
    返回 {"rows", "dup", "inverted", "jump_offenders"(截断前10)}。
    """
    n = len(df)
    if n == 0:
        return {"rows": 0, "dup": 0, "inverted": 0, "jump_offenders": []}
    dts = df["datetime"].astype(str).tolist()
    closes = pd.to_numeric(df["close"], errors="coerce").tolist()

    dup = int(df["datetime"].duplicated().sum())
    inverted = sum(1 for i in range(n - 1) if dts[i] >= dts[i + 1])

    offenders = []
    for i in range(1, n):
        prev, cur = closes[i - 1], closes[i]
        if not (np.isfinite(prev) and np.isfinite(cur)) or prev <= 0 or cur <= 0:
            continue
        ret = cur / prev - 1.0
        if abs(ret) > JUMP_THRESHOLD:
            offenders.append({"datetime": dts[i], "ret": round(ret, 4)})
    return {"rows": n, "dup": dup, "inverted": inverted,
            "jump_offenders": offenders[:JUMP_OFFENDER_CAP]}
