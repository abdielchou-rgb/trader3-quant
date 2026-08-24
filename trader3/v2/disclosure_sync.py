"""3号交易员 v2.1 — 预约披露日历同步（akshare 预约披露时间表 → 本地显式公告日历）

接口实测（akshare 1.18.81，2026-08-24）：
- ak.stock_yysj_em(symbol="沪深A股", date="YYYYMMDD") 可用，date 仅接受报告期末日
  （03-31 / 06-30 / 09-30 / 12-31；传 20260831 这类非期末日会在 akshare 内部报 TypeError）
- 返回 5208 行 × 8 列：['序号','股票代码','股票简称','首次预约时间',
  '一次变更日期','二次变更日期','三次变更日期','实际披露时间']（NaT=未披露/未变更）

存储：data/disclosure_calendar.json，结构 {code: {quarter: announce_date}}。
AnnouncementCalendar 优先消费显式值，缺失时回退规律推断。

用法：
    python -m trader3.v2.disclosure_sync --quarter 2026-06-30 [--codes 600519,000858]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)

# 测试可通过环境变量重定向存储路径
_PATH_ENV = "TRADER3_DISCLOSURE_CALENDAR"

# 报告期合法月末（EM 接口仅接受这些日期）
_QUARTER_ENDS = {"03-31", "06-30", "09-30", "12-31"}

# 公告日取值优先级：实际披露 > 三次变更 > … > 首次预约
_DATE_COLS_PRIORITY = ["实际披露时间", "三次变更日期", "二次变更日期", "一次变更日期", "首次预约时间"]

_PRIMARY = "stock_yysj_em"


def default_calendar_path() -> Path:
    env = os.environ.get(_PATH_ENV)
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data" / "disclosure_calendar.json"


def load_explicit_calendar(path: Path | None = None) -> dict[str, dict[str, str]]:
    """读取显式公告日历 {code: {quarter: announce_date}}；缺失/损坏返回 {}"""
    p = Path(path) if path else default_calendar_path()
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.warning("披露日历文件损坏，忽略：%s", p)
        return {}


def save_explicit_calendar(data: dict, path: Path | None = None) -> Path:
    """原子写：临时文件 + os.replace"""
    p = Path(path) if path else default_calendar_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(p.parent), prefix=p.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=0, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, str(p))
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    return p


def _norm_code(code) -> str:
    c = str(code).replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
    return c.zfill(6)[:6]


def _pick_announce_date(row: dict) -> str | None:
    """按优先级取首个非空日期 → ISO 字符串"""
    for col in _DATE_COLS_PRIORITY:
        val = row.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        try:
            if pd.isna(val):
                continue
        except (TypeError, ValueError):
            pass
        ts = pd.Timestamp(val)
        if pd.isna(ts):
            continue
        return ts.strftime("%Y-%m-%d")
    return None


def sync_disclosure_dates(quarter: str, codes: list[str] | None = None) -> dict:
    """拉取报告期 quarter 的预约披露表并写入显式公告日历。

    - 幂等：同 (code, quarter) 重写覆盖；合并保留其他报告期条目
    - 原子写：临时文件 + os.replace
    - 容错：主接口失败 logger.warning 并返回 synced=0，不抛出
    """
    q = (quarter or "").strip()
    md = q[5:10] if len(q) >= 10 else ""
    if len(q) != 10 or md not in _QUARTER_ENDS:
        logger.warning("非法报告期 %r（需报告期末日，如 2026-06-30），跳过同步", quarter)
        return {"synced": 0, "quarter": quarter, "source": None,
                "path": str(default_calendar_path())}

    wanted: set | None = None
    if codes:
        wanted = {_norm_code(c) for c in codes}

    try:
        df = ak.stock_yysj_em(symbol="沪深A股", date=q.replace("-", ""))
    except Exception as exc:
        logger.warning("预约披露主源 %s 失败（quarter=%s）：%s", _PRIMARY, q, exc)
        return {"synced": 0, "quarter": q, "source": None,
                "path": str(default_calendar_path())}

    updates: dict[str, str] = {}
    skipped_no_date = 0
    for row in df.to_dict("records"):
        code = _norm_code(row.get("股票代码"))
        if wanted is not None and code not in wanted:
            continue
        ann = _pick_announce_date(row)
        if ann:
            updates[code] = ann
        else:
            skipped_no_date += 1

    calendar = load_explicit_calendar()
    for code, ann in updates.items():
        calendar.setdefault(code, {})[q] = ann  # 同 key 覆盖 → 幂等
    path = save_explicit_calendar(calendar)

    logger.info("披露日历同步完成 quarter=%s synced=%d path=%s", q, len(updates), path)
    return {
        "synced": len(updates),
        "skipped_no_date": skipped_no_date,
        "quarter": q,
        "source": _PRIMARY,
        "path": str(path),
        "total_codes": len(calendar),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trader3.v2.disclosure_sync",
        description="同步预约披露时间表到本地公告日历（防前视用）",
    )
    parser.add_argument("--quarter", required=True,
                        help="报告期，如 2026-06-30（须为季末日）")
    parser.add_argument("--codes", default=None,
                        help="逗号分隔股票代码，缺省全市场")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    codes = [c.strip() for c in args.codes.split(",") if c.strip()] if args.codes else None
    res = sync_disclosure_dates(args.quarter, codes=codes)
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
