#!/usr/bin/env python3
"""
experiments_list.py — 查看 GP 进化实验清单（evolve/experiments/index.jsonl）。

用法:
    python scripts/experiments_list.py              # 最近 10 条实验
    python scripts/experiments_list.py --limit 20   # 最近 20 条
    python scripts/experiments_list.py --best       # 按当次最优 fitness 降序排列
"""

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_INDEX = _ROOT / "evolve" / "experiments" / "index.jsonl"

_HEADER = (
    f"{'#':>4}  {'ts':<19}  {'universe':<10}  {'gen×pop':>9}  {'n':>4}  "
    f"{'train':<23}  {'fitness':>8}  {'ic':>7}  {'icir':>6}  {'mono':>5}  "
    f"{'ls':>6}  {'sel':>3}  {'dv':<10}  {'git':<8}  expr_best"
)


def _load_rows() -> list:
    if not _INDEX.exists():
        return []
    rows = []
    for line in _INDEX.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip().lstrip("﻿")
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _fmt_float(v, width) -> str:
    try:
        return f"{float(v):>{width}.4f}"
    except (TypeError, ValueError):
        return f"{'-':>{width}}"


def main() -> None:
    parser = argparse.ArgumentParser(description="查看进化实验清单")
    parser.add_argument("--limit", type=int, default=10, help="显示条数（默认 10）")
    parser.add_argument(
        "--best", action="store_true",
        help="按当次最优 fitness_best 降序排列（默认按时间正序取最近 N 条）",
    )
    args = parser.parse_args()

    rows = _load_rows()
    if not rows:
        print(f"(无实验记录: {_INDEX})")
        return

    if args.best:
        rows.sort(key=lambda r: float(r.get("fitness_best") or -999), reverse=True)
        shown = rows[: max(args.limit, 0)] if args.limit > 0 else rows
    else:
        shown = rows[-args.limit:] if args.limit > 0 else rows

    print(_HEADER)
    for i, r in enumerate(shown, 1):
        train = f"{r.get('train_start', '?')}~{r.get('train_end', '?')}"
        expr = str(r.get("expr_best", ""))
        print(
            f"{i:>4}  {str(r.get('ts', ''))[:19]:<19}  "
            f"{str(r.get('universe', '')):<10}  "
            f"{r.get('gen', '?')}×{r.get('pop', '?'):>5}  "
            f"{r.get('n_stocks', '-'):>4}  {train:<23}  "
            f"{_fmt_float(r.get('fitness_best'), 8)}  "
            f"{_fmt_float(r.get('ic'), 7)}  {_fmt_float(r.get('icir'), 6)}  "
            f"{_fmt_float(r.get('mono'), 5)}  {_fmt_float(r.get('ls'), 6)}  "
            f"{r.get('selected_count', '-'):>3}  "
            f"{str(r.get('data_version', '')):<10}  "
            f"{str(r.get('git_sha', '')):<8}  {expr}"
        )
    print(f"\n共 {len(rows)} 条记录，显示 {len(shown)} 条")


if __name__ == "__main__":
    main()
