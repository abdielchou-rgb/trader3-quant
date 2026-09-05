#!/usr/bin/env python3
"""
run_headfix_batches.py — 分批执行 repair_dirty_head_rows（防 MCP/外层超时）。

断点续跑：每批 800 只，幂等（_fix_one 非脏字段自动跳过）。
进度持久化到 <data_dir>/_headfix_progress.json（已完成下标集合）。
备份统一写到 <data_dir>/_backup_headfix_<首跑时间戳>/（批间共享）。

用法：
  python scripts/run_headfix_batches.py              # 从断点继续，直到全部完成
  python scripts/run_headfix_batches.py --batch 400  # 单批最多修 N 只后退出
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from repair_dirty_head_rows import _fix_one  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=800, help="单次调用最多修复只数")
    ap.add_argument("--list-file", default="/tmp/final_list.json")
    args = ap.parse_args()

    from trader3.data_provider import QlibDataProvider

    dp = QlibDataProvider()
    data_dir = dp.data_dir
    codes = json.load(open(args.list_file, encoding="utf-8"))
    print(f"名单: {len(codes)} 只", flush=True)

    # 断点续跑状态
    prog_path = os.path.join(data_dir, "_headfix_progress.json")
    done = set()
    backup_root = ""
    if os.path.exists(prog_path):
        st = json.load(open(prog_path, encoding="utf-8"))
        done = set(st.get("done", []))
        backup_root = st.get("backup_root", "")
    if not backup_root:
        backup_root = os.path.join(data_dir, f"_backup_headfix_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(backup_root, exist_ok=True)

    todo = [c for c in codes if c not in done]
    print(f"已完成 {len(done)}，剩余 {len(todo)}。备份: {backup_root}", flush=True)

    t0 = time.time()
    fixed = 0
    bad: list[str] = []
    for i, c in enumerate(todo):
        if i >= args.batch:
            break
        code, changed, ok = _fix_one(data_dir, c, backup_root, apply=True)
        if changed:
            fixed += 1
        if not ok:
            bad.append(code)
        done.add(c)
        if (i + 1) % 100 == 0:
            json.dump({"done": sorted(done), "backup_root": backup_root},
                      open(prog_path, "w", encoding="utf-8"))
            print(f"  批内 {i+1}/{len(todo)} 累计真改 {fixed} 耗时 {time.time()-t0:.0f}s", flush=True)

    json.dump({"done": sorted(done), "backup_root": backup_root},
              open(prog_path, "w", encoding="utf-8"))
    print(f"本轮: 处理 {min(args.batch, len(todo))} 只，真改 {fixed}，坏 {bad}", flush=True)
    remaining = [c for c in codes if c not in done]
    print(f"剩余 {len(remaining)} 只（再跑一次续批）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
