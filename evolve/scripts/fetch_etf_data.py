#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_etf_data.py — 拉取主流 A股 ETF 日线数据（akshare）

用法（本机有网时运行，需安装 akshare）:
    pip install akshare
    python fetch_etf_data.py --out ../../evolve/data/etf --n 50

输出:
    evolve/data/etf/{code}.csv   每只 ETF 一个 CSV (date,open,high,low,close,volume)

在沙箱/无网环境不可运行；仅在本机执行。
"""

import argparse
import os
import time
from pathlib import Path


# 主流 ETF 列表（代码, 名称）
DEFAULT_ETFS = [
    ("510300", "沪深300ETF"),
    ("510500", "中证500ETF"),
    ("510050", "上证50ETF"),
    ("159915", "创业板ETF"),
    ("588000", "科创50ETF"),
    ("512100", "中证1000ETF"),
    ("512880", "证券ETF"),
    ("512690", "酒ETF"),
    ("515030", "新能源车ETF"),
    ("515790", "光伏ETF"),
    ("512480", "半导体ETF"),
    ("159995", "芯片ETF"),
    ("512170", "医疗ETF"),
    ("512010", "医药ETF"),
    ("515220", "煤炭ETF"),
    ("512800", "银行ETF"),
    ("512000", "券商ETF"),
    ("159928", "消费ETF"),
    ("512660", "军工ETF"),
    ("516160", "新能源ETF"),
    ("159869", "游戏ETF"),
    ("512720", "计算机ETF"),
    ("515050", "5GETF"),
    ("159992", "创新药ETF"),
    ("515000", "科技ETF"),
    ("510880", "红利ETF"),
    ("512890", "红利低波ETF"),
    ("159941", "纳指ETF"),
    ("513100", "纳指ETF易方达"),
    ("513500", "标普500ETF"),
    ("518880", "黄金ETF"),
    ("511010", "国债ETF"),
    ("511260", "十年国债ETF"),
    ("159920", "恒生ETF"),
    ("510900", "H股ETF"),
    ("512200", "房地产ETF"),
    ("512400", "有色金属ETF"),
    ("159905", "深红利ETF"),
    ("512010", "医药ETF"),
    ("515880", "通信ETF"),
    ("516110", "汽车ETF"),
    ("159865", "养殖ETF"),
    ("512690", "酒ETF"),
    ("515210", "钢铁ETF"),
    ("512580", "环保ETF"),
    ("159755", "电池ETF"),
    ("516010", "游戏动漫ETF"),
    ("512170", "医疗ETF"),
    ("159825", "农业ETF"),
    ("515650", "消费50ETF"),
]


def fetch_etf(code: str, name: str, start: str = "2018-01-01") -> tuple:
    """拉取单只 ETF 日线，返回 (df, code)"""
    import akshare as ak

    try:
        df = ak.fund_etf_hist_em(
            symbol=code,
            period="daily",
            start_date=start.replace("-", ""),
            end_date="20300101",
            adjust="qfq",
        )
        # 标准化列
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume",
        })
        keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]
        return df, code
    except Exception as e:
        print(f"  [SKIP] {code} {name}: {e}")
        return None, code


def main():
    parser = argparse.ArgumentParser(description="拉取 A股 ETF 日线")
    parser.add_argument("--out", default="../../evolve/data/etf", help="输出目录")
    parser.add_argument("--n", type=int, default=50, help="拉取数量")
    parser.add_argument("--start", default="2018-01-01", help="起始日期")
    parser.add_argument("--codes", nargs="*", help="指定代码列表（覆盖默认）")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    etfs = args.codes if args.codes else [c for c, _ in DEFAULT_ETFS[:args.n]]
    name_map = dict(DEFAULT_ETFS)

    ok = 0
    fail = 0
    for i, code in enumerate(etfs):
        print(f"[{i+1}/{len(etfs)}] {code} {name_map.get(code, '')}", flush=True)
        df, c = fetch_etf(code, name_map.get(code, ""), args.start)
        if df is not None and len(df) > 100:
            csv_path = out_dir / f"{c}.csv"
            df.to_csv(csv_path, index=False)
            ok += 1
            print(f"  -> {len(df)} 行, 保存 {csv_path}")
        else:
            fail += 1
        time.sleep(0.5)  # 避免请求过频

    print(f"\n完成: 成功 {ok}, 失败 {fail}")
    print(f"数据目录: {out_dir}")


if __name__ == "__main__":
    main()