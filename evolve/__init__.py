"""
3号交易员 — 策略进化工厂 (evolve)

GP 遗传编程自动挖因子 → 筛选出有用策略 → 供 2号分析师/3号交易员引用。

子模块:
  core/gp.py         表达式树 + 遗传操作 + 适应度
  core/parser.py     表达式解析
  core/evolution.py  进化引擎
  core/data_loader.py 数据加载（qlib/ETF）
  scripts/fetch_etf_data.py  akshare 拉取 ETF 数据（本机有网时）
  run_evolution.py   主入口（marvis 调用）
  selection.py       策略筛选门禁
"""

__version__ = "0.1.0"