"""3号交易员 v2.0 — 端到端演示（采集+触发）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ['AKSHARE_NO_PROXY'] = '1'

from trader3.v2.collector import DataCollector
from trader3.v2.trigger import get_trigger_engine
from trader3.v2.watchlist import get_watchlist
from trader3.v2.events import get_event_library

# 1. 重建自选股
wl = get_watchlist()
for code, name in [('600519', '贵州茅台'), ('000858', '五粮液'), ('300750', '宁德时代')]:
    wl.add(code, name)
wl.transition('600519', '关注', '已跟踪')
wl.transition('000858', '关注', '已跟踪')
wl.close()
print("自选股已建立: 600519/000858/300750")

# 2. 采集真实事件
collector = DataCollector()
for code in ['600519', '000858', '300750']:
    n = collector.collect_stock(code, source='news')
    print(f"采集 {code} 新闻/公告: {n} 条")
n2 = collector.collect_flow_events()
print(f"采集市场事件(涨停池): {n2} 条")
collector.close()

# 3. 三因子扫描（催化自动从事件库读取）
engine = get_trigger_engine()
wl = get_watchlist()
items = wl.list()
results = engine.scan(items)
print()
for r in results:
    mark = 'TRIGGER' if r.triggered else 'watch'
    print(f"{mark} {r.code} 催化{r.catalyst_score:.2f} 估值{r.valuation_score:.2f} 技术{r.tech_score:.2f} 信号{r.score:.2f}")
    print(f"   {r.reason}")
wl.close()

# 4. 事件库状态
lib = get_event_library()
print()
print(f"事件库总数: {lib.count()}")
lib.close()