"""股本/市值/comps 验证"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["AKSHARE_NO_PROXY"] = "1"

from trader3.v2.market_data import get_quote
from trader3.v2.comps import get_comps_analyzer

q = get_quote("600519")
print("茅台快照:", q.get("name"), "价", q.get("price"),
      "市值", q.get("market_cap", 0) / 1e8, "亿",
      "股本", q.get("total_share", 0) / 1e8, "亿")

t = get_comps_analyzer().analyze("600519", industry="白酒")
print("comps:", t.target_name, "| 结论:", t.conclusion)
for r in t.peers:
    tag = "T" if r.is_target else "P"
    print(f"  [{tag}] {r.code} {r.name} P/E={r.pe:.1f} EV/EBITDA={r.ev_ebitda:.1f} EV/Rev={r.ev_revenue:.1f}")

# 其他股票
for c in ["000858", "300750"]:
    q2 = get_quote(c)
    print(f"{c} {q2.get('name')} 价{q2.get('price')} 市值{q2.get('market_cap',0)/1e8:.0f}亿 股本{q2.get('total_share',0)/1e8:.1f}亿")