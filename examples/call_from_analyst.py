"""
2号分析师 调用 3号交易员 完整工作流示例

展示在写报告时如何嵌入调用全部 10 个 Tool：
回测验证、WFA、组合优化、情境路由、交易成本、执行计划、
信号验证、市场状态、估值锚、基本面评分卡。

每个 Tool 返回统一的 Trader3Response，可直接把 summary / key_metrics 引用进报告。
输出为候选信号，非投资建议。
"""
from trader3 import Trader3

# 初始化 3号交易员
t3 = Trader3()

# ═══════════════════════════════════════════
# 场景 1：回测验证（M1）
# ═══════════════════════════════════════════
print("=" * 70)
print("【场景1】策略回测验证 → run_backtest")
print("=" * 70)

result = t3.run_backtest(start_date="2020-01-01", end_date="2025-12-31")
print(f"\n  >>> 结论: {result.summary}")
print(f"  >>> 年化收益: {result.key_metrics.get('年化收益', 0):.1%}")
print(f"  >>> 夏普: {result.key_metrics.get('夏普比', 0):.2f}")
print(f"  >>> 最大回撤: {result.key_metrics.get('最大回撤', 0):.1%}")
print(f"  >>> 净值曲线点数: {len(result.data.equity_curve)}")
print("\n  报告引用格式:")
print(f'    "3号交易员回测，过去 6 年年化 {result.key_metrics.get("年化收益", 0):.1%}，'
      f'夏普 {result.key_metrics.get("夏普比", 0):.2f}，最大回撤 {result.key_metrics.get("最大回撤", 0):.1%}"')

# ═══════════════════════════════════════════
# 场景 2：WFA 过拟合检测（M1）
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("【场景2】Walk-Forward 过拟合检测")
print("=" * 70)

result = t3.walk_forward_analysis(train_window=252, test_window=63)
print(f"\n  >>> 结论: {result.summary}")
print("\n  报告引用格式:")
print(f'    "3号交易员 WFA 样本外年化 {result.key_metrics.get("样本外收益", 0):.1%}，'
      f'过拟合概率 {result.key_metrics.get("过拟合概率", 0):.0%}"')

# ═══════════════════════════════════════════
# 场景 3：组合优化（M2）— 三种方法
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("【场景3】组合优化 → optimize_portfolio（三种方法）")
print("=" * 70)

signals = {
    "000001.SZ": 80, "000002.SZ": 72, "000003.SZ": 65, "000004.SZ": 60,
    "000005.SZ": 55, "000006.SZ": 50, "000007.SZ": 45, "000008.SZ": 40,
}
for method in ("risk_budget", "mean_variance", "black_litterman"):
    result = t3.optimize_portfolio(signals=signals, method=method)
    print(f"  [{method}] {result.summary}")

# ═══════════════════════════════════════════
# 场景 4：情境路由（M2）
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("【场景4】情境路由 → regime_aware_allocation")
print("=" * 70)

result = t3.regime_aware_allocation(
    signals={"000001.SZ": 80, "000002.SZ": 72, "000003.SZ": 65},
    regime_probs={"ranging": 0.6, "bearish": 0.3, "trending_up": 0.1},
    regime_weights={
        "ranging": {"000001.SZ": 0.4, "000002.SZ": 0.4, "000003.SZ": 0.2},
        "bearish": {"000001.SZ": 0.2, "000002.SZ": 0.3, "000003.SZ": 0.5},
        "trending_up": {"000001.SZ": 0.5, "000002.SZ": 0.3, "000003.SZ": 0.2},
    },
)
print(f"\n  >>> 结论: {result.summary}")
print(f"  >>> 建议仓位: {result.key_metrics.get('建议仓位', 0):.0%}")

# ═══════════════════════════════════════════
# 场景 5：交易成本估算（M3）— 买入 vs 卖出
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("【场景5】交易成本估算 → estimate_transaction_cost")
print("=" * 70)

buy = t3.estimate_transaction_cost(
    orders=[{"symbol": "301150.SZ", "side": "buy", "value_cny": 5000000}]
)
sell = t3.estimate_transaction_cost(
    orders=[{"symbol": "301150.SZ", "side": "sell", "value_cny": 5000000}]
)
print(f"  买入: {buy.summary}")
print(f"  卖出: {sell.summary}（含印花税 {sell.key_metrics.get('印花税(bp)', 0):.0f}bp）")
print("\n  报告引用格式:")
print(f'    "建仓 500 万，3号交易员估算预期交易成本 {buy.key_metrics.get("总成本(bp)", 0):.0f}bp'
      f'（约 ¥{buy.key_metrics.get("总成本(元)", 0):,.0f}），建议分批执行"')

# ═══════════════════════════════════════════
# 场景 6：执行计划（M3）
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("【场景6】生成执行计划 → generate_execution_plan")
print("=" * 70)

result = t3.generate_execution_plan(
    target_weights={"000001.SZ": 0.3, "000002.SZ": 0.2, "000003.SZ": 0.1},
    algorithm="adaptive_vwap", urgency="normal",
)
print(f"\n  >>> 结论: {result.summary}")
print(f"  >>> 首片: {result.data.slices[0]}")
print(f"  >>> 风控熔断线: {result.data.risk_limits}")

# ═══════════════════════════════════════════
# 场景 7：信号验证（M4）
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("【场景7】信号验证 → validate_signal")
print("=" * 70)

result = t3.validate_signal(signal_name="动量因子")
print(f"\n  >>> 结论: {result.summary}")
print("\n  报告引用格式:")
print(f'    "3号交易员验证，该因子 ICIR={result.key_metrics.get("ICIR", 0):.2f}，'
      f'半衰期 {result.key_metrics.get("半衰期(月)", 0):.1f} 个月，拥挤度低"')

# ═══════════════════════════════════════════
# 场景 8：市场状态诊断（M4）
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("【场景8】市场状态诊断 → diagnose_market_regime")
print("=" * 70)

result = t3.diagnose_market_regime()
print(f"\n  >>> 结论: {result.summary}")
print(f"  >>> 状态概率: {result.data.regime_probabilities}")
print(f"  >>> 建议仓位: {result.key_metrics.get('建议仓位', 0):.0%}")
print("\n  报告引用格式:")
print(f'    "3号交易员判定当前为 {result.key_metrics.get("当前状态", "?")} '
      f'（P={result.key_metrics.get("最大概率", 0):.0%}），'
      f'建议仓位 {result.key_metrics.get("建议仓位", 0):.0%}"')

# ═══════════════════════════════════════════
# 场景 9：估值锚（M5）
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("【场景9】估值锚 → valuation_anchor")
print("=" * 70)

result = t3.valuation_anchor(codes=["301150.SZ"])
print(f"\n  >>> 结论: {result.summary}")
print("\n  报告引用格式:")
print(f'    "3号交易员 DCF 测算，Base 目标价 ¥{result.key_metrics.get("Base", 0):.1f}，'
      f'Bull ¥{result.key_metrics.get("Bull", 0):.1f}，Bear ¥{result.key_metrics.get("Bear", 0):.1f}，'
      f'隐含收益率 {result.key_metrics.get("隐含收益率", 0):.1%}"')

# ═══════════════════════════════════════════
# 场景 10：基本面评分卡（M5）
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("【场景10】基本面评分卡 → fundamental_scorecard")
print("=" * 70)

result = t3.fundamental_scorecard(codes=["000001.SZ"], template="quality_growth")
print(f"\n  >>> 结论: {result.summary}")
print(f"  >>> 红旗预警: {result.data.red_flags}")

# ═══════════════════════════════════════════
# 汇总与审计
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("【审计】门禁状态与调用历史")
print("=" * 70)

gates = t3.gates.summary()
print(f"  门禁: enabled={gates['enabled']}")

history = t3.get_call_history()
print(f"  最近调用记录 {len(history)} 条（按耗时排序 Top5）:")
for h in history[:5]:
    print(f"    - {h['tool']:>28} elapsed={h.get('elapsed', 0):.3f}s success={h.get('success')}")

print("\n" + "=" * 70)
print("⚠ 以上全部输出为候选信号，非投资建议。")
print("=" * 70)
