---
name: "3号交易员"
description: "注册 3号交易员（trader3）为 2号分析师的量化交易子系统。当用户要求回测策略、验证因子、估算交易成本、生成执行计划、组合优化、市场状态诊断、估值锚、基本面评分卡时触发。"
---

# 3号交易员 — 2号分析师注册 Skill

将 **3号交易员**（`trader3` 包，M0–M7 全量实现）注册为 2号分析师可随时调用的量化交易子系统。
所有输出为**候选信号，非投资建议**。

## 触发短语

- "用3号交易员算一下"
- "帮我回测"
- "验证这个因子"
- "算估值"
- "市场状态"
- "最优权重 / 组合优化"
- "交易成本 / 滑点 / 执行计划"
- "基本面评分卡"

## 导入

```python
from trader3 import Trader3

t3 = Trader3()                      # 默认门禁关闭
t3 = Trader3(gates_enabled=True)    # 写正式报告前开启 IronGate 门禁
```

## 统一返回格式 Trader3Response

每个 Tool 返回同一结构，写报告时按字段引用：

| 字段 | 用途 |
|------|------|
| `summary` | 一句话结论 → 报告正文 |
| `key_metrics` | 关键指标（中文标签）→ 表格 |
| `charts` | 图表规格（line/bar/heatmap/table）→ 可视化 |
| `caveats` | 局限性/假设 → 脚注 |
| `metadata.gate_results` / `gates_passed` | 门禁检查 → 风控附录 |

## 10 个 Tool 与示例调用

### 1. run_backtest — 回测
```python
r = t3.run_backtest(start_date="2020-01-01", end_date="2025-12-31")
# summary: "[未命名策略] 2020-01-01~2025-12-31 年化 6.5%, 夏普 0.31, 超额 4.7%, 最大回撤 -34.5%"
# data: BacktestReport（净值曲线/归因/统计检验）；结果按 sha256 缓存
```

### 2. walk_forward_analysis — 过拟合检测
```python
r = t3.walk_forward_analysis(train_window=252, test_window=63)
# summary: "WFA 样本外年化 18.6%, 夏普 0.86, 过拟合概率 52%, 参数稳定性 79%"
```

### 3. optimize_portfolio — 组合优化（三种方法）
```python
r = t3.optimize_portfolio(
    signals={"000001.SZ": 80, "000002.SZ": 72, "000003.SZ": 65},
    method="risk_budget",          # risk_budget / mean_variance / black_litterman
)
# summary: "优化完成 [风险平价]: 预期年化 -0.2%, 波动 13.8%, 夏普 -0.02, 换手成本 6bp"
# data: OptimizationResult（目标权重/因子暴露/换手成本）
```

### 4. regime_aware_allocation — 情境路由
```python
r = t3.regime_aware_allocation(
    signals={"000001.SZ": 80, "000002.SZ": 72, "000003.SZ": 65},
    regime_probs={"ranging": 0.6, "bearish": 0.3, "trending_up": 0.1},
    regime_weights={...},           # 每状态下的资产权重映射
)
# summary: "情境路由: 当前状态「ranging」(P=60%), 综合 3 个标的权重, 预期夏普 0.02, 建议仓位 55%"
```

### 5. estimate_transaction_cost — 交易成本 TCA
```python
r = t3.estimate_transaction_cost(
    orders=[{"symbol": "301150.SZ", "side": "buy", "value_cny": 5000000}]
)
# summary: "预期总成本 70bp (¥34,994), 其中冲击成本 64bp 为主力贡献"
# 注意：A股印花税 10bp 仅卖出时收取（买入为 0）
```

### 6. generate_execution_plan — 分时执行计划
```python
r = t3.generate_execution_plan(
    target_weights={"000001.SZ": 0.3, "000002.SZ": 0.2},
    algorithm="adaptive_vwap",      # twap / vwap / is / adaptive_vwap
    urgency="normal",               # low / normal / high
)
# summary: "执行计划 (adaptive_vwap): 12 笔切片, 预期完成率 95%, 预期成本 20bp"
# data: ExecutionPlan（分时切片 + 风控熔断线）
```

### 7. validate_signal — 因子/信号验证
```python
r = t3.validate_signal(signal_name="动量因子")
# summary: "[动量因子（合成数据）] ICIR=0.396, 分组单调性 75%, 半衰期 12.0个月, 拥挤度 0.14, 多空年化 38.0%"
# data: SignalValidationReport（IC时序/五分组收益/条件有效性）
# 门槛参考: ICIR>0.3 有效; 半衰期<2个月 警惕衰减; 拥挤度>0.7 拥挤
```

### 8. diagnose_market_regime — 市场状态（HMM）
```python
r = t3.diagnose_market_regime()
# summary: "当前状态: ranging（合成数据） (P=100%), 熵=0.01, 建议仓位 60%, 检测方法: HMM (custom EM)"
# data: RegimeDiagnosis（状态概率和为1/关键指标/历史类比/策略建议/注意力动量）
```

### 9. valuation_anchor — 估值锚
```python
r = t3.valuation_anchor(codes=["301150.SZ"])
# summary: "加权目标价 ¥50.2, Base ¥52.3 / Bull ¥68.1 / Bear ¥38.7, 隐含收益率 12.0%"
# data: ValuationReport（DCF/PE/PB-ROE/EV-EBITDA + 敏感性网格）
```

### 10. fundamental_scorecard — 基本面评分卡
```python
r = t3.fundamental_scorecard(codes=["000001.SZ"], template="quality_growth")
# summary: "[质量成长] 总分 7.2/10, 高于同行均值 6.5, 2 个红旗预警"
# data: ScorecardReport（六维评分 + 红旗预警 + 同业对比）
```

## 报告引用格式

把 `summary` 与 `key_metrics` 直接引用进报告，保持中文标签一致：

```python
print(f'    结论：{r.summary}')
print(f'    "3号交易员 DCF 测算，Base 目标价 ¥{r.key_metrics["Base"]:.1f}，'
      f'Bull ¥{r.key_metrics["Bull"]:.1f}，Bear ¥{r.key_metrics["Bear"]:.1f}"')
print(f'    "3号交易员验证，该因子 ICIR={r.key_metrics["ICIR"]:.2f}，半衰期 '
      f'{r.key_metrics["半衰期(月)"]:.1f} 个月，拥挤度低"')
```

## 门禁与审计

```python
t3 = Trader3(gates_enabled=True)
r = t3.run_backtest()
r.metadata["gate_results"]   # [{'check_name', 'passed', 'score'}, ...]
r.metadata["gates_passed"]   # True/False
t3.get_call_history()        # 最近 50 条调用记录（审计）
```

## 边界与合规

- 所有输出是**候选信号，非投资建议**，报告正文需注明。
- M1–M4 为真实引擎；M5 估值/评分卡当前为确定性模板数据（接口契约完整，真实财务数据待接入）。
- 合成数据回测不代表真实表现，实盘前需接入真实数据（见 README Roadmap）。
- 调用前建议 `t3.state.set_data_version(...)` / `get_data_version()` 校验数据新鲜度。
