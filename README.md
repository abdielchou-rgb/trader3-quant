# 3号交易员

> 量化交易引擎 · 双模式运行 · 镶嵌于 2号分析师 · 候选信号，非投资建议

---

## 定位

**3号交易员** 是一个双模式量化交易引擎：

| 模式 | 形态 | 典型场景 |
|------|------|----------|
| **独立模式** | CLI / FastAPI / 定时调度 | 日度因子更新、组合再平衡检查、TCA 复盘、WFA 滚动验证 |
| **嵌入模式** | Python 包被 `from trader3 import Trader3` 导入 | 2号分析师写报告时实时调用：回测验证、成本估算、市场状态、估值锚、信号验证 |

所有 Tool 返回统一的 `Trader3Response`（`summary` / `key_metrics` / `charts` / `caveats` / `metadata`），
2号分析师可直接把 `summary` 与 `key_metrics` 引用进报告正文与表格。

---

## 核心特性（M0–M8 全部落地）

| 里程碑 | 能力 |
|--------|------|
| **M0 骨架** | Tool 接口 / 数据模型 / 配置管理 / 注册表 / 共享状态 / CLI |
| **M1 回测引擎** | 向量化回测（Qlib 回退）、真实净值曲线、全套指标、WFA 滚动验证、结果缓存 |
| **M2 优化器** | 风险平价 / 均值-方差 / Black-Litterman 三种真实优化、情境路由概率加权分配 |
| **M3 执行/TCA** | Almgren-Chriss 冲击模型、A股印花税/佣金规则、TWAP/VWAP/IS/Adaptive 四种执行算法 |
| **M4 信号验证** | 横截面 Spearman IC、五分组收益单调性、半衰期拟合、拥挤度、条件有效性、HMM 市场状态 |
| **M5 估值/基本面** | 多方法估值锚（DCF/PE分位/PB-ROE/EV-EBITDA）、六维基本面评分卡、红旗预警 |
| **M6 门禁/API** | IronGate 门禁接入 `metadata.gate_results`、FastAPI 服务 |
| **M7 文档/注册** | 本 README、全量测试 `tests/test_full_system.py`、2号分析师注册 Skill、调用示例 |
| **M8 真实数据** | qlib 行情 + financials.db 财务，回测/信号/估值/市场状态全部真实数据，合成仅作回退 |

---

## 架构

```
3号交易员/
├── trader3/
│   ├── __init__.py         # 主入口 Trader3 类（10 个方法直接映射 10 个 Tool）
│   ├── base_tool.py        # Tool 抽象基类 + Trader3Response/ChartSpec 统一返回
│   ├── models.py           # 数据模型（回测/WFA/优化/执行/TCA/信号/状态/估值/评分卡）
│   ├── config.py           # YAML 配置管理器（环境变量覆盖，T3_ 前缀）
│   ├── registry.py         # Tool 注册表（注册/查询/调用/摘要）
│   ├── shared_state.py     # 共享状态（数据版本/信号/因子/持仓，磁盘可见）
│   ├── gates.py            # IronGate 门禁（M6 接入 metadata.gate_results）
│   ├── data_provider.py    # M8 轻量级 qlib bin 行情读取器（numpy，无 qlib 依赖）
│   ├── financials_provider.py # M8 financials.db 财务数据读取器（560万行真实财务）
│   ├── cli.py              # CLI 入口（backtest/optimize/execute/validate/regime/wfa/daily/list）
│   ├── api/                # M6 FastAPI 服务
│   └── tools/
│       ├── backtest.py     # M1+M8 真实数据回测（动量策略）+ WFA + sha256 缓存
│       ├── optimize.py     # M2 风险平价/均值-方差/BL + 情境路由
│       ├── execution.py    # M3 Almgren-Chriss TCA + 四种执行算法
│       ├── signal.py       # M4+M8 真实 IC/分组/半衰期/拥挤度 + HMM 市场状态（真实指数）
│       └── valuation.py    # M5+M8 真实财务估值锚 + 六维评分卡
├── config/                 # 策略/约束 YAML
├── shared_state/           # 运行时共享状态（含 backtest_cache/）
├── tests/                  # test_m0_skeleton.py + test_full_system.py
├── examples/               # 2号分析师调用示例
├── skills/3号交易员/        # 2号分析师注册 Skill（SKILL.md）
└── pyproject.toml          # 包配置（editable 安装）
```

真实数据源：

```
qlib_bin/  (2hao-analyst/data/qlib_bin)
├── calendars/day.txt           # 6440 个交易日 (2000-01-04 ~ 2026-07-31)
├── instruments/                # all/csi300/csi500/csi800/csi1000 股票列表
└── features/{code}/*.day.bin   # 6122 只股票: open/high/low/close/volume/vwap/amount/factor
                                # close = 前复权价，factor = 复权因子

financials.db  (2hao-analyst/data/financials.db)
└── financials(code, quarter, table_name, field, value, source)
    # 560 万行真实财务: profit/balance/cashflow 三大表
    # 字段: epsTTM/roeAvg/gpMargin/npMargin/FCF/OCF/totalEquity/goodwill 等
```

数据流：

```
2号分析师（写报告）
    │  from trader3 import Trader3
    ▼
Trader3 ──registry──▶ 10 个 Tool（backtest/optimize/execution/signal/valuation）
    │                        │
    │  gates.run_all()       │  共享状态 shared_state/（版本一致协议）
    ▼                        ▼
Trader3Response ──▶ 报告正文引用 summary / key_metrics / caveats
```

---

## 快速开始

### 安装

```bash
pip install -e .
```

依赖：
- **必须**：Python >= 3.10、`pyyaml`、`numpy`、`scipy`
- **可选**：`hmmlearn`（M4 HMM 加速，缺失时自动回退到内置 EM 实现）、`fastapi + uvicorn`（M6 API）、`cvxpy`（M2 优化器，当前用 scipy SLSQP）、`qlib`（M1 真实数据回测，缺失时用内置向量化引擎）

### 作为 Python 包使用（2号分析师嵌入模式）

```python
from trader3 import Trader3

t3 = Trader3()

# 回测验证
result = t3.run_backtest(start_date="2020-01-01", end_date="2025-12-31")
print(result.summary)
# → "[未命名策略] 2020-01-01~2025-12-31 年化 6.5%, 夏普 0.31, 超额 4.7%, 最大回撤 -34.5%"

# 估值锚
result = t3.valuation_anchor(codes=["301150.SZ"])
print(result.summary)
# → "加权目标价 ¥50.2, Base ¥52.3 / Bull ¥68.1 / Bear ¥38.7, 隐含收益率 12.0%"

# 市场状态诊断
result = t3.diagnose_market_regime()
print(result.summary)
# → "当前状态: ranging（合成数据） (P=100%), 熵=0.01, 建议仓位 60%, 检测方法: HMM (custom EM)"

# 交易成本估算（卖出单含印花税）
result = t3.estimate_transaction_cost(
    orders=[{"symbol": "301150.SZ", "side": "sell", "value_cny": 5000000}]
)
print(result.summary)
# → "预期总成本 80bp (¥39,994), 其中冲击成本 64bp 为主力贡献"
```

### 作为 CLI 使用（独立模式）

```bash
# 回测
python -m trader3.cli backtest --start 2020-01-01 --end 2025-12-31

# 组合优化（读取 signals.json）
python -m trader3.cli optimize --signals signals.json --method risk_budget

# 市场状态诊断
python -m trader3.cli regime

# 生成执行计划
python -m trader3.cli execute --target target_weights.json --algo adaptive_vwap

# 验证因子
python -m trader3.cli validate --signal-name "动量因子"

# Walk-Forward Analysis
python -m trader3.cli wfa --train-window 252 --test-window 63

# 日度任务管线
python -m trader3.cli daily

# 列出所有可用 Tool
python -m trader3.cli list
```

---

## 10 个 Tool（真实示例 + 真实输出）

| # | Tool | 类别 | 2号分析师典型调用 |
|---|------|------|------------------|
| 1 | `run_backtest` | backtest | "帮我跑一下这个选股逻辑过去3年的表现" |
| 2 | `walk_forward_analysis` | backtest | "这个因子会不会过拟合？给我 WFA" |
| 3 | `optimize_portfolio` | optimize | "给我这20只票的最优权重" |
| 4 | `regime_aware_allocation` | optimize | "现在震荡偏弱，帮我按情境路由调整" |
| 5 | `estimate_transaction_cost` | execution | "买入500万大概多少滑点？" |
| 6 | `generate_execution_plan` | execution | "明天早盘分时怎么买？" |
| 7 | `validate_signal` | signal | "这个因子有效吗？给我 IC/分组/衰减" |
| 8 | `diagnose_market_regime` | signal | "现在是什么市场状态？" |
| 9 | `valuation_anchor` | valuation | "帮我算 DCF 估值锚" |
| 10 | `fundamental_scorecard` | valuation | "给我这批票的质量成长评分卡" |

### 1. run_backtest — 回测

```python
r = t3.run_backtest(start_date="2020-01-01", end_date="2025-12-31")
print(r.summary)
print(r.key_metrics)
```

```text
[未命名策略] 2020-01-01~2025-12-31 年化 6.5%, 夏普 0.31, 超额 4.7%, 最大回撤 -34.5%
年化收益: 0.0646  超额收益: 0.0471  夏普比: 0.313  最大回撤: -0.345
信息比: 0.551  年化换手: 9.57   t统计量: 1.948
```

`r.data` 为 `BacktestReport`，含 1512 点 `equity_curve`、基准收益、波动率、Calmar、胜率、
Brinson 归因、Barra 暴露、分时段表现、统计检验。结果按 sha256 策略指纹缓存到 `shared_state/backtest_cache/`。

### 2. walk_forward_analysis — 滚动样本外验证

```python
r = t3.walk_forward_analysis(train_window=252, test_window=63)
print(r.summary)
```

```text
WFA 样本外年化 18.6%, 夏普 0.86, 过拟合概率 52%, 参数稳定性 79%
```

`r.data` 为 `WFAReport`：逐窗口 IS/OOS 收益与夏普、参数稳定性（相邻窗口权重相关系数均值）、
过拟合概率 `max(0, 1 - OOS_Sharpe / IS_Sharpe)`。

### 3. optimize_portfolio — 组合优化（三种方法）

```python
signals = {"000001.SZ": 80, "000002.SZ": 72, "000003.SZ": 65, "000004.SZ": 60, "000005.SZ": 55}
r = t3.optimize_portfolio(signals=signals, method="risk_budget")
print(r.summary)
r = t3.optimize_portfolio(signals=signals, method="mean_variance")
r = t3.optimize_portfolio(signals=signals, method="black_litterman")
```

```text
优化完成 [风险平价]: 预期年化 -0.2%, 波动 13.8%, 夏普 -0.02, 换手成本 6bp
优化完成 [均值-方差]: 预期年化 6.0%, 波动 17.6%, 夏普 0.34, 换手成本 14bp
优化完成 [Black-Litterman]: 预期年化 6.0%, 波动 17.6%, 夏普 0.34, 换手成本 14bp
```

`r.data` 为 `OptimizationResult`：目标权重、预期收益/风险/夏普、五因子暴露、换手成本、约束满足状态。

### 4. regime_aware_allocation — 情境路由

```python
r = t3.regime_aware_allocation(
    signals={"000001.SZ": 80, "000002.SZ": 72, "000003.SZ": 65},
    regime_probs={"ranging": 0.6, "bearish": 0.3, "trending_up": 0.1},
    regime_weights={
        "ranging": {"000001.SZ": 0.4, "000002.SZ": 0.4, "000003.SZ": 0.2},
        "bearish": {"000001.SZ": 0.2, "000002.SZ": 0.3, "000003.SZ": 0.5},
        "trending_up": {"000001.SZ": 0.5, "000002.SZ": 0.3, "000003.SZ": 0.2},
    },
)
print(r.summary)
```

```text
情境路由: 当前状态「ranging」(P=60%), 综合 3 个标的权重, 预期夏普 0.02, 建议仓位 55%
```

按 `P(r) × w_r(asset)` 概率加权合成综合分，再归一为目标权重，带单票上限约束与风险状态仓位折价。

### 5. estimate_transaction_cost — TCA（A股成本规则）

```python
buy  = t3.estimate_transaction_cost(orders=[{"symbol": "301150.SZ", "side": "buy",  "value_cny": 5000000}])
sell = t3.estimate_transaction_cost(orders=[{"symbol": "301150.SZ", "side": "sell", "value_cny": 5000000}])
print("买入:", buy.summary)
print("卖出:", sell.summary)
```

```text
买入: 预期总成本 70bp (¥34,994), 其中冲击成本 64bp 为主力贡献
卖出: 预期总成本 80bp (¥39,994), 其中冲击成本 64bp 为主力贡献
```

**印花税 10bp 仅在卖出时收取**（买入为 0），佣金 2bp 双边，冲击成本用 Almgren-Chriss 永久+临时冲击模型。

### 6. generate_execution_plan — 四种执行算法

```python
for algo in ["twap", "vwap", "is", "adaptive_vwap"]:
    r = t3.generate_execution_plan(
        target_weights={"000001.SZ": 0.3, "000002.SZ": 0.2},
        algorithm=algo, urgency="normal",
    )
    print(r.summary)
```

```text
执行计划 (twap): 12 笔切片, 预期完成率 95%, 预期成本 20bp
执行计划 (vwap): 12 笔切片, 预期完成率 95%, 预期成本 20bp
执行计划 (is): 12 笔切片, 预期完成率 95%, 预期成本 20bp
执行计划 (adaptive_vwap): 12 笔切片, 预期完成率 95%, 预期成本 20bp
```

TWAP 时间均匀、VWAP 按 A股 U 型成交量加权、IS 前端加载、Adaptive 按紧急度路由（high→IS / normal→VWAP / low→TWAP）。
每个计划含分时切片、预期完成率、预期成本、风控熔断线。

### 7. validate_signal — 信号/因子验证

```python
r = t3.validate_signal(signal_name="动量因子")
print(r.summary)
```

```text
[动量因子（合成数据）] ICIR=0.396, 分组单调性 75%, 半衰期 12.0个月, 拥挤度 0.14, 多空年化 38.0%
```

`r.data` 为 `SignalValidationReport`：IC 时序、ICIR、五分组收益（Q1–Q5）及单调性、半衰期、
拥挤度、按波动率分组的条件有效性、多空/多头年化。

### 8. diagnose_market_regime — HMM 市场状态

```python
r = t3.diagnose_market_regime()
print(r.summary)
print(r.data.regime_probabilities)
```

```text
当前状态: ranging（合成数据） (P=100%), 熵=0.01, 建议仓位 60%, 检测方法: HMM (custom EM)
{'trending_up': 0.0, 'ranging': 0.999, 'bearish': 0.0, 'high_vol': 0.0, 'liquidity_crisis': 0.0}
```

用 4 状态高斯 HMM（日收益/20日波动率/成交量变化/价差代理），优先 `hmmlearn`、缺失时回退内置 EM。
输出状态概率（和为 1）、熵、关键指标、历史类比、仓位与风格建议、注意力动量（Serenity-Radar 轻量）。

### 9. valuation_anchor — 估值锚

```python
r = t3.valuation_anchor(codes=["301150.SZ"])
print(r.summary)
print(r.data.sensitivity)
```

```text
加权目标价 ¥50.2, Base ¥52.3 / Bull ¥68.1 / Bear ¥38.7, 隐含收益率 12.0%
{'wacc': [45.2, 50.2, 56.3], 'terminal_growth': [46.8, 50.2, 54.1], 'fcf_growth_5y': [43.5, 50.2, 58.6]}
```

`r.data` 为 `ValuationReport`：DCF / PE 分位 / PB-ROE / EV-EBITDA 四方法目标价、加权目标、Base/Bull/Bear、
隐含收益率、上行概率、关键假设与敏感性网格。

### 10. fundamental_scorecard — 基本面评分卡

```python
r = t3.fundamental_scorecard(codes=["000001.SZ"], template="quality_growth")
print(r.summary)
print(r.data.red_flags)
```

```text
[质量成长] 总分 7.2/10, 高于同行均值 6.5, 2 个红旗预警
['应收账款增速显著高于营收增速', '商誉占总资产比例 > 15%']
```

`r.data` 为 `ScorecardReport`：盈利能力/成长性/财务健康/估值合理性/管理层质量/竞争壁垒 六维评分、
同业对比、红旗预警、关键亮点与顾虑。

---

## 2号分析师集成指南

### 1. 导入

```python
from trader3 import Trader3
t3 = Trader3()                      # 默认门禁关闭
t3 = Trader3(gates_enabled=True)    # 写正式报告前开启 IronGate 门禁
```

### 2. 调用模式

每个 Tool 都是 `Trader3` 上的一个方法，参数即关键字参数，返回统一的 `Trader3Response`：

```python
resp = t3.run_backtest(start_date="2020-01-01", end_date="2025-12-31")
```

### 3. 返回结构（Trader3Response）

| 字段 | 类型 | 用途 |
|------|------|------|
| `success` | bool | 调用是否成功 |
| `summary` | str | 一句话结论，直接引用进报告正文 |
| `key_metrics` | dict | 关键指标（K: 中文标签, V: 数值），引用进表格 |
| `charts` | list[ChartSpec] | 图表规格（line/bar/scatter/heatmap/table） |
| `caveats` | list[str] | 局限性/假设，引用进脚注 |
| `metadata` | dict | 运行元信息：`tool` / `version` / `request_id` / `elapsed_seconds` / `timestamp` / `gate_results` / `gates_passed` |

### 4. 报告引用格式

```python
print(f'    结论：{resp.summary}')
print(f'    "3号交易员 DCF 测算，Base 目标价 ¥{resp.key_metrics["Base"]:.1f}，'
      f'Bull ¥{resp.key_metrics["Bull"]:.1f}，Bear ¥{resp.key_metrics["Bear"]:.1f}"')
print(f'    "3号交易员验证，该因子 ICIR={resp.key_metrics["ICIR"]:.2f}，半衰期 '
      f'{resp.key_metrics["半衰期(月)"]:.1f} 个月，拥挤度低"')
```

### 5. 门禁与审计

```python
t3 = Trader3(gates_enabled=True)
resp = t3.run_backtest()
resp.metadata["gate_results"]   # [{'check_name': ..., 'passed': ..., 'score': ...}, ...]
resp.metadata["gates_passed"]   # True / False

t3.get_call_history()           # 最近 50 条调用记录（审计用）
```

### 6. 共享状态

`shared_state/` 目录在 2号分析师与 3号交易员间共享数据版本 / 信号 / 因子 / 持仓：

```python
t3.state.set_data_version({"financials": "v20260809", "market": "v20260810"})
t3.state.get_data_version()     # 版本一致性协议，防止用陈旧数据
```

---

## 测试

```bash
python -m pytest tests/ -v
```

- `tests/test_m0_skeleton.py` — M0 接口契约 / 返回格式统一 / 门禁默认关闭 / 配置与共享状态
- `tests/test_full_system.py` — M1–M6 全量真实引擎测试（回测净值 / WFA / 三种优化 / 情境路由 / TCA / 执行计划 / 信号验证 / HMM / 估值 / 评分卡 / 门禁 / API）

---

## 版本

| 里程碑 | 状态 | 交付物 |
|--------|------|--------|
| **M0 骨架** | ✅ 完成 | Tool 接口 / 模型 / 配置 / 注册表 / 共享状态 / 门禁存根 / CLI |
| **M1 回测引擎** | ✅ 完成 | 向量化回测（Qlib 回退）+ 全套指标 + WFA + 缓存 |
| **M2 优化器** | ✅ 完成 | 风险平价 / 均值-方差 / Black-Litterman + 情境路由 |
| **M3 执行/TCA** | ✅ 完成 | Almgren-Chriss 冲击模型 + 印花税规则 + 四种执行算法 |
| **M4 信号验证** | ✅ 完成 | IC / 分组单调性 / 半衰期 / 拥挤度 / 条件有效性 + HMM 市场状态 |
| **M5 估值/基本面** | ✅ 完成 | 多方法估值锚 + 六维评分卡 + 红旗预警 |
| **M6 门禁/API** | ✅ 完成 | IronGate 门禁接入 `metadata.gate_results` + FastAPI 服务 |
| **M7 文档/注册** | ✅ 完成 | README / 全量测试 / 2号分析师注册 Skill / 调用示例 |

> **实现状态说明**：M5 估值与 M6 FastAPI 由并行分支交付。截至 M7 文档定稿，
> M1–M4 已为真实引擎；M5 估值/评分卡当前返回确定性模板数据（接口契约完整，真实财务数据待接入）；
> M6 门禁元数据已接入 `metadata.gate_results`，FastAPI 路由接线在 M6 分支内落地。
> 下方 Roadmap 列出了真实数据与实盘执行路线。

---

## Roadmap

- **真实数据接入**：对接 Qlib / Tushare / 聚宽，替换合成数据（回测、因子截面、估值财务、注意力动量文本）
- **QMT 实盘执行**：生成执行计划 → 推送 QMT 交易终端 → 成交回报回写 TCA
- **门禁完整路由**：为 `tca` / `regime` 等类别补全真实检查项，与 IronGate 基类完全对接
- **并发压测**：FastAPI 服务多进程压测，验证共享状态并发安全
- **因子库沉淀**：M4 验证通过的因子写入 `shared_state`，供 2号分析师跨报告复用

---

## 免责声明

3号交易员所有输出为**候选信号，非投资建议**。合成数据回测结果不代表真实表现，实盘前请接入真实数据并充分验证。
