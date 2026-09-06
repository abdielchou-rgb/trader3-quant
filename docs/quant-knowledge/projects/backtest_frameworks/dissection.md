# 五大开源回测/交易框架解剖

> 来源：网络检索（官方文档 + GitHub API 核验）
> 日期：2026-09-06

## 一句话总览

| 框架 | 本质 | 一句话 |
|------|------|--------|
| vectorbt (9k★) | 研究者向量化 | 把策略压成矩阵一次算完，秒级参数扫描 |
| backtesting.py (8.9k★) | 研究者半事件化 | bar 循环 + 向量账本，API 极简 |
| Lean (21.5k★) | 机构级事件驱动 | 全可插拔模型，研究→实盘同算法 |
| nautilus_trader (28.4k★) | 生产级事件驱动 | Rust-native 确定性引擎，研究/实盘同内核 |
| freqtrade (54k★) | 加密运营平台 | 声明式信号 + hyperopt + dry-run（仅 crypto） |

## 撮合模型对比（最关键维度）

| 框架 | 撮合时点 | 防前视 | 盘口 |
|------|----------|--------|------|
| vectorbt | 信号与成交同 bar 同价（默认 close）⚠️ | 需自 fshift(1)+Open | 无 OB |
| backtesting.py | 市价单默认下一根 bar 开盘成交 | 限价用 bar 内 High/Low 触碰 | 无 OB |
| Lean | 市价单仅市时成交 + **陈旧价纪律** | 限价单次根 bar 穿透 + 跳空按 open | EquityFillModel 明言 OHLC 无法建模深度部分成交 |
| nautilus | 确定性事件驱动 | 全 TIF + 条件单 + OCO + iceberg | 有 OB（L2） |
| freqtrade | 信号收盘评估，入场按开盘价 | **显式盘内时序假设**清单 | 无 OB |

**freqtrade 最值得借鉴**：`--timeframe-detail 5m` —— 只对有持仓/信号的活动 K 线用低周期数据重放，模拟止损/ROI 触发。跑同策略 L1 vs L2 结果差异大 = 策略在吃撮合假设红利。

## 架构取舍本质

- **向量化**（vectorbt）：决策写成无状态矩阵换计算规模；前提是决策无路径依赖；防前视靠用户纪律
- **事件驱动**（nautilus/Lean）：显式建模订单生命周期与成交时序换真实性 + 实盘复用；慢

**不是互斥，是两层**：指标/信号层向量化，执行/成交层事件化。

## trader3 该吸收什么（按优先级）

| # | 迁移设计 | 来源 | 优先级 |
|---|----------|------|--------|
| 1 | 成交时点纪律化：决策 bar 收盘信号 → 成交下一 bar 开盘，杜绝同 bar 出同 bar 成交 | backtesting/Lean stale-price | **P0** |
| 2 | FillModel 接口分层：市价/限价/止损 + 跳空处理 + 涨跌停封板不成交 | Lean EquityFillModel | **P0** |
| 3 | 订单生命周期对象化：OrderIntent → Order + OrderEvent 异步回执 | nautilus/Lean | **P1** |
| 4 | 两档撮合：L1 bar 级 OHLC 触碰；L2 仅活动 K 线分钟重放 + 容量检查 | freqtrade | **P1** |
| 5 | A股费用非对称函数化（卖出印花税+过户费+最低佣金） | backtesting | **P1** |
| 6 | 指标层向量化 + 预计算全参量列 + optuna | freqtrade/vectorbt | **P2** |

## Order book 模拟：别过度设计

**结论：做"分层确定性撮合 + 可切换"，把 OB 留给证明需要它的策略。**

- **L0 快速网格**：按 bar 规则撮合（默认），大参数扫描速度优先
- **L1 bar 级 OHLC 触碰**（日常研究默认）：限价/止损检查 High/Low/Open + A股涨跌停约束 + 非对称费用
- **L2 选择性盘内重放**：只对活动 bar 回放分钟线 + 参与率容量检查（拦截"虚假大单全成交"）
- 护栏：L0/L1/L2 一键切换报告 PnL 差，差异>10-15% 提示"策略在吃撮合假设"
- 金句：**没有 L2 逐笔数据就别假装有盘口**（Lean EquityFillModel 的 TODO 是最好的警句）

## 源码该看哪里

- vectorbt: `vectorbt/portfolio/nb/simulate_nb.py`（引擎心脏）
- backtesting.py: `backtesting/_broker.py` + `backtesting.py`
- Lean: `Common/Orders/Fills/EquityFillModel.cs`（一个文件讲清全部成交语义）
- nautilus: `crates/`（v2 Rust）+ examples/backtest
- freqtrade: docs "Assumptions made by backtesting" 一节
