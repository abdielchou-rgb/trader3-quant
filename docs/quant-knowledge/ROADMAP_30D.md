# trader3 迁移路线图 — 从解剖到落地的 30 天行动计划

> 基于第一轮解剖结论。原则：低算力、小步验证、每 3 天一个可跑通增量。

## Phase 1（Day 1-7）：研究档案层 + 回测纪律（P0）

| Day | 任务 | 产出 | 验收 |
|-----|------|------|------|
| 1 | 新建 `trader3/obs/experiment.py`：轻量 Experiment/Recorder（jsonl 落盘 + 可检索） | 实验记录层 | 记一次 run 可 search |
| 2-3 | 给现有 factor 评估加 seed/参数/指标自动记录 | run 档案 | 两次 run 可对比 |
| 4-5 | **成交时点纪律**：backtest 默认决策收盘→成交下一 bar 开盘；加 `trade_on_close` 开关 | 撮合时序修正 | 对照旧结果看差异 |
| 6-7 | **L1 撮合分层**：限价/止损 bar 内 High/Low 触碰 + 涨停不成交/跌停不卖出 | 撮合引擎增强 | 单测覆盖涨停拦单 |

## Phase 2（Day 8-16）：GP 进化升级（P0，纯 CPU）

| Day | 任务 | 产出 | 验收 |
|-----|------|------|------|
| 8-9 | `trader3/factor/labeling.py`：三重屏障 + meta-label 自研（对齐 mlfinlab 列名） | 标签模块 | 单测：涨跌停污染样本正确标 0 |
| 10 | `trader3/backtest/rigor.py`：PSR / DSR / MTRL 自研 | 严谨性模块 | DSR(试10次) > PSR |
| 11-12 | evolve 加**协同适应度**：组合级 RankIC + 残差互相关惩罚（P_orth 空间） | 协同 GP | 挖出因子两两相关<0.5 |
| 13-14 | evolve 加**验收闸门**：候选入库前过 DSR(n_trials=累计) + 滚动 ICIR | 因子库闸门 | 假因子被拦 |
| 15-16 | 真实 qlib 数据跑协同 GP + 闸门全链路冒烟 | 实证 | 正交因子入库 |

## Phase 3（Day 17-24）：因子库 + 对照研究

| Day | 任务 | 产出 | 验收 |
|-----|------|------|------|
| 17-18 | 迁移 Alpha158 骨架（6 大类因子，除以当期值归一）到 factor library | Alpha158 复刻 | 与 qlib 算的 IC 相关 >0.95 |
| 19-20 | LightGBM 选股管线：Alpha158 → LGBM → IC 上报 | ML 基线 | 20 seed 均值 |
| 21-22 | 对照：GP 协同因子 vs Alpha158+LGBM vs 深度小模型 | 三腿对照 | 记录哪个真赢 |
| 23-24 | **唯一性权重**（事件样本时启用）+ 补 CPCV | 样本权重 | 加权 vs 不加权差异 |

## Phase 4（Day 25-30）：LLM 提案器试点 + 复盘

| Day | 任务 | 产出 | 验收 |
|-----|------|------|------|
| 25-26 | LLM 生成候选表达式/模板（低频 API）→ 本地 GP 闸门验证 | LLM 提案试点 | 幻觉被拒率记录 |
| 27-28 | 财报 LLM 因子试点（可选）：文本判断当低相关因子 | 另类 alpha | 与量价因子相关性 |
| 29-30 | 全景复盘：三腿对照结论、更新 CLAUDE.md、写季度路线图 | 复盘文档 | 有数据支撑 |

## 明确不做（止损线）

- ❌ 真 order book 撮合（L2 逐笔数据都没有）
- ❌ import mlfinlab（已闭源空壳，自研最小实现）
- ❌ 本地训大 LLM / 复现 DeepMind AlphaEvolve
- ❌ 分布式 / Rust 重写

## 验证纪律（全程）

- 每步 TDD：先失败测试 → 最小实现 → 全绿
- 每个新模块过质量门：ruff + mypy（mypy files 白名单更新）
- 每次进化实验记录 experiment run（Phase 1 建的档案层）
- 测试基线推进时同步 CLAUDE.md
