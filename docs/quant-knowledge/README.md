# Quant 全景认知库 — 主索引

> 目标：站在巨人肩膀上，系统化吸收顶级项目、论文、方法论
> 原则：带着手术刀解剖，产出结构化笔记 + 迁移计划，不空读
> 建立：2026-09-06 ｜ 解剖第一轮完成（4 个项目并行）

---

## 🗺️ 项目解剖地图

| 状态 | 项目 | 定位 | 解剖文档 | 核心吸收点 |
|------|------|------|----------|------------|
| ✅ 完成 | **qlib** | 端到端量化平台 | `projects/qlib/dissection.md` | 表达式 AST、窗口自推导、learn/infer 防泄漏、Recorder 实验链 |
| ✅ 完成 | **mlfinlab/AFML** | 金融 ML 方法论 | `projects/mlfinlab/dissection.md` | 三重屏障/meta-label/唯一性权重/CPCV/DSR/PBO |
| ✅ 完成 | **5 大回测框架** | 横向对比 | `projects/backtest_frameworks/dissection.md` | 分层撮合(L0/L1/L2)、成交时点纪律、FillModel 分层 |
| ✅ 完成 | **因子挖掘论文** | 22 篇已核实 | `projects/alpha_research/dissection.md` | AlphaGen 协同目标、验收闸门、RD-Agent-Quant |
| ⏳ 待解剖 | **riskfolio-lib** | 组合优化 | - | HRP/NCO/CVaR/BL |
| ⏳ 待解剖 | **alphagen 源码** | 协同因子 RL | - | calc_pool_* 奖励实现 |
| ⏳ 待解剖 | **qlib 回测源码** | 撮合细节 | - | HighFrequencyExecutor / 成本模型 |

## 🔑 第一轮解剖的三大战略结论

1. **trader3 缺的不是"更好的算子"，是"研究档案层"**：运行时 telemetry（已有）与死的可检索实验记录（缺）是两个正交维度。优先补轻量 experiment log（仿 qlib Recorder 三段式但自研）。
2. **GP 进化缺的不是更多因子，是"验收闸门 + 协同目标"**：把适应度从单因子 IC 升级为组合级 RankIC + 残差空间互相关惩罚；候选入库前过 Deflated Sharpe（n_trials=累计个体评估数）+ 20 seed 均值上报。
3. **回测缺的不是盘口，是"成交时点纪律 + 分层撮合"**：默认决策收盘→成交下一开盘；L0/L1/L2 可切换 + PnL 差异护栏；没有 L2 逐笔数据就别假装有盘口。

## 迁移路线图（30 天行动）

见 `ROADMAP_30D.md`

## 教训（本会话踩坑）

- Windows PowerShell 无 `&&`/`cd /d`/`~` 展开 → 统一用 workdir 参数 + 全路径
- 项目 CLAUDE.md 曾被调试脚本误覆盖 → 每次 git 提交前 `git diff --stat` 检查
- clone 到中文路径 + 网络不稳 → 用 `D:\Temp\opencode` + fetch.py 镜像更稳
