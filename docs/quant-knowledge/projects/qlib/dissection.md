# qlib 解剖 — 数据管线 + 工作流/实验记录

> 来源：本机源码 `D:\Temp\opencode\qlib_meta\qlib-main`（只读解剖）
> 日期：2026-09-06

## 一句话结论

qlib 最值得吸收的不是算子库，而是 **「表达式树 = 可序列化延迟计算图 + 每节点自报回看窗口 + 全局 datetime 定位器」** 三层耦合设计 —— 让"我要 T~T+N 数据"自动翻译成"底层多取 L 天历史"，同时满足无回看与无泄漏。

## 架构数据流（文字版）

```
存储层(.bin 二进制) → Provider(日历/成分/特征/PIT)
  → 表达式树(Feature叶子 + Elem/Pair/Rolling算子, 运算符重载构造 AST)
  → ExpressionD.expression() 按 (start-lft, end+rgt) 扩窗取数再截回   ← 无泄漏单一事实来源
  → DataHandlerLP 三道加工: _data → _infer / _learn   (learn 处理器 is_for_infer()=False)
  → DatasetH 按时间元组切 train/valid/test            (切分与加工解耦)
  → model.fit / predict
  → workflow: R.start → SignalRecord(pred.pkl) → SigAnaRecord(IC/IR) → PortAnaRecord(回测+风险)
```

## 核心设计模式

| # | 模式 | 佐证 | trader3 迁移 |
|---|------|------|-------------|
| P1 | 表达式=可序列化 AST，`__str__` 即 DSL，缓存 key=表达式串 | base.py 运算符重载 | factor_dsl 升级为 AST 中间表示 |
| P2 | 窗口自推导：get_extended_window_size 精确/ get_longest_back_rolling 哨兵 | ops.py:757-778 | panel_builder 预热天数自动化 |
| P3 | learn/infer 双轨 + processor fit 窗口锁定防泄漏 | handler.py:382-611, processor.py:196-245 | 归一化只 fit 训练段 |
| P4 | 切分=纯时间区间，防泄漏在 processor 不在切分 | dataset/__init__.py | 概念解耦 |
| P5 | Recorder 接口抽象 + 异步日志 + 自动记 argv/git diff | recorder.py | 新建 obs/experiment.py |
| P6 | RecordTemp 产物依赖链 + 幂等 + `<PRED>` 占位 | record_temp.py | SignalRecord→IC→Backtest 级联 |
| P7 | MultiPass 打乱首日 init score 消初始仓位敏感 | record_temp.py:575 | backtest_cv 可补 |

## Alpha158/360 因子族谱

- **Alpha360**：过去 60 日 OHLCV 全部用最新 close/volume 归一（相对价格原始输入），60×6=360 列，**无手工因子**，喂纯 DL。
- **Alpha158**：9 kbar + 4 price + 29 滚动族 × 5 窗 = 158，分 6 大类：
  1. K线单日形态（KMID/KLEN/KUP/KLOW...）
  2. 趋势/动量（ROC/MA/BETA/RSQR/RESI/RSV/IMAX...）
  3. 波动/分位（STD/QTLU/QTLD）
  4. 价量相关（CORR/CORD）
  5. RSI 类累计涨跌（SUMP/SUMN/SUMD）
  6. 量能统计（VMA/VSTD/WVMA/VOLUME）
- 元设计：归一化一律除以"当期值"（close 或 volume+ε），无量纲、跨标的可比。

## 与 trader3 差距

| 维度 | trader3 | 缺失 | 吸收点 |
|------|---------|------|--------|
| 因子计算 | GP DSL + factor_dsl 编译缓存 | 无节点级历史窗口自推导 | P1/P2 |
| 回测严谨性 | purged k-fold + embargo + 缩水夏普（已有！） | 无统一处理器防泄漏链 | P3 |
| 结果落盘 | factor_registry.json + sha256 缓存 | 不可检索/不可跨 run 对比 | Recorder+MLflow |
| 实时遥测 | Prometheus 自实现 ✓ | 缺研究档案（死的可检索记录） | P5/P6 |

**关键洞察**：trader3 观测栈是"活的运行时仪表盘"（telemetry），缺"死的、可检索的研究档案"（experiment log）。二者互补。

## 反直觉发现

1. ewm/expanding 的"无限回看"被数学截断成 `log(1e-6)/log(1-α)` 天
2. Alpha360 反直觉：360 维里没任何手工因子，特征工程交给模型
3. PortAnaRecord 回测结束时间自动回拨一天（"最后一天"边界坑）
4. 污染教训：项目 CLAUDE.md 曾被误覆盖，需定期 `git diff HEAD -- CLAUDE.md` 自查
