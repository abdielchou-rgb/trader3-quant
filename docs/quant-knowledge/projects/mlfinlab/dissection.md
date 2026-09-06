# mlfinlab / López de Prado 方法论解剖

> 来源：网络检索（GitHub API + CrossRef 核验）
> 日期：2026-09-06

## ⚠️ 前置关键发现

**开源仓库 master 已被掏空成"空壳"**：核心文件（labeling.py / cross_validation.py）全是纯 docstring 桩（函数体只有 pass），但 docstring 精确标注了每个函数对应的 AFML 书 Snippet 编号与页码——成了唯一仍开源可用的"算法说明书"。README 自述降级为 issue 收集站，License 改 "All rights reserved"（闭源商业化），2023-10 停更。

**结论：trader3 不应 import mlfinlab，应依据 AFML 书 Snippet 规范自研最小实现**，但接口/列名约定（t1/trgt/pt/sl/side/bin）是最佳对齐参照。

## A. 三重屏障标签法（AFML Ch.3）

- 每个进场事件同时架 3 条退出边界：上屏障(止盈=pt×σ)、下屏障(止损=sl×σ)、垂直屏障(最长持有)
- 标签 = 实际最先被触碰的边界：上→+1，下→-1，到期→0
- σ 用日波动率缩放 → 跨波动率/牛熊可比
- **A股价值**：涨跌停污染固定持有期标签；T+1 买入当日不可卖 → 垂直屏障最短 1 日；σ 缩放让高波动妖股与低波动白马公平可比
- 反直觉：标签区间不等长 → 相邻样本重叠 → 破坏 IID → 引出样本权重 + Purged CV

## B. Meta-Labeling（元标签）

- 两层：主模型预测方向(side)，元模型预测"这笔交易扣费后是否盈利"(二分类 0/1)
- 价值：解耦"方向对错"与"该不该下单"，把风控内嵌进模型而非后置规则
- **A股落地**：GP 因子给主方向 → 元特征=信号强度+波动+流动性+大盘+成本 → p_meta>θ 才真开仓 → 降换手、删掉手续费+滑点后不赚钱的交易

## C. 样本权重（唯一性权重）

```
并发度 concurrency[t] = Σ_i D[t,i]   # 同根bar上同时"活着"的样本数
唯一性 uniqueness[i] = Σ_{t∈区间} (1/concurrency[t]) / 区间长
权重 w_i = uniqueness[i]（可叠加时间衰减/收益权重）
```

- 金融标签有区间 → 重叠样本共享同一段价格路径 → 按 IID 训练 = 信息重复计数
- **A股价值**：N股×T日面板，相邻标签窗口重叠 + 全市场同日共享一根指数 bar
- 纯截面打分（每 bar 一行）无此问题 —— 关键取舍点

## D. 回测严谨性三件套

1. **CPCV/CSCV**：T 观测拆 S 块，穷举 C(S,S/2) 个"半训练/半测试"组合 → 一整族 OOS 路径而非一条，既防前视又暴露单路径偶然性
2. **PBO**：`λ=ln(ω/(1-ω))`，ω=IS冠军在OOS的相对位次；P(λ<0) = IS最优策略OOS垫底概率
3. **Deflated Sharpe**：
```
SR* = √V[{SR_n}]·[(1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(N·e))]   # N=试验次数, γ≈0.5772
DSR = PSR(SR*)；试过N次后"最大夏普"的期望本身就>0 → 门槛自动抬高
```

- **A股价值**：GP 一代几百个体×几百代 = 数万次试验，不惩罚试验次数任何夏普都是虚的

## 落地到 trader3 的接口草案

```
trader3/factor/labeling.py        # A+B：三重屏障 + meta-label
trader3/factor/sample_weights.py  # C：唯一性/时间衰减权重
trader3/backtest/cv.py            # D：PurgedKFold / CombinatorialPurgedKFold
trader3/backtest/rigor.py         # D：PSR / DSR / MTRL / PBO
```

### GP 进化接入（4 个洞）

1. **PBO/DSR 当候选因子入库门控**：n_trials = 整个进化实验累计个体评估数（非单代）
2. **CPCV 取代单次 train/test 评 GP 因子**：取 OOS 路径族中位数而非一条最佳曲线
3. **三重屏障 + meta-label 给 GP 配执行层**：GP 进化"方向表达式"，meta 判"是否值得开仓"——两层目标解耦
4. **唯一性权重做 GP 预处理**：防代际间隐性信息复用（同一段价格路径被多代重复捡到）

## 论文清单（已核实）

| 文献 | 年份 | 贡献 | 置信 |
|------|------|------|------|
| AFML (López de Prado, Wiley) | 2018 | 方法论母本 | 高 |
| The Deflated Sharpe Ratio (JPM 40(5)) | 2014 | DSR/PSR | 高（DOI 核实） |
| Probability of Backtest Overfitting (JCF) | 2016 | CSCV/PBO | 高（DOI 核实） |
| 10 Reasons Most ML Funds Fail (JPM) | 2018 | 落地纪律清单 | 高 |
| Empirical Asset Pricing via ML (RFS 33(5)) | 2020 | ML 因子组合跑赢线性 | 高 |
