# 自动因子挖掘 / Alpha 进化 / ML 选股 — 调研报告

> 来源：Semantic Scholar + OpenAlex + GitHub API + Bing（arxiv/DDG 本机被阻断）
> 日期：2026-09-06 ｜ 主清单 22 篇均已核实存在

## ⚠️ 先纠 4 个事实点

1. **arXiv 2201.09745 ≠ AlphaGen**——AlphaGen 正确编号是 **2306.12964**（KDD'23, Yu Shuo et al., 中科院计算所）
2. **"AlphaX (Google GNN)" 未能核实**——最接近的是 Qlib 系 HIST/IGMTF
3. **"微软 AlphaEvolve" 未能核实**——检索到的 AlphaEvolve 是 **DeepMind 2025 LLM+进化编程智能体**（2506.13131），非金融，但架构最启发我们
4. Gen-Alpha / AlphaGPT 等名称未确认公开版本

## 研究路线图（低算力个人）

| 级别 | 内容 | 算力 | 时间 |
|------|------|------|------|
| **L1 公式化因子强化** | 现有 GP 从"挖单因子"升级为"挖协同集合"：组合级 IC 适应度 + 互相关惩罚 + 层级搜索空间 + QD 多样性 | 纯 CPU | 1-3 月 |
| **L2 深度打分器** | Alpha158 喂 LightGBM/XGBoost/MLP + 多 seed 上报；深度时序做对照 | 单卡 ≤8GB | 6-12 月 |
| **L3 LLM 因子** | LLM 当"提案者"→本地 GP 当"验证者"拒绝幻觉（Chain-of-Alpha/RD-Agent-Quant 思路） | LLM API | 12-24 月 |

> QLib 官榜显示：**Alpha158 表格上 XGBoost/Linear 常强于深度模型，原始量价 Alpha360 上深度模型占优**——数据形态决定模型选择。

## 精选论文（22 篇已核实）

### G1 公式化 Alpha 挖掘（我们的直接对照区）
- **AlphaGen** (2306.12964, KDD'23)：RL+掩码动作搜协同因子集合，奖励面向组合级——**我们最该抄的下一个功能**
- **AutoAlpha** (2002.08245, 2020)：层级进化挖因子
- **AlphaForge** (2406.18394, AAAI'25)：挖 + 动态组合框架
- **QuantFactor REINFORCE** (IEEE TSP 2025, DOI 10.1109/tsp.2025.3576781)：方差上界 REINFORCE 挖稳态因子——把滚动 IC 方差写进我们 evolve 目标
- **Chain-of-Alpha** (2508.06312, 2025)：LLM 链式推理挖 alpha
- **AlphaEvolve** (2506.13131, DeepMind 2025)：LLM 提案 + 进化搜索（算力警示 + 架构借鉴）
- **101 Formulaic Alphas** (1601.00991, Kakushadze)：因子模板参照系

### G2 深度/GNN
- **qlib** (2009.11189)、**TRA** (2106.12950)、**HIST** (2110.13716 概念图)、**IGMTF** (2109.06489)

### G3 LLM 读财报
- **Financial Statement Analysis with LLMs** (2407.17866, 2024)：财报文本判断有超额预测力——最便宜另类 alpha 增量
- **AlphaFin** (2403.12582)：RAG 财务分析

### G5 LLM 多智能体因子工厂
- **RD-Agent-Quant** (2505.15155, NeurIPS'25)：因子与模型联合进化自循环（<10 成本达 2× ARR 官方声明）
- **RD-Agent** (2505.14738)：通用数据科学版

## evolve 升级建议（按优先级）

- **S1 (P0)**：适应度从"单因子 IC"→"组合级 RankIC + 与已有因子/Barra 残差互相关惩罚"。P_orth 正交残差化正好提供已占用方向先验，fitness 应作用在残差空间上。锚点：AlphaGen 协同目标
- **S2 (P0)**：加"验收闸门"：滚动 IS/OOS + 20 次多 seed 均值±std + 半衰期/滚动 ICIR + 最大相关上限（QLib 上报纪律）
- **S3 (P1)**：搜索空间结构化（层级模板 + bloat 惩罚）+ Quality-Diversity 保多样性防早熟
- **S4 (P1-P2)**：LLM 当变异算子混合路线，本地闸门拒绝幻觉
- **S5 (P2)**：P_orth 风格剥离 vs HIST 式概念中性化对照实验

## 未核实风险

- arxiv 全文被阻断 → 各论文"一句话核心"基于标题+摘要，机制级需后续抓 PDF
- X 社区热点未能直抓（网络限制）
- "深度学习因子失效"争议未检索到严肃锚点；最强反方证据 = QLib 官榜可复核数据点
