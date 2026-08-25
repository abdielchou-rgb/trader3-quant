# evolve 工厂 v3 调研：AlphaGen 与 RD-Agent(Q)

- 调研日期: 2026-08-25
- 目的: 为 GP 因子工厂的下一代（v3）选型提供依据
- 结论先行: **双轨引入**——AlphaGen 作为"协同因子挖掘器"与现有 GP 并联，
  RD-Agent(Q) 作为研究流程自动化骨架试点；两者产出一律过 trader3 既有门禁
  （CPCV/DSR/衰减监控）后才可入 selected.json

---

## 一、AlphaGen（中科院计算所 MLDM 组）

| 项 | 内容 |
|---|---|
| 出处 | KDD 2023 ADS《Generating Synergistic Formulaic Alpha Collections via RL》(arXiv:2306.12964)；GitHub ICT-FinD-Lab/alphagen ~1.2k stars，持续维护 |
| 2026 进展 | HARLA（LLM 辅助混合式公式因子发现）发表于 Frontiers of Computer Science 2026 |
| 核心思想 | **以下游组合模型表现作为 RL(PPO) 的 reward** 直接优化"协同因子集合"，而非逐个因子的独立 IC——传统 GP/gplearn 挖出的因子彼此高度相关，组合后增益枯竭 |
| 数据接口 | `alphagen_qlib` 子模块原生读 qlib 数据；特征集 {open,close,high,low,volume,vwap} 与 trader3 面板字段**完全一致**；实验市场即 CSI300/CSI500 |
| 基线对照 | 内含修改版 gplearn（GP 单因子 IC 适应度）与 DSO |

### 对 trader3 的意义（差距诊断）

evolve 工厂当前的选择逻辑 = 单因子 fitness 排序 + token-Jaccard/相关性去重。
AlphaGen 论文的核心论断正是这种"先挖后合"流程的结构性缺陷：
> "传统方法逐个挖 alpha，忽略它们之后会被组合使用的事实"

基线卡实证也支持这一点：TopK 等权合成(K=2) 未跑赢最优单因子（F2 拖累），
说明我们的合成层缺乏"协同性"目标。ICIR 加权是改良，但仍是事后加权；
AlphaGen 是把协同性放进搜索目标。

### 引入方案（v3-α）

```
新候选源: AlphaGen PPO miner（qlib 数据适配层对接现有 qlib_bin）
     ↓ 产出的因子表达式集合
既有门禁: CPCV(新) → DSR → 衰减监控 → 行业暴露检查
     ↓
selected.json（GP 与 AlphaGen 两源合并排序）
```

工程量评估：主要工作在 qlib_bin→alphagen_qlib 数据适配器（格式已知）+
GPU/长时训练环境。表达式语法与 evolve/core/parser 高度相似（同为算子树），
可写转换器统一入库。

## 二、RD-Agent(Q)（Microsoft）

| 项 | 内容 |
|---|---|
| 出处 | NeurIPS 2025《R&D-Agent-Quant》；GitHub microsoft/RD-Agent ~14.2k stars，MIT 许可，v0.8.0 (2025-11) 活跃 |
| 定位 | **首个数据中心多智能体量化全栈研发框架**：Research(假设生成/知识森林) ↔ Development(Co-STEER 代码生成+实盘回测验证) ↔ Feedback(多臂老虎机调度方向) |
| 场景 | fin_factor / fin_model / fin_quant(因子×模型联合进化) / **fin_factor_report（读财报自动提因子）** |
| 实证 | <$10 成本实现约 2× 于经典因子库的 ARR、因子数少 70%；超越深度时序模型 |
| LLM 后端敏感性 | o1 > GPT-4.1 > 其余；**gpt-4o-mini 推理能力弱导致表现差** ——与 BCID 项目"4o-mini 提取需人工校准"的经验互证 |
| 底座 | 原生构建于 qlib 之上 |

### 对 trader3 的意义（差距诊断）

RD-Agent(Q) 自动化的正是我们手工维护的研究闭环（假设→代码→回测→反馈→调度），
且同样以 qlib 为底座。差异在于它把"研究方向选择"交给多臂老虎机、把代码生成
交给 Co-STEER——这是 v3 若追求全自动进化的现成骨架，避免自研调度器。

### 引入方案（v3-β，试点级）

```
rdagent fin_factor --scenario=trader3   # 数据侧替换为 qlib_bin 只读适配
```
- 试点目标：对比 RD-Agent 产出的因子库 vs 本地 GP 工厂在相同 OOS 层的表现
- 成本项：LLM API 费用（论文口径 <$10/轮）；需配置强推理后端（o1/GPT-4.1 级）
- 风险：其 Validation 单元自带回测语义，可能与 trader3 门禁冲突——原则：
  **它的产物仍视为候选，最终裁决权在本地方禁**

## 三、邻近生态速览

| 项目 | 会议 | 增量点 | 备注 |
|---|---|---|---|
| AlphaForge | AAAI 2025 | 挖掘+**动态组合**（按日选择因子子集） | 解决静态等权合成缺陷的另一思路 |
| AlphaQCM | fork 衍生 | 分布式 RL 改进 reward 密度 | 与下条负结果同因 |
| alpha-harness 个人实验 | 2026 | **负结果**：LLM pick-mode warm-start 会稀释 PPO 的 reward 密度反而伤 IC；compose-mode 才有效 | 引以为戒：LLM 先验注入方式很关键 |
| HARLA | FCS 2026 | LLM 辅助混合发现（AlphaGen 同组续作） | 关注其 prompt 设计 |

## 四、风险与成本

1. AlphaGen RL 训练需要 GPU 与小时级训练时长；CPU 上不可行
2. RD-Agent 强依赖高质量 LLM 后端（4o-mini 级别不够），API 成本随迭代轮数线性增长
3. 两者的因子语法均为自有算子集，需要写 trader3 parser 双向转换器并加测试
4. **纪律红线**：无论哪条产线挖出的因子，都必须通过本地
   CPCV + DSR + 衰减监控 + 行业暴露四道门禁才能进 selected.json——
   这套诚实性基建是我们相对这些学术原型的差异化优势，不能为速度让路

## 五、决策建议

1. 短期（本季度）：不引框架，先做 **AlphaGen 数据适配器 PoC**（只读对接，
   小规模 GPU 冒烟），验证其协同因子在 csi1000 OOS 的表现是否复现论文结论
2. 中期：若 PoC 成立 → evolve v3 = "GP(探索) + AlphaGen(协同) 双矿并行 +
   统一门禁"；RD-Agent(Q) 作为研究流程自动化试点单独立项
3. 持续跟踪：HARLA / AlphaForge 动态组合 / AlphaQCM 分布式 RL
