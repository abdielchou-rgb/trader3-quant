# Marvis 执行任务：策略进化工厂

> 给 marvis 的具体执行任务。目标：用 GP 遗传编程 + 3号交易员真实数据，进化出有用的交易策略，并把 ETF 策略纳入。
>
> **输出**：`evolve/strategies/selected.json`（筛选后的策略库）+ `evolution_summary.json`（进化摘要）
> **候选信号，非投资建议。**

---

## 一、整体架构

```
3号交易员/evolve/                   策略进化工厂
├── core/gp.py                      GP 引擎（表达式树/遗传操作/适应度）
├── core/parser.py                  表达式解析
├── core/evolution.py               进化引擎（选择/交叉/变异/精英保留）
├── core/data_loader.py             数据加载（qlib 股票 / ETF CSV）
├── core/selection.py               策略筛选门禁（6道门槛）
├── scripts/fetch_etf_data.py       akshare 拉取 ETF 数据（本机有网时）
├── run_evolution.py                主入口
├── data/etf/                       ETF CSV 数据目录
├── evolution_log/                  每代最优记录
└── strategies/                     ★ 最终产出
    ├── selected.json               筛选通过的策略
    └── evolution_summary.json      进化摘要
```

## 二、第一阶段：股票策略进化（已跑通）

**用 qlib 真实行情跑 GP 进化，筛选出有效量价因子策略。**

```bash
cd D:\Claude\projects\3号交易员

# 基础跑法（推荐起始参数）
python evolve/run_evolution.py --source qlib --universe csi300 --n-stocks 60 --gen 15 --pop 50 --top-k 5

# 更大规模（耗时更长）
python evolve/run_evolution.py --source qlib --universe csi300 --n-stocks 100 --gen 30 --pop 80 --top-k 10

# 换股票池（中证500/中证1000）
python evolve/run_evolution.py --source qlib --universe csi500 --n-stocks 80 --gen 20 --pop 60
python evolve/run_evolution.py --source qlib --universe csi1000 --n-stocks 100 --gen 20 --pop 60
```

**已验证结果**（12代×40种群×60股，~64秒）：
```
📊 进化完成: qlib/csi300 | 12代×40 | 64s | 评估候选:52 通过:3
[1] add(log(vwap), mul(mul(low, mul(log(ts_corr(high,high,20)), ts_std(delay(close,10),5))), log(zscore(volume))))
    IC=0.053 ICIR=0.277 单调性=75% 多空年化=105% ✅
[2] add(log(vwap), mul(sub(volume, sqrt(ts_min(ts_mean(vwap,20),5))), log(zscore(volume))))
    IC=0.051 ICIR=0.275 单调性=75% 多空年化=96% ✅
[3] add(log(vwap), mul(sub(volume, log(vwap)), log(zscore(volume))))
    IC=0.050 ICIR=0.273 单调性=75% 多空年化=97% ✅
```

## 三、第二阶段：ETF 策略进化

**需要先在**有网的本机**拉取 ETF 数据，再进 ETF 面板进化。**

### 3.1 拉取 ETF 数据（本机执行，需 akshare）

```bash
# 本机安装 akshare
pip install akshare

# 拉取 50 只主流 ETF 日线 → evolve/data/etf/
python evolve/scripts/fetch_etf_data.py --out evolve/data/etf --n 50

# 或者只拉特定代码
python evolve/scripts/fetch_etf_data.py --codes 510300 510500 510050 159915 588000
```

产出：`evolve/data/etf/{code}.csv`，每只 ETF 一个文件：
```
date,open,high,low,close,volume
2020-01-02,3.2,3.25,3.15,3.22,1000000
```

### 3.2 ETF 进化

```bash
# ETF 面板进化（需先拉数据）
python evolve/run_evolution.py --source etf --etf-dir evolve/data/etf --n-stocks 30 --gen 15 --pop 50 --top-k 5
```

## 四、第三阶段：进化策略 → 3号交易员回测验证

筛选出的策略表达式可以直接给 3号交易员回测（需把表达式转成信号）。

**对接思路**：`evolve/strategies/selected.json` 里的每个 `expr`，可以用 `core.gp.evaluate()` 生成信号，喂给 `trader3` 的 `run_backtest` 或 `validate_signal` 做最终验证。

```python
# 思路示例（marvis 执行时按此扩展）
import sys; sys.path.insert(0, 'evolve')
from core.gp import evaluate, parse_expr
from core.data_loader import load_qlib_panel

panel, fwd = load_qlib_panel(universe='csi300', n_stocks=60)
node = parse_expr("ts_corr(low, zscore(volume), 10)")
signal = evaluate(node, panel)
# → 把 signal 交给 trader3.run_backtest 验证
```

## 五、筛选门禁说明

筛选出的策略必须全部通过 6 道门槛：

| 门槛 | 阈值 | 说明 |
|------|------|------|
| IC | \|IC\| > 0.02 | 相关性门槛 |
| ICIR | > 0.15 | 稳定性（A股量价实际水平） |
| 单调性 | > 0.4 | 分组收益单调 |
| 多空年化 | > 0.10 | 年化多空差 > 10% |
| 复杂度 | < 25 节点 | 可解释性 |
| 去重 | 相似度 < 0.7 | 与已选策略去重 |

## 六、Marvis 每日/定期任务建议

```
每日（盘后）：
  1. 检查数据新鲜度（qlib_bin 是否更新到最新）
  2. 跑一次股票进化（--gen 15 --pop 50，~2-3分钟）
  3. 新增策略过筛选门禁 → 追加到 selected.json
  4. 更新 evolution_summary.json

每周：
  1. 拉取 ETF 数据（本机有网）
  2. ETF 进化一轮
  3. 用 top 策略喂给 3号交易员回测验证
  4. 沉淀到策略库，供 2号分析师引用
```

## 七、参数调优指南

| 参数 | 影响 | 建议 |
|------|------|------|
| `--gen` | 进化代数，越大越深但过拟合风险高 | 15-30 |
| `--pop` | 种群大小，越大多样性越好 | 50-100 |
| `--n-stocks` | 股票数，越大 IC 越稳但越慢 | 60-100 |
| `--seed` | 随机种子，固定可复现 | 42 |

## 八、注意事项

- **过拟合风险**：进化会倾向复杂表达式。门槛里的复杂度限制（<25节点）就是防这个。
- **数据窗口**：当前用 2020-2024 训练，建议换窗口验证（如 2018-2022）看是否过拟合。
- **ETF 数据**：需要本机跑 `fetch_etf_data.py` 拉取，沙箱里没有 akshare。
- **诚实标注**：所有进化出的策略是**候选信号**，必须经过 3号交易员回测 + 门禁双重验证才能进入 2号分析师报告。
