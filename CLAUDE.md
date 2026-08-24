# 3号交易员 — Agent 行为约束

> 约束你（Agent）在 **3号交易员 (trader3)** 项目中的行为。核心：**量化工程方法论** + **数据纪律**。
> 全局 skill `engineering-methodology`（Matt Pocock 方法论浓缩版）在任何编码/调试/审查任务时自动生效，本文件是其在 trader3 的具体化。

---

## 项目定位

双模式量化交易引擎：
1. **独立模式** — CLI / FastAPI / 定时调度（`python -m trader3.cli ...`）
2. **嵌入模式** — 2号分析师 `from trader3 import Trader3` 调用 10 个工具

## 目录结构

```
trader3/            # 核心包（10 个 Tool）
evolve/             # 策略进化工厂（GP 挖因子，marvis 跑）
config/             # 策略/约束 YAML
shared_state/       # 运行时共享状态（与 2hao 共享）
tests/              # test_m0_skeleton.py + test_full_system.py
skills/             # 2hao 注册 Skill
examples/           # 调用示例
```

## 数据源（重要）

- **行情**: `D:\Claude\projects\2hao-analyst\data\qlib_bin\` — 6440 交易日 / 6122 股票，numpy 直接读（无需 qlib 安装）
  - `close` = **前复权价**，`factor` = 复权因子
- **财务**: `D:\Claude\projects\2hao-analyst\data\financials.db` — 560 万行 / 5259 股票，最新 2026-06-30
- **数据纪律**: 真实数据优先，合成数据仅回退且必须标注"（合成数据）"

## 代码工程方法论（改 trader3 代码时）

### 1. 改代码 → TDD
- 先写失败测试 → 再写恰好通过的最小实现 → 一次一片
- 测试通过公共接口（`Trader3` 方法 / Tool 的 `execute`），不测私有实现
- 现有测试在 `tests/`，改完必须 `python -m pytest tests/ -v` 全绿

### 2. 报错 / 回测异常 / Gate 失败 → 先建反馈环再修
- **铁律：无根因调查不修复**。先建一个能复现 bug 的最小命令/测试
- 假设必须可证伪；修复前先写回归测试
- 修完清理临时探针，把正确假设写进 commit

### 3. 审查代码 → 双轴并行
- **Standards 轴**：是否符合现有代码风格（base_tool/registry/gates 模式）
- **Spec 轴**：是否实现任务要求
- 两轴分开报告，不合并

### 4. 设计新 Tool / 改接口 → 先商定 seam
- 新 Tool 必须继承 `BaseTool`，返回统一 `Trader3Response`
- 输入/输出 schema 先定，再实现

### 5. 策略进化 → 过门禁
- evolve 产物必须过 6 道筛选门禁（IC/ICIR/单调性/多空/复杂度/去重）
- 进化策略是**候选信号**，须经 `run_backtest` 回测验证才能进 2hao 报告

---

## 铁律

1. **数据必须带来源** — 真实数据 / 合成数据标注清楚，禁止编造
2. **门禁必须过** — 回测/信号/估值产出过 IronGate，`gates_passed` 为 false 要说明原因
3. **量化输出是候选信号，非投资建议**
4. **测试必须绿** — 改代码后 `pytest` 全绿才能交付

---

## 关键命令

```bash
# 测试
python -m pytest tests/ -v

# 冒烟验证
python -c "from trader3 import Trader3; t=Trader3(); print(t.diagnose_market_regime().summary)"

# 回测 / 市场状态 / 估值
python -m trader3.cli backtest --start 2020-01-01 --end 2024-12-31
python -m trader3.cli regime
python -c "from trader3 import Trader3; print(Trader3().valuation_anchor(codes=['600519']).summary)"

# 策略进化
python evolve/run_evolution.py --source qlib --universe csi300 --gen 15 --pop 50
```
