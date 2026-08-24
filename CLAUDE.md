# 3号交易员 — Agent 行为约束

> 约束你（Agent）在 **3号交易员 (trader3)** 项目中的行为。核心：**量化工程方法论** + **数据纪律**。
> 本文件反映 2026-08 深度审计修复后的真实状态（三轮修复，117 tests 全绿）。

---

## 项目定位

双模式量化交易引擎：
1. **独立模式** — CLI / FastAPI（默认绑 127.0.0.1，`TRADER3_API_KEY` 设置后强制鉴权）/ 定时调度
2. **嵌入模式** — 2号分析师 `from trader3 import Trader3` 调用 10 个工具

## 目录结构

```
trader3/            # 核心包（10 个 Tool，全部经 IronGate）
  tools/            #   signal / valuation / backtest / optimize / execution
  v2/               # 新版子系统：事件采集/自选股/触发/风控链/纸面交易/comps TTM
  api/              # FastAPI（X-API-Key 鉴权）
evolve/             # GP 因子工厂（fwd 已修正为前向收益；训练≤2023/验证≥2024 强制切分）
config/             # 策略/约束 YAML
scripts/            # update_market_data.py（qlib_bin 增量管线）等
shared_state/       # 原子写共享状态 + paper/（纸面账户）+ _quarantine_pre_audit/
tests/              # 117 个测试；testpaths 已在 pyproject 隔离
```

## 数据源（重要）

- **行情**: `D:\Claude\projects\2hao-analyst\data\qlib_bin\` — numpy bin（close=前复权）
  - **已激活增量管线**: `python scripts/update_market_data.py --apply`
    （指数锚点对齐重建 + day.txt 扩展 + 自动备份/校验/回滚 + data_version 盖章；
    成分股严格追加用 `--stocks`，对齐漂移的股票一律拒绝写入）
  - **bin 加载契约**（data_provider.load_stock）: 剥离首尾 ≤0 占位 → 按 all.txt 上市锚点
    对齐日历；首值>10000 或长度偏差>5 抛 ValueError（调用方须容错跳过该股）
- **财务**: 同目录 `financials.db` — balance 表含 shortLoan/longLoan/bondPayable/cashAssets
  （comps 真 EV 可用）；无公告日字段的表必须走 v2.AnnouncementCalendar 做 asof 对齐
- **数据纪律**: 真实数据优先；合成数据仅回退且必须标注"（合成数据）"；
  估值白名单模式会禁用关键输入为合成值的方法并打"[演示值·禁止引用]"

## 数据时序纪律（审计核心教训）

1. **回测/信号**: t 日收盘产生信号 → **t+1 生效**（pending_weights 模式），成本计生效日
2. **股票池**: 必须 `instruments(universe, asof_date=start)` 按时段过滤，禁止全集回测
3. **涨跌停**: 执行日按板块幅度拦截（主板9.8%/创业科创19.5%/北交29%），停牌=内部0占位
4. **evolve fwd**: `fwd[i_prev] = c[i_curr]/c[i_prev]-1`（信号日挂次日收益）；负 delay 禁止

## 门禁语义（IronGate）

- Gate2 半衰期单位=交易日（阈值42期）；Gate3 独立复检权重向量不信自报布尔
- Gate6 数据新鲜度: data_version.json 缺失时回退校验 qlib day.txt mtime
- Gate7: 产出显式 used_synthetic=True 而 caveats 未标注 → 拦截
- 绕过门禁的路径（直调 .execute() / gates_enabled=False）仅限调试，产物不得外流

## 代码工程方法论（改 trader3 代码时）

### 1. 改代码 → TDD
先写失败测试 → 最小实现 → `py -3.11 -m pytest tests/ -q` 全绿才能交付。
注意 backtest 缓存指纹含 CODE_VERSION——改报告结构/引擎语义时必须递增。

### 2. 报错/异常 → 先建反馈环再修
铁律：无根因调查不修复。修复前先写回归测试。

### 3. 审查代码 → 双轴并行
Standards 轴（风格）与 Spec 轴（需求）分开报告。

### 4. 新 Tool 必须继承 BaseTool，返回 Trader3Response，输入输出 schema 先定

### 5. 策略进化 → 过门禁
进化是候选信号；selected.json 须经 validate（样本外窗口）+ run_backtest 双确认才可进报告。

---

## 铁律

1. **数据必须带来源** — 真实/合成标注清楚，禁止编造；伪造指标（如硬编码 PE/融资余额）一律删除
2. **门禁必须过** — gates_passed=false 的产出要说明原因，下游不得静默采纳
3. **量化输出是候选信号，非投资建议**
4. **测试必须绿** — 当前基线 117 passed

---

## 关键命令

```bash
# 测试
py -3.11 -m pytest tests/ -q

# 冒烟
python -c "from trader3 import Trader3; print(Trader3().diagnose_market_regime().summary)"

# 回测 / 状态 / 估值 / WFA
python -m trader3.cli backtest --start 2024-01-01 --end 2024-12-31
python -m trader3.cli backtest --config "config/3号交易员_core.yaml" --start 2024-01-01
python -m trader3.cli regime

# v2 触发扫描（dry_run 不落库）与每日管线
python -m trader3.v2.cli_v2 trigger --dry-run
python tools/daily_routine.py          # 含 qlib_bin 增量更新 + 进化任务

# 数据增量更新（指数默认；成分股加 --stocks）
python scripts/update_market_data.py --apply [--stocks]

# API 服务（TRADER3_API_KEY 设置后强制鉴权）
uvicorn trader3.api.server:app --host 127.0.0.1 --port 8000
```

## 已知边界（诚实声明）

- WFA 未计交易成本/涨跌停；ST 名单依赖行情名称字段（缺名 fail-closed）
- 卖出触发仅入纸面账；extra_sources 默认关闭需环境变量开启
- financials.db 无公告日列，asof 防前视依赖 AnnouncementCalendar 覆盖度
