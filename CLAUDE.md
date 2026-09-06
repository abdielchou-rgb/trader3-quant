# 3号交易员 — Agent 行为约束

> 约束你（Agent）在 **3号交易员 (trader3)** 项目中的行为。核心：**量化工程方法论** + **数据纪律**。
> 本文件反映 2026-09 机构级工程瓶颈攻坚后的真实状态（545 tests 基线）。

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
tests/              # 602 个测试（600 passed / 1 skipped 基线）；testpaths 已在 pyproject 隔离
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
4. **测试必须绿** — 当前基线 600 passed / 1 skipped

---

## 关键命令

```bash
# 质量门（提交前必过）
py -3.11 -m ruff check .
py -3.11 -m mypy
py -3.11 -m pytest tests/ -q

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
python tools/daily_routine.py          # 含 qlib_bin 增量更新 + 进化任务 + 影子对账（配置后）

# 深度模型进化（LSTM 候选并入 GP 同一门禁）
py -3.11 evolve/run_evolution.py --universe csi300 --gen 15 --pop 50 --deep

# 影子对账（配置 QMT_PATH/QMT_ACCOUNT/SHADOW_TARGETS 后由 daily_routine 自动跑）
py -3.11 -c "from trader3.v2.shadow_reconcile import run_shadow_reconcile; \
  import os; print(run_shadow_reconcile(targets_path=os.environ['SHADOW_TARGETS'], \
  qmt_path=os.environ['QMT_PATH'], account_id=os.environ['QMT_ACCOUNT'], simulated=True)['mode'])"

# 数据增量更新（指数默认；成分股加 --stocks）
python scripts/update_market_data.py --apply [--stocks]

# API 服务（TRADER3_API_KEY 设置后强制鉴权）
uvicorn trader3.api.server:app --host 127.0.0.1 --port 8000
```

## 已知边界（诚实声明）

- ~~WFA 未计交易成本/涨跌停~~ **已修复**（post-audit-5）：WFA 测试段首日计入单边换手成本并套用涨跌停约束，见 tests/test_fix_signal_source.py
- ST 名单依赖行情名称字段（缺名 fail-closed）
- 卖出触发仅入纸面账；extra_sources 默认关闭需环境变量开启
- financials.db 无公告日列，asof 防前视依赖 AnnouncementCalendar 覆盖度
- **实盘链路（2026-09 全量推进）**：
  - `trader3/v2/live/qmt_broker.py` — QMT/miniQMT 适配器（xtquant 缺失 fail-fast；simulated=True 离线 SIM 撮合，订单显式打标 QMT-SIM）
  - `trader3/v2/shadow_reconcile.py` — 影子对账（ShadowBroker 拦截在先，落盘 shared_state/shadow/shadow_run.json）
  - `tools/daily_routine.py` Step 5.5 — 配置 QMT_PATH/QMT_ACCOUNT/SHADOW_TARGETS 环境变量后每日影子对账（缺配置静默跳过）
  - ShadowBroker + runner（`python -m trader3.v2.runner --shadow`）为既有能力；CTP 适配器离线走 SIMULATED
  - **以上均未经过真实资金环境验证**（QMT 终端联调待做）
- **深度模型（2026-09 全量推进）**：`evolve/core/deep_model.py` LSTM 打分器（torch 2.13 CPU，无前视/确定性/截尾一致性测试覆盖）；
  `run_evolution.py --deep` 把 LSTM 候选并入 GP 候选同一筛选门禁（真实 csi300 数据冒烟通过：backend=torch，IC 由 StrategySelector 判定）；
  torch 缺失时确定性 numpy ridge 回退（显式打标 fallback_used，禁标 LSTM）
- **机构级工程层（2026-09 瓶颈攻坚，全部 TDD）**：
  - **PIT 双时间戳**（`trader3/data/pit_loader.py`）：financial_pit 表 report_date+publish_timestamp 双列，
    asof 检索严格按披露时刻切片（7 测试：年报 4/30 边界、更正公告、法定截止保守回填）；旧库无公告日时按法定最晚日回填（保守：宁可晚可见不可穿越）
  - **GP 正交残差化**（`evolve/core/orthogonal_fitness.py`）：P_orth 预计算投影剥离 Barra 风格共线，
    残差 Rank-IC 为边际增量 Alpha（6 测试：纯风格克隆杀、增量 Alpha 留、独立因子不误伤）；
    StrategySelector 新增可选 orthogonality 门禁（注入 barra_styles+forward_returns 即激活）
  - **订单状态机+WAL**（`trader3/v2/live/order_state.py` + `robust_qmt_executor.py`）：9 态白名单转移、
    终态吸收幂等、先日志后动作 fsync、断线对账（远程事实收敛/幽灵单 FAILED_LOST 隔离）（8 测试）
  - **多日参与率执行**（`trader3/tools/execution_flow.py`）：5% 参与率上限顺延、AC 非线性冲击方向感知、
    整手约束、流动性不足诚实报 unfilled（9 测试）
- **机构层接线收尾（2026-09 P1-P4 全量推进）**：
  - **PIT 真库已灌**：`data/financials_pit.db`（565 万行，脚本 `scripts/backfill_pit_db.py`，
    法定截止保守回填 + 200 次 asof 随机抽查零穿越）；外部用户可用同脚本从自己的 financials.db 重建
  - **正交门禁已通电**：`evolve/core/style_exposures.py`（末截面 mom20/size/vol20 标准暴露）
    + `run_evolution.py --orthogonal`；真实 csi300 冒烟 13 候选全部被共线拦截（诚实结果）
  - **WAL 已接 QMTBroker**：`QMTBroker(config, wal_path=...)` 注入即启用先日志后动作路径；
    崩溃重放/整手拒绝不落盘/对账收敛全部回归覆盖（tests/test_qmt_wal_wiring.py）
  - **WFA 参与率约束已接**：`_run_wfa_rolling(daily_volumes=panel["amount"], capital=...)`
    ——首日调仓超 5% 日成交额部分按现金截断（保守下界）+ 成本照提；无 amount 面板时历史行为不变
- **生产级架构层（2026-09 六维推进，双模同构+硬风控+可观测，全部 TDD）**：
  - **统一事件模型**（`trader3/runtime/events.py`）：Bar/Tick/Signal/OrderIntent/Fill 不可变 dataclass；
    Fill 双时间戳（exchange_ts/local_ts）
  - **双模同构运行时**（`trader3/runtime/`）：DualModeStrategy 只面向事件+快照零 SDK 依赖；
    ReplayRuntime（历史流+TimestampGuard 前视守卫，未来事件拒消费）与 LiveRuntime（网关回调）
    驱动同一份策略代码 —— Train-Serving Skew 结构性根治
  - **前置硬风控网关**（`trader3/risk/gateway.py`）：单笔限额/集中度（增量口径）/自成交/OTR 熔断/
    回撤 KillSwitch/急停按钮/幂等单号（DUPLICATE_ID）+ 确定性单号生成器（日+策略+序列可复现）
  - **持仓漂移挂起**（`trader3/risk/drift_halt.py`）：对账差异达阈值 → 自动挂起开仓（平仓放行），
    resolve() 人工解除
  - **可观测性**（`trader3/obs/metrics.py`）：Prometheus 文本格式零依赖实现（Counter/Gauge/Histogram），
    API `/metrics` 端点已挂
  - **容器化**（`Dockerfile` 多阶段 + `docker-compose.yml`）：非 root 运行、api 长驻 +
    daily-routine 批处理分容器、/data 与 shared_state 卷持久化
  - **有意不做**（诚实边界）：Level-2 ring buffer/ZeroMQ/Rust 扩展是 tick 级 HFT 设施，
    本引擎日频 A股口径下属过度设计；需要时再评估
- **生产层真接线（2026-09 继续推进，F1-F5）**：
  - **端到端同构闭环**：LiveRuntime.attach(gateway, executor, metrics) —— 意图→风控→WAL→
    fill 回报回流账户+双时间戳延迟指标（tests/test_pipeline_e2e.py）
  - **既有链路接风控**：OrderManager.submit(risk_gateway=, account=) 可选钩子 ——
    超限单拒在 broker 前（REJECTED+metadata.risk_reason），无 gateway 历史行为不变；
    gateway.check_broker_order 提供 Order→OrderIntent 适配（tests/test_order_manager_risk.py）
  - **遥测真接线**：`trader3/obs/telemetry.py` 全局单例 registry；OrderManager 下单/拒单、
    DriftHalt 挂起/恢复、API /metrics 全部共享该单例 —— 指标不再只是口头声明
  - **漂移告警**：DriftHaltEngine 挂起/解除 → notify.send_notification + telemetry gauge（幂等不重复轰炸）
  - **Docker 实测通过**：本机 daemon 构建（~130s）+ 容器内 uvicorn 起 API、
    `GET /metrics` 返回 200 + 指标文本；补 uvicorn 到 requirements（曾注释缺失致容器秒退）
- **生产层全量推进（2026-09 G1-G5）**：
  - **即时模式**：LiveRuntime.set_immediate(strategy) → push_bar 当场触发全链路（策略→风控→执行），
    RLock 串行化行情回调；fill 幂等（重复 client_order_id 去重）、方向语义 sell 减仓、
    乱序回报先到先记（券商事实优先）（tests/test_live_immediate.py）
  - **集中度存量口径**：AccountSnapshot.position_values 提供时用 (已持市值+本单名义)/权益，
    缺失回退增量口径（tests/test_concentration_upgrade.py）
  - **size 真实市值注入**：style_exposures(shares=) 用 log(close×shares)，缺失回退成交额代理
    （tests/test_size_upgrade.py）
  - **Compose 端口改 127.0.0.1:8100**（避开本机 coolify 占 8000），compose up 实测 healthy
  - quality gates：ruff clean / mypy 0 errors（20 文件）/ pytest 600 passed / 1 skipped
