# 3号交易员 → marvis 执行指令（v2.1 剩余工作）

> 交接对象：marvis
> 交接日期：2026-08-23
> 目标：把 3号交易员 v2.1 剩余工作全部落地并运行起来
> 项目位置：`D:\Claude\projects\3号交易员\`

---

## 一、当前状态（已完成）

**已落地（v2.1）：**
- `trader3/v2/risk_rules.py` —— 事前风控规则链（黑名单/单笔限额/仓位/价格带/流控）
- `trader3/v2/execution_realism.py` —— T+1 三态撮合 + 涨跌停拦截
- `trader3/v2/announcement_calendar.py` —— **财务事件表防前视**（刚落地）
- `trader3/v2/market_data.py` —— 股本/市值/行情三通道
- `trader3/v2/comps.py` —— 可比公司分析
- `trader3/v2/collector.py` + `sources.py` + `extra_sources.py` —— 多源采集
- `trader3/v2/trigger.py` —— 三因子触发（催化×预期差×技术）
- `trader3/v2/watchlist.py` —— 自选股状态机
- `trader3/v2/daily_pipeline.py` + `cli_v2.py` —— 每日管线 + CLI
- `evolve/` —— GP 进化工厂（已加 parsimony 防过拟合）
- 测试：**55/55 全绿**

**已深度吸收（12项目）：** ai-berkshire / AI-Trader / serenity / qlib / vnpy / alphagen / TradeMaster / TradingAgents-CN / fooltrader / QUANTAXIS / backtrader / ai_quant_trade（吸收清单见 `docs/五项目深层吸收清单.md` 与 `docs/fooltrader吸收与顶级项目清单.md`）

---

## 二、剩余工作清单（marvis 执行）

### 批次1：QUANTAXIS/backtrader 深读落地（已深读，待编码）

**1. QUANTAXIS 模式 ②③④ → 落地：**
- [ ] 统一行情列访问器 `.open/.close/.volume` 属性壳 → 参考 `QUANTAXIS/QADataStruct.py`
- [ ] T+1 三态已做（`execution_realism.py`）——校验与 QUANTAXIS 的 `his/today/frozen` 一致
- [ ] 增量更新「查尾续拉」→ 对齐 `QUANTAXIS/QASU/save_tdx.py`，把 `sync_qlib_data.py` 升级成通用 `sync_all.py`

**2. backtrader 模式 → 落地：**
- [ ] 可插拔费用模型（佣金/印花税做成参数而非固定率）→ 参考 backtrader `CommissionInfo`
- [ ] 涨跌停成交量限制（Filler 按量撮合）→ 参考 backtrader `Fillers`

### 批次2：财务防前视接入（已落地 announcement_calendar，待接入）

- [ ] `trigger.py` 读财务时改用 `financials_asof(code, asof_date)`（防前视）
- [ ] `valuation_anchor` 读财务时同步改
- [ ] `comps.py` 读财务时同步改
- [ ] `backtest.py` 回测读财务时同步改（历史财务按公告日对齐）

**验证标准：** 用 `cal.latest_asof("600519", "2024-01-15")` 应返回 `2023-09-30`（一季报未披露）、`2024-05-15` 应返回 `2024-03-31`。

### 批次3：收尾 + 运行

- [ ] 全量回归 `python -m pytest tests/` → 55+ 全绿
- [ ] 每日管线端到端：`python -m trader3.v2.daily_pipeline --codes 600519,000858,300750`
- [ ] CLI 冒烟：`python -m trader3.v2.cli_v2 watchlist / trigger / comps`
- [ ] 写 `docs/v2.1落地报告.md`（交付说明）

---

## 三、关键环境

```bash
cd D:\Claude\projects\3号交易员
# 编辑器一律用（沙箱 python）：
python3 -c "import sys; sys.path.insert(0,'.'); from trader3 import Trader3; t=Trader3(); print('trader3 OK')"
# 测试：
python -m pytest tests/ -v
# CLI：
python -m trader3.v2.cli_v2 watchlist
python -m trader3.v2.cli_v2 trigger
python -m trader3.v2.cli_v2 comps --code 600519 --industry 白酒
```

数据源：
- 行情：`D:\2hao-analyst\data\qlib_bin\`（无法更新时用本机 `run_local_sources.py`）
- 财务：`D:\2hao-analyst\data\financials.db`（560万行）
- 股本/市值：`trader3/v2/market_data.py`（东财 push2 实时）

---

## 四、执行纪律（铁律）

1. **数据带来源**——真实数据/合成数据标注清楚，禁止编造
2. **测试必绿**——改完必须 `pytest` 全绿才能交付
3. **门禁必过**——回测/信号/估值产出过 IronGate
4. **候选信号**——所有产出是候选信号非投资建议
5. **防前视**——读财务一律走 `announcement_calendar.financials_asof`

---

## 五、DoD（完成定义）

- [ ] 批次1（QUANTAXIS Filler + backtrader 费用可插拔）落地
- [ ] 批次2（财务防前视接入 trigger/估值/comps/回测）落地
- [ ] 批次3（全量回归 + 每日管线端到端 + CLI 冒烟）通过
- [ ] `docs/v2.1落地报告.md` 产出