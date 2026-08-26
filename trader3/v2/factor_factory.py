"""
LLM 因子工厂（Factor Factory）—— 自主因子挖掘闭环。

对标 AlphaAgent / QuantGPT / RD-Agent 的「假设→生成→回测→抗过拟合→入库」代理循环：

1. 提出假设：离线模板（动量/反转/量价/波动率…）+ 参数变异；可选 LLM 生成新表达式。
2. 落地回测：对候选因子做扩张窗口 OOS RankIC 与多空收益；并叠加 Purged-CV
   （embargo + 标签 purge 的时序 walk-forward）逐折 RankIC 与组合式回测，
   输出 OOS t 统计量、稳定性与 Deflated Sharpe Ratio（多重检验校正）。
3. 抗过拟合闸门：IC > 阈值、扩张窗口或 Purged-CV 的 t 统计量 > 1.96、正 IC 占比、
   与库内因子冗余度 < 上限、且非 NaN；（可选 dsr_min / cv_strict 进一步加严）。
4. 入库：通过闸门的因子写入 Registry（JSON 持久化），供管线 `factor_exprs` 复用；失败候选触发变异/精炼（refine）。

LLM 路径为可选：提供 `llm_fn(prompt)->str` 时启用；无 Key 时纯离线闭环同样可运行。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("trader3.v2.factor_factory")


@dataclass
class FactorMetrics:
    expr: str
    hypothesis: str
    ic: float = float("nan")
    ir: float = float("nan")
    tstat: float = float("nan")
    ic_pos_ratio: float = float("nan")
    turnover: float = float("nan")
    sharpe: float = float("nan")
    max_dd: float = float("nan")
    corr_with_existing: float = 0.0
    cv_ic: float = float("nan")        # Purged-CV 逐折 IC 均值
    cv_ic_t: float = float("nan")      # CV 聚合 OOS t 统计量（更保守）
    cv_stability: float = float("nan")  # 1 - std/|mean|，越接近 1 越稳
    deflated_sharpe: float = 0.0       # 多重检验校正后的夏普
    passed: bool = False
    reason: str = ""
    generation: int = 0


@dataclass
class FactorCandidate:
    expr: str
    hypothesis: str
    generation: int = 0


# (假设描述, 表达式模板, 参数网格) —— {L} 被替换为时延/窗口
_TEMPLATE_SPECS: list[tuple[str, str, list[int]]] = [
    ("短期动量", "sub(close, delay(close, {L}))", [5, 10, 20]),
    ("短期反转", "sub(delay(close, {L}), close)", [5, 10, 20]),
    ("成交量变化率", "div(volume, delay(volume, {L}))", [5, 10, 20]),
    ("成交额动量", "sub(amount, delay(amount, {L}))", [5, 10, 20]),
    ("收益率波动率", "ts_std(close, {L})", [10, 20]),
    ("价格乖离度", "sub(close, ts_mean(close, {L}))", [20, 60]),
    ("量价共振", "mul(sub(close, delay(close, {L})), rank(volume))", [5, 10]),
    ("VWAP 偏离", "sub(close, vwap)", [0]),
    ("日内振幅", "sub(high, low)", [0]),
    ("截面强度", "zscore(close)", [0]),
]


class FactorRegistry:
    """通过闸门的因子持久化仓库（JSON）。"""

    def __init__(self, path: str | Path = "shared_state/factor_registry.json"):
        self.path = Path(path)
        self.items: list[FactorMetrics] = []
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.items = [FactorMetrics(**d) for d in json.loads(self.path.read_text("utf-8"))]
            except Exception as e:  # noqa: BLE001
                logger.warning("[factor_factory] 仓库读取失败: %s", e)
                self.items = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [vars(m) for m in self.items]
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

    def add(self, m: FactorMetrics) -> None:
        if any(x.expr == m.expr for x in self.items):
            return
        self.items.append(m)

    def exprs(self) -> list[str]:
        return [m.expr for m in self.items]

    def as_factor_exprs(self) -> dict[str, str]:
        return {f"ff{i}": m.expr for i, m in enumerate(self.items)}


@dataclass
class FactorFactoryConfig:
    ic_min: float = 0.01          # 最低 RankIC
    tstat_min: float = 1.96       # t 统计量下限
    ic_pos_min: float = 0.6       # 正 IC 窗口占比下限
    corr_max: float = 0.7        # 与库内因子最大冗余度上限
    min_train: int = 60          # 扩张窗口最小训练长度
    top_n: int = 20               # 单轮最多评估候选数
    n_propose: int = 12          # 单轮模板提议数（参数展开后截断）
    # Purged-CV / 组合式回测（P0-4：严谨 OOS 验证）
    cv_splits: int = 5
    cv_embargo: int = 5
    cv_horizon: int = 5
    cv_top_frac: float = 0.2
    cv_bottom_frac: float = 0.2
    cv_strict: bool = False       # True：仅以 Purged-CV t 为准（生产推荐，杜绝乐观偏差）
    dsr_min: float = 0.0          # 缩水夏普下限（>0 时启用硬闸）


def _row_rank_ic(fr: pd.Series, ff: pd.Series) -> float:
    m = fr.notna() & ff.notna()
    if m.sum() < 5:
        return float("nan")
    a = fr[m].rank()
    b = ff[m].rank()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a.values, b.values)[0, 1])


def _ooss_backtest(factor: pd.DataFrame, fwd: pd.DataFrame,
                   min_train: int = 60) -> tuple[np.ndarray, np.ndarray, float]:
    """扩张窗口 OOS 回测：返回 (逐期 RankIC, 多空收益, 换手率)。"""
    dates = list(factor.index)
    T = len(dates)
    ics: list[float] = []
    ls: list[float] = []
    prev_ranks: pd.Series | None = None
    turn_sum = 0.0
    turn_n = 0
    for start in range(min_train, T):
        train = factor.iloc[:start]
        stack = train.stack()
        mu = stack.mean()
        sd = stack.std() + 1e-9
        t = dates[start]
        fr = (factor.loc[t] - mu) / sd
        ff = fwd.loc[t]
        ic = _row_rank_ic(fr, ff)
        if not np.isnan(ic):
            ics.append(ic)
        fr_r = fr.rank()
        q = fr_r.quantile(0.8) if fr_r.notna().any() else float("nan")
        if not np.isnan(q):
            longs = fr_r >= q
            shorts = fr_r <= fr_r.quantile(0.2)
            if longs.sum() and shorts.sum() and ff.notna().any():
                ls.append(float(ff[longs].mean() - ff[shorts].mean()))
        if prev_ranks is not None:
            mm = fr_r.notna() & prev_ranks.notna()
            if mm.sum() > 0:
                turn_sum += float((fr_r[mm] - prev_ranks[mm]).abs().mean())
                turn_n += 1
        prev_ranks = fr_r
    return np.array(ics), np.array(ls), (turn_sum / turn_n if turn_n else 0.0)


class FactorFactory:
    """自主因子挖掘闭环。"""

    def __init__(self, registry: FactorRegistry | None = None,
                 config: FactorFactoryConfig | None = None):
        from trader3.v2.factor_dsl import get_dsl
        self.dsl = get_dsl()
        self.registry = registry or FactorRegistry()
        self.config = config or FactorFactoryConfig()
        self._registry_series_cache: dict[str, pd.DataFrame] = {}

    # ── 提出候选 ───────────────────────────────────────
    def propose_offline(self, n: int | None = None) -> list[FactorCandidate]:
        n = n or self.config.n_propose
        cands: list[FactorCandidate] = []
        for hypo, tmpl, grid in _TEMPLATE_SPECS:
            for L in grid:
                expr = tmpl.format(L=L) if "{L}" in tmpl else tmpl
                cands.append(FactorCandidate(expr=expr, hypothesis=f"{hypo} (L={L})"))
                if len(cands) >= n:
                    return cands
        return cands

    def propose_llm(self, n: int, llm_fn) -> list[FactorCandidate]:
        """可选 LLM 提议：llm_fn(prompt)->str，解析 'expr:' / 'hypothesis:' 行。"""
        prompt = (
            "你是量化研究员。请提出 "
            + str(n)
            + " 个全新横截面因子表达式（基于 open/high/low/close/"
            "volume/vwap/amount 字段，可用算子 abs/add/sub/mul/div/log/sqrt/rank/zscore/"
            "delay/ts_mean/ts_std/ts_min/ts_max/ts_corr）。每个因子一行：\n"
            "expr: <表达式>\nhypothesis: <一句话逻辑>\n"
        )
        try:
            out = llm_fn(prompt)
        except Exception as e:  # noqa: BLE001
            logger.warning("[factor_factory] LLM 提议失败: %s", e)
            return []
        cands: list[FactorCandidate] = []
        cur_expr = cur_hypo = None
        for line in out.splitlines():
            line = line.strip()
            if line.lower().startswith("expr:"):
                cur_expr = line.split(":", 1)[1].strip()
            elif line.lower().startswith("hypothesis:"):
                cur_hypo = line.split(":", 1)[1].strip()
            if cur_expr and cur_hypo:
                cands.append(FactorCandidate(expr=cur_expr, hypothesis=cur_hypo))
                cur_expr = cur_hypo = None
        return cands

    def _mutate(self, cand: FactorCandidate) -> list[FactorCandidate]:
        """离线精炼：对失败候选做符号翻转 / 时延替换，生成变异体。"""
        out: list[FactorCandidate] = []
        expr = cand.expr
        if expr.startswith("sub("):
            flipped = "sub(" + expr[4:]
            out.append(FactorCandidate(expr=flipped, hypothesis=cand.hypothesis + " [翻转]",
                                        generation=cand.generation + 1))
        for m in re.finditer(r"delay\((\w+),\s*(\d+)\)", expr):
            fld, lg = m.group(1), int(m.group(2))
            for nl in (max(1, lg // 2), lg * 2):
                ne = expr[:m.start()] + f"delay({fld}, {nl})" + expr[m.end():]
                out.append(FactorCandidate(expr=ne, hypothesis=cand.hypothesis + f" [L={nl}]",
                                            generation=cand.generation + 1))
        return out

    # ── 评估单个候选 ───────────────────────────────────
    def evaluate(self, cand: FactorCandidate, panel: pd.DataFrame,
                 fwd: pd.DataFrame) -> FactorMetrics:
        cfg = self.config
        try:
            factor = self.dsl.full_series(cand.expr, panel)
        except Exception as e:
            return FactorMetrics(expr=cand.expr, hypothesis=cand.hypothesis,
                                 passed=False, reason=f"DSL 编译失败: {e}",
                                 generation=cand.generation)
        if factor.shape[0] < cfg.min_train + 5:
            return FactorMetrics(expr=cand.expr, hypothesis=cand.hypothesis,
                                 passed=False, reason="样本不足", generation=cand.generation)
        ics, ls, turnover = _ooss_backtest(factor, fwd, cfg.min_train)
        if len(ics) < 3:
            return FactorMetrics(expr=cand.expr, hypothesis=cand.hypothesis,
                                 passed=False, reason="有效窗口过少", generation=cand.generation)
        ic_mean = float(ics.mean())
        ic_std = float(ics.std())
        tstat = ic_mean / (ic_std / np.sqrt(len(ics))) if ic_std > 0 else 0.0
        pos_ratio = float((ics > 0).mean())
        sharpe = float(ls.mean() / ls.std() * np.sqrt(252)) if ls.std() > 0 else 0.0
        cum = np.cumprod(1 + ls)
        run_max = np.maximum.accumulate(cum)
        dd = (run_max - cum) / run_max
        max_dd = float(dd.max()) if len(dd) else 0.0

        # Purged-CV + 组合式回测（严谨 OOS，防泄漏/乐观）
        from trader3.v2.backtest_cv import combinatorial_backtest, purged_cv_rankic
        cv = purged_cv_rankic(factor, fwd, cfg.cv_splits, cfg.cv_embargo,
                              cfg.min_train, cfg.cv_horizon)
        n_trials = max(2, cfg.cv_splits * cfg.top_n)
        cb = combinatorial_backtest(factor, fwd, cfg.cv_splits, cfg.cv_embargo,
                                    cfg.cv_top_frac, cfg.cv_bottom_frac,
                                    cfg.min_train, cfg.cv_horizon, n_trials)

        corr = self._redundancy(factor, panel)
        # OOS t 统计量：扩张窗口或 Purged-CV 任一达标即视为具备样本外预测力
        # （生产环境建议置 cv_strict=True，仅以 CV 为准以杜绝泄漏/乐观偏差）
        tstat_eff = cv["ic_t"] if cfg.cv_strict else max(tstat, cv["ic_t"])
        passed = bool(
            np.isfinite(ic_mean) and ic_mean >= cfg.ic_min
            and tstat_eff >= cfg.tstat_min
            and pos_ratio >= cfg.ic_pos_min and corr <= cfg.corr_max
            and (cfg.dsr_min <= 0 or cb["deflated_sharpe"] >= cfg.dsr_min)
        )
        reason = ("ok" if passed else
                  self._fail_reason(ic_mean, tstat_eff, pos_ratio, corr, cfg))
        return FactorMetrics(
            expr=cand.expr, hypothesis=cand.hypothesis,
            ic=round(ic_mean, 4), ir=round(ic_mean / ic_std, 4) if ic_std > 0 else 0.0,
            tstat=round(float(tstat), 3), ic_pos_ratio=round(pos_ratio, 3),
            turnover=round(float(turnover), 3), sharpe=round(sharpe, 3),
            max_dd=round(max_dd, 3), corr_with_existing=round(float(corr), 3),
            cv_ic=round(cv["ic_mean"], 4), cv_ic_t=round(cv["ic_t"], 3),
            cv_stability=round(cv["stability"], 3),
            deflated_sharpe=round(cb["deflated_sharpe"], 3),
            passed=passed, reason=reason, generation=cand.generation)

    def _redundancy(self, factor: pd.DataFrame, panel: pd.DataFrame) -> float:
        if not self.registry.items:
            return 0.0
        f_r = factor.rank(axis=1)
        mx = 0.0
        for expr in self.registry.exprs():
            other = self._registry_series_cache.get(expr)
            if other is None:
                try:
                    other = self.dsl.full_series(expr, panel)
                    self._registry_series_cache[expr] = other
                except Exception:
                    continue
            o_r = other.rank(axis=1)
            c = f_r.corrwith(o_r, axis=1).mean()
            if np.isfinite(c) and abs(c) > mx:
                mx = abs(c)
        return mx

    @staticmethod
    def _fail_reason(ic, tstat, pos, corr, cfg) -> str:
        if not np.isfinite(ic):
            return "IC 非有限"
        if ic < cfg.ic_min:
            return f"IC={ic:.3f}<{cfg.ic_min}"
        if tstat < cfg.tstat_min:
            return f"t={tstat:.2f}<{cfg.tstat_min}"
        if pos < cfg.ic_pos_min:
            return f"正IC占比={pos:.2f}<{cfg.ic_pos_min}"
        if corr > cfg.corr_max:
            return f"冗余度={corr:.2f}>{cfg.corr_max}"
        return "fail"

    # ── 主循环 ────────────────────────────────────────
    def run(self, panel: pd.DataFrame, fwd: pd.DataFrame,
            use_llm: bool = False, llm_fn=None) -> list[FactorMetrics]:
        """单轮：提出 → 评估 → 入库通过者。"""
        cands = self.propose_offline(self.config.n_propose)
        if use_llm and llm_fn:
            cands += self.propose_llm(self.config.n_propose, llm_fn)
        accepted = self._evaluate_all(cands, panel, fwd)
        self.registry.save()
        return accepted

    def run_with_refine(self, panel: pd.DataFrame, fwd: pd.DataFrame,
                        generations: int = 2, use_llm: bool = False,
                        llm_fn=None) -> list[FactorMetrics]:
        """多轮精炼：失败候选变异后重评估，逐轮积累入库。"""
        cands = self.propose_offline(self.config.n_propose)
        if use_llm and llm_fn:
            cands += self.propose_llm(self.config.n_propose, llm_fn)
        accepted: list[FactorMetrics] = []
        pool = list(cands)
        for _ in range(generations):
            new_accepted = self._evaluate_all(pool, panel, fwd)
            accepted.extend(new_accepted)
            failed = [c for c in pool if not self._last_pass(c, new_accepted)]
            if not failed:
                break
            pool = []
            for c in failed:
                pool += self._mutate(c)[:2]
            pool = pool[: self.config.n_propose]
        self.registry.save()
        return accepted

    def _evaluate_all(self, cands: list[FactorCandidate],
                      panel: pd.DataFrame, fwd: pd.DataFrame) -> list[FactorMetrics]:
        accepted: list[FactorMetrics] = []
        for c in cands:
            m = self.evaluate(c, panel, fwd)
            if m.passed:
                self.registry.add(m)
                accepted.append(m)
        return accepted

    def _last_pass(self, cand: FactorCandidate, accepted: list[FactorMetrics]) -> bool:
        return any(m.expr == cand.expr and m.passed for m in accepted)


def run_factor_factory(panel: pd.DataFrame, fwd: pd.DataFrame | None = None,
                       *, use_llm: bool = False, llm_fn=None,
                       generations: int = 1,
                       config: FactorFactoryConfig | None = None,
                       registry_path: str | Path = "shared_state/factor_registry.json"
                       ) -> tuple[list[FactorMetrics], FactorRegistry]:
    """便利入口：在 panel 上跑因子工厂，返回 (通过因子, 仓库)。"""
    fwd = fwd if fwd is not None else _default_forward(panel)
    fac = FactorFactory(FactorRegistry(registry_path), config)
    if generations > 1:
        accepted = fac.run_with_refine(panel, fwd, generations=generations,
                                       use_llm=use_llm, llm_fn=llm_fn)
    else:
        accepted = fac.run(panel, fwd, use_llm=use_llm, llm_fn=llm_fn)
    return accepted, fac.registry


def _default_forward(panel: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    close = panel.xs("close", axis=1, level=1)
    return close.pct_change(horizon).shift(-horizon)
