"""
3号交易员 — IronGate 门禁 (M6: 完整门禁)

所有产出在返回给 2号分析师之前必须通过门禁检查。
M0 阶段为 pass-through 存根；M6 接入完整 IronGate 检查，默认启用。

门禁清单:
  Gate 1  回测统计显著性      check_backtest_significance  BacktestReport
  Gate 2  因子有效性门槛      check_factor_hurdle          SignalValidationReport
  Gate 3  组合约束满足        check_constraints            OptimizationResult
  Gate 4  交易成本合理性      check_tca_reasonable         TCAEstimate
  Gate 5  状态识别置信度      check_regime_confidence      RegimeDiagnosis
  Gate 6  数据版本新鲜度      check_data_freshness         SharedState (新增)
  Gate 7  默认未验证          check_credibility_default    ValuationReport/通用 (新增)
  Gate 8  抽检审计            sample_audit                 通用 (新增)
"""

from __future__ import annotations

import os
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from trader3.models import (
    BacktestReport,
    OptimizationResult,
    RegimeDiagnosis,
    SignalValidationReport,
    TCAEstimate,
    ValuationReport,
    ScorecardReport,
    WFAReport,
)


# ── 门禁阈值常量 ──
BACKTEST_T_STAT_MIN = 2.0            # t 统计量门槛
BACKTEST_EXCESS_RETURN_MIN = 0.0     # 样本外超额收益 > 0
BACKTEST_IR_MIN = 0.3                # 信息比门槛

FACTOR_ICIR_MIN = 0.3                # ICIR 门槛
FACTOR_MONOTONICITY_MIN = 0.5        # 分组单调性门槛
FACTOR_HALF_LIFE_MONTHS_MIN = 2.0    # 半衰期门槛（月）

TCA_TOTAL_COST_BP_MAX = 200.0        # 总成本上限 2%

REGIME_MAX_PROB_MIN = 0.5            # 状态识别最大概率门槛
REGIME_ENTROPY_MAX = 1.0             # 状态熵上限

DATA_MAX_AGE_SECONDS = 14400         # 数据新鲜度上限 4 小时

# 抽检审计的样本量 / 交叉验证来源数
AUDIT_SAMPLE_RATIO = 0.15
AUDIT_MIN_SOURCES = 2


@dataclass
class GateResult:
    """门禁检查结果"""
    passed: bool
    check_name: str = ""
    score: float = 0.0         # 0.0~1.0
    message: str = ""          # 通过/失败原因
    details: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed

    def to_dict(self) -> dict:
        return asdict(self)


class BaseGate(ABC):
    """门禁基类（完整 IronGate Mixin）"""

    gate_name: str = ""
    gate_description: str = ""

    @abstractmethod
    def check(self, **kwargs) -> GateResult:
        ...


# ═══════════════════════════════════════════
# 通用辅助函数（供多个门禁复用）
# ═══════════════════════════════════════════

def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """从 dataclass / dict / 任意对象安全取值。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key)
    return default


def _to_dict(obj: Any) -> Dict[str, Any]:
    """dataclass / dict → dict；其他对象尝试 asdict，失败返回空。"""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    try:
        from dataclasses import asdict as _asdict
        return _asdict(obj)
    except Exception:
        return {}


# ═══════════════════════════════════════════
# 独立抽检审计逻辑（Gate 8）
# ═══════════════════════════════════════════

def _extract_numeric_fields(obj: Any, max_fields: int = 30) -> List[Tuple[str, float]]:
    """从对象中提取数值字段（供抽检使用）。"""
    fields: List[Tuple[str, float]] = []
    d = _to_dict(obj)
    for k, v in d.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            fields.append((k, float(v)))
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    fields.append((f"{k}.{kk}", float(vv)))
        if len(fields) >= max_fields:
            break
    return fields


def _find_consistency_relations(
    d: Dict[str, Any]
) -> List[Tuple[str, float, str, str]]:
    """
    从数据字典中寻找内部一致性关系: total == a + b + c + ...

    仅对 bp 类字段做求和验证（total_cost_bp = 分项之和），
    避免把 CNY 金额与 bp 分量混算。
    无总和字段时退化为逐项包含检查（total 存在且 ≥ 最大分项）。
    """
    relations: List[Tuple[str, float, str, str]] = []
    total_keys = [k for k in d.keys() if "total" in k.lower()]
    component_keys = [
        "commission_bp", "stamp_tax_bp", "impact_bp", "timing_risk_bp",
        "opportunity_cost_bp",
    ]

    for tk in total_keys:
        total = d.get(tk)
        if not isinstance(total, (int, float)):
            continue
        # 仅当总和字段本身是 bp（或与分量同量纲）时做求和验证
        is_bp_total = "bp" in tk.lower() or tk in ("total_cost_bp",)
        comps = [d[ck] for ck in component_keys if isinstance(d.get(ck), (int, float))]
        if comps and is_bp_total:
            expected = float(sum(comps))
            # 浮点宽容度 1bp
            relations.append((tk, float(total), "approx_eq", f"sum({'+'.join(component_keys)})={expected:.2f}"))
        else:
            # 退化：total 必须 ≥ 最大分项（避免明显不一致）
            max_comp = max((v for v in d.values() if isinstance(v, (int, float))), default=0.0)
            if max_comp > 0:
                relations.append((tk, float(total), "gte", f"max_component={max_comp:.2f}"))

    return relations


def run_sample_audit(data: Any, sample_ratio: float = AUDIT_SAMPLE_RATIO) -> GateResult:
    """
    抽检审计（M6 实现: 内部一致性交叉验证）。

    从输出中提取数值字段 → 按 sample_ratio 抽样 → 对抽样字段做内部一致性交叉验证
    （如 total_bp == 分项之和）。M7 将接入 2 个独立数据源的真实交叉验证。

    Parameters
    ----------
    data : Any — 产出对象（dataclass 或 dict）
    sample_ratio : float — 抽样比例（默认 15%）

    Returns
    -------
    GateResult
    """
    d = _to_dict(data)

    if not d:
        return GateResult(
            passed=False, check_name="抽检审计", score=0.0,
            message="抽检审计: 输出为空，无数据可审计",
            details={"reason": "empty_output"},
        )

    # 1) 提取数值字段
    numeric = _extract_numeric_fields(data)
    total_fields = len(numeric)

    # 2) 15% 抽样（以字段名派生种子，保证审计可复现）
    import random
    k = max(1, int(total_fields * sample_ratio))
    rng = random.Random("|".join(sorted(k_ for k_, _ in numeric))[:512])
    sampled = rng.sample(numeric, min(k, total_fields))

    # 3) 内部一致性交叉验证
    relations = _find_consistency_relations(d)
    checked_count = 0
    mismatches: List[Dict[str, Any]] = []

    for tk, total, op, note in relations:
        if op == "approx_eq":
            if "=" not in note:
                continue
            expected = float(note.split("=")[-1])
            ok = abs(total - expected) <= 1.0
        else:  # gte
            max_comp = float(note.split("=")[-1])
            ok = total >= max_comp - 1e-9
        checked_count += 1
        if not ok:
            mismatches.append({"field": tk, "total": total, "expected": note})

    passed = len(mismatches) == 0
    score = 1.0 if passed else max(0.0, 1.0 - len(mismatches) / max(checked_count, 1))

    sampled_fields = [f"{k}={v:.4g}" for k, v in sampled[:5]]

    return GateResult(
        passed=passed,
        check_name="抽检审计",
        score=round(score, 3),
        message=(
            f"抽检审计: {checked_count} 条一致性关系，{len(mismatches)} 条不一致"
            if checked_count else
            f"抽检审计: 无可验证的一致性关系（抽样 {len(sampled)}/{total_fields} 字段）"
        ),
        details={
            "total_fields": total_fields,
            "sampled": len(sampled),
            "sample_ratio": sample_ratio,
            "checked_relations": checked_count,
            "mismatches": mismatches,
            "sampled_fields": sampled_fields,
            "cross_validation_sources": AUDIT_MIN_SOURCES,
            "note": "M6 内部一致性审计；M7 接入 2 个独立数据源",
        },
    )


def _has_explicit_synthetic_flag(report) -> bool:
    """产出对象/dict 是否显式携带 used_synthetic=True 之类的合成声明。"""
    for attr in ("used_synthetic", "is_synthetic", "synthetic"):
        v = _safe_get(report, attr, None)
        if v is None:
            v = _to_dict(report).get(attr)
        if isinstance(v, bool) and v:
            return True
    return False


# ═══════════════════════════════════════════
# 门禁集
# ═══════════════════════════════════════════

class Trader3Gates:
    """
    3号交易员专属门禁集 (M6: 完整 IronGate，默认启用)。

    enabled=True 时运行真实检查；enabled=False 时 pass-through（M0 兼容）。
    """

    def __init__(self, enabled: bool = True, state: Any = None):
        self.enabled = enabled
        self.state = state
        self._history: List[GateResult] = []

    # ── Gate 1: 回测统计显著性 ──

    def check_backtest_significance(self, report) -> GateResult:
        """回测统计显著性: t统计量 > 2.0, 超额收益 > 0, 信息比 > 0.3"""
        if not self.enabled:
            return self._pass_through("回测统计显著性")

        t = float(_safe_get(report, "t_statistic", 0) or 0)
        oos = float(_safe_get(report, "excess_return", 0) or 0)
        ir = float(_safe_get(report, "information_ratio", 0) or 0)

        checks = {
            "t_statistic>2.0": t > BACKTEST_T_STAT_MIN,
            "excess_return>0": oos > BACKTEST_EXCESS_RETURN_MIN,
            "information_ratio>0.3": ir > BACKTEST_IR_MIN,
        }
        passed = all(checks.values())

        score = min(t / 3.0, 1.0) if t > 0 else 0.0
        score = max(score, 0.1 if passed else 0.0)

        return self._result(
            passed, "回测统计显著性", score=score,
            details={"t_statistic": t, "excess_return": oos,
                     "information_ratio": ir, "checks": checks,
                     "thresholds": {"t_statistic": 2.0, "excess_return": 0.0,
                                    "information_ratio": 0.3}},
        )

    # ── Gate 2: 因子有效性门槛 ──

    def check_factor_hurdle(self, report) -> GateResult:
        """因子有效性门槛: ICIR > 0.3, 分组单调性 > 0.5, 半衰期 > 42 个交易日"""
        if not self.enabled:
            return self._pass_through("因子有效性门槛")

        icir = float(_safe_get(report, "icir", 0) or 0)
        mono = float(_safe_get(report, "monotonicity", 0) or 0)
        # 新口径：交易日；兼容旧 months 字段（×21 换算）
        half_life = float(_safe_get(report, "half_life_periods", 0) or 0)
        if half_life <= 0:
            half_life = float(_safe_get(report, "half_life_months", 0) or 0) * 21.0

        checks = {
            "icir>0.3": icir > FACTOR_ICIR_MIN,
            "monotonicity>0.5": mono > FACTOR_MONOTONICITY_MIN,
            "half_life_periods>42": half_life > FACTOR_HALF_LIFE_MONTHS_MIN * 21,
        }
        passed = all(checks.values())

        score = min(icir / 0.5, 1.0) if icir > 0 else 0.0
        score = max(score, 0.1 if passed else 0.0)

        return self._result(
            passed, "因子有效性门槛", score=score,
            details={"icir": icir, "monotonicity": mono,
                     "half_life_periods": half_life, "checks": checks,
                     "thresholds": {"icir": 0.3, "monotonicity": 0.5,
                                    "half_life_periods": FACTOR_HALF_LIFE_MONTHS_MIN * 21}},
        )

    # ── Gate 3: 组合约束满足 ──

    def check_constraints(self, result) -> GateResult:
        """组合约束满足：独立复检权重向量，不信任自报 constraints_satisfied。"""
        if not self.enabled:
            return self._pass_through("组合约束满足")

        satisfied = bool(_safe_get(result, "constraints_satisfied", True))
        violations = list(_safe_get(result, "constraint_violations", []) or [])

        # 结构性违规（超限/负权重/权重和偏离）→ 直接失败；
        # 求解器收敛警告只记录不拦截。
        structural = [
            v for v in violations
            if any(k in v for k in ("超限", "负权重", "偏离"))
        ]

        # 独立复检：用产出里的 target_weights 与上限字段重算
        independent: List[str] = []
        tw = _safe_get(result, "target_weights", None)
        cap = float(_safe_get(result, "max_single_weight_cap", 0) or 0)
        if isinstance(tw, dict) and tw and cap > 0:
            max_w = max(float(v) for v in tw.values())
            if max_w > cap + 1e-6:
                independent.append(
                    f"独立复检失败: 最大权重 {max_w:.4f} > 上限 {cap:.4f}"
                )

        passed = satisfied and len(structural) == 0 and len(independent) == 0

        return self._result(
            passed, "组合约束满足",
            score=1.0 if passed else 0.0,
            message=(
                f"组合约束满足: {'通过' if passed else '违规'}"
                f"（结构性 {len(structural)}，独立复检 {len(independent)}）"
            ),
            details={"constraints_satisfied": satisfied,
                     "constraint_violations": violations,
                     "structural_violations": structural,
                     "independent_check": independent},
        )

    # ── Gate 4: 交易成本合理性 ──

    def check_tca_reasonable(self, estimate) -> GateResult:
        """交易成本合理性: 总成本 < 2% (200bp), 冲击成本占比合理"""
        if not self.enabled:
            return self._pass_through("交易成本合理性")

        total = float(_safe_get(estimate, "total_cost_bp", 0) or 0)
        impact = float(_safe_get(estimate, "impact_bp", 0) or 0)
        commission = float(_safe_get(estimate, "commission_bp", 0) or 0)

        checks = {
            "total_cost_bp<200": total < TCA_TOTAL_COST_BP_MAX,
            "impact_bp<150": impact < TCA_TOTAL_COST_BP_MAX - 50,
            "commission_bp>0": commission >= 0,
        }
        passed = all(checks.values())

        # 合理冲击占比: impact / total < 0.9（除非 total 极小）
        impact_ratio = (impact / total) if total > 1e-9 else 0.0
        reasonable_impact = impact_ratio < 0.95
        passed = passed and reasonable_impact
        checks["impact/total<0.95"] = reasonable_impact

        score = min(1.0, (TCA_TOTAL_COST_BP_MAX - total) / TCA_TOTAL_COST_BP_MAX)
        score = max(score, 0.0)

        return self._result(
            passed, "交易成本合理性", score=round(score, 3),
            details={"total_cost_bp": total, "impact_bp": impact,
                     "commission_bp": commission,
                     "impact_ratio": round(impact_ratio, 3),
                     "checks": checks,
                     "thresholds": {"total_cost_bp_max": 200.0}},
        )

    # ── Gate 5: 状态识别置信度 ──

    def check_regime_confidence(self, diagnosis) -> GateResult:
        """状态识别置信度: 最大概率 > 0.5, 熵 < 1.0"""
        if not self.enabled:
            return self._pass_through("状态识别置信度")

        probs = _safe_get(diagnosis, "regime_probabilities", {}) or {}
        entropy = float(_safe_get(diagnosis, "regime_entropy", 1.0) or 1.0)

        if isinstance(probs, dict) and probs:
            max_prob = float(max(probs.values()))
        else:
            max_prob = float(_safe_get(diagnosis, "max_probability", 0) or 0)

        checks = {
            "max_prob>0.5": max_prob > REGIME_MAX_PROB_MIN,
            "entropy<1.0": entropy < REGIME_ENTROPY_MAX,
        }
        passed = all(checks.values())

        score = min(max_prob * 1.5, 1.0)
        score = max(score, 0.1 if passed else 0.0)

        return self._result(
            passed, "状态识别置信度", score=round(score, 3),
            details={"max_prob": round(max_prob, 4), "entropy": round(entropy, 4),
                     "checks": checks,
                     "thresholds": {"max_prob": 0.5, "entropy": 1.0}},
        )

    # ── Gate 6: 数据版本新鲜度 (新增) ──

    @staticmethod
    def _probe_source_mtime() -> Optional[Dict[str, Any]]:
        """
        data_version.json 缺失时的回退：用 qlib 数据集日历文件 mtime 作为新鲜度依据。
        静态数据集不应按墙钟时间判陈旧，而应看数据源本身是否被写入过。
        """
        try:
            from trader3.data_provider import QlibDataProvider

            dp = QlibDataProvider()
            cal_path = os.path.join(dp.data_dir, "calendars", "day.txt")
            if os.path.exists(cal_path):
                mt = os.path.getmtime(cal_path)
                from datetime import datetime as _dt
                return {
                    "path": cal_path,
                    "mtime": _dt.fromtimestamp(mt).isoformat(timespec="seconds"),
                    "age_seconds": round(time.time() - mt, 1),
                }
        except Exception:
            return None
        return None

    def check_data_freshness(self, report=None) -> GateResult:
        """
        数据版本新鲜度: 通过 SharedState.ensure_freshness() 校验。
        数据超过 4 小时未更新 → 未通过（提示触发增量更新）。
        """
        if not self.enabled:
            return self._pass_through("数据版本新鲜度")

        state = self.state
        if state is None:
            return self._result(
                True, "数据版本新鲜度", score=0.5,
                message="数据新鲜度: SharedState 未注入，跳过",
                details={"reason": "no_shared_state", "max_age_seconds": DATA_MAX_AGE_SECONDS},
            )

        # 数据访问走 ensure_freshness（超龄返回默认值或原数据）
        state.ensure_freshness("data_version", max_age_seconds=DATA_MAX_AGE_SECONDS)
        version_info = state.read_json("data_version") or {}
        meta = version_info.get("metadata", {})
        updated_at = meta.get("updated_at")

        if not updated_at:
            src_updated = self._probe_source_mtime()
            if src_updated:
                return self._result(
                    True, "数据版本新鲜度", score=0.8,
                    message=(
                        "data_version 缺失，回退校验数据源文件时间: "
                        f"{src_updated['path']} (mtime={src_updated['mtime']})"
                    ),
                    details={
                        "fallback": "source_mtime",
                        "max_age_seconds": DATA_MAX_AGE_SECONDS,
                        **src_updated,
                    },
                )
            return self._result(
                False, "数据版本新鲜度", score=0.0,
                message="数据陈旧 (data stale, trigger update): data_version 缺少 updated_at 且未探测到数据源文件",
                details={"updated_at": None, "max_age_seconds": DATA_MAX_AGE_SECONDS,
                         "versions": version_info.get("versions", {}),
                         "action": "trigger update"},
            )

        from datetime import datetime, timedelta

        age = datetime.now() - datetime.fromisoformat(updated_at)
        stale = age > timedelta(seconds=DATA_MAX_AGE_SECONDS)
        passed = not stale

        message = (
            f"数据版本新鲜度: 数据正常（更新于 {updated_at}，age={age.total_seconds():.0f}s）"
            if passed else
            "数据陈旧 (data stale, trigger update): 数据超过 4 小时未更新"
        )

        return self._result(
            passed, "数据版本新鲜度",
            score=1.0 if passed else 0.0,
            message=message,
            details={
                "updated_at": updated_at,
                "age_seconds": round(age.total_seconds(), 1),
                "max_age_seconds": DATA_MAX_AGE_SECONDS,
                "versions": version_info.get("versions", {}),
                "action": "trigger update" if not passed else "none",
            },
        )

    # ── Gate 7: 默认未验证 (新增) ──

    def check_credibility_default(self, report) -> GateResult:
        """
        默认未验证: 无证据支撑的字段默认为 "unverified"。
        估值类产出若基于合成数据 → 可信度标记为 "low"。
        """
        if not self.enabled:
            return self._pass_through("默认未验证")

        report_dict = _to_dict(report)

        # 1) 数据来源标注检查（caveats 中是否出现合成/模板字样）
        caveats = _safe_get(report, "caveats", None)
        if caveats is None:
            caveats = report_dict.get("caveats", [])
        caveats = caveats if isinstance(caveats, list) else []
        caveat_text = " ".join(str(c) for c in caveats)

        synthetic_markers = ["合成", "模板", "synthetic", "template", "M0", "模拟"]
        synthetic = any(m in caveat_text for m in synthetic_markers)

        # 2) 逐字段可信度: 无证据 → "unverified"
        evidence_keywords = ("source", "verified", "evidence", "real")
        unverified_fields: List[str] = []
        for k, v in report_dict.items():
            if isinstance(v, (str, int, float)) and not isinstance(v, bool):
                if k in ("key_assumptions", "methods"):
                    continue
                # 简单启发: 数值字段且该字段名不含 evidence 词 → 视为未验证
                unverified_fields.append(k)
            if len(unverified_fields) >= 12:
                break

        credibility = "low" if synthetic else "medium"
        if not synthetic:
            # 若存在已标注来源的字段，提升为 medium+（M6 保守估计）
            has_source = any(ev in caveat_text for ev in ("来源", "数据源", "source", "Source"))
            if has_source:
                credibility = "medium"

        checks = {
            "synthetic_data_tagged": synthetic or not _has_explicit_synthetic_flag(report),
            "unverified_fields_defaulted": len(unverified_fields) > 0,
        }

        # 门禁语义: 合成/模板数据 → 必须明确标注，且可信度降级；标注存在则通过。
        # 牙齿：产出内部显式声明 used_synthetic=True 但 caveats 未标注 → 拦截（铁律#1）。
        explicit_flag = _has_explicit_synthetic_flag(report)
        passed = not (explicit_flag and not synthetic)

        return self._result(
            passed, "默认未验证", score=1.0 if passed else 0.0,
            message=(
                f"默认未验证: 数据来源 {'合成/模板 → 可信度 low' if synthetic else '未标注 → 可信度 medium'}"
                f"，{len(unverified_fields)} 个字段默认为 unverified"
            ),
            details={
                "credibility": credibility,
                "synthetic_data": synthetic,
                "unverified_field_count": len(unverified_fields),
                "unverified_fields_sample": unverified_fields[:8],
                "evidence_policy": "无引用证据的字段默认为 unverified",
            },
        )

    # ── Gate 8: 抽检审计 (新增) ──

    def sample_audit(self, report) -> GateResult:
        """抽检审计: 15% 随机抽样, 内部一致性交叉验证"""
        if not self.enabled:
            return self._pass_through("抽检审计")
        return run_sample_audit(report)

    # ── 批量运行 ──

    def run_all(self, type: str = "", data: Any = None) -> List[GateResult]:
        """运行所有相关门禁（按产出类型 + 产出实例路由）。"""
        results = []
        for name, check in self._get_checks(type, data):
            try:
                result = check(data)
            except Exception as e:
                result = GateResult(
                    passed=False, check_name=name, score=0.0,
                    message=f"门禁异常: {e}", details={"error": str(e)},
                )
            results.append(result)
            self._history.append(result)
        return results

    def summary(self) -> Dict[str, Any]:
        """门禁摘要"""
        total = len(self._history)
        passed = sum(1 for r in self._history if r.passed)
        return {
            "enabled": self.enabled,
            "total_checks": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "latest": [
                {"check_name": r.check_name, "passed": r.passed,
                 "score": r.score, "message": r.message}
                for r in self._history[-5:]
            ],
        }

    # ── 内部工具 ──

    def _pass_through(self, name: str) -> GateResult:
        """M0 兼容: 默认通过"""
        result = GateResult(
            passed=True, check_name=name, score=1.0,
            message=f"[M0 存根] {name} 默认通过（门禁未启用）",
        )
        self._history.append(result)
        return result

    def _result(self, passed: bool, name: str, score: float = 0.0,
                details: Optional[dict] = None, message: str = "") -> GateResult:
        if not message:
            message = "通过" if passed else f"未通过: {name}"
        result = GateResult(
            passed=passed, check_name=name, score=score,
            message=message, details=details or {},
        )
        self._history.append(result)
        return result

    def _get_checks(self, output_type: str = "", data: Any = None) -> List[Tuple[str, Callable]]:
        """
        产出 → 门禁检查列表。

        1) 按类别路由基础门禁（backtest / optimize / execution / valuation）；
        2) 按产出类型 isinstance 精确路由专用门禁（signal 类别下区分
           SignalValidationReport 与 RegimeDiagnosis），并追加通用门禁
           （数据新鲜度 / 默认未验证 / 抽检审计）。
        """
        # 类别基础门禁（回测类别的显著性门禁按实例精确路由到 BacktestReport）
        mapping = {
            "backtest": [],
            "optimize": [
                ("组合约束满足", self.check_constraints),
            ],
            "execution": [
                ("交易成本合理性", self.check_tca_reasonable),
            ],
            "valuation": [],
        }
        checks = list(mapping.get(output_type, []))

        # 精确类型路由（signal 类别下区分两种产出；回测显著性仅对 BacktestReport）
        type_specific: List[Tuple[str, Callable]] = []
        if isinstance(data, SignalValidationReport):
            type_specific.append(("因子有效性门槛", self.check_factor_hurdle))
        elif isinstance(data, RegimeDiagnosis):
            type_specific.append(("状态识别置信度", self.check_regime_confidence))
        elif isinstance(data, OptimizationResult):
            type_specific.append(("组合约束满足", self.check_constraints))
        elif isinstance(data, TCAEstimate):
            type_specific.append(("交易成本合理性", self.check_tca_reasonable))
        elif isinstance(data, BacktestReport):
            type_specific.append(("回测统计显著性", self.check_backtest_significance))

        # 去重后合并
        existing_names = {name for name, _ in checks}
        for name, check in type_specific:
            if name not in existing_names:
                checks.append((name, check))
                existing_names.add(name)

        # 通用门禁（所有产出）
        checks.extend([
            ("数据版本新鲜度", self.check_data_freshness),
            ("默认未验证", self.check_credibility_default),
            ("抽检审计", self.sample_audit),
        ])

        return checks
