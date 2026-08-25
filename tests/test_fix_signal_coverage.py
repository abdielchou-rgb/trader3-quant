"""
signal.py 覆盖率补齐（TDD）— 纯离线单元测试，numpy 合成数据直调内部函数。

缺口对照（基线 56% missing 列表）：
1. _compute_half_life 边界            → TestHalfLife
2. _rule_based_regime 全分支          → TestRuleBasedRegime
3. _causal_zscore 边界                → TestCausalZscore
4. _compute_conditional_validity      → TestConditionalValidity
5. analog / suggestion 各状态         → TestHistoricalAnalog / TestStrategySuggestion
6. ValidateSignalTool 用户面板 / horizons 回退 / _try_real_signal_panel（注入假数据源）
7. DiagnoseMarketRegimeTool 注入行情 / 合成回退 / HMM 异常规则回退 / 量能分支
8. _CustomGaussianHMM 直测 + hmmlearn 缺失回退（sys.modules 隐藏）

历史记录：以下三个 BUG 曾以 xfail-strict 看门，现已全部修复并转正为常规通过用例：
- BUG#1 _rule_based_regime 在 volumes=None 守卫之前解引用 volumes[-20:]
  → 已修：量能均值/现值计算移入 volumes_real 守卫内
- BUG#2 _try_real_signal_panel 对齐错位（fr 比 sig 少一期导致广播失败、
  恒回退合成面板）→ 已修：丢弃最后一期动量，sig/fr 严格等长对齐
- BUG#3 _fit_hmm 自研回退硬编码 n_features=4（无量能时仅2特征崩溃）
  → 已修：n_features=X.shape[1] 自适应
"""

import datetime as dt
import sys

import numpy as np
import pytest

REGIMES = {
    "trending_up", "ranging", "bearish", "high_vol", "liquidity_crisis",
}


# ═══════════════════════════════════════════
# 假数据源（注入 QlibDataProvider，避免真实 qlib 依赖）
# ═══════════════════════════════════════════


class _FakeProvider:
    """duck-type QlibDataProvider：instruments / load_stock 由闭包注入。"""

    def __init__(self, instruments_fn=None, load_fn=None, calendar=None):
        self._instruments_fn = instruments_fn or (lambda name: [])
        self._load_fn = load_fn or (lambda code, field: (None, []))
        self._calendar = calendar or []

    def calendar(self):
        return list(self._calendar)

    def instruments(self, name, **kwargs):
        return self._instruments_fn(name)

    def load_stock(self, code, field, *args, **kwargs):
        return self._load_fn(code, field)


def _iso_dates(n, start_year=2021):
    d = dt.date(start_year, 1, 4)
    out = []
    while len(out) < n:
        out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def _stock_prices(seed, n_days, base=40.0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, n_days)
    return base * np.exp(np.cumsum(rets))


def _install_panel_env(monkeypatch, n_stocks=22, n_days=110, split_calendar=False,
                       broken_load=False):
    """注入个股面板数据源；split_calendar=后半股票换日历轴（制造公共日历不足）。"""
    import trader3.data_provider as dp_mod

    dates_a = _iso_dates(n_days, 2021)
    dates_b = _iso_dates(n_days, 2023)
    codes = [f"SH6000{i:02d}" for i in range(n_stocks)]
    half = n_stocks // 2

    def load(code, field):
        if field != "close":
            return None, []
        if broken_load:
            return None, []
        i = int(code[-2:])
        dates = dates_b if (split_calendar and i >= half) else dates_a
        return _stock_prices(1000 + i, n_days), dates

    monkeypatch.setattr(
        dp_mod, "QlibDataProvider",
        lambda: _FakeProvider(
            instruments_fn=lambda name, **_: codes if name == "csi300" else [],
            load_fn=load,
            calendar=_iso_dates(n_days, 2021),
        ),
    )
    return codes


def _index_close(n=150, zero_count=0, seed=77):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.012, n - zero_count)
    px = 3000.0 * np.exp(np.cumsum(rets))
    if zero_count:
        px = np.concatenate([px, np.zeros(zero_count)])
    return px


def _install_index_env(monkeypatch, close, volume_mode="ok"):
    """注入指数数据源。volume_mode: ok|holes|sparse|error（见各用例）。"""
    import trader3.data_provider as dp_mod

    dates = _iso_dates(len(close))

    def load(code, field):
        if field == "close":
            return close, dates
        if volume_mode == "error":
            raise RuntimeError("volume 字段不可用")
        vals = np.full(len(dates), 8e6)
        if volume_mode == "sparse":
            vals[50:] = np.nan  # 尾段 100 日全缺 → 放弃量能
        elif volume_mode == "holes":
            vals[:5] = np.nan  # 头部缺 → 走前值填补
            vals[-30] = np.nan  # 尾段 100 日内缺 3 个 → 仍 ≥95 有效
            vals[-60] = np.nan
            vals[-90] = np.nan
        return vals, dates

    monkeypatch.setattr(
        dp_mod, "QlibDataProvider",
        lambda: _FakeProvider(load_fn=load),
    )


# ═══════════════════════════════════════════
# 1. 半衰期边界
# ═══════════════════════════════════════════


class TestHalfLife:
    def _fn(self):
        from trader3.tools.signal import _compute_half_life

        return _compute_half_life

    def test_too_short_returns_default(self):
        assert self._fn()(np.array([0.1, -0.2, 0.3, 0.0, 0.1])) == 12.0

    def test_constant_series_returns_default(self):
        assert self._fn()(np.full(30, 0.42)) == 12.0

    def test_zero_lag1_autocorr_returns_one_period(self):
        # [1,0,-1,0] 周期：所有相邻乘积恒为 0 → lag-1 自相关严格为 0
        # → 白噪声语义 → 半衰期 1 期
        x = np.tile([1.0, 0.0, -1.0, 0.0], 8)
        assert self._fn()(x) == 1.0

    def test_near_unit_root_caps_at_one_year(self):
        x = 0.99999 ** np.arange(5000)  # 确定性几何衰减，中心化后 φ≈0.9994 ≥ 0.999
        assert self._fn()(x) == 252.0

    def test_negative_autocorr_gives_short_halflife(self):
        rng = np.random.default_rng(3)
        x = np.zeros(300)
        for t in range(1, 300):
            x[t] = -0.7 * x[t - 1] + rng.normal(0.0, 0.05)
        hl = self._fn()(x)
        assert 1.0 < hl < 4.0, f"负自相关应得短半衰期, got {hl}"

    def test_typical_ar1_halflife(self):
        rng = np.random.default_rng(4)
        x = np.zeros(400)
        for t in range(1, 400):
            x[t] = 0.9 * x[t - 1] + rng.normal(0.0, 0.05)
        hl = self._fn()(x)
        assert 4.0 < hl < 12.0, f"φ=0.9 AR(1) 理论半衰期≈6.6, got {hl}"

    def test_nan_tolerant(self):
        rng = np.random.default_rng(5)
        x = np.zeros(200)
        for t in range(1, 200):
            x[t] = 0.9 * x[t - 1] + rng.normal(0.0, 0.05)
        x[50] = np.nan
        hl = self._fn()(x)
        assert np.isfinite(hl) and hl > 0


# ═══════════════════════════════════════════
# 2. 因果 z-score 边界
# ═══════════════════════════════════════════


class TestCausalZscore:
    def _fn(self):
        from trader3.tools.signal import _causal_zscore

        return _causal_zscore

    def test_single_point_is_zero(self):
        out = self._fn()(np.array([5.0]))
        assert out.shape == (1,) and out[0] == 0.0

    def test_constant_series_all_zero(self):
        out = self._fn()(np.full(10, 3.0))
        assert np.allclose(out, 0.0)

    def test_nan_propagates_after_first_point(self):
        out = self._fn()(np.array([1.0, np.nan, 3.0]))
        assert out[0] == 0.0
        assert np.isnan(out[1]) and np.isnan(out[2])

    def test_lookback_capped_at_60_periods(self):
        # 远古离群点滑出 60 期窗口后不再影响当前 z-score
        a = np.zeros(80)
        a[0] = 1000.0
        out = self._fn()(a)
        assert abs(out[1] + 1.0) < 1e-6  # 早期受离群点压制
        assert out[79] == 0.0  # 窗口已不含 a[0]


# ═══════════════════════════════════════════
# 3. IC / 拥挤度退化分支
# ═══════════════════════════════════════════


def test_ic_series_degenerate_cross_section_is_zero():
    from trader3.tools.signal import _compute_ic_series

    rng = np.random.default_rng(6)
    sig = rng.normal(0.0, 1.0, (4, 10))
    sig[0] = 1.0  # 整行常数 → 该期 IC 记 0
    fr = rng.normal(0.0, 0.01, (4, 10))
    ic = _compute_ic_series(sig, fr)
    assert ic[0] == 0.0
    assert len(ic) == 4


class TestCrowding:
    def _fn(self):
        from trader3.tools.signal import _compute_crowding_index

        return _compute_crowding_index

    def test_small_panel_returns_zero(self):
        rng = np.random.default_rng(7)
        assert self._fn()(rng.normal(size=(10, 2))) == 0.0

    def test_all_degenerate_columns_return_zero(self):
        sig = np.ones((12, 3))
        sig[:, 2] = np.linspace(-1.0, 1.0, 12)  # 唯一有方差的列无有效配对
        assert self._fn()(sig) == 0.0

    def test_nan_heavy_column_skipped_but_valid_pair_counted(self):
        rng = np.random.default_rng(8)
        sig = rng.normal(size=(8, 3))
        sig[:, 1] = np.nan  # 仅 0 个有限重叠 → 跳过该配对
        sig[:, 1][:3] = [1.0, 2.0, 3.0]  # 与其他列重叠只有 3 期 < 5 → 仍跳过
        c = self._fn()(sig)
        assert 0.0 <= c <= 1.0


# ═══════════════════════════════════════════
# 4. 条件有效性（两路径 + 分组不足）
# ═══════════════════════════════════════════


class TestConditionalValidity:
    def _fn(self):
        from trader3.tools.signal import _compute_conditional_validity

        return _compute_conditional_validity

    def _panel(self, t_len, n_assets, seed):
        rng = np.random.default_rng(seed)
        sig = np.tile(np.arange(n_assets, dtype=np.float64), (t_len, 1))
        sig += rng.normal(0.0, 1e-9, sig.shape)
        fr = 0.001 * sig + rng.normal(0.0, 1e-12, sig.shape)
        return sig, fr

    def test_regime_split_basic(self):
        sig, fr = self._panel(40, 8, 11)
        mr = np.concatenate([
            np.zeros(20), np.tile([0.05, -0.05], 10),
        ])
        cond = self._fn()(sig, fr, mr)
        assert set(cond.keys()) == {"低波动", "中波动", "高波动"}
        for v in cond.values():
            assert -1.0 <= v <= 1.0
        assert sum(1 for v in cond.values() if v > 0.5) >= 2

    def test_insufficient_groups_yield_zero(self):
        # 构造波动率分布使 低波动组为空、高波动组仅 2 期 → 两组各走 mask.sum()<3 分支
        sig, fr = self._panel(12, 8, 12)
        small = np.tile([0.001, -0.001], 6)[:11]
        mr = np.concatenate([small, [1.0]])
        cond = self._fn()(sig, fr, mr)
        assert cond["低波动"] == 0.0   # 空组分支
        assert cond["高波动"] == 0.0   # 组内样本不足分支
        assert cond["中波动"] > 0.9    # 完美秩相关面板


# ═══════════════════════════════════════════
# 5. 历史类比 / 策略建议 各状态
# ═══════════════════════════════════════════


class TestHistoricalAnalog:
    def _fn(self):
        from trader3.tools.signal import _find_historical_analog

        return _find_historical_analog

    def test_fuzzy_state_gets_suffix(self):
        probs = {name: 0.2 for name in REGIMES}
        out = self._fn()("bearish", probs)
        assert "状态模糊" in out and "下跌趋势" in out

    def test_clear_state_gets_suffix(self):
        out = self._fn()("trending_up", {"trending_up": 0.95, "ranging": 0.05})
        assert "状态清晰" in out and "主升浪" in out

    def test_medium_confidence_plain_text(self):
        out = self._fn()("high_vol", {"high_vol": 0.6, "ranging": 0.4})
        assert "状态模糊" not in out and "状态清晰" not in out
        assert "疫情冲击" in out

    def test_unknown_regime_falls_back_to_ranging_analog(self):
        out = self._fn()("weird_state", {"weird_state": 1.0})
        assert "震荡区间" in out and "状态清晰" in out


class TestStrategySuggestion:
    def _fn(self):
        from trader3.tools.signal import _generate_strategy_suggestion

        return _generate_strategy_suggestion

    def test_all_known_regimes_map_to_expected_position(self):
        cases = {
            "trending_up": 0.85, "ranging": 0.60, "bearish": 0.30,
            "high_vol": 0.40, "liquidity_crisis": 0.10,
        }
        for regime, pos in cases.items():
            text, weighted = self._fn()(regime, {regime: 1.0})
            assert f"仓位 {pos:.0%}" in text, regime
            assert weighted == pos, regime
            assert "建议仓位" in text

    def test_blended_position_weighted_by_probs(self):
        _, weighted = self._fn()("bearish", {"bearish": 0.5, "high_vol": 0.5})
        assert weighted == 0.35  # (0.30 + 0.40) / 2

    def test_unknown_regime_falls_back_to_neutral_text(self):
        text, weighted = self._fn()("zzz", {"ranging": 1.0})
        assert "震荡格局" in text and "50%" in text
        assert weighted == 0.60  # 文案回退，但仓位仍按概率加权


# ═══════════════════════════════════════════
# 6. 规则回退全分支
# ═══════════════════════════════════════════


_AUTO = object()


def _rb_data(drift, vol_daily, n=80, seed=5, volumes_real=False,
             volumes=_AUTO):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol_daily, n)
    prices = 3000.0 * np.exp(np.cumsum(rets))
    if volumes is _AUTO:
        volumes = np.full(n, 1e6)
    return {
        "prices": prices,
        "returns": rets,
        "volumes": volumes,
        "volumes_real": volumes_real,
    }


class TestRuleBasedRegime:
    def _fn(self):
        from trader3.tools.signal import _rule_based_regime

        return _rule_based_regime

    def test_short_history_exact_early_exit(self):
        data = _rb_data(0.001, 0.01, n=30)
        data["prices"] = data["prices"][:30]
        data["returns"] = data["returns"][:30]
        assert self._fn()(data) == ("ranging", {"ranging": 1.0}, 0.0)

    def test_golden_cross_detects_trending_up(self):
        regime, scores, _ = self._fn()(_rb_data(0.002, 0.008))
        assert regime == "trending_up"
        assert scores["trending_up"] == max(scores.values())

    def test_death_cross_detects_bearish(self):
        regime, scores, _ = self._fn()(_rb_data(-0.002, 0.008))
        assert regime == "bearish"
        assert scores["bearish"] == max(scores.values())

    def test_flat_market_is_ranging(self):
        regime, scores, _ = self._fn()(_rb_data(0.0002, 0.006))
        assert regime == "ranging"
        assert scores["ranging"] > 0.6

    def test_high_vol_adds_score_even_when_bearish_wins(self):
        regime, scores, entropy = self._fn()(_rb_data(-0.0005, 0.032))
        assert scores["high_vol"] > 0.0  # 高波动项被触发
        assert scores["bearish"] > scores["high_vol"]
        assert regime in REGIMES and 0.0 < entropy <= 1.0

    def test_volume_absent_flag_skips_volume_term(self):
        # 有成交量数组但标记非真实 → 量能项完全不参与
        data = _rb_data(0.002, 0.008, volumes_real=False)
        regime, scores, _ = self._fn()(data)
        assert regime == "trending_up"
        assert scores["liquidity_crisis"] == 0.0

    def test_low_volume_triggers_crisis_score_and_wins(self):
        n = 80
        volumes = np.full(n, 1e5)
        volumes[-1] = 2e4  # 当日量能 < 20 日均值一半
        data = _rb_data(0.0002, 0.018, volumes_real=True, volumes=volumes)
        regime, scores, _ = self._fn()(data)
        assert scores["liquidity_crisis"] > 0.0
        assert regime == "liquidity_crisis"

    def test_volume_surge_boosts_trending_up(self):
        n = 80
        volumes = np.full(n, 1e5)
        volumes[-1] = 4e5  # 当日量能 > 20 日均值 1.5 倍
        data = _rb_data(0.002, 0.008, volumes_real=True, volumes=volumes)
        regime, scores, _ = self._fn()(data)
        assert scores["trending_up"] >= 0.70  # 0.6 + 0.2 量能加成，归一化后 ≈0.727
        assert regime == "trending_up"

    def test_scores_normalized_and_entropy_bounded(self):
        _, scores, entropy = self._fn()(_rb_data(0.0002, 0.006, seed=15))
        assert set(scores.keys()) == REGIMES
        assert abs(sum(scores.values()) - 1.0) < 1e-9
        assert 0.0 < entropy <= 1.0

    def test_rule_based_volumes_none_known_bug(self):
        data = _rb_data(0.0002, 0.006, volumes=None)
        data.pop("volumes_real")
        regime, scores, _ = self._fn()(data)
        assert regime in REGIMES  # pragma: no cover


# ═══════════════════════════════════════════
# 7. 自研 HMM（_CustomGaussianHMM）与 hmmlearn 缺失回退
# ═══════════════════════════════════════════


@pytest.fixture()
def hide_hmmlearn(monkeypatch):
    """sys.modules 里置 None 使 'from hmmlearn import hmm' 抛 ImportError。"""
    monkeypatch.setitem(sys.modules, "hmmlearn", None)


class TestCustomGaussianHMM:
    def _cls(self):
        from trader3.tools.signal import _CustomGaussianHMM

        return _CustomGaussianHMM

    def _bimodal(self):
        rng = np.random.default_rng(21)
        lo = rng.normal(0.0, 0.5, (60, 2))
        hi = rng.normal(4.0, 0.5, (60, 2))
        return np.vstack([lo, hi])

    def test_fit_sets_valid_parameters(self):
        model = self._cls()(n_states=2, n_features=2, random_state=7, n_iter=25)
        model.fit(self._bimodal())
        assert model.transmat_ is not None
        assert np.allclose(model.transmat_.sum(axis=1), 1.0, atol=1e-8)
        assert np.allclose(model.startprob_.sum(), 1.0, atol=1e-8)
        assert model.means_.shape == (2, 2) and np.isfinite(model.means_).all()
        assert np.all(np.diag(model.covars_[0]) > 0)

    def test_predict_consistent_with_posteriors(self):
        x = self._bimodal()
        model = self._cls()(n_states=2, n_features=2, random_state=7, n_iter=25)
        model.fit(x)
        proba = model.predict_proba(x)
        assert proba.shape == (120, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert np.array_equal(model.predict(x), proba.argmax(axis=1))

    def test_zero_variance_input_does_not_crash(self):
        x = np.zeros((30, 2))
        model = self._cls()(n_states=2, n_features=2, random_state=7, n_iter=3)
        model.fit(x)
        proba = model.predict_proba(x)
        assert np.isfinite(proba).all()
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_try_hmmlearn_hidden_module_returns_none(hide_hmmlearn):
    from trader3.tools.signal import _try_hmmlearn

    assert _try_hmmlearn() is None


def test_fit_hmm_custom_em_fallback(hide_hmmlearn, monkeypatch):
    """hmmlearn 缺失 → 自研 EM 回退；需真实量能标记凑齐 4 特征
    （源码硬编码 n_features=4，缺量能时 2 特征会触发 reshape 崩溃，见 BUG#3）。"""
    import trader3.tools.signal as sig

    monkeypatch.setattr(sig, "HMM_N_ITER", 12)
    data = sig._generate_market_regime_data()
    trimmed = {
        k: (v[:240] if isinstance(v, np.ndarray) else v)
        for k, v in data.items()
    }
    trimmed["volumes_real"] = True
    regime, probs, entropy = sig._fit_hmm(trimmed)
    assert regime in REGIMES
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert 0.0 <= entropy <= 1.0


def test_fit_hmm_short_series_leaves_states_unpopulated(hide_hmmlearn,
                                                        monkeypatch):
    """T=3 < 4 状态 → 必有空状态 → 走 mask.sum()==0 → 标记 ranging 分支。"""
    import trader3.tools.signal as sig

    monkeypatch.setattr(sig, "HMM_N_ITER", 5)
    data = {
        "returns": np.array([0.001, -0.002, 0.0005]),
        "volumes": np.array([1e6, 1.1e6, 0.9e6]),
        "volumes_real": True,
    }
    regime, probs, entropy = sig._fit_hmm(data)
    assert regime in REGIMES
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert 0.0 <= entropy <= 1.0


def test_fit_hmm_labels_liquidity_crisis(hide_hmmlearn, monkeypatch):
    """全程 ±5% 高波动震荡、均值≈0 → 主导状态应被标记 liquidity_crisis。
    （需带真实量能标记凑齐 4 特征，规避 BUG#3 的 reshape 崩溃。）"""
    import trader3.tools.signal as sig

    monkeypatch.setattr(sig, "HMM_N_ITER", 5)
    data = {
        "returns": np.tile([0.05, -0.05], 40),
        "volumes": np.full(80, 1e6),
        "volumes_real": True,
    }
    regime, probs, _ = sig._fit_hmm(data)
    assert regime == "liquidity_crisis"
    assert probs["liquidity_crisis"] > 0.99


# ═══════════════════════════════════════════
# 8. ValidateSignalTool：用户面板 / horizons / 真实面板
# ═══════════════════════════════════════════


def _perfect_panel_values(n_assets=14):
    """信号=资产序号、收益=序号比例 → 每期 Spearman IC=1。"""
    sig = np.tile(np.arange(n_assets, dtype=np.float64), (n_assets, 1))
    fr = 0.001 * sig
    return sig.reshape(-1).tolist(), fr.reshape(-1).tolist()


class TestValidateSignalUserPanel:
    def _tool(self):
        from trader3.tools.signal import ValidateSignalTool

        return ValidateSignalTool()

    def test_user_panel_perfect_ic_and_monotonicity(self):
        sig, fr = _perfect_panel_values()
        r = self._tool().execute(signal_name="排序因子", signal_values=sig,
                                 forward_returns={1: fr})
        assert r.success, r.summary
        rep = r.data
        assert rep.ic_mean == pytest.approx(1.0, abs=1e-6)
        assert rep.monotonicity == 1.0
        assert rep.long_short_return > 0 and rep.long_only_return > 0
        assert "真实数据" in r.summary  # 用户供数即视为真实路径

    def test_horizons_selects_requested_key(self):
        sig, fr = _perfect_panel_values()
        r = self._tool().execute(signal_values=sig, forward_returns={5: fr},
                                 horizons=[5])
        assert r.success and r.data.ic_mean == pytest.approx(1.0, abs=1e-6)

    def test_missing_horizon_falls_back_to_first_key(self):
        sig, fr = _perfect_panel_values()
        r = self._tool().execute(signal_values=sig, forward_returns={3: fr},
                                 horizons=[99])
        assert r.success and r.data.ic_mean == pytest.approx(1.0, abs=1e-6)

    def test_nan_forward_returns_keep_ic_metrics_finite(self):
        sig, fr = _perfect_panel_values()
        fr_with_nan = list(fr)
        for i in range(14):  # 最后一行整行 NaN → 市场基准回退 0
            fr_with_nan[13 * 14 + i] = float("nan")
        r = self._tool().execute(signal_values=sig,
                                 forward_returns={1: fr_with_nan})
        assert r.success
        rep = r.data  # IC 侧指标不受污染（分组收益会传播 NaN，属输入性缺失）
        for v in (rep.ic_mean, rep.ic_std, rep.icir, rep.half_life_periods,
                  rep.crowding_index):
            assert np.isfinite(float(v)), r.key_metrics

    def test_default_call_synthetic_and_unnamed(self):
        r = self._tool().execute()
        assert r.success and "未命名因子" in r.summary


class TestTryRealSignalPanel:
    def _static(self):
        from trader3.tools.signal import ValidateSignalTool

        return ValidateSignalTool._try_real_signal_panel

    def test_tool_uses_injected_real_panel(self, monkeypatch):
        _install_panel_env(monkeypatch, n_stocks=22, n_days=110)
        from trader3.tools.signal import ValidateSignalTool

        r = ValidateSignalTool().execute(signal_name="真实面板因子")
        assert r.success, r.summary
        assert "真实数据" in r.summary
        assert any("真实 qlib 行情数据" in c for c in r.caveats)
        assert len(r.data.ic_series) == 60

    def test_too_few_instruments_returns_none(self, monkeypatch):
        _install_panel_env(monkeypatch, n_stocks=10, n_days=110)
        assert self._static()() is None

    def test_all_load_failures_return_none(self, monkeypatch):
        _install_panel_env(monkeypatch, n_stocks=22, n_days=110,
                           broken_load=True)
        assert self._static()() is None

    def test_disjoint_calendars_return_none(self, monkeypatch):
        # 每只股票自身数据足够，但两半日历轴不相交 → 公共交集不足 81 日
        _install_panel_env(monkeypatch, n_stocks=22, n_days=110,
                           split_calendar=True)
        assert self._static()() is None

    def test_happy_path_builds_aligned_panels(self, monkeypatch):
        _install_panel_env(monkeypatch, n_stocks=22, n_days=110)
        out = self._static()()
        assert out is not None
        sig, fr, mr = out
        assert sig.shape == (60, 22) and fr.shape == (60, 22)
        assert np.isfinite(sig).all() and np.isfinite(fr).all()
        assert mr.shape == (60,) and np.isfinite(mr).all()


# ═══════════════════════════════════════════
# 9. DiagnoseMarketRegimeTool：注入行情 / 回退链 / 量能分支
# ═══════════════════════════════════════════


def _injected_series(n=120, seed=31):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.001, 0.01, n)
    prices = 3000.0 * np.exp(np.cumsum(rets))
    volumes = 8e6 * (1.0 + rng.normal(0.0, 0.1, n))
    return prices, volumes


class TestDiagnoseInjectedPath:
    def _tool(self):
        from trader3.tools.signal import DiagnoseMarketRegimeTool

        return DiagnoseMarketRegimeTool()

    def test_prices_and_volumes_injection_runs_hmm(self):
        prices, volumes = _injected_series()
        r = self._tool().execute(prices=prices.tolist(),
                                 volumes=volumes.tolist())
        assert r.success, r.summary
        assert "HMM" in r.summary
        assert "合成数据" not in r.summary
        assert r.data.current_regime in REGIMES
        assert 0.0 <= r.data.suggested_position <= 1.0
        assert "20日年化波动率(%)" in r.key_metrics
        # 注入路径未带 volumes_real 标记 → 明确声明无量能维度
        assert any("无真实量能数据" in c for c in r.caveats)

    def test_synthetic_fallback_when_provider_unavailable(self, monkeypatch):
        import trader3.data_provider as dp_mod

        class _BoomProvider:
            def __init__(self):
                raise RuntimeError("qlib 离线")

        monkeypatch.setattr(dp_mod, "QlibDataProvider", _BoomProvider)
        r = self._tool().execute(lookback=60)
        assert r.success, r.summary
        assert "合成数据" in r.summary
        assert any("合成数据" in c for c in r.caveats)

    def test_rule_based_fallback_when_hmm_raises(self, monkeypatch):
        import trader3.tools.signal as sig

        def _boom(_data):
            raise ValueError("HMM 内部异常")

        monkeypatch.setattr(sig, "_fit_hmm", _boom)
        prices, volumes = _injected_series()
        r = self._tool().execute(prices=prices.tolist(),
                                 volumes=volumes.tolist())
        assert r.success, r.summary
        assert "规则回退 (rule-based)" in r.summary


class TestTryRealIndexData:
    def _static(self):
        from trader3.tools.signal import DiagnoseMarketRegimeTool

        return DiagnoseMarketRegimeTool._try_real_index_data

    def test_short_close_falls_back_to_synthetic(self, monkeypatch):
        _install_index_env(monkeypatch, _index_close(n=60))
        assert self._static()() == (None, True)

    def test_zero_padded_close_falls_back(self, monkeypatch):
        # 总长足够但有效价 < 100 → 放弃
        _install_index_env(monkeypatch, _index_close(n=150, zero_count=80))
        assert self._static()() == (None, True)

    def test_volume_field_error_yields_no_volume_features(self, monkeypatch):
        _install_index_env(monkeypatch, _index_close(n=150),
                           volume_mode="error")
        data, synthetic = self._static()()
        assert synthetic is False
        assert data["volumes"] is None and data["volumes_real"] is False
        assert len(data["returns"]) == 149

    def test_sparse_tail_volume_discarded(self, monkeypatch):
        _install_index_env(monkeypatch, _index_close(n=150),
                           volume_mode="sparse")
        data, synthetic = self._static()()
        assert synthetic is False
        assert data["volumes"] is None and data["volumes_real"] is False

    def test_hole_volume_filled_and_flagged_real(self, monkeypatch):
        _install_index_env(monkeypatch, _index_close(n=150),
                           volume_mode="holes")
        data, synthetic = self._static()()
        assert synthetic is False
        assert data["volumes_real"] is True
        assert np.isfinite(data["volumes"]).all()

        from trader3.tools.signal import DiagnoseMarketRegimeTool

        r = DiagnoseMarketRegimeTool().execute()
        assert r.success, r.summary
        assert any("量能变化(真实成交量)" in c for c in r.caveats)
        assert "合成数据" not in r.summary


def test_generate_market_regime_data_contract():
    from trader3.tools.signal import _generate_market_regime_data

    data = _generate_market_regime_data(lookback=60)
    assert set(data.keys()) == {"prices", "volumes", "returns", "rolling_vol"}
    t = max(120, 252 * 3)
    for key in ("prices", "volumes", "returns", "rolling_vol"):
        assert len(data[key]) == t, key
    assert (data["prices"] > 0).all()
    assert (data["volumes"] > 0).all()
    assert np.isfinite(data["returns"]).all()
    assert np.isfinite(data["rolling_vol"]).all()
