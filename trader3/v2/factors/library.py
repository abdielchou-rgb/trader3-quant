from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Factor:
    name: str
    description: str
    category: str
    params: dict[str, Any]
    formula: str


class FactorBase(ABC):
    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        pass

    def get_factor_info(self) -> Factor:
        return Factor(
            name=self.__class__.__name__,
            description=self.__doc__ or "",
            category=getattr(self, "category", "unknown"),
            params=self.params,
            formula=getattr(self, "formula", ""),
        )


class Momentum(FactorBase):
    category = "momentum"
    formula = "close / close.shift(n) - 1"

    def __init__(self, window: int = 20):
        super().__init__(window=window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        return data["close"] / data["close"].shift(n) - 1


class MeanReversion(FactorBase):
    category = "mean_reversion"
    formula = "(close - close.rolling(n).mean()) / close.rolling(n).std()"

    def __init__(self, window: int = 20):
        super().__init__(window=window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        ma = data["close"].rolling(n).mean()
        std = data["close"].rolling(n).std()
        return (data["close"] - ma) / std


class RSIFactor(FactorBase):
    category = "momentum"
    formula = "100 - 100 / (1 + RS)"

    def __init__(self, window: int = 14):
        super().__init__(window=window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(n).mean()
        loss = -delta.where(delta < 0, 0).rolling(n).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)


class MACDFactor(FactorBase):
    category = "trend"
    formula = "EMA(12) - EMA(26)"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(fast=fast, slow=slow, signal=signal)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        fast = data["close"].ewm(span=self.params["fast"]).mean()
        slow = data["close"].ewm(span=self.params["slow"]).mean()
        return fast - slow


class BollingerBands(FactorBase):
    category = "volatility"
    formula = "(close - MA) / (2 * std)"

    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(window=window, num_std=num_std)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        k = self.params["num_std"]
        ma = data["close"].rolling(n).mean()
        std = data["close"].rolling(n).std()
        upper = ma + k * std
        lower = ma - k * std
        return (data["close"] - ma) / (upper - lower)


class VolumeWeightedMomentum(FactorBase):
    category = "volume"
    formula = "sum(volume * return) / sum(volume)"

    def __init__(self, window: int = 20):
        super().__init__(window=window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        ret = data["close"].pct_change()
        vwap_ret = (data["volume"] * ret).rolling(n).sum() / data["volume"].rolling(n).sum()
        return vwap_ret


class VolumeRateOfChange(FactorBase):
    category = "volume"
    formula = "volume / volume.shift(n) - 1"

    def __init__(self, window: int = 10):
        super().__init__(window=window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        return data["volume"] / data["volume"].shift(n) - 1


class OBVFactor(FactorBase):
    category = "volume"
    formula = "cumsum(sign(close_diff) * volume)"

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close_diff = data["close"].diff()
        sign = np.sign(close_diff).fillna(0)
        return (sign * data["volume"]).cumsum()


class VWAPDeviation(FactorBase):
    category = "microstructure"
    formula = "(close - VWAP) / VWAP"

    def __init__(self, window: int = 1):
        super().__init__(window=window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "vwap" in data.columns:
            vwap = data["vwap"]
        else:
            typical = (data["high"] + data["low"] + data["close"]) / 3
            vwap = (typical * data["volume"]).cumsum() / data["volume"].cumsum()
        return (data["close"] - vwap) / vwap


class IlliquidityFactor(FactorBase):
    category = "microstructure"
    formula = "abs(return) / volume"

    def __init__(self, window: int = 20):
        super().__init__(window=window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        ret = data["close"].pct_change().abs()
        return (ret / data["volume"]).rolling(n).mean()


class TurnoverFactor(FactorBase):
    category = "microstructure"
    formula = "volume / shares_outstanding"

    def __init__(self, window: int = 20):
        super().__init__(window=window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        if "shares_outstanding" in data.columns:
            turnover = data["volume"] / data["shares_outstanding"]
        else:
            turnover = data["volume"] / 1e8
        return turnover.rolling(n).mean()


class Alpha1(FactorBase):
    category = "alpha101"
    formula = "rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2), 5)) - 0.5"

    def __init__(self, window: int = 20):
        super().__init__(window=window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        returns = data["close"].pct_change()
        cond = returns < 0
        val = np.where(cond, returns.rolling(n).std(), data["close"])
        signed_pow = np.sign(val) * val ** 2
        ts_argmax = signed_pow.rolling(5).apply(lambda x: np.argmax(x) if len(x) == 5 else np.nan, raw=True)
        return ts_argmax.rank(pct=True) - 0.5


class Alpha2(FactorBase):
    category = "alpha101"
    formula = "(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"

    def __init__(self, window: int = 6):
        super().__init__(window=window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        vol_delta = np.log(data["volume"]).diff(2)
        ret = (data["close"] - data["open"]) / data["open"]
        corr = vol_delta.rolling(n).corr(ret)
        return -1 * corr.rank(pct=True)


class Alpha3(FactorBase):
    category = "alpha101"
    formula = "(-1 * correlation(rank(open), rank(volume), 10))"

    def __init__(self, window: int = 10):
        super().__init__(window=window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        corr = data["open"].rolling(n).corr(data["volume"])
        return -1 * corr.rank(pct=True)


class Alpha4(FactorBase):
    category = "alpha101"
    formula = "(-1 * Ts_Rank(rank(low), 9))"

    def __init__(self, window: int = 9):
        super().__init__(window=window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        rank_low = data["low"].rank(pct=True)
        ts_rank = rank_low.rolling(n).apply(lambda x: x.rank(pct=True).iloc[-1] if len(x) == n else np.nan, raw=True)
        return -1 * ts_rank


class Alpha5(FactorBase):
    category = "alpha101"
    formula = "(rank(open) - rank(close)) * rank(volume)"

    def compute(self, data: pd.DataFrame) -> pd.Series:
        rank_open = data["open"].rank(pct=True)
        rank_close = data["close"].rank(pct=True)
        rank_vol = data["volume"].rank(pct=True)
        return (rank_open - rank_close) * rank_vol


class Alpha6(FactorBase):
    category = "alpha101"
    formula = "(-1 * correlation(open, volume, 10))"

    def __init__(self, window: int = 10):
        super().__init__(window=window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        return -1 * data["open"].rolling(n).corr(data["volume"])


class Alpha7(FactorBase):
    category = "alpha101"
    formula = "((adv20 < volume) ? ((-1 * Ts_Rank(abs(delta(close, 1)), 5)) * rank(delta(close, 1))) : -1)"

    def __init__(self, window: int = 20):
        super().__init__(window=window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        adv20 = data["volume"].rolling(n).mean()
        cond = adv20 < data["volume"]
        delta_close = data["close"].diff(1)
        ts_rank = delta_close.abs().rolling(5).apply(
            lambda x: x.rank(pct=True).iloc[-1] if len(x) == 5 else np.nan, raw=True
        )
        val = -1 * ts_rank * delta_close.rank(pct=True)
        return np.where(cond, val, -1)


class Alpha8(FactorBase):
    category = "alpha101"
    formula = "(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay(sum(open, 5) * sum(returns, 5), 10))))"

    def __init__(self, window: int = 5):
        super().__init__(window=window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        returns = data["close"].pct_change()
        sum_open = data["open"].rolling(n).sum()
        sum_ret = returns.rolling(n).sum()
        val = sum_open * sum_ret - (sum_open * sum_ret).shift(10)
        return -1 * val.rank(pct=True)


class Alpha9(FactorBase):
    category = "alpha101"
    formula = "((0 < Ts_Min(delta(close, 1), 5)) ? delta(close, 1) : ((Ts_Max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))"

    def __init__(self, window: int = 5):
        super().__init__(window=window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        delta = data["close"].diff(1)
        ts_min = delta.rolling(n).min()
        ts_max = delta.rolling(n).max()
        cond1 = ts_min > 0
        cond2 = ts_max < 0
        return np.where(cond1, delta, np.where(cond2, delta, -delta))


class Alpha10(FactorBase):
    category = "alpha101"
    formula = "rank(((0 < Ts_Min(delta(close, 1), 4)) ? delta(close, 1) : ((Ts_Max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))"

    def __init__(self, window: int = 4):
        super().__init__(window=window)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        delta = data["close"].diff(1)
        ts_min = delta.rolling(n).min()
        ts_max = delta.rolling(n).max()
        cond1 = ts_min > 0
        cond2 = ts_max < 0
        val = np.where(cond1, delta, np.where(cond2, delta, -delta))
        return val.rank(pct=True)


FACTOR_REGISTRY = {
    "momentum": Momentum,
    "mean_reversion": MeanReversion,
    "rsi": RSIFactor,
    "macd": MACDFactor,
    "bollinger": BollingerBands,
    "vwm": VolumeWeightedMomentum,
    "vol_roc": VolumeRateOfChange,
    "obv": OBVFactor,
    "vwap_dev": VWAPDeviation,
    "illiquidity": IlliquidityFactor,
    "turnover": TurnoverFactor,
    "alpha1": Alpha1,
    "alpha2": Alpha2,
    "alpha3": Alpha3,
    "alpha4": Alpha4,
    "alpha5": Alpha5,
    "alpha6": Alpha6,
    "alpha7": Alpha7,
    "alpha8": Alpha8,
    "alpha9": Alpha9,
    "alpha10": Alpha10,
}

FACTOR_META = {
    name: cls().get_factor_info() for name, cls in FACTOR_REGISTRY.items()
}


def get_factor(name: str, **params) -> FactorBase:
    if name not in FACTOR_REGISTRY:
        raise ValueError(f"Unknown factor: {name}. Available: {list(FACTOR_REGISTRY.keys())}")
    return FACTOR_REGISTRY[name](**params)


def list_factors(category: str | None = None) -> list[Factor]:
    factors = list(FACTOR_META.values())
    if category:
        factors = [f for f in factors if f.category == category]
    return factors


def compute_factor(data: pd.DataFrame, factor_name: str, **params) -> pd.Series:
    factor = get_factor(factor_name, **params)
    return factor.compute(data)


def compute_all_factors(data: pd.DataFrame, categories: list[str] | None = None) -> pd.DataFrame:
    results = {}
    for name, cls in FACTOR_REGISTRY.items():
        meta = FACTOR_META[name]
        if categories and meta.category not in categories:
            continue
        try:
            factor = cls()
            results[name] = factor.compute(data)
        except Exception:
            pass
    return pd.DataFrame(results, index=data.index)
