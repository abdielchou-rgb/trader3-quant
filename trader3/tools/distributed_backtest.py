import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import dask
    import dask.distributed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


@dataclass
class BacktestConfig:
    symbols: list[str]
    start_date: str
    end_date: str
    strategy_params: dict[str, Any]
    initial_cash: float = 1_000_000.0
    commission: float = 0.0003
    slippage_bps: float = 1.0


@dataclass
class BacktestResult:
    config: BacktestConfig
    metrics: dict[str, float]
    equity_curve: pd.Series
    trades: list[dict]
    runtime_seconds: float


def generate_param_grid(param_space: dict[str, list]) -> list[dict]:
    keys = list(param_space.keys())
    values = list(param_space.values())
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo, strict=False)) for combo in combinations]


def run_single_backtest(config: BacktestConfig) -> BacktestResult:
    start_time = time.time()

    np.random.seed(42)
    n_days = 252
    dates = pd.date_range(config.start_date, periods=n_days, freq='D')

    returns = np.random.randn(n_days) * 0.015 + 0.0005
    equity = config.initial_cash * np.cumprod(1 + returns)

    trades = []
    position = 0
    cash = config.initial_cash

    for i, (date, _ret) in enumerate(zip(dates, returns, strict=False)):
        signal = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
        if signal != 0 and position == 0:
            qty = cash * 0.1 / (equity[i] if i > 0 else config.initial_cash)
            price = equity[i] if i > 0 else config.initial_cash
            cost = qty * price * (1 + config.commission + config.slippage_bps / 10000)
            if cost <= cash:
                position = qty if signal == 1 else -qty
                cash -= cost
                trades.append({
                    "date": date.isoformat(),
                    "side": "buy" if signal == 1 else "sell",
                    "quantity": abs(qty),
                    "price": price,
                })
        elif signal == 0 and position != 0:
            price = equity[i] if i > 0 else config.initial_cash
            proceeds = abs(position) * price * (1 - config.commission - config.slippage_bps / 10000)
            cash += proceeds
            trades.append({
                "date": date.isoformat(),
                "side": "sell" if position > 0 else "buy",
                "quantity": abs(position),
                "price": price,
            })
            position = 0

    final_equity = cash + (position * equity[-1] if position != 0 else 0)
    total_return = (final_equity - config.initial_cash) / config.initial_cash

    daily_returns = pd.Series(equity).pct_change().dropna()
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    max_dd = (pd.Series(equity) / pd.Series(equity).cummax() - 1).min()

    metrics = {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "num_trades": len(trades),
        "win_rate": sum(1 for t in trades if t["side"] == "sell") / len(trades) if trades else 0,
    }

    runtime = time.time() - start_time

    return BacktestResult(
        config=config,
        metrics=metrics,
        equity_curve=pd.Series(equity, index=dates),
        trades=trades,
        runtime_seconds=runtime,
    )


def run_grid_search(param_space: dict[str, list], base_config: BacktestConfig, max_workers: int = 4) -> list[BacktestResult]:
    param_grid = generate_param_grid(param_space)
    configs = [
        BacktestConfig(
            symbols=base_config.symbols,
            start_date=base_config.start_date,
            end_date=base_config.end_date,
            strategy_params={**base_config.strategy_params, **params},
            initial_cash=base_config.initial_cash,
            commission=base_config.commission,
            slippage_bps=base_config.slippage_bps,
        )
        for params in param_grid
    ]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_backtest, cfg) for cfg in configs]
        for future in futures:
            results.append(future.result())

    return results


if RAY_AVAILABLE:
    @ray.remote
    def ray_backtest_task(config: BacktestConfig) -> BacktestResult:
        return run_single_backtest(config)

    def run_ray_grid_search(param_space: dict[str, list], base_config: BacktestConfig, num_workers: int = 4) -> list[BacktestResult]:
        ray.init(num_cpus=num_workers, ignore_reinit_error=True)
        try:
            param_grid = generate_param_grid(param_space)
            configs = [
                BacktestConfig(
                    symbols=base_config.symbols,
                    start_date=base_config.start_date,
                    end_date=base_config.end_date,
                    strategy_params={**base_config.strategy_params, **params},
                    initial_cash=base_config.initial_cash,
                    commission=base_config.commission,
                    slippage_bps=base_config.slippage_bps,
                )
                for params in param_grid
            ]
            futures = [ray_backtest_task.remote(cfg) for cfg in configs]
            results = ray.get(futures)
            return results
        finally:
            ray.shutdown()


if DASK_AVAILABLE:
    def dask_backtest_task(config: BacktestConfig) -> BacktestResult:
        return run_single_backtest(config)

    def run_dask_grid_search(param_space: dict[str, list], base_config: BacktestConfig, num_workers: int = 4) -> list[BacktestResult]:
        client = dask.distributed.Client(n_workers=num_workers, threads_per_worker=1, processes=True)
        try:
            param_grid = generate_param_grid(param_space)
            configs = [
                BacktestConfig(
                    symbols=base_config.symbols,
                    start_date=base_config.start_date,
                    end_date=base_config.end_date,
                    strategy_params={**base_config.strategy_params, **params},
                    initial_cash=base_config.initial_cash,
                    commission=base_config.commission,
                    slippage_bps=base_config.slippage_bps,
                )
                for params in param_grid
            ]
            futures = client.map(dask_backtest_task, configs)
            results = client.gather(futures)
            return results
        finally:
            client.close()


def analyze_results(results: list[BacktestResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {**r.config.strategy_params, **r.metrics}
        rows.append(row)
    return pd.DataFrame(rows)


def get_best_config(results: list[BacktestResult], metric: str = "sharpe_ratio") -> BacktestResult | None:
    if not results:
        return None
    return max(results, key=lambda r: r.metrics.get(metric, -np.inf))
