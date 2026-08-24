"""
3号交易员 — CLI 入口
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from trader3 import Trader3


def main():
    # Windows GBK 控制台容错：无法编码的字符替换而非崩溃
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(
        description="3号交易员 — 独立量化交易引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  trader3 backtest --config config/strategies/3号交易员_core.yaml
  trader3 optimize --signals signals.json --constraints config/constraints.yaml
  trader3 execute --target target_weights.json --current current_pos.json
  trader3 validate --signal-name "动量因子" --values values.json
  trader3 regime
  trader3 daily                    # 运行日度任务管线
        """,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # backtest
    bt_parser = subparsers.add_parser("backtest", help="运行回测")
    bt_parser.add_argument("--config", "-c", type=str, default="", help="策略配置文件路径")
    bt_parser.add_argument("--start", type=str, default="2020-01-01", help="开始日期")
    bt_parser.add_argument("--end", type=str, default="2025-12-31", help="结束日期")
    bt_parser.add_argument("--benchmark", type=str, default="000300.SH", help="基准指数")

    # optimize
    op_parser = subparsers.add_parser("optimize", help="组合优化")
    op_parser.add_argument("--signals", type=str, required=True, help="信号文件 (JSON)")
    op_parser.add_argument("--constraints", type=str, default="", help="约束配置文件")
    op_parser.add_argument("--method", type=str, default="risk_budget", help="优化方法")

    # execute
    ex_parser = subparsers.add_parser("execute", help="生成执行计划")
    ex_parser.add_argument("--target", type=str, required=True, help="目标权重文件 (JSON)")
    ex_parser.add_argument("--current", type=str, default="", help="当前权重文件 (JSON)")
    ex_parser.add_argument("--algo", type=str, default="adaptive_vwap", help="执行算法")
    ex_parser.add_argument("--urgency", type=str, default="normal", help="紧急度")

    # validate
    vl_parser = subparsers.add_parser("validate", help="验证信号")
    vl_parser.add_argument("--signal-name", type=str, required=True, help="信号名称")
    vl_parser.add_argument("--values", type=str, help="信号值文件 (JSON)")
    vl_parser.add_argument("--returns", type=str, help="收益数据文件 (JSON)")

    # regime
    subparsers.add_parser("regime", help="诊断市场状态")

    # wfa
    wfa_parser = subparsers.add_parser("wfa", help="Walk-Forward Analysis")
    wfa_parser.add_argument("--config", type=str, default="", help="策略配置文件")
    wfa_parser.add_argument("--train-window", type=int, default=252, help="训练窗口")
    wfa_parser.add_argument("--test-window", type=int, default=63, help="测试窗口")

    # daily
    subparsers.add_parser("daily", help="运行日度任务管线")

    # list
    subparsers.add_parser("list", help="列出所有可用 Tool")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    t3 = Trader3()

    if args.command == "backtest":
        _run_backtest(t3, args)
    elif args.command == "optimize":
        _run_optimize(t3, args)
    elif args.command == "execute":
        _run_execute(t3, args)
    elif args.command == "validate":
        _run_validate(t3, args)
    elif args.command == "regime":
        _run_regime(t3, args)
    elif args.command == "wfa":
        _run_wfa(t3, args)
    elif args.command == "daily":
        _run_daily(t3, args)
    elif args.command == "list":
        _run_list(t3, args)


def _output(result, verbose: bool = False):
    """格式化输出"""
    print(f"\n{'='*60}")
    print(f"[{result.metadata.get('tool', '?')}] {result.summary}")
    print(f"{'='*60}")
    if verbose and result.key_metrics:
        print("\n关键指标:")
        for k, v in result.key_metrics.items():
            print(f"  {k}: {v}")
    if result.caveats:
        print("\n注意事项:")
        for c in result.caveats:
            print(f"  ⚠ {c}")
    if verbose and result.metadata:
        print(f"\n[元信息] version={result.metadata.get('version')} "
              f"elapsed={result.metadata.get('elapsed_seconds', 0):.2f}s "
              f"request_id={result.metadata.get('request_id')}")


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_backtest(t3, args):
    kwargs = {"start_date": args.start, "end_date": args.end, "benchmark": args.benchmark}
    if args.config:
        try:
            kwargs["strategy_config"] = _load_strategy_config(args.config)
        except Exception as e:
            print(f"❌ 策略配置解析失败: {args.config}\n  {e}")
            sys.exit(2)
    result = t3.run_backtest(**kwargs)
    _output(result, args.verbose)


def _load_strategy_config(path: str) -> StrategyConfig:  # noqa: F821 -- StrategyConfig 在函数内延迟导入，模块级无法解析
    """完整解析策略 YAML（metadata/universe/factors/constraints/risk_model/optimizer）"""
    from dataclasses import fields as dc_fields

    from trader3.config import load_yaml
    from trader3.models import FactorConfig, StrategyConfig

    config = load_yaml(path)
    if not isinstance(config, dict):
        raise ValueError("策略配置必须是 YAML 映射")
    meta = config.get("metadata") or {}

    factor_keys = {f.name for f in dc_fields(FactorConfig)}
    factors = []
    for raw in config.get("factors") or []:
        unknown = set(raw) - factor_keys
        if unknown:
            raise ValueError(f"因子配置含未知字段: {sorted(unknown)} (因子 {raw.get('name')})")
        factors.append(FactorConfig(**raw))

    return StrategyConfig(
        name=str(meta.get("name", "")),
        version=str(meta.get("version", "0.1.0")),
        factors=factors,
        universe=config.get("universe") or {},
        risk_model=config.get("risk_model") or {},
        optimizer=config.get("optimizer") or {},
        params={
            "constraints": config.get("constraints") or {},
            "author": meta.get("author", ""),
            "created": meta.get("created", ""),
        },
    )


def _run_optimize(t3, args):
    signals = _load_json(args.signals)
    result = t3.optimize_portfolio(signals=signals, method=args.method)
    _output(result, args.verbose)


def _run_execute(t3, args):
    target = _load_json(args.target)
    current = _load_json(args.current) if args.current else {}
    result = t3.generate_execution_plan(target_weights=target, current_weights=current,
                                         algorithm=args.algo, urgency=args.urgency)
    _output(result, args.verbose)


def _run_validate(t3, args):
    values = _load_json(args.values) if args.values else None
    returns = _load_json(args.returns) if args.returns else None
    result = t3.validate_signal(signal_name=args.signal_name, signal_values=values,
                                 forward_returns=returns)
    _output(result, args.verbose)


def _run_regime(t3, args):
    result = t3.diagnose_market_regime()
    _output(result, True)


def _run_wfa(t3, args):
    result = t3.walk_forward_analysis(train_window=args.train_window,
                                       test_window=args.test_window)
    _output(result, args.verbose)


def _run_daily(t3, args):
    """日度任务管线"""
    print("\n📋 运行日度任务管线...")
    print("  [1/4] 市场状态诊断...")
    result = t3.diagnose_market_regime()
    print(f"  状态: {result.summary}")
    print("  [2/4] v2 自选股触发扫描: python -m trader3.v2.cli_v2 scan")
    print("  [3/4] 数据同步: python -m trader3.v2.sync_all")
    print("  [4/4] 因子进化(可选): python evolve/run_evolution.py --help")
    print("✅ 日度管线核心步骤完成\n")


def _run_list(t3, args):
    tools = t3.list_tools()
    print("\n📦 3号交易员 — 可用 Tool 清单\n")
    print(f"{'名称':<30} {'类别':<15} {'版本':<10} {'描述'}")
    print("-" * 100)
    for t in tools:
        print(f"{t['name']:<30} {t['category']:<15} {t['version']:<10} {t['description']}")
    print()


if __name__ == "__main__":
    main()
