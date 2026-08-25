"""
CLI 冒烟测试（离线、零风险命令，subprocess 逐条执行）：

1. python -m trader3.cli list → 输出含 run_backtest
2. help 类零风险命令（regime --help / --verbose backtest --help / cli_v2 --help）
3. validate --values NUL（不存在文件）→ exit 非 0 且 stderr 无 Traceback

已知缺陷（本任务禁改 trader3/cli.py，暂以 xfail 记录）：
- cli._load_json 未捕获异常，Windows 下 open("NUL") 打开空设备成功后
  json.load 抛 JSONDecodeError → 当前带 Traceback 退出。
- 修复方式：trader3/cli.py:125 _run_validate/_load_json 捕获 (OSError, ValueError)
  后 print 错误并 sys.exit(2)。修复后移除对应 xfail 标记即自动转绿。

环境注记：解释器用 sys.executable（与 pytest 同一 Python 3.11，含项目依赖；
PATH 上的 `python` 可能指向无依赖的其他 venv）。子进程 env 设
PYTHONIOENCODING=utf-8 以规避 Windows GBK 控制台编码问题。全部离线，
不实跑 regime/backtest。
"""
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TIMEOUT = 120


def _run(module_args: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", *module_args],
        cwd=str(_PROJECT_ROOT),
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=_TIMEOUT,
        check=False,
    )


def test_cli_list_lists_tools():
    p = _run(["trader3.cli", "list"])
    assert p.returncode == 0, p.stderr
    assert "run_backtest" in p.stdout


def test_cli_regime_help_zero_risk():
    p = _run(["trader3.cli", "regime", "--help"])
    assert p.returncode == 0, p.stderr
    assert "usage" in p.stdout.lower()


def test_cli_verbose_backtest_help():
    p = _run(["trader3.cli", "--verbose", "backtest", "--help"])
    assert p.returncode == 0, p.stderr
    assert "--config" in p.stdout


def test_cli_v2_help():
    p = _run(["trader3.v2.cli_v2", "--help"])
    assert p.returncode == 0, p.stderr
    assert "watchlist" in p.stdout


def test_cli_validate_missing_values_file_exit_nonzero():
    p = _run(["trader3.cli", "--verbose", "validate",
              "--signal-name", "smoke", "--values", "NUL"])
    assert p.returncode != 0, p.stdout


def test_cli_validate_missing_values_file_stderr_no_traceback():
    p = _run(["trader3.cli", "--verbose", "validate",
              "--signal-name", "smoke", "--values", "NUL"])
    assert "Traceback" not in (p.stderr or "")
