"""
3号交易员 — Tool 抽象基类 & 统一返回格式
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════
# 统一返回格式
# ═══════════════════════════════════════════

@dataclass
class ChartSpec:
    """图表规格，供上层渲染用"""
    chart_type: str          # line / bar / scatter / heatmap / table
    title: str
    data: Any                # 结构化数据，渲染层自行解析
    x_label: str = ""
    y_label: str = ""
    description: str = ""    # 图表说明（用于报告正文引用）
    source: str = ""         # 数据来源标注

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trader3Response:
    """
    所有 Tool 统一返回格式。
    2号分析师可直接引用 summary / key_metrics / charts / caveats / metadata。
    """
    success: bool
    data: Any = None                        # 核心数据
    summary: str = ""                       # 一句话结论（给正文引用）
    key_metrics: Dict[str, float] = field(default_factory=dict)  # 关键指标（给表格引用）
    charts: List[ChartSpec] = field(default_factory=list)         # 图表规格（给可视化）
    caveats: List[str] = field(default_factory=list)              # 局限性/假设（给脚注）
    metadata: Dict[str, Any] = field(default_factory=dict)        # 运行元信息

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "summary": self.summary,
            "key_metrics": self.key_metrics,
            "charts": [c.to_dict() for c in self.charts],
            "caveats": self.caveats,
            "metadata": self.metadata,
        }

    @staticmethod
    def error(message: str, **kwargs) -> "Trader3Response":
        """构造失败响应"""
        return Trader3Response(success=False, summary=message, **kwargs)


# ═══════════════════════════════════════════
# Tool 抽象基类
# ═══════════════════════════════════════════

class BaseTool(ABC):
    """所有 3号交易员 Tool 的基类"""

    # Tool 元信息（子类覆写）
    tool_name: str = ""
    tool_description: str = ""
    tool_version: str = "0.1.0"
    tool_category: str = ""  # backtest / optimize / execution / signal / valuation

    def __init__(self):
        self._call_history: List[dict] = []

    @abstractmethod
    def execute(self, **kwargs) -> Trader3Response:
        """执行 Tool 逻辑"""
        ...

    def __call__(self, **kwargs) -> Trader3Response:
        """包装 execute：计时 + 记录 + 填充 metadata"""
        start = time.time()
        request_id = str(uuid.uuid4())[:8]

        try:
            result = self.execute(**kwargs)
        except Exception as e:
            result = Trader3Response(
                success=False,
                summary=f"[{self.tool_name}] 执行异常: {e}",
                caveats=[str(e)],
            )

        elapsed = time.time() - start
        result.metadata.update({
            "tool": self.tool_name,
            "version": self.tool_version,
            "request_id": request_id,
            "elapsed_seconds": max(round(elapsed, 3), 0.001),  # 最低 0.001s 避免浮点精度为 0
            "timestamp": datetime.now().isoformat(),
        })

        self._call_history.append({
            "request_id": request_id,
            "kwargs": {k: v for k, v in kwargs.items() if not k.startswith("_")},
            "success": result.success,
            "elapsed": elapsed,
        })

        return result

    def get_call_history(self, last_n: int = 10) -> List[dict]:
        return self._call_history[-last_n:]