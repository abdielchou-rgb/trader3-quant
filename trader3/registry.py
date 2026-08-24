"""
3号交易员 — Tool 注册表
"""

from __future__ import annotations

import builtins

from trader3.base_tool import BaseTool, Trader3Response


class ToolRegistry:
    """Tool 注册表（管理所有注册的 Tool）"""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._categories: dict[str, list[str]] = {}

    def register(self, tool: BaseTool) -> None:
        """注册一个 Tool 实例"""
        name = tool.tool_name
        if not name:
            raise ValueError("Tool must have a non-empty tool_name")
        self._tools[name] = tool
        cat = tool.tool_category or "uncategorized"
        if cat not in self._categories:
            self._categories[cat] = []
        self._categories[cat].append(name)

    def get(self, name: str) -> BaseTool | None:
        """按名称获取 Tool"""
        return self._tools.get(name)

    def list(self, category: str | None = None) -> builtins.list[str]:
        """列出 Tool（可按类别筛选）"""
        if category:
            return self._categories.get(category, [])
        return list(self._tools.keys())

    def call(self, name: str, **kwargs) -> Trader3Response:
        """调用一个 Tool"""
        tool = self.get(name)
        if tool is None:
            return Trader3Response.error(f"Tool '{name}' not found")
        return tool(**kwargs)

    def categories(self) -> dict[str, builtins.list[str]]:
        return dict(self._categories)

    def summary(self) -> builtins.list[dict]:
        """所有 Tool 摘要（给 2号分析师看）"""
        result = []
        for name, tool in sorted(self._tools.items()):
            result.append({
                "name": name,
                "description": tool.tool_description,
                "version": tool.tool_version,
                "category": tool.tool_category,
            })
        return result
