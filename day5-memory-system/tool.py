"""Day 4 Tool 系统的 Python 移植 — Tool 抽象基类和 ToolRegistry."""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    def __init__(self, name: str, description: str, parameters: dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

    @abstractmethod
    async def execute(self, args: dict[str, Any]) -> str: ...

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get_all_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    async def execute(self, name: str, args: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        last_error = None
        for _ in range(3):
            try:
                return await tool.execute(args)
            except Exception as e:
                last_error = e
        raise last_error  # type: ignore[misc]
