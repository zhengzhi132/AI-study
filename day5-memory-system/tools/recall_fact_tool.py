"""召回事实工具 — Agent 搜索长期记忆."""

from tool import Tool


class RecallFactTool(Tool):
    def __init__(self, memory_manager):
        super().__init__(
            name="recall_fact",
            description="搜索长期记忆，查找与查询相关的用户事实、偏好和历史信息。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询，用于在长期记忆中查找相关信息",
                    },
                },
                "required": ["query"],
            },
        )
        self._memory_manager = memory_manager

    async def execute(self, args: dict) -> str:
        return await self._memory_manager.recall_fact(args["query"])
