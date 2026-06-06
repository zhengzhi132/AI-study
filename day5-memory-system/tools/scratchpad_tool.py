"""暂存器工具 — Agent 读写工作记忆."""

from tool import Tool


class ScratchpadTool(Tool):
    def __init__(self, memory_manager):
        super().__init__(
            name="scratchpad",
            description="读写工作记忆暂存器。用于在多步骤任务中保存中间结果、记录当前进度。",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "操作类型: write、read、list",
                        "enum": ["write", "read", "list"],
                    },
                    "key": {
                        "type": "string",
                        "description": "键名（write 和 read 操作需要）",
                    },
                    "value": {
                        "type": "string",
                        "description": "要保存的值（仅 write 操作需要）",
                    },
                },
                "required": ["action"],
            },
        )
        self._memory_manager = memory_manager

    async def execute(self, args: dict) -> str:
        action = args["action"]
        if action == "write":
            return await self._memory_manager.update_scratchpad(
                args.get("key", ""), args.get("value", "")
            )
        elif action == "read":
            return await self._memory_manager.get_scratchpad(args.get("key", ""))
        elif action == "list":
            return await self._memory_manager.list_scratchpad()
        return f"未知操作: {action}，支持 write / read / list"
