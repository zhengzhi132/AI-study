"""保存事实工具 — Agent 显式保存信息到长期记忆."""

from tool import Tool


class SaveFactTool(Tool):
    def __init__(self, memory_manager):
        super().__init__(
            name="save_fact",
            description="将一个关于用户的重要事实或偏好保存到长期记忆中，供未来对话使用。",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "要保存的事实内容，例如：用户喜欢简洁的回答、用户住在北京",
                    },
                },
                "required": ["content"],
            },
        )
        self._memory_manager = memory_manager

    async def execute(self, args: dict) -> str:
        content = args["content"]
        mid = await self._memory_manager.save_fact(content)
        return f"已保存到长期记忆 (id: {mid}): {content}"
