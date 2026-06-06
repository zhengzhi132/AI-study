"""搜索工具 — 模拟搜索结果."""

from tool import Tool

MOCK_SEARCH = {
    "北京户外": "故宫、长城、颐和园、香山公园、奥林匹克森林公园",
    "北京景点": "故宫、天坛、颐和园、圆明园、798艺术区",
    "户外徒步": "香山、百望山、凤凰岭、京西古道、妙峰山",
    "骑行": "长安街沿线、奥体公园环线、十三陵水库、雁栖湖",
    "周末活动": "故宫特展、798艺术展、国家大剧院演出、三里屯市集",
}


class SearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="search",
            description="搜索网页，获取最新信息、新闻或百科知识",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"],
            },
        )

    async def execute(self, args: dict) -> str:
        query = args.get("query", "")
        for key, result in MOCK_SEARCH.items():
            if key in query or query in key:
                return f'关于"{query}"的搜索结果: {result}'
        return f'关于"{query}"的搜索结果: 相关户外活动、景点和文化场所'
