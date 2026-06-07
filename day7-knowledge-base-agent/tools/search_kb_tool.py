# -*- coding: utf-8 -*-
"""SearchKBTool — 封装 ChromaDBStore 为 Day 4 Tool 接口。

由 agent.py 在加载 Day 5/6 模块后传入 Tool 基类和 ChromaDBStore 实例。
此文件不自行导入跨目录模块，避免 importlib 路径冲突。
"""


def create_search_kb_tool(Tool, store):
    """工厂函数：延迟绑定 Tool 基类和 ChromaDBStore 实例。"""

    class SearchKBTool(Tool):
        def __init__(self, store):
            self._store = store
            super().__init__(
                name="search_kb",
                description=(
                    "搜索知识库中的文档，获取相关信息。"
                    "当用户问技术问题、概念解释或任何需要参考文档的问题时，先用此工具检索。"
                    "返回最相关的文档片段及其来源。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询，用自然语言描述要查找的内容",
                        },
                    },
                    "required": ["query"],
                },
            )

        async def execute(self, args: dict) -> str:
            query = str(args.get("query", ""))
            if not query:
                return "错误: query 参数不能为空"
            results = self._store.search(query, top_k=3)
            if not results:
                return "知识库中未找到相关内容。建议使用 search 工具搜索网络。"
            lines = []
            for i, r in enumerate(results):
                src = r["metadata"].get("source", "unknown")
                dist = r.get("distance", 0)
                lines.append(
                    f"[来源 {i + 1}: {src} (dist={dist:.3f})]\n{r['document']}"
                )
            return "\n\n---\n\n".join(lines)

    return SearchKBTool(store)
