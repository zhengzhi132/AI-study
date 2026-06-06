"""长期记忆 — 持久化用户事实/偏好，支持跨会话检索."""

import json
import time
from openai import OpenAI
from interfaces import IVectorStore, MemoryEntry, MemorySearchResult


class LongTermMemory:
    def __init__(self, store: IVectorStore):
        self._store = store
        self._conversation_id = f"conv_{int(time.time())}"

    async def add_fact(self, content: str, source: str, importance: float = 0.5) -> str:
        # 简单去重
        existing = await self._store.search(content, 1)
        if existing and existing[0].score > 0.8:
            return existing[0].entry.id

        entry = MemoryEntry(
            id="",
            content=content,
            source=source,
            conversation_id=self._conversation_id,
            timestamp=time.time(),
            importance=importance,
        )
        return await self._store.add(entry)

    async def search_relevant(self, query: str, top_k: int = 3) -> list[MemorySearchResult]:
        return await self._store.search(query, top_k)

    async def get_all_facts(self) -> list[MemoryEntry]:
        return self._store.get_all()

    async def extract_facts_from_conversation(
        self, client: OpenAI, model: str, messages: list[dict]
    ) -> list[str]:
        text = "\n".join(
            f"[{m['role']}]: {m.get('content', '')}"
            for m in messages
            if m.get("role") in ("user", "assistant")
        )
        if not text.strip():
            return []

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个知识提取器。从以下对话中提取关于用户的重要事实、偏好和习惯。\n"
                            "只提取可跨会话保留的长期信息。以 JSON 数组格式返回，每个元素是一个事实字符串。\n"
                            "示例: [\"用户喜欢简洁的回答\", \"用户住在北京\", \"用户偏好TypeScript编程\"]\n"
                            "不要包含会话特定的临时信息。只返回 JSON 数组，不要其他文字。"
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=300,
            )
            raw = response.choices[0].message.content or "[]"
            json_str = raw.replace("```json", "").replace("```", "").strip()
            facts: list[str] = json.loads(json_str)
            for fact in facts:
                if fact and fact.strip():
                    await self.add_fact(fact.strip(), "extracted")
            return facts
        except Exception:
            return []

    def format_for_prompt(self, results: list[MemorySearchResult]) -> str:
        relevant = [r for r in results if r.score > 0]
        if not relevant:
            return ""
        lines = ["[关于用户的长期记忆]"]
        for r in relevant:
            lines.append(f"- {r.entry.content}")
        return "\n".join(lines)

    def set_conversation_id(self, cid: str) -> None:
        self._conversation_id = cid

    async def delete_fact(self, entry_id: str) -> None:
        await self._store.delete(entry_id)

    def stats(self) -> dict:
        return {"total_facts": self._store.size()}
