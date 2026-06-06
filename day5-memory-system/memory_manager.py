"""MemoryManager — 编排短期/长期/工作记忆的统一接口."""

from openai import OpenAI
from short_term_memory import ShortTermMemory
from long_term_memory import LongTermMemory
from working_memory import WorkingMemory
from interfaces import IVectorStore, SummarizationConfig


class MemoryManager:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        vector_store: IVectorStore | None = None,
        short_term_config: SummarizationConfig | None = None,
    ):
        self._client = client
        self._model = model
        self.short_term = ShortTermMemory(short_term_config)
        self.long_term = LongTermMemory(vector_store)  # type: ignore[arg-type]
        self.working = WorkingMemory()
        self._base_system_prompt = ""

    async def initialize(self, system_prompt: str) -> None:
        self._base_system_prompt = system_prompt
        self.short_term.clear()
        augmented = await self._build_augmented_system_prompt("")
        self.short_term.set_system_prompt(augmented)

    async def pre_process(self, user_input: str) -> list[dict]:
        self.short_term.add({"role": "user", "content": user_input})

        if self.short_term.is_over_threshold():
            await self.short_term.summarize(self._client, self._model)

        augmented = await self._build_augmented_system_prompt(user_input)
        self.short_term.set_system_prompt(augmented)

        return self.short_term.get_all()

    def post_process(self, response: dict) -> None:
        self.short_term.add(response)

    async def finalize(self) -> None:
        messages = self.short_term.get_all()
        await self.long_term.extract_facts_from_conversation(self._client, self._model, messages)
        self.working.end_task()

    async def _build_augmented_system_prompt(self, user_input: str) -> str:
        parts = [self._base_system_prompt]

        if user_input:
            relevant = await self.long_term.search_relevant(user_input, 3)
            mem_text = self.long_term.format_for_prompt(relevant)
            if mem_text:
                parts.append("\n" + mem_text)
        else:
            all_facts = await self.long_term.get_all_facts()
            if all_facts:
                from interfaces import MemorySearchResult
                pseudo_results = [MemorySearchResult(entry=f, score=0.5) for f in all_facts]
                mem_text = self.long_term.format_for_prompt(pseudo_results)
                if mem_text:
                    parts.append("\n" + mem_text)

        wm_text = self.working.format_for_prompt()
        if wm_text:
            parts.append("\n" + wm_text)

        return "\n".join(parts)

    # === Memory Tool Helpers ===

    async def save_fact(self, content: str) -> str:
        return await self.long_term.add_fact(content, "explicit")

    async def recall_fact(self, query: str) -> str:
        results = await self.long_term.search_relevant(query, 5)
        if not results:
            return "未找到相关的长期记忆。"
        lines = []
        for r in results:
            if r.score > 0:
                lines.append(f"[相关性 {r.score:.2f}] {r.entry.content}")
        return "\n".join(lines)

    async def update_scratchpad(self, key: str, value: str) -> str:
        self.working.set(key, value)
        return f"已保存: {key} = {value}"

    async def get_scratchpad(self, key: str) -> str:
        val = self.working.get(key)
        return val if val is not None else f'键 "{key}" 不存在'

    async def list_scratchpad(self) -> str:
        all_entries = self.working.get_all()
        if not all_entries:
            return "工作记忆为空。"
        return "\n".join(f"{e.key}: {e.value}" for e in all_entries)

    def stats(self) -> dict:
        return {
            "short_term": self.short_term.stats(),
            "long_term": self.long_term.stats(),
            "working": self.working.stats(),
        }
