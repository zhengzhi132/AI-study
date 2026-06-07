# -*- coding: utf-8 -*-
"""Day 7 ReAct Agent — 整合 Day 5 记忆系统 + Day 6 RAG 检索。"""

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

from openai import OpenAI

_PROJECT = Path(__file__).resolve().parent.parent
_DAY5 = _PROJECT / "day5-memory-system"
_DAY6 = _PROJECT / "day6-rag-basics"
_DAY7 = Path(__file__).resolve().parent

# Day 5 gets permanent path priority: "from interfaces import" resolves to Day 5
sys.path.insert(0, str(_DAY5))


def _load_from(name: str, path: Path, source_dir: Path):
    sys.path.insert(0, str(source_dir))
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    finally:
        sys.path.pop(0)


# Load Day 6 modules FIRST (before Day 5's interfaces.py pollutes path)
_day6_embed = _load_from("_day6_embed", _DAY6 / "embedding.py", _DAY6)
_day6_chroma = _load_from("_day6_chroma_store", _DAY6 / "chroma_store.py", _DAY6)
_day6_chunker = _load_from("_day6_chunker", _DAY6 / "chunker.py", _DAY6)
_day6_doc_idx = _load_from("_day6_doc_indexer", _DAY6 / "document_indexer.py", _DAY6)

# Day 5 modules: _DAY5 is on sys.path so "from interfaces import" works
_day5_tool = _load_from("_day5_tool", _DAY5 / "tool.py", _DAY5)
_day5_mem = _load_from("_day5_memory_manager", _DAY5 / "memory_manager.py", _DAY5)
_day5_store = _load_from("_day5_inmem_store", _DAY5 / "in_memory_vector_store.py", _DAY5)
_day5_weather = _load_from("_day5_weather", _DAY5 / "tools" / "weather_tool.py", _DAY5)
_day5_search = _load_from("_day5_search", _DAY5 / "tools" / "search_tool.py", _DAY5)
_day5_save = _load_from("_day5_save_fact", _DAY5 / "tools" / "save_fact_tool.py", _DAY5)
_day5_recall = _load_from("_day5_recall_fact", _DAY5 / "tools" / "recall_fact_tool.py", _DAY5)
_day5_scratch = _load_from("_day5_scratchpad", _DAY5 / "tools" / "scratchpad_tool.py", _DAY5)

# Day 6 exports
EmbeddingService = _day6_embed.EmbeddingService
EmbeddingConfig = _day6_embed.EmbeddingConfig
ChromaDBStore = _day6_chroma.ChromaDBStore
index_documents_day6 = _day6_doc_idx.index_documents

# Day 5 exports
ToolRegistry = _day5_tool.ToolRegistry
_tool_base = _day5_tool.Tool
MemoryManager = _day5_mem.MemoryManager
InMemoryVectorStore = _day5_store.InMemoryVectorStore
WeatherTool = _day5_weather.WeatherTool
SearchTool = _day5_search.SearchTool
SaveFactTool = _day5_save.SaveFactTool
RecallFactTool = _day5_recall.RecallFactTool
ScratchpadTool = _day5_scratch.ScratchpadTool

# Day 7
_day7_kb = _load_from("_day7_search_kb", _DAY7 / "tools" / "search_kb_tool.py", _DAY7)
create_search_kb_tool = _day7_kb.create_search_kb_tool

# ── System Prompt ───────────────────────────────────────────

SYSTEM_PROMPT = """你是一个个人知识库 AI 助手。规则：

1. 知识检索优先: 用户问技术问题或概念时，先用 search_kb 检索知识库
2. 网络搜索兜底: search_kb 返回空时，才用 search 搜索网络
3. 记住用户偏好: 了解到用户的重要信息时用 save_fact 保存
4. 个性化回答: 回答前用 recall_fact 查用户偏好
5. 多步任务: 用 scratchpad 暂存中间结果
6. 拿到所有工具数据后才给出最终答案"""

# ── Agent ────────────────────────────────────────────────────


async def run_turn(
    client: OpenAI,
    model: str,
    memory_manager: MemoryManager,
    registry: ToolRegistry,
    user_input: str,
    label: str = "",
) -> str:
    if label:
        print(f"\n{'=' * 60}")
        print(f"{label}")
        print(f"{'=' * 60}")

    print(f"[User] {user_input}")

    await memory_manager.initialize(SYSTEM_PROMPT)
    await memory_manager.pre_process(user_input)

    while True:
        messages = memory_manager.short_term.get_all()

        resp = client.chat.completions.create(
            model=model, messages=messages, tools=registry.get_all_schemas(),
        )

        choice = resp.choices[0]
        memory_manager.post_process(choice.message.model_dump())

        print(f"\n[Thought] finish_reason={choice.finish_reason}")

        if choice.finish_reason == "stop":
            answer = choice.message.content or "(no answer)"
            print(f"[Answer] {answer}")
            await memory_manager.finalize()
            return answer

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                name = tc.function.name
                args_str = tc.function.arguments
                print(f"[Action] {name}({args_str})")
                try:
                    result = await registry.execute(name, json.loads(args_str))
                except Exception as e:
                    result = f"error: {e}"
                preview = result[:200] + "..." if len(result) > 200 else result
                print(f"[Observation] {preview}")
                memory_manager.short_term.add({
                    "role": "tool", "tool_call_id": tc.id, "content": result,
                })


def build_registry(
    memory_manager: MemoryManager, kb_store: ChromaDBStore
) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(WeatherTool())
    reg.register(SearchTool())
    reg.register(create_search_kb_tool(_tool_base, kb_store))
    reg.register(SaveFactTool(memory_manager))
    reg.register(RecallFactTool(memory_manager))
    reg.register(ScratchpadTool(memory_manager))
    return reg
