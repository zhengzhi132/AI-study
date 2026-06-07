# -*- coding: utf-8 -*-
"""Day 7 Demo — 个人知识库 Agent 端到端演示。"""

import asyncio
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

from openai import OpenAI

from agent import (
    run_turn,
    build_registry,
    MemoryManager,
    InMemoryVectorStore,
    EmbeddingService,
    EmbeddingConfig,
    ChromaDBStore,
    index_documents_day6 as index_documents,
)

# ── Config ──────────────────────────────────────────────────

_DAY7 = Path(__file__).resolve().parent
DOC_DIR = str(_DAY7 / "documents")
CHROMA_DIR = str(_DAY7 / "chroma_data")

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY", "sk-xx"),
)
MODEL = "deepseek-v4-flash"


async def main():
    print("\nDay 7 -- Personal Knowledge Base Agent Demo\n")

    # ============================================================
    # Phase 1: Index documents
    # ============================================================
    print("=" * 60)
    print("[Phase 1] Document Indexing")
    print("=" * 60)

    embed_svc = EmbeddingService(client, EmbeddingConfig())
    kb_store = ChromaDBStore(
        embed_svc, collection_name="day7_demo", persist_dir=CHROMA_DIR
    )

    result = await index_documents(
        embed_svc, kb_store, DOC_DIR, chunk_size=400, chunk_overlap=80,
    )
    print(
        f"Indexed: {result['file_count']} files -> {result['chunk_count']} chunks"
    )
    for s in result["sources"]:
        print(f"  - {s}")

    # ============================================================
    # Phase 2: Multi-turn conversation
    # ============================================================
    print(f"\n{'=' * 60}")
    print("[Phase 2] Multi-turn Conversation")
    print("=" * 60)

    vec_store = InMemoryVectorStore()
    mem_mgr = MemoryManager(client, MODEL, vector_store=vec_store)
    registry = build_registry(mem_mgr, kb_store)

    # Turn 1: Self-introduction -> save_fact
    await run_turn(
        client, MODEL, mem_mgr, registry,
        "你好！我是小王，一名 Python 后端工程师，最近在学 asyncio。我喜欢简洁的技术解释。",
        "Turn 1: Build user profile",
    )

    # Turn 2: KB question -> search_kb
    await run_turn(
        client, MODEL, mem_mgr, registry,
        "GIL 是什么？它为什么导致 Python 多线程性能差？",
        "Turn 2: KB retrieval (GIL)",
    )

    # Turn 3: Personalized recommendation -> recall_fact + search_kb
    await run_turn(
        client, MODEL, mem_mgr, registry,
        "根据我的背景，推荐一个适合我深入学习的 Python 并发主题",
        "Turn 3: Personalized recommendation",
    )

    # ============================================================
    # Final Stats
    # ============================================================
    print(f"\n{'=' * 60}")
    print("[Stats]")
    print("=" * 60)
    s = mem_mgr.stats()
    print(f"  Short-term messages: {s['short_term']['message_count']}")
    print(f"  Long-term facts: {s['long_term']['total_facts']}")
    print(f"  KB chunks indexed: {kb_store.count()}")
    print(f"  Working memory entries: {s['working']['entry_count']}")
    print(f"\nDay 7 demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
