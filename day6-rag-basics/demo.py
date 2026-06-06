# -*- coding: utf-8 -*-
"""Day 6 RAG demo -- document index + Jaccard vs Embedding + RAG QA."""

import asyncio
import os
import re
import sys
import time as _time
from pathlib import Path

# Fix Windows GBK console: force UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

from openai import OpenAI

from chunker import load_documents, chunk_documents
from embedding import EmbeddingService, EmbeddingConfig
from chroma_store import ChromaDBStore
from document_indexer import index_documents
from rag_agent import rag_query


# -- Jaccard keyword search (Day 5 style, inlined) --

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "because", "but", "and", "or", "if", "while",
    "about", "up", "out", "now", "also", "me", "i", "you", "he", "she",
    "it", "we", "they", "my", "your", "his", "her", "its", "our", "their",
    "mine", "yours", "hers", "ours", "theirs", "this", "that", "these", "those",
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "什么", "怎么", "如何", "哪", "吗", "啊", "吧", "呢", "哦", "嗯",
    "可以", "需要", "应该", "已经", "还是", "或者", "因为", "所以",
    "但是", "而且", "然后", "虽然", "如果", "这个", "那个", "哪个",
}

_SEG = re.compile(
    r"[\s" r"，,。.！!？?：:；;"
    r"、()（）" r"\[\]《》<>" r"/\\|@#$%^&*+=~`"
    r"\"\"''" r"\-]+"
)


def _kw(text: str) -> list[str]:
    words = _SEG.split(text)
    return list(dict.fromkeys(
        w.strip().lower() for w in words
        if len(w.strip()) >= 2 and w.strip().lower() not in _STOP and not w.strip().isdigit()
    ))


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    u = sa | sb
    return len(sa & sb) / len(u) if u else 0.0


class _JaccardStore:
    def __init__(self):
        self.entries: list[dict] = []

    def add(self, text: str):
        self.entries.append({"text": text, "keywords": _kw(text)})

    def search(self, query: str, top_k: int) -> list[dict]:
        qk = _kw(query)
        scored = [{"text": e["text"], "score": _jaccard(qk, e["keywords"])} for e in self.entries]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


DOC_DIR = str(Path(__file__).resolve().parent / "documents")
CHROMA_DIR = str(Path(__file__).resolve().parent / "chroma_data")

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY", "sk-xx"),
)
MODEL = "deepseek-v4-flash"


async def main():
    print("\nDay 6 -- RAG Basics Demo\n")

    # ============================================================
    # Phase 1: Offline Indexing
    # ============================================================
    print("=" * 60)
    print("[Phase 1] Document Indexing")
    print("=" * 60)

    docs = load_documents(DOC_DIR)
    print(f"\nFound {len(docs)} documents:")
    for d in docs:
        print(f"  - {d['filename']} ({len(d['text'])} chars)")

    chunks = chunk_documents(docs, strategy="recursive", chunk_size=400, overlap=80)
    print(f"\nRecursive chunking -> {len(chunks)} chunks (chunk_size=400, overlap=80):")
    for c in chunks[:5]:
        src = c.metadata["source"]
        preview = c.content[:60].replace("\n", " ")
        print(f"  [{c.chunk_index}] {src}: {preview}...")

    embed_service = EmbeddingService(client, EmbeddingConfig())
    store = ChromaDBStore(embed_service, collection_name="day6_demo", persist_dir=CHROMA_DIR)

    result = await index_documents(
        embed_service, store, DOC_DIR,
        chunk_size=400, chunk_overlap=80, strategy="recursive",
    )
    print(f"\nIndex complete: {result['file_count']} files -> {result['chunk_count']} chunks")

    # ============================================================
    # Phase 2: Jaccard vs ChromaDB Semantic Search
    # ============================================================
    print(f"\n{'=' * 60}")
    print("[Phase 2] Jaccard (Day 5) vs ChromaDB Embedding (Day 6)")
    print("=" * 60)

    jaccard_store = _JaccardStore()
    for c in chunks:
        jaccard_store.add(c.content)

    test_queries = [
        "Python 多线程性能为什么差",
        "async await 怎么用",
        "应该选 threading 还是 multiprocessing",
    ]

    for query in test_queries:
        print(f'\nQuery: "{query}"')

        j_results = jaccard_store.search(query, 3)
        print(f"  [Jaccard Day 5]:")
        for i, r in enumerate(j_results):
            preview = r["text"][:60].replace("\n", " ")
            print(f"    [{i+1}] score={r['score']:.3f} | {preview}...")

        c_results = store.search(query, 3)
        print(f"  [ChromaDB Day 6]:")
        for i, r in enumerate(c_results):
            preview = r["document"][:60].replace("\n", " ")
            print(f"    [{i+1}] dist={r.get('distance', 0):.3f} | {preview}...")

    # ============================================================
    # Phase 3: End-to-End RAG QA
    # ============================================================
    print(f"\n{'=' * 60}")
    print("[Phase 3] RAG QA (retrieve + augment + generate)")
    print("=" * 60)

    questions = [
        "GIL 是什么？它为什么会影响 Python 多线程性能？",
        "asyncio 和 threading 应该怎么选择？",
    ]

    for q in questions:
        await rag_query(client, MODEL, store, q, top_k=3, verbose=True)
        print(f"\n{'-' * 40}")

    # ============================================================
    # Final Stats
    # ============================================================
    print(f"\n{'=' * 60}")
    print("[Stats] Day 6 Final State")
    print(f"{'=' * 60}")
    print(f"  ChromaDB chunks: {store.count()}")
    print(f"  Jaccard entries: {len(jaccard_store.entries)}")
    print(f"  ChromaDB path: {CHROMA_DIR}")
    print(f"\nDay 6 RAG demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
