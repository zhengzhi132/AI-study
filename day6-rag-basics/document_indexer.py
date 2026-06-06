"""文档索引器 — 离线批量分块 + 向量化入库。"""

from openai import OpenAI
from chunker import load_documents, chunk_documents
from embedding import EmbeddingService
from chroma_store import ChromaDBStore


async def index_documents(
    embed_service: EmbeddingService,
    store: ChromaDBStore,
    doc_dir: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    strategy: str = "recursive",
    clear_first: bool = True,
) -> dict:
    if clear_first:
        store.clear()

    docs = load_documents(doc_dir)
    if not docs:
        return {"file_count": 0, "chunk_count": 0, "sources": []}

    chunks = chunk_documents(docs, strategy=strategy, chunk_size=chunk_size, overlap=chunk_overlap)
    store.add_chunks(chunks)

    return {
        "file_count": len(docs),
        "chunk_count": len(chunks),
        "sources": [d["filename"] for d in docs],
    }
