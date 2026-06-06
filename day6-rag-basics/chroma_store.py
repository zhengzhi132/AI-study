"""ChromaDBStore — 用 ChromaDB + OpenAI Embedding 实现语义向量检索。

这是 Day 5 → Day 6 的核心迁移：同一套存储 + 检索模式，
底层从 Jaccard 关键词匹配升级为 ChromaDB 语义向量检索。
"""

import chromadb
from embedding import EmbeddingService


class ChromaDBStore:
    """基于 ChromaDB 的向量存储。

    刻意不继承 Day 5 的 IVectorStore（ChromaDB 是同步 API，IVectorStore 是 async）。
    生产环境用 asyncio.to_thread 包装即可适配。
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        collection_name: str = "rag_docs",
        persist_dir: str = "./chroma_data",
    ):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedding = embedding_service
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list) -> list[str]:
        """批量添加文档 chunks 及其元数据。"""
        if not chunks:
            return []
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self._collection.add(
            ids=ids,
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        return ids

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """语义检索 — 自动将 query 转为向量再搜索。

        Returns:
            [{"id": ..., "document": ..., "metadata": ..., "distance": ...}, ...]
        """
        results = self._collection.query(query_texts=[query], n_results=top_k)
        return [
            {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

    def delete_by_source(self, source: str) -> None:
        """按文件名删除旧 chunks（更新文档时使用）。"""
        self._collection.delete(where={"source": source})

    def clear(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._collection.count()
