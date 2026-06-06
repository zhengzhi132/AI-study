"""Embedding 服务 — OpenAI text-embedding-3 封装。"""

from openai import OpenAI
from interfaces import EmbeddingConfig


class EmbeddingService:
    def __init__(self, client: OpenAI, config: EmbeddingConfig | None = None):
        self._client = client
        self._config = config or EmbeddingConfig()

    async def embed_single(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self._config.model,
            input=text,
            dimensions=self._config.dimensions,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self._config.batch_size):
            batch = texts[i : i + self._config.batch_size]
            response = self._client.embeddings.create(
                model=self._config.model,
                input=batch,
                dimensions=self._config.dimensions,
            )
            all_vectors.extend([d.embedding for d in response.data])
        return all_vectors

    async def embed(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(texts, str):
            return await self.embed_single(texts)
        return await self.embed_batch(texts)
