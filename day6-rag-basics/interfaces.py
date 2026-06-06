from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    content: str
    metadata: dict
    chunk_index: int


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float


@dataclass
class ChunkConfig:
    chunk_size: int = 500
    chunk_overlap: int = 100


@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 20


@dataclass
class RagConfig:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    top_k: int = 5
    similarity_threshold: float = 0.0
