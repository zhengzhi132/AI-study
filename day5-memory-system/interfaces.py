"""共享数据类和接口定义."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ========== Summarization Config ==========

@dataclass
class SummarizationConfig:
    max_tokens: int = 4000
    keep_last_n: int = 6
    system_prompt_tokens: int = 500


# ========== Long-Term Memory ==========

@dataclass
class MemoryEntry:
    id: str
    content: str
    source: str  # "extracted" or "explicit"
    conversation_id: str
    timestamp: float
    importance: float = 0.5
    keywords: list[str] = field(default_factory=list)


@dataclass
class MemorySearchResult:
    entry: MemoryEntry
    score: float


class IVectorStore(ABC):
    @abstractmethod
    async def add(self, entry: MemoryEntry) -> str: ...

    @abstractmethod
    async def search(self, query: str, top_k: int) -> list[MemorySearchResult]: ...

    @abstractmethod
    async def delete(self, entry_id: str) -> None: ...

    @abstractmethod
    def size(self) -> int: ...

    @abstractmethod
    def get_all(self) -> list[MemoryEntry]: ...


# ========== Working Memory ==========

@dataclass
class WorkingMemoryEntry:
    key: str
    value: str
    timestamp: float
