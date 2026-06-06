"""基于 Jaccard 关键词相似度的内存向量存储 — 零外部依赖."""

import re
from interfaces import IVectorStore, MemoryEntry, MemorySearchResult

STOP_WORDS = {
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
    "这里", "那里", "哪里", "这样", "那样", "怎样", "为什么",
}

_SEGMENT_RE = re.compile(
    r"[\s"
    r"，,。.！!？?：:；;"
    r"、()（）"
    r"\[\]《》<>"
    r"/\\|@#$%^&*+=~`"
    r"“”‘’"  # ""''
    r"\-]+"
)


def _extract_keywords(text: str) -> list[str]:
    words = _SEGMENT_RE.split(text)
    keywords: list[str] = []
    for word in words:
        trimmed = word.strip().lower()
        if len(trimmed) < 2:
            continue
        if trimmed in STOP_WORDS:
            continue
        if trimmed.isdigit():
            continue
        keywords.append(trimmed)
    return list(dict.fromkeys(keywords))  # 去重保序


def _jaccard(a: list[str], b: list[str]) -> float:
    set_a, set_b = set(a), set(b)
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


class InMemoryVectorStore(IVectorStore):
    def __init__(self):
        self._entries: list[MemoryEntry] = []
        self._next_id = 1

    async def add(self, entry: MemoryEntry) -> str:
        if not entry.id:
            entry.id = f"mem_{self._next_id}"
            self._next_id += 1
        entry.keywords = _extract_keywords(entry.content)
        self._entries.append(entry)
        return entry.id

    async def search(self, query: str, top_k: int) -> list[MemorySearchResult]:
        query_kw = _extract_keywords(query)
        scored = [
            MemorySearchResult(entry=e, score=_jaccard(query_kw, e.keywords))
            for e in self._entries
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    async def delete(self, entry_id: str) -> None:
        self._entries = [e for e in self._entries if e.id != entry_id]

    def size(self) -> int:
        return len(self._entries)

    def get_all(self) -> list[MemoryEntry]:
        return list(self._entries)
