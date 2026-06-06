"""文档分块 — 支持固定大小 / 句子感知 / 递归分块三种策略。"""

import re
from pathlib import Path
from interfaces import DocumentChunk


def load_markdown(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8")


def load_documents(doc_dir: str, glob_pattern: str = "*.md") -> list[dict]:
    docs = []
    for fp in Path(doc_dir).glob(glob_pattern):
        docs.append({"filename": fp.name, "text": fp.read_text(encoding="utf-8")})
    return docs


# ========== 策略 1: 固定大小分块 ==========

def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# ========== 策略 2: 句子感知分块 ==========

_SENTENCE_RE = re.compile(r"(?<=[。！？.!?])\s*")


def sentence_chunk(text: str, max_chars: int = 1000) -> list[str]:
    sentences = _SENTENCE_RE.split(text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > max_chars and current:
            chunks.append(current.strip())
            current = sent
        else:
            current += sent
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ========== 策略 3: 递归分块 ==========

_SEPARATORS = ["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", " "]


def recursive_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    return _recursive_split(text, _SEPARATORS, chunk_size, overlap)


def _recursive_split(
    text: str, separators: list[str], chunk_size: int, overlap: int
) -> list[str]:
    if not separators or len(text) <= chunk_size:
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        return fixed_size_chunk(text, chunk_size, overlap)

    sep = separators[0]
    remaining = separators[1:]
    parts = text.split(sep)
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = current + (sep if current else "") + part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current.strip())
            if len(part) > chunk_size:
                sub = _recursive_split(part, remaining, chunk_size, overlap)
                chunks.extend(sub)
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current.strip())
    return chunks


# ========== 统一入口 ==========

def chunk_documents(
    docs: list[dict],
    strategy: str = "recursive",
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[DocumentChunk]:
    chunk_func = {
        "fixed": fixed_size_chunk,
        "sentence": lambda t: sentence_chunk(t, chunk_size),
        "recursive": recursive_chunk,
    }[strategy]

    all_chunks: list[DocumentChunk] = []
    for doc in docs:
        texts = (
            chunk_func(doc["text"], chunk_size, overlap)
            if strategy != "sentence"
            else chunk_func(doc["text"])
        )
        for idx, text in enumerate(texts):
            if text.strip():
                all_chunks.append(DocumentChunk(
                    content=text.strip(),
                    metadata={"source": doc["filename"], "chunk_idx": idx},
                    chunk_index=len(all_chunks),
                ))
    return all_chunks
