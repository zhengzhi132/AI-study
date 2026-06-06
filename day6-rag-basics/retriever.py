"""检索策略 — 余弦相似度 + MMR 去重 + 后处理。"""

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def mmr_rerank(
    query_vec: list[float],
    doc_vecs: list[list[float]],
    doc_texts: list[str],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[str]:
    """最大边际相关性 — 选相关且多样的结果。

    lambda_param: 1 = 纯相关性，0 = 纯多样性
    """
    selected: list[int] = []
    remaining = list(range(len(doc_vecs)))

    for _ in range(min(top_k, len(doc_vecs))):
        best_score = float("-inf")
        best_idx = -1
        for i in remaining:
            relevance = cosine_similarity(query_vec, doc_vecs[i])
            diversity = max(
                (cosine_similarity(doc_vecs[i], doc_vecs[j]) for j in selected),
                default=0.0,
            )
            mmr = lambda_param * relevance - (1 - lambda_param) * diversity
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [doc_texts[i] for i in selected]


def deduplicate_by_ngram(chunks: list[dict], threshold: float = 0.9) -> list[dict]:
    """用 3-gram Jaccard 去重，高度重叠的 chunk 只保留第一个。"""

    def _ngrams(s: str, n: int = 3) -> set[str]:
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    kept: list[dict] = []
    for chunk in chunks:
        is_dup = False
        ngrams_a = _ngrams(chunk["document"])
        for existing in kept:
            ngrams_b = _ngrams(existing["document"])
            union = ngrams_a | ngrams_b
            if not union:
                continue
            if len(ngrams_a & ngrams_b) / len(union) > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(chunk)
    return kept
