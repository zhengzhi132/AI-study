"""RAG Agent — 完整 pipeline：检索 → 上下文增强 → LLM 生成。"""

from openai import OpenAI
from chroma_store import ChromaDBStore

RAG_PROMPT_TEMPLATE = """你是一个知识助手。请基于以下参考资料回答用户问题。

规则：
1. 如果参考资料包含答案，基于资料回答并注明来源（格式：[来源 N]）
2. 如果参考资料不足以回答问题，诚实说明并建议用户补充信息
3. 不要编造参考资料中没有的信息

---
## 参考资料

{context}

---
## 用户问题

{question}

请回答："""


def format_chunks_for_prompt(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results):
        source = r["metadata"].get("source", "unknown")
        parts.append(f"[来源 {i + 1}: {source}]\n{r['document']}")
    return "\n\n".join(parts)


def build_rag_prompt(chunks: list[dict], question: str) -> str:
    context = format_chunks_for_prompt(chunks)
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


async def rag_query(
    client: OpenAI,
    model: str,
    store: ChromaDBStore,
    question: str,
    top_k: int = 5,
    verbose: bool = True,
) -> dict:
    """端到端 RAG 查询。

    Returns:
        {"answer": str, "sources": list[dict]}
    """
    # 1. 检索
    results = store.search(question, top_k=top_k)
    if verbose:
        print(f"\n[Retrieved] {len(results)} relevant chunks:")
        for i, r in enumerate(results):
            source = r["metadata"].get("source", "?")
            preview = r["document"][:80].replace("\n", " ")
            print(f"  [{i + 1}] {source} | distance={r['distance']:.3f} | {preview}...")

    # 2. 增强 + 3. 生成
    prompt = build_rag_prompt(results, question)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content or "(无回答)"
    if verbose:
        print(f"\n[RAG Answer]\n{answer}")

    return {
        "answer": answer,
        "sources": [
            {
                "source": r["metadata"].get("source", "unknown"),
                "preview": r["document"][:200],
                "distance": r.get("distance", 0),
            }
            for r in results
        ],
    }
