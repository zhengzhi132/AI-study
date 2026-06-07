#1. 分块 2. 向量化 3. 存储和检索 4. 生成回答
# 
# 分块策略：
# 1. 固定大小分块：将文本按固定字符数分割，
# 2. 句子感知分块：根据句子边界分割，确保每块包含完整句子，
# 3. 递归分块：先按大段落分割，再按小段落，最后按句子分割，确保每块尽可能大但不超过限制。 

_SEPARATORS = ["\n\n", "\n", "。", ".", "!", "？", "?", "；", ";", " "]

def chunk_text(text ,size = 500, overlap = 100):
    if len(text) <= size:
        return [text]
    sentences = text.split("\n\n")
    chunks = []
    
    for sent in sentences:
        if len(sent) < size:
            chunks.append(sent)
        else:
            paras = sent.replace("。","。\n").split("\n")
            current = ""
            for para in paras:
                if len(current) + len(para) > size and current:
                    chunks.append(current)
                    # overlap: 保留 current 尾部作为下一块开头
                    if len(current) > overlap:
                        current = current[-overlap:] + para
                    else:
                        current = para
                    
                else:
                    current += para
            if current:
                chunks.append(current)
    return chunks

class EmbeddingService:
    def __init__(self, client, model_name: str = "text-embedding-3-small"):
        self.client = client
        self.model_name = model_name
    def embed(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(input=text, model=self.model_name)
        return resp.data[0].embedding
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(input=texts, model=self.model_name)
        return [item.embedding for item in resp.data]
    
import chromadb

#初始化
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_kb")

#写入
collection.add(ids=["1"], documents=["这是一个测试文本"], metadatas=[{"source": "test.txt"}])

#查询
results = collection.query(query_texts=["测试"], n_results=1)

#RAG流程
def rag_query(llm_client, model, store, question):

    # 1. 检索 — n_results 不是 top_k
    raw = store.query(query_texts=[question], n_results=5)

    # 2. 增强 — ChromaDB 返回二维列表，需要还原结构
    ids = raw["ids"][0]
    docs = raw["documents"][0]
    metas = raw["metadatas"][0]

    context = "\n\n".join(
        f"[来源 {i + 1}: {metas[i].get('source', '?')}]\n{docs[i]}"
        for i in range(len(ids))
    )
    prompt = f"参考资料：\n{context}\n\n问题: {question}\n请基于参考资料回答。"

    # 3. 生成
    resp = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content