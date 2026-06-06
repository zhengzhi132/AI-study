# Day 6 — RAG 基础 知识点详解

> 从 Day 5 的关键词匹配到语义检索：Embedding → 向量检索 → 上下文增强 完整链路

---

## 目录

- [一、什么是 RAG — 不只是"搜索+LLM"](#一什么是-rag--不只是搜索llm)
  - [1.1 RAG 解决的核心问题](#11-rag-解决的核心问题)
  - [1.2 完整链路总览](#12-完整链路总览)
- [二、文档分块（Chunking）— 检索精度的第一关](#二文档分块chunking--检索精度的第一关)
  - [2.1 块太大 vs 块太小的权衡](#21-块太大-vs-块太小的权衡)
  - [2.2 四种分块策略](#22-四种分块策略)
  - [2.3 Chunk Overlap（重叠窗口）](#23-chunk-overlap重叠窗口)
  - [2.4 元数据的重要性](#24-元数据的重要性)
- [三、Embedding — 把文字变成可比较的向量](#三embedding--把文字变成可比较的向量)
  - [3.1 从 Jaccard 到 Embedding：一个质的飞跃](#31-从-jaccard-到-embedding一个质的飞跃)
  - [3.2 什么是 Embedding 向量](#32-什么是-embedding-向量)
  - [3.3 主流 Embedding 模型对比](#33-主流-embedding-模型对比)
  - [3.4 OpenAI Embedding API 调用详解](#34-openai-embedding-api-调用详解)
  - [3.5 深入：Embedding 模型的训练原理](#35-深入embedding-模型的训练原理)
  - [3.6 嵌入维度的选择](#36-嵌入维度的选择)
- [四、向量数据库 — ChromaDB](#四向量数据库--chromadb)
  - [4.1 为什么需要向量数据库](#41-为什么需要向量数据库)
  - [4.2 ChromaDB 核心概念](#42-chromadb-核心概念)
  - [4.3 ChromaDB 基本操作](#43-chromadb-基本操作)
  - [4.4 ChromaDB 的 Embedding Function](#44-chromadb-的-embedding-function)
  - [4.5 替换 Day 5 的 InMemoryVectorStore](#45-替换-day-5-的-inmemoryvectorstore)
- [五、检索策略](#五检索策略)
  - [5.1 相似度检索（Similarity Search）](#51-相似度检索similarity-search)
  - [5.2 最大边际相关性（MMR）— 避免重复检索](#52-最大边际相关性mmr--避免重复检索)
  - [5.3 混合检索（Hybrid Search）](#53-混合检索hybrid-search)
  - [5.4 检索结果的后处理](#54-检索结果的后处理)
- [六、上下文增强（Context Augmentation）](#六上下文增强context-augmentation)
  - [6.1 Prompt 模板设计](#61-prompt-模板设计)
  - [6.2 引用来源的重要性](#62-引用来源的重要性)
  - [6.3 Token 预算分配](#63-token-预算分配)
  - [6.4 多轮对话中的 RAG](#64-多轮对话中的-rag)
- [七、Day 6 代码架构设计](#七day-6-代码架构设计)
  - [7.1 与 Day 5 的关系](#71-与-day-5-的关系)
  - [7.2 文件结构](#72-文件结构)
  - [7.3 核心数据流](#73-核心数据流)
  - [7.4 关键设计决策](#74-关键设计决策)
- [八、从 Day 5 到 Day 6 的迁移路径](#八从-day-5-到-day-6-的迁移路径)
- [九、Day 6 核心概念清单](#九day-6-核心概念清单)
- [十、常见问题与进阶方向](#十常见问题与进阶方向)

---

## 一、什么是 RAG — 不只是"搜索+LLM"

### 1.1 RAG 解决的核心问题

**LLM 的三大局限**：

| 问题 | 具体表现 | RAG 如何解决 |
|------|---------|------------|
| **知识截止** | 训练数据到某个日期，之后的事不知道 | 检索外部文档，注入最新信息 |
| **幻觉** | 编造不存在的事实、数据、论文 | 答案必须基于检索到的原文 |
| **私有知识盲区** | 不知道你的公司内部文档、个人笔记 | 你的文档库就是"外部记忆" |

**RAG = Retrieval-Augmented Generation = 检索增强生成**

这个名字本身就说明了三个步骤：**检索**（找到相关文档）→ **增强**（把文档片段注入 prompt）→ **生成**（LLM 基于注入的上下文回答问题）。

### 1.2 完整链路总览

```
文档库 (Markdown / PDF / 网页)
  │
  ▼
┌──────────────┐
│  1. 分块      │  把长文档切成小块（chunks）
│  (Chunking)  │
└──────┬───────┘
       │ chunks: ["chunk1", "chunk2", ..., "chunkN"]
       ▼
┌──────────────┐
│  2. 向量化    │  每个 chunk → API 调用 → 1536维向量
│  (Embedding) │  写入 ChromaDB
└──────┬───────┘
       │
       │  ┌──────────────┐
       │  │  ChromaDB    │  ← 向量数据库（持久化存储）
       │  └──────────────┘
       │
       ▼ (用户提问时)
┌──────────────┐
│  3. 查询向量化│  user question → API 调用 → 1536维向量
└──────┬───────┘
       │ query_vector
       ▼
┌──────────────┐
│  4. 相似检索  │  query_vector vs 所有 chunk_vectors
│  (Retrieval) │  返回 top-K 最相似的 chunks
└──────┬───────┘
       │ top-K chunks
       ▼
┌──────────────┐
│  5. 增强生成  │  prompt = 系统指令 + chunks上下文 + 用户问题
│  (Generation)│  LLM 基于上下文回答
└──────────────┘
```

**和 Day 5 的关键区别**：Day 5 的长期记忆存的是"关于用户的事实"（用户偏好、个人信息），Day 6 的 RAG 存的是"外部知识"（文档内容）。前者是 user profile，后者是 document store。两者在架构上完全同构——都存储、都检索、都注入 prompt——只是存储的内容不同。

---

## 二、文档分块（Chunking）— 检索精度的第一关

### 2.1 块太大 vs 块太小的权衡

这是 RAG 系统**最关键的调参**。没有标准答案，取决于你的文档类型和下游任务。

| | 大块 (1000-2000 tokens) | 小块 (100-300 tokens) |
|---|---|---|
| **优点** | 上下文完整，LLM 能理解完整论述 | 检索精准，噪声少 |
| **缺点** | 检索可能返回不相关内容，浪费 token | 割裂上下文，语义碎片化 |
| **适合场景** | 技术文档、论文（需要完整推理链） | FAQ、代码片段（问什么答什么） |

**例子：同一个文档的两种切法**

```
原文: "Python 的 GIL（全局解释器锁）限制了多线程的并行执行。
       这意味着 CPU 密集型任务在 Python 多线程下不会更快。
       解决方案包括使用 multiprocessing 模块或
       用 C 扩展绕过 GIL。"

大块切法 (1024 tokens, overlap=200):
  Chunk 1: 整段都在一个chunk里 → 检索到就能理解完整因果链

小块切法 (256 tokens, overlap=50):
  Chunk 1: "Python 的 GIL（全局解释器锁）限制了多线程的并行执行。"
  Chunk 2: "这意味着 CPU 密集型任务在 Python 多线程下不会更快。"
  Chunk 3: "解决方案包括使用 multiprocessing 模块或..."
  → 搜索"Python 并发"可能只命中 Chunk 2，丢了前因后果
```

### 2.2 四种分块策略

**策略 1: 固定大小分块（Fixed-size）— 最简单，最常用**

```python
def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """按字符数等分，是最直接的策略。"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # overlap 保证跨块连续性
    return chunks
```

优点：简单、可控、适合大多数场景。缺点：可能在句子中间截断。

**策略 2: 句子感知分块（Sentence-aware）**

```python
import re

def sentence_chunk(text: str, max_chars: int = 1000) -> list[str]:
    """以句子为最小单元，尽量不在句子中间切断。"""
    sentences = re.split(r'(?<=[。！？.!?])\s*', text)
    chunks = []
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
```

优点：语义边界完整，不会截断句子。缺点：块大小不固定，可能产生超大或超小块。

**策略 3: 递归分块（Recursive）— LangChain 的默认策略**

```
设定 chunk_size=1000, separators=["\n\n", "\n", "。", ".", " "]

1. 先用 "\n\n" (段落) 分 → 如果某段 > 1000
2. 再用 "\n" (换行) 分 → 如果某行 > 1000
3. 再用 "。" (句子) 分 → 如果某句 > 1000
4. 再用 " " (词) 分  → 如果还 > 1000
5. 硬截断
```

这是最灵活的策略——尽量在自然边界切分，优先保持段落完整，实在不行再降级。

**策略 4: 语义分块（Semantic）— 最智能但也最贵**

用 LLM 或专门的模型判断"话题边界"在哪里。适合高质量要求的场景，但每个 chunk 的边界检测本身就要额外的模型调用。

### 2.3 Chunk Overlap（重叠窗口）

```
chunk_size=500, overlap=100

Chunk 1: ┌──────────────────────────┐
         │  A B C D E F G H I J    │
         └──────────────────────────┘
                 └──overlap──┘
Chunk 2:         ┌──────────────────────────┐
                 │  G H I J K L M N O P Q  │
                 └──────────────────────────┘
```

**为什么需要 overlap？** 如果关键信息恰好横跨两个 chunk 的边界，没有 overlap 就会丢失。比如"Python GIL 限制了多线程"这句话如果被切成两半，分别在 Chunk N 和 Chunk N+1 中，两者都不包含完整语义，检索时不会被高分召回。

**overlap 设多大？** 经验值：chunk_size 的 10-20%。对于 500 字符的 chunk，overlap=50-100。

### 2.4 元数据的重要性

每个 chunk 不只是文本，还要带元数据：

```python
@dataclass
class DocumentChunk:
    content: str
    metadata: dict  # {"source": "python_gil.md", "page": 3, "section": "并发"}
    chunk_index: int
```

元数据的用途：
- **追溯原文**：用户要求"给我看原文"时，指向具体文件/页码
- **过滤检索**：只在特定来源中搜索（"只查技术文档，不查会议记录"）
- **去重和更新**：文档变更时，按 source 删除旧 chunks 再写入新的

---

## 三、Embedding — 把文字变成可比较的向量

### 3.1 从 Jaccard 到 Embedding：一个质的飞跃

回顾 Day 5 的痛点（[day5/in_memory_vector_store.py](../day5-memory-system/in_memory_vector_store.py)）：

```python
# Jaccard 检索 "用户所在城市"
# query: "用户所在城市"  → 关键词: {用户, 所在, 城市}
# entry: "用户住在北京"  → 关键词: {用户, 住在, 北京}
# Jaccard = {"用户"} / {"用户", "所在", "城市", "住在", "北京"} = 1/5 = 0.2 ← 低分
# 结果：找不到 → Agent 回答"我不了解你" ❌
```

同样的问题，用 embedding（把文字映射到向量空间）：

```
"用户所在城市" → [0.12, -0.34, 0.56, ..., -0.09]  (1536维)
"用户住在北京" → [0.11, -0.32, 0.58, ..., -0.11]  (1536维)

余弦相似度 ≈ 0.92 ← 高分！正确召回 ✅
```

**根本区别**：Jaccard 看"字符有没有重叠"，embedding 看"意思像不像"。后者是语义级别的匹配。

### 3.2 什么是 Embedding 向量

**本质**：把一段文字压缩成一个固定长度的浮点数数组（向量），这段文字的**语义信息**被编码在这个数组的数值关系里。

```
"今天天气真好" ──→ text-embedding-3-small ──→ [0.023, -0.451, 0.782, ..., 0.134]
                                                    ↑
                                          1536 个浮点数（1536 维空间中的一个点）
```

**核心性质**：语义相似的文字，在向量空间中距离近；语义无关的文字，距离远。

```
距离近（相似）：
  "Python 多线程性能差"   ←──4°──→  "GIL 限制了 Python 并发"

距离远（不相关）：
  "Python 多线程性能差"   ←──87°──→  "今天晚饭吃什么"
```

相似度用**余弦相似度（Cosine Similarity）**衡量：

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """两个向量夹角的余弦值。1.0 = 方向完全相同, 0 = 正交, -1 = 相反。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = (sum(x * x for x in a)) ** 0.5
    norm_b = (sum(x * x for x in b)) ** 0.5
    return dot / (norm_a * norm_b)
```

余弦相似度为 1.0 意味着两个向量指向完全相同的方向（语义等价），0 意味着毫无关系（正交），-1 意味着语义相反。

### 3.3 主流 Embedding 模型对比

| 模型 | 维度 | 最大输入 | 中文支持 | 费用 | 适用场景 |
|------|------|---------|---------|------|---------|
| `text-embedding-3-small` | 512/1536 | 8191 tokens | 好 | ~$0.02/1M tokens | 通用，性价比最高 |
| `text-embedding-3-large` | 256/1024/3072 | 8191 tokens | 好 | ~$0.13/1M tokens | 高精度要求 |
| `BGE-M3` (BAAI) | 1024 | 8192 tokens | **极好** | 免费（本地部署） | 中文场景首选 |
| `jina-embeddings-v3` | 1024 | 8192 tokens | 好 | 有免费额度 | 多语言 |
| `bge-large-zh-v1.5` | 1024 | 512 tokens | **极好** | 免费（本地部署） | 纯中文轻量场景 |

**为什么课程用 `text-embedding-3-small`？** 不需要本地 GPU，API 调用即可，维度适中（1536），成本极低。BGE 系列更适合中文但对硬件有要求，Day 6 聚焦概念而非工程部署。

### 3.4 OpenAI Embedding API 调用详解

```python
from openai import OpenAI

client = OpenAI()  # 自动读 OPENAI_API_KEY

# 单条文本
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Python 的 GIL 限制了多线程性能",
    dimensions=1536,  # 可选：不传则用模型默认维度
)
vector = response.data[0].embedding  # list[float], 长度 1536

# 批量文本（省钱！一次请求处理多条）
texts = ["chunk1...", "chunk2...", "chunk3..."]
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts,  # 最多 2048 条/请求
)
vectors = [d.embedding for d in response.data]  # 按输入顺序返回
```

**关键细节**：

1. **`dimensions` 参数**：`text-embedding-3-small` 原生输出 1536 维，但你可以降到 512 维。维度越低，存储越小、检索越快，但精度也越低。1536 → 512 是最常见的降维选择，精度损失 <5%。

2. **批量处理**：一次请求发 20-50 条文本比逐条发送快得多，而且计费按 token 算（不按请求次数），不增加成本。

3. **token 限制**：每条输入不能超过 8191 tokens。如果 chunk 超大，需要先截断。

### 3.5 深入：Embedding 模型的训练原理

> 以下内容为进阶知识，不影响 Day 6 代码实现。

**第一代：Word2Vec（2013）— 基于"一个词的含义由它周围的词决定"**

```
训练语料: "我 今天 开 汽车 去 上班"

以"汽车"为中心，窗口=2：
  输入: ("开", "去")（上下文词 one-hot 编码）
  目标: "汽车"（要预测的中心词 one-hot 编码）
  
神经网络:
  Input(2×vocab_size) → Hidden(300维) → Output(vocab_size)
  
训练完毕，丢弃输出层。取输入权重矩阵的每一行 → 就是每个词的 embedding。

为什么"汽车"和"轿车"接近？
→ 它们在语料中总是出现在相同的上下文 ("开 _ 去", "买 _ 了")
→ 梯度下降迫使它们产生相似的隐藏层激活值
→ 相似激活值 = 相近的向量
```

**第二代：BERT（2018）— 上下文相关**

Word2Vec 缺陷：同一个"银行"只有一个固定向量，不分"存钱的银行"还是"河岸"。

```python
# Word2Vec: "银行" 永远返回同一个向量
embed("我去银行存钱")["银行"] == embed("我在银行钓鱼")["银行"]  # True

# BERT: 每个词融入周围上下文，不同语境不同向量
bert_embed("我去银行存钱")["银行"] != bert_embed("我在银行钓鱼")["银行"]  # 不同!
```

BERT 的核心机制是 **Masked Language Model + Transformer 自注意力**：

```
输入:  "我 昨天 [MASK] 了一辆汽车"  → 预测 [MASK] = "买"

整个句子过 12 层 Transformer。自注意力让每个词都"看到"句子中的其他所有词，
计算加权融合。最终每个位置输出的向量 = 融合了整个句子上下文后的表示。
```

**第三代：对比学习（2020s）— 专为检索优化**

现代的 `text-embedding-3-small`、BGE 等不再是"顺带产出向量"，而是**直接以相似度为目标**训练：

```
训练数据: (query, 正例, 负例) × 数十亿组

query:  "如何在 Python 中读取文件"
正例:   "使用 open() 函数打开文件并读取内容"    ← 相关的
负例:   "巴西在 2022 年世界杯中止步八强"        ← 不相关的

训练目标（InfoNCE Loss / 对比损失）:
  让 cos_sim(query, 正例) → 尽可能接近 1
  让 cos_sim(query, 负例) → 尽可能接近 0

本质: 在向量空间中，把"相关的"拉近，把"不相关的"推开。
重复数十亿次后，模型学会映射语义。
```

训练数据集通常是 `(query, positive, negative)` 三元组：

| 来源 | 构造方式 |
|------|---------|
| 搜索点击日志 | query + 用户点击的页面（正例）+ 未点击的页面（负例） |
| 问答对 | question + answer（正例）+ 随机其他 answer（负例） |
| 文档标题-段落 | 标题 + 所属段落（正例）+ 其他段落（负例） |

**推理（Inference）阶段**：
```
已训练好的编码器权重固定不变

"汽车" ──▶ 编码器 ──▶ [0.23, -0.45, 0.78, ..., 0.12]
"轿车" ──▶ 编码器 ──▶ [0.21, -0.43, 0.80, ..., 0.15]  ← 距离 0.04, 很近
```

把"编码器"理解为一个固定的数学函数：同样的输入永远产生同样的输出。它不是查表，不会有"数据库—不需要查询就能直接匹配"的问题。向量存储的职责是存下这些固定输出，然后在查询时做最近邻搜索——这才是"查找"发生的环节。

### 3.6 嵌入维度的选择

```python
# 维度对比
small_1536 = create_embedding("text", dimensions=1536)  # 高精度，存储大
small_512  = create_embedding("text", dimensions=512)   # 精度略降，存储是 1/3
```

| 维度 | 单条存储 | 10 万条存储 | 精度 | 推荐场景 |
|------|---------|-----------|------|---------|
| 1536 | ~6 KB | ~600 MB | 基准 | 文档量 < 10万，精度优先 |
| 512 | ~2 KB | ~200 MB | 降 ~3-5% | 文档量大，速度/成本优先 |

`text-embedding-3-small` 使用 Matryoshka 表示学习（Matryoshka Representation Learning, MRL）技术，在训练时就让模型学会按维度排序信息重要性——前 512 维编码最重要信息，后面维度是细节补充。因此直接截取前 512 维就近似等于低维版本，精度损失很小。这是 MRL 的核心贡献：**一个模型，多个精度级别**。

---

## 四、向量数据库 — ChromaDB

### 4.1 为什么需要向量数据库

10 万个 chunks × 1536 维 × 4 bytes = ~600 MB。如果每次查询都对 10 万个向量做暴力余弦相似度计算，延迟在**秒级**。

向量数据库的核心价值：
1. **索引加速**：用 HNSW（Hierarchical Navigable Small World）等近似最近邻算法，10 万个向量也能在毫秒级返回
2. **持久化**：数据写到磁盘，重启不丢
3. **元数据过滤**：`where={"source": "tech_docs"}` 先过滤再检索
4. **自动管理 embedding**：可以内置 embedding function，存文本自动转向量

### 4.2 ChromaDB 核心概念

```
ChromaDB
  └── Collection（集合/表）
       ├── ids:       ["doc1_chunk0", "doc1_chunk1", ...]
       ├── documents: ["分块文本1...", "分块文本2...", ...]
       ├── embeddings:[[0.12, -0.34, ...], [0.09, 0.56, ...], ...]
       └── metadatas: [{"source": "a.md", "chunk": 0}, ...]
```

**ChromaDB 的数据模型极其简单**：一个 Collection 就是一张四列的表（ids、documents、embeddings、metadatas）。没有复杂的 schema、没有外键、没有 join。

### 4.3 ChromaDB 基本操作

```python
import chromadb
from chromadb.config import Settings

# 创建客户端
client = chromadb.PersistentClient(path="./chroma_data")  # 持久化到磁盘
# client = chromadb.Client()  # 仅内存，关闭即丢失

# 创建或获取 collection
collection = client.get_or_create_collection(
    name="tech_docs",
    metadata={"hnsw:space": "cosine"},  # 用余弦距离（默认是 L2）
)

# 添加文档（ChromaDB 自动调用 embedding function）
collection.add(
    ids=["doc1_c0", "doc1_c1", "doc1_c2"],
    documents=["chunk 0 text...", "chunk 1 text...", "chunk 2 text..."],
    metadatas=[
        {"source": "python_gil.md", "chunk_idx": 0},
        {"source": "python_gil.md", "chunk_idx": 1},
        {"source": "python_gil.md", "chunk_idx": 2},
    ],
)

# 查询（自动将 query 转为向量再检索）
results = collection.query(
    query_texts=["Python 并发性能问题"],  # 自动 embedding
    n_results=3,                          # top-3
    where={"source": "python_gil.md"},    # 可选：元数据过滤
)

# results["documents"][0] → ["chunk 2...", "chunk 0...", "chunk 1..."]
# results["distances"][0] → [0.12, 0.23, 0.31]  (越小越相似，如果用 cosine)
# results["metadatas"][0] → [{"source":..., "chunk_idx":2}, ...]
# results["ids"][0]       → ["doc1_c2", "doc1_c0", "doc1_c1"]

# 删除
collection.delete(ids=["doc1_c0"])  # 删除指定条目
# collection.delete(where={"source": "python_gil.md"})  # 按元数据批量删除
```

### 4.4 ChromaDB 的 Embedding Function

ChromaDB 内置了多种 embedding function，自动处理"文本 → 向量"转换：

```python
import chromadb
from chromadb.utils import embedding_functions

# 方案 A: OpenAI embedding（Day 6 使用这个）
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="sk-xxx",
    model_name="text-embedding-3-small",
    dimensions=1536,
)

# 方案 B: 本地 Sentence Transformer（免费，需 GPU）
sentence_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-large-zh-v1.5",  # 中文最优
)

# 创建 collection 时绑定
collection = client.create_collection(
    name="docs",
    embedding_function=openai_ef,  # add/query 时自动转向量
)

# 之后 add() 和 query() 都不用手动调 embedding API
collection.add(ids=["id1"], documents=["文本..."])  # Chroma 自动调用 OpenAI API
results = collection.query(query_texts=["问题?"], n_results=5)  # 自动 embedding
```

### 4.5 替换 Day 5 的 InMemoryVectorStore

回顾 Day 5 的 [interfaces.py](../day5-memory-system/interfaces.py) 中的 `IVectorStore`：

```python
class IVectorStore(ABC):
    async def add(self, entry: MemoryEntry) -> str: ...
    async def search(self, query: str, top_k: int) -> list[MemorySearchResult]: ...
    async def delete(self, entry_id: str) -> None: ...
    def size(self) -> int: ...
    def get_all(self) -> list[MemoryEntry]: ...
```

Day 6 创建一个 `ChromaDBStore(IVectorStore)`，用 ChromaDB 替换关键词匹配：

```python
class ChromaDBStore(IVectorStore):
    """用 ChromaDB + OpenAI Embedding 实现 IVectorStore 接口"""

    def __init__(self, collection_name: str, persist_dir: str = "./chroma_data"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="text-embedding-3-small",
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(self, entry: MemoryEntry) -> str:
        entry_id = entry.id or f"mem_{uuid.uuid4().hex[:8]}"
        self._collection.add(
            ids=[entry_id],
            documents=[entry.content],
            metadatas=[{"source": entry.source, "conversation_id": entry.conversation_id}],
        )
        return entry_id

    async def search(self, query: str, top_k: int) -> list[MemorySearchResult]:
        results = self._collection.query(query_texts=[query], n_results=top_k)
        # 转换 ChromaDB 返回格式 → MemorySearchResult
        ...
```

**与 Day 5 的关键不同**：`ChromaDBStore.search()` 底层走的是 `cosine_similarity(embedding(query), embedding(each_doc))`，而不是 `jaccard(query_keywords, entry_keywords)`。这就是语义检索替代关键词检索——接口完全不变，行为彻底升级。

---

## 五、检索策略

### 5.1 相似度检索（Similarity Search）

最基本也最常用的方式。查询向量和所有文档向量算余弦相似度，取 top-K。

```python
results = collection.query(
    query_texts=["Python 并发编程最佳实践"],
    n_results=5,
)
```

**局限**：可能返回 5 条几乎相同的内容（比如同一段话被切成多个 chunk，全部高分召回），浪费上下文空间。

### 5.2 最大边际相关性（MMR）— 避免重复检索

**MMR = Maximal Marginal Relevance**。核心思想：选择与查询相关**且**彼此之间不相似的文档。

```
普通 top-5 检索:
  Chunk 1 (score 0.95) ← 同段话被切成
  Chunk 2 (score 0.94) ← 多个 chunk，全召回
  Chunk 3 (score 0.93) ← 浪费 token
  Chunk 4 (score 0.81)
  Chunk 5 (score 0.79)

MMR 检索 (λ=0.7):
  Chunk 1 (score 0.95)         ← 与查询最相关
  Chunk 4 (score 0.81, 多样)   ← 与 Chunk 1 不重复
  Chunk 7 (score 0.76, 多样)   ← 补充不同角度
  Chunk 12 (score 0.72, 多样)
  Chunk 15 (score 0.68, 多样)
```

公式：`MMR = λ × relevance_to_query - (1-λ) × max_similarity_to_already_selected`

- `λ` 控制"相关性 vs 多样性"的权衡。`λ=1` 退化为普通 top-K，`λ=0` 只看多样性。
- 生产环境常用 `λ=0.7`（偏重相关，兼顾多样）。

```python
# ChromaDB 不直接支持 MMR，需要手动实现或切换到支持 MMR 的向量库
# 伪代码
def mmr_search(query_vec, doc_vecs, doc_texts, top_k=5, lambda_param=0.7):
    selected = []
    remaining = list(range(len(doc_vecs)))
    
    for _ in range(top_k):
        best_score = -float("inf")
        best_idx = None
        for i in remaining:
            relevance = cosine_similarity(query_vec, doc_vecs[i])
            diversity = max(
                [cosine_similarity(doc_vecs[i], doc_vecs[j]) for j in selected],
                default=0
            )
            mmr = lambda_param * relevance - (1 - lambda_param) * diversity
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return [doc_texts[i] for i in selected]
```

### 5.3 混合检索（Hybrid Search）

**问题**：纯向量检索（dense）擅长语义匹配，但遇到精确名词（人名、API 名、代码）可能漏。纯关键词检索（sparse，如 BM25）擅长精确匹配，但不理解同义词。

**解决方案**：两种结果融合。

```python
# 伪代码
def hybrid_search(query, collection, alpha=0.5):
    # 1. 向量检索（语义）
    dense_results = collection.query(query_texts=[query], n_results=10)
    
    # 2. 关键词检索（精确匹配）
    sparse_results = bm25_search(query, all_documents, top_k=10)
    
    # 3. 分数融合（Reciprocal Rank Fusion）
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    
    # alpha 控制权重：1.0 = 纯向量, 0.0 = 纯关键词
    return fused[:5]
```

Reciprocal Rank Fusion (RRF) 不需要知道每个分数的绝对大小，只关心排名：

```python
def reciprocal_rank_fusion(results_a, results_b, k=60):
    """k=60 是经验值：排名越靠前，融合分越高。"""
    scores = {}
    for rank, (doc_id, _) in enumerate(results_a):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(results_b):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 5.4 检索结果的后处理

```python
# 后处理流水线
retrieved = collection.query(query_texts=[query], n_results=10)

# 1. 去重（Jaccard 相似度 > 0.9 的 chunk 只保留分数最高的）
deduped = deduplicate_chunks(retrieved, threshold=0.9)

# 2. 相关性阈值过滤（分数太低 = 噪声，不注入 prompt）
filtered = [r for r in deduped if r["score"] > 0.7]

# 3. 排序 + 截断
final = sorted(filtered, key=lambda r: r["score"], reverse=True)[:5]
```

注意：第二步的阈值（0.7）是一个魔法数字，需要根据你的数据、模型和场景调参。通用的做法是先打印一批 query 的 score 分布，观察"正例"和"负例"的自然分界点在哪里。

---

## 六、上下文增强（Context Augmentation）

### 6.1 Prompt 模板设计

检索到的 chunks 如何注入 prompt？核心原则：**把 chunks 放在 user message 的最前面，用明确的标记分隔**。

```python
RAG_PROMPT_TEMPLATE = """你是一个知识助手。请基于以下参考资料回答用户问题。

规则：
1. 如果参考资料包含答案，基于资料回答并注明引用来源
2. 如果参考资料不足以回答问题，诚实说明并建议用户补充信息
3. 不要编造参考资料中没有的信息

---
## 参考资料

{context}

---
## 用户问题

{question}

请回答："""
```

**为什么要用明确的分隔标记（`---`）？** LLM 对格式敏感，清晰的结构标记比纯文本拼接更能让模型区分"参考材料"和"用户问题"。

### 6.2 引用来源的重要性

```python
# 每个 chunk 都标注来源
formatted_chunks = []
for i, chunk in enumerate(retrieved_chunks):
    formatted_chunks.append(
        f"[来源 {i+1}: {chunk.metadata['source']}, "
        f"章节 {chunk.metadata.get('section', 'N/A')}]\n"
        f"{chunk.content}"
    )

context = "\n\n".join(formatted_chunks)
```

引用来源不只是"好习惯"，它在生产环境中是**必须的**：
- **可审计**：用户能回到原文验证答案
- **可调试**：检索到了但不相关 → chunking 问题；检索到了且相关但答案错了 → prompt 或 LLM 问题
- **建立信任**：用户看到引用，更相信 AI 没编造

### 6.3 Token 预算分配

LLM 的上下文窗口是有限的，你需要显式分配预算：

| 组件 | 预算占比 | 说明 |
|------|---------|------|
| System prompt | ~10% | 角色定义 + 规则 |
| 检索到的 chunks | ~60% | RAG 的核心价值所在 |
| 对话历史 | ~20% | 多轮对话需要保留 |
| 用户当前问题 + LLM 回答空间 | ~10% | 预留生成空间 |

```python
def allocate_token_budget(max_tokens: int, num_chunks: int) -> int:
    """计算每个 chunk 平均可用 token。"""
    retrieval_budget = int(max_tokens * 0.6)
    return retrieval_budget // num_chunks
```

### 6.4 多轮对话中的 RAG

单轮和多轮有本质区别：

```python
# 单轮：直接用问题检索
query = "Python 多线程性能为什么差"  # 原样检索即可

# 多轮：需要对问题做"上下文化改写"
# 对话：
#   用户: "什么是 GIL？"
#   助手: "GIL 是全局解释器锁..."
#   用户: "那怎么解决它的性能问题？"  ← "它" 指 GIL，"性能问题" 指多线程

# 直接用"那怎么解决它的性能问题"检索 → 效果很差
# 需要先把问题改写为独立的、自包含的版本
rewritten = "如何解决 Python GIL 全局解释器锁导致的多线程性能问题"
```

解决方案：在检索前多用一次 LLM 调用，把模糊的追问改写为完整的独立问题。这是多轮 RAG 的标配操作。

```python
async def rewrite_query(client, history, current_question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 用便宜模型即可
        messages=[
            {"role": "system", "content": "将用户的多轮对话追问改写为一个独立的、自包含的完整问题。"},
            {"role": "user", "content": f"对话历史:\n{history}\n\n当前问题: {current_question}\n\n改写后的问题:"},
        ],
    )
    return response.choices[0].message.content
```

---

## 七、Day 6 代码架构设计

### 7.1 与 Day 5 的关系

Day 5 和 Day 6 不是替代关系，是**互补关系**：

```
Day 5 — 记忆系统                 Day 6 — RAG 系统
─────────────────────────       ──────────────────────
目的: 记住用户是谁              目的: 回答基于文档的问题
存储: 关于用户的事实/偏好       存储: 外部文档的 chunks
检索: 查"用户喜欢什么"         检索: 查"文档里怎么说"
注入: 注入到 system prompt      注入: 注入到 user prompt 前面

            两个系统共用一个体系结构——
            IVectorStore 接口定义了
            add / search / delete / size / get_all
```

Day 6 不是"替换 InMemoryVectorStore"，而是**新增 ChromaDBStore**——前者继续用来存用户事实，后者用来存文档 chunks。两个 `IVectorStore` 实例，各自服务不同的记忆/知识类型。

### 7.2 文件结构

```
day6-rag-basics/
├── STUDY-GUIDE.md              # 本文件
├── interfaces.py               # 复用 Day 5 的 IVectorStore + 新增 RAG 专用类型
├── chunker.py                  # 文档加载 + 分块策略 (fixed / recursive / sentence)
├── embedding.py                # OpenAI Embedding API 封装
├── chroma_store.py             # ChromaDBStore — 实现 IVectorStore 接口
├── retriever.py                # 检索策略 (similarity / MMR / hybrid)
├── rag_agent.py                # RAG Agent — 完整 pipeline: 查询 → 检索 → 增强 → 生成
├── document_indexer.py         # 批量索引文档的入口脚本
└── demo.py                     # 演示：对比 Jaccard vs Embedding 检索效果
```

### 7.3 核心数据流

```
# === 离线阶段：索引文档 ===
documents/                        # 3-5 篇 Markdown 文件
  │
  ▼
chunker.chunk_documents()         # 分块 + 元数据提取
  │ chunks: list[DocumentChunk]
  ▼
embedding.batch_embed()           # API 调用 → 向量数组
  │ vectors: list[list[float]]
  ▼
chroma_store.add_chunks()         # 存入 ChromaDB
  │ 持久化到 ./chroma_data/

# === 在线阶段：回答用户问题 ===
user_question
  │
  ▼
retriever.retrieve()              # query → embedding → ChromaDB.search
  │ top-K chunks
  ▼
rag_agent.build_prompt()          # template(chunks, question) → augmented_prompt
  │
  ▼
llm.chat.completions.create()    # 生成回答
  │ answer
  ▼
用户看到回答（带来源引用）
```

### 7.4 关键设计决策

**决策 1: 离线索引 vs 在线检索分离**

`document_indexer.py`（离线）和 `rag_agent.py`（在线）是两个独立的文件。因为索引只需做一次（或文档更新时重新做），检索每次对话都要做。把两者分开避免混在一起。

**决策 2: 复用 Day 5 的 IVectorStore 接口**

`ChromaDBStore` 实现 Day 5 定义的 `IVectorStore` 接口。这意味着：
- Day 5 的 `LongTermMemory` 可以无缝切换为 ChromaDB 后端
- Day 7 的"个人知识库 Agent"可以同时使用两类 Store（事实 Store + 文档 Store）
- 接口的统一是关键——上层代码不需要知道底层是内存还是 ChromaDB

**决策 3: embedding function 封装**

```python
class EmbeddingService:
    def __init__(self, client: OpenAI, model: str, dimensions: int):
        ...
    async def embed_single(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

封装成独立服务类，ChromaDBStore 和 Demo 都可以用。也方便未来切换模型（改一行即可）。

---

## 八、从 Day 5 到 Day 6 的迁移路径

Day 6 不是从头开始，而是 Day 5 的自然扩展：

| Day 5 | Day 6 | 变化 |
|-------|-------|------|
| `InMemoryVectorStore` | `ChromaDBStore` | 内存 → 磁盘持久化，关键词 → 语义向量 |
| `MemoryEntry` (用户事实) | `DocumentChunk` (文档片段) | 同样的结构，不同的语义 |
| `_jaccard()` 相似度 | `cosine_similarity()` | 字符匹配 → 向量空间距离 |
| `_extract_keywords()` | `embedding_api()` | 本地关键词提取 → API 调用 |
| `LongTermMemory` | `DocumentIndexer` | 事实管理 → 文档索引 |
| `MemoryManager` | `RAGAgent` | 记忆编排 → 检索编排 |

核心接口 `IVectorStore` 保持不变。这是依赖倒置原则最有力的展示——换底层实现，上层代码零改动。

---

## 九、Day 6 核心概念清单

1. **RAG 完整链路**: 分块 → 向量化 → 存储 → 检索 → 增强 → 生成
2. **Chunking 策略**: fixed-size / sentence-aware / recursive / semantic，以及 overlap 的作用
3. **块大小权衡**: 太大噪声多 → 太小丢上下文。需要按文档类型调参
4. **Embedding 原理**: 对比学习训练编码器，语义相似 → 向量空间距离近
5. **余弦相似度**: `cos(a,b) = a·b / (|a|×|b|)`，值域 [-1, 1]，语义搜索标准度量
6. **ChromaDB**: Collection 模型，PersistentClient，内置 embedding function
7. **MMR 检索**: 相关且多样的结果，避免重复 chunks 浪费 token
8. **混合检索**: dense (语义) + sparse (关键词) 互补
9. **Prompt 增强**: 明确的上下文标记，引用来源，token 预算分配
10. **多轮 RAG**: 对追问做完整改写后再检索
11. **IVectorStore 复用**: Day 5 的抽象接口，Day 6 新增 ChromaDB 实现
12. **离线/在线分离**: 索引一次，检索多次

---

## 十、常见问题与进阶方向

**Q: ChromaDB vs FAISS vs Pinecone 怎么选？**
ChromaDB 适合本地开发和小规模（<100k 文档），零配置。FAISS 适合大规模（百万级），需要手动管理。Pinecone 是全托管云服务，适合不想维护基础设施的团队。

**Q: 检索到的 chunks 不相关怎么办？**
按优先级排查：(1) chunk 太大导致混入无关内容 → 减小 chunk_size；(2) embedding 模型不适合你的语言/领域 → 换 BGE 或其他；(3) 用户问题太模糊需要改写 → 加 query rewriting 步骤；(4) 向量检索本身的局限 → 加混合检索。

**Q: 怎么知道我的 RAG 系统好不好？**
准备 20-50 个问答对做 eval。对每个问题：检查检索到的 top-5 chunks 是否包含答案（检索准确率），再检查 LLM 最终回答是否正确（端到端准确率）。两个指标分开看——前者排查 chunking/embedding，后者排查 prompt/LLM。

**Q: 我的文档是 PDF 怎么办？**
用 `PyMuPDF`（fitz）或 `pdfplumber` 提取文本，然后和 Markdown 一样走分块→向量化→存储的流程。Week 2 的 Day 7（综合项目）会具体实现。

---

> **Day 7 预告**：整合 Day 1-6 的全部内容——ReAct Agent + Tool System + Memory System + RAG，做一个完整的个人知识库 Agent。
