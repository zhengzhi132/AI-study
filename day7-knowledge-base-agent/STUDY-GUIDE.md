# Day 7 — 个人知识库 Agent 知识点详解

> 对照 [demo.py](demo.py) 运行结果阅读，理解如何将 Day 1-6 的独立模块整合为一个完整系统。

---

## 目录

- [一、Day 7 在学什么](#一day-7-在学什么)
  - [1.1 从零件到整机](#11-从零件到整机)
  - [1.2 整合了什么](#12-整合了什么)
- [二、双向量存储架构 — InMemoryVectorStore + ChromaDB 并行](#二双向量存储架构--inmemoryvectorstore--chromadb-并行)
  - [2.1 为什么需要两个 Store](#21-为什么需要两个-store)
  - [2.2 两套 Store 的生命周期](#22-两套-store-的生命周期)
  - [2.3 设计考量：为什么不合并](#23-设计考量为什么不合并)
- [三、跨模块加载 — importlib 的实战应用](#三跨模块加载--importlib-的实战应用)
  - [3.1 问题：两个 interfaces.py 的命名冲突](#31-问题两个-interfacespy-的命名冲突)
  - [3.2 解决：sys.path 隔离 + importlib 加载](#32-解决syspath-隔离--importlib-加载)
  - [3.3 替代方案对比](#33-替代方案对比)
- [四、工具系统的扩展 — SearchKBTool 的工厂模式](#四工具系统的扩展--searchkbtool-的工厂模式)
  - [4.1 为什么用工厂函数](#41-为什么用工厂函数)
  - [4.2 Agent 的 6 个工具一览](#42-agent-的-6-个工具一览)
- [五、检索策略 — "先 KB 后网络"的调度逻辑](#五检索策略--先-kb-后网络的调度逻辑)
  - [5.1 System Prompt 的角色](#51-system-prompt-的角色)
  - [5.2 实际调度流程分析](#52-实际调度流程分析)
  - [5.3 这就是 ReAct 的强大之处](#53-这就是-react-的强大之处)
- [六、端到端数据流](#六端到端数据流)
  - [6.1 离线阶段：索引文档](#61-离线阶段索引文档)
  - [6.2 在线阶段：多轮对话](#62-在线阶段多轮对话)
  - [6.3 Demo 三回合解析](#63-demo-三回合解析)
- [七、Day 7 核心概念清单](#七day-7-核心概念清单)
- [八、Week 1 总结：你构建了什么](#八week-1-总结你构建了什么)

---

## 一、Day 7 在学什么

### 1.1 从零件到整机

Day 1-6 每两天解决一个独立问题：LLM API 调用、Function Calling、ReAct 推理循环、可扩展工具系统、三层记忆架构、RAG 检索增强生成。Day 7 不做新理论，而是做**系统集成**——把独立模块拼成可用的完整系统，解决模块间的命名冲突、接口适配、生命周期协调。

### 1.2 整合了什么

```
┌─────────────────────────────────────────────────────┐
│                   Day 7 Agent                        │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Day 4       │  │  Day 5       │  │  Day 6     │ │
│  │  ToolRegistry│  │  MemoryManager│  │  ChromaDB  │ │
│  │  (6 tools)   │  │  (3-tier mem) │  │  (KB store) │ │
│  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │
│         │                 │                 │        │
│         └─────────────────┼─────────────────┘        │
│                           │                          │
│                    ┌──────┴──────┐                   │
│                    │  ReAct Loop │                   │
│                    │  (Day 3)    │                   │
│                    └─────────────┘                   │
└─────────────────────────────────────────────────────┘
```

---

## 二、双向量存储架构 — InMemoryVectorStore + ChromaDB 并行

### 2.1 为什么需要两个 Store

Day 7 的 Agent 同时运行两个独立的向量存储：

| | InMemoryVectorStore (Day 5) | ChromaDBStore (Day 6) |
|---|---|---|
| **存储内容** | 用户事实/偏好 | 文档 chunks |
| **存储介质** | 内存（进程内） | 磁盘（SQLite + HNSW 索引） |
| **检索算法** | Jaccard 关键词相似度 | 余弦相似度（语义向量） |
| **生命周期** | 进程重启即丢失 | 持久化，跨重启保留 |
| **谁来管理** | MemoryManager.long_term | SearchKBTool |
| **触发方式** | 自动（finalize 提取） | 显式（Agent 调用工具） |

两者各司其职：

```
"你记得我学什么吗？"  →  recall_fact  →  InMemoryVectorStore  →  "用户正在学 asyncio"
"GIL 是什么？"       →  search_kb    →  ChromaDBStore        →  python_gil.md 的 chunk
```

一个记住"用户是谁"，一个检索"文档说了什么"——两个完全不同的信息域。

### 2.2 两套 Store 的生命周期

```
Agent 启动
  │
  ├── InMemoryVectorStore()     ← 空白，每次启动重新 build
  ├── MemoryManager(store)      ← 包装短期/长期/工作记忆
  │
  └── ChromaDBStore(persist_dir) ← 从磁盘恢复，已有历史索引数据
       
对话进行中...
       
  ├── save_fact → InMemoryVectorStore (已保存 mem_1, mem_2, ...)
  └── search_kb → ChromaDBStore    (始终可检索已索引的文档)
       
Agent 关闭
       
  ├── InMemoryVectorStore → 数据消失（下次启动重新 build）
  └── ChromaDBStore → 数据持久（chroma_data/ 目录）
```

### 2.3 设计考量：为什么不合并

1. **用户事实需要频繁更新**：每轮对话后 LLM 提取新事实，`add` 操作高频。ChromaDB 的磁盘写入比内存慢
2. **语义空间不同**：用户事实（"喜欢简洁回答"）和文档内容（"GIL 是互斥锁..."）在 embedding 空间里距离很远，混在一起会互相污染检索结果
3. **教学目的**：保留两套 Store 让你直观感受"什么时候用什么存储方案"

---

## 三、跨模块加载 — importlib 的实战应用

### 3.1 问题：两个 interfaces.py 的命名冲突

```
day5-memory-system/
  └── interfaces.py   ← 定义 MemoryEntry, IVectorStore, SummarizationConfig

day6-rag-basics/
  └── interfaces.py   ← 定义 DocumentChunk, RetrievalResult, ChunkConfig
```

`from interfaces import X` 在 Day 5 和 Day 6 里分别导入不同的东西。当你要在一个文件里同时导入 Day 5 和 Day 6 的模块时，Python 的模块缓存（`sys.modules`）只认**一个** `"interfaces"`。

### 3.2 解决：sys.path 隔离 + importlib 加载

[agent.py](agent.py) 的加载策略：

```python
# Step 1: Day 5 获得永久路径优先权
sys.path.insert(0, str(_DAY5))

# Step 2: 用 importlib 加载每个 Day 6 模块，
#         每个模块有临时路径 + 独立模块名
_day6_chroma = _load_from("_day6_chroma_store", _DAY6 / "chroma_store.py", _DAY6)

# _load_from 内部：临时加 _DAY6 到 path，加载完移除
def _load_from(name, path, source_dir):
    sys.path.insert(0, str(source_dir))  # Day 6 模块能看到 Day 6 的 interfaces
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    finally:
        sys.path.pop(0)  # 恢复，防止污染 Day 5 的导入
```

核心技巧：每个模块有独立的 `sys.path` 上下文和独立的缓存名，加载完立刻恢复 path。

### 3.3 替代方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| importlib (当前) | 无需修改源模块 | 代码啰嗦，理解成本高 |
| 重命名 interfaces.py | 最简单 | 破坏 Day 5/6 的独立性 |
| 做成 pip 包 | 标准做法 | 需要 setup.py |
| 统一为一个 interfaces.py | 零冲突 | 耦合 Day 5 和 Day 6 |

Day 7 用 importlib 不仅是解决冲突，也是教学——让你理解 `sys.modules` 缓存、`sys.path` 搜索顺序、`spec_from_file_location` API。

---

## 四、工具系统的扩展 — SearchKBTool 的工厂模式

### 4.1 为什么用工厂函数

[search_kb_tool.py](tools/search_kb_tool.py) 不直接定义类，而是：

```python
def create_search_kb_tool(Tool, store):
    class SearchKBTool(Tool):
        ...
    return SearchKBTool(store)
```

原因：`Tool` 基类来自 Day 5，`ChromaDBStore` 来自 Day 6。如果子模块内部自己做 import，又会出现 interfaces.py 冲突。工厂函数把"依赖解析"推迟到调用方——这是**依赖注入**在模块加载层面的应用。

### 4.2 Agent 的 6 个工具一览

| 工具 | 来源 | 用途 | 存储后端 |
|------|------|------|---------|
| `weather` | Day 4 | 查天气（mock） | 无 |
| `search` | Day 4 | 网络搜索（mock） | 无 |
| `search_kb` | **Day 7 NEW** | 检索知识库文档 | ChromaDBStore |
| `save_fact` | Day 5 | 保存用户偏好 | InMemoryVectorStore |
| `recall_fact` | Day 5 | 查询用户记忆 | InMemoryVectorStore |
| `scratchpad` | Day 5 | 工作暂存器 | WorkingMemory |

6 个工具，3 个来源，2 个存储后端——全部通过统一的 `ToolRegistry` 管理。

---

## 五、检索策略 — "先 KB 后网络"的调度逻辑

### 5.1 System Prompt 的角色

```python
SYSTEM_PROMPT = """规则：
1. 知识检索优先: 先用 search_kb 检索知识库
2. 网络搜索兜底: search_kb 返回空时，才用 search 搜索网络
...
"""
```

策略在 prompt 里，执行在 loop 里——不需要硬编码 `if kb_empty: web_search()`。

### 5.2 实际调度流程分析（Demo Turn 2）

```
User: "GIL 是什么？"

Step 1 — LLM 判断: 这是技术概念问题
Step 2 — Action: search_kb("Python GIL 全局解释器锁")
Step 3 — Observation: 返回 python_gil.md 的 chunk
Step 4 — Action: recall_fact("用户偏好 回答风格")
Step 5 — Observation: 空（Jaccard 没匹配到）
Step 6 — LLM 判断: KB 已返回内容 → 不需要调 search
Step 7 — Answer: 基于 KB 结果详细回答
```

### 5.3 这就是 ReAct 的强大之处

如果以后扩展到 1000 篇文档，或加入 PDF 解析，`search_kb` 实现会变复杂，但 Agent 的调度逻辑不需要改——LLM 仍然根据 system prompt 和工具描述自主选择调用顺序。

---

## 六、端到端数据流

### 6.1 离线阶段：索引文档

```
documents/*.md
  → chunker.chunk_documents() → list[DocumentChunk]
  → ChromaDBStore.add_chunks() → chroma_data/ (磁盘持久化)
```

### 6.2 在线阶段：多轮对话

```
User Input
  → MemoryManager.pre_process()
    → ShortTermMemory.add(user_msg)     追加到对话历史
    → check token → summarize?          必要时压缩
    → LongTermMemory.searchRelevant()   检索用户偏好 → 注入 system prompt
  → ReAct Loop:
    → LLM 决策 → search_kb/recall_fact/save_fact/weather/search/scratchpad
    → 工具执行 → 结果反馈
    → finish_reason="stop" → 最终答案
  → MemoryManager.finalize()
    → LongTermMemory.extract_facts()    LLM 提取本轮事实
    → WorkingMemory.end_task()
```

### 6.3 Demo 三回合解析

**Turn 1 — 建立画像**: save_fact × 2 + finalize LLM 提取 → 6 条长期记忆

**Turn 2 — KB 检索**: search_kb 返回 python_gil.md → recall_fact 找到"喜欢简洁" → 给出简洁版 GIL 解释

**Turn 3 — 个性化推荐**: recall_fact 找到"Python 后端" + search_kb 返回并发文档 → 推荐 asyncio（精确匹配用户背景）

Turn 3 最能体现"记忆 + 检索"的协同价值——没有 recall_fact，Agent 不知道用户是后端；没有 search_kb，只能泛泛而谈。两者结合，精准推荐。

---

## 七、Day 7 核心概念清单

1. **系统集成**: 独立模块组合为完整系统，解决接口适配和生命周期协调
2. **双向量存储**: InMemoryVectorStore (用户事实) + ChromaDBStore (文档索引) 各司其职
3. **importlib 模块加载**: `sys.modules` 缓存、`sys.path` 搜索、`spec_from_file_location` API
4. **命名冲突解决**: 两个 `interfaces.py` 通过 sys.path 隔离 + 独立模块名解决
5. **工厂模式 + 依赖注入**: 推迟对象创建到调用方，避免子模块自己做跨目录导入
6. **Prompt 驱动调度**: 检索策略在 system prompt 中，LLM 自主决策工具调用顺序
7. **记忆 + RAG 协同**: recall_fact (用户画像) + search_kb (文档检索) 组合出个性化答案
8. **工具系统扩展性**: 新增 search_kb 不改 Agent 循环，只改 ToolRegistry
9. **离线/在线分离**: 文档索引离线完成，在线阶段只做检索和生成

---

## 八、Week 1 总结：你构建了什么

从 Day 1 到 Day 7 的进阶路径：

```
Day 1 ─── LLM 调用 (API basics)
Day 2 ─── Tool Use (function calling)
Day 3 ─── ReAct Loop (thought → action → observation)
Day 4 ─── Tool System (abstract Tool + ToolRegistry)
Day 5 ─── Memory System (short/long/working, IVectorStore)
Day 6 ─── RAG System (chunking → embedding → ChromaDB)
Day 7 ─── Integration (6 tools + 2 stores + 3-tier memory)
```

你现在掌握的能力：

- 从零写出 ReAct Agent（不需要框架）
- 设计可扩展的工具系统（Tool 抽象类 + ToolRegistry）
- 实现三层记忆架构（Token 管理 + 摘要压缩 + 向量存储 + 事实提取）
- 搭建完整 RAG pipeline（分块 → embedding → 检索 → 增强生成）
- 组合独立模块为统一 Agent（importlib 跨目录加载 + 双 Store 并行）

Week 2 将进入工程化——MCP 协议、多 Agent 协作、评估系统、生产化部署。但核心原理你已经在 Week 1 全部手写过一遍了。
