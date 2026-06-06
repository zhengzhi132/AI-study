# Day 5 — 记忆系统 知识点详解

> 对照 [demo.py](demo.py) 运行结果阅读，理解 Agent 记忆系统的三层架构。

---

## 一、为什么 Agent 需要记忆？

普通 LLM 调用是**无状态的**——每次请求都像第一次见面。Day 1-4 的 Agent 虽然能调用工具，但：

- 关掉程序再开，它完全不记得你是谁
- 对话太长，消息数组塞爆上下文窗口，直接报错
- 多步任务中，中间结果散落在消息列表里，难以追踪

记忆系统解决的就是这三个问题，对应三种记忆类型：

| 记忆类型 | 解决问题 | 生命周期 | 文件 |
|---------|---------|---------|------|
| 短期记忆 | 上下文窗口溢出 | 单次会话 | [short_term_memory.py](short_term_memory.py) |
| 长期记忆 | 跨会话遗忘 | 持久化 | [long_term_memory.py](long_term_memory.py) |
| 工作记忆 | 中间状态追踪 | 单次任务 | [working_memory.py](working_memory.py) |

---

## 二、短期记忆（ShortTermMemory）

**文件**: [short_term_memory.py](short_term_memory.py)

### 知识点 1: Token 估算

```python
def estimate_tokens(self) -> int:
    chars = sum(self._msg_chars(m) for m in self._messages)
    return (chars + 3) // 4  # ceil(chars / 4)
```

**原理**: LLM 用 token（词元）计费和控制上下文。通用的经验法则是：**英文约 4 字符 = 1 token，中文约 1.5-2 字符 = 1 token**。`chars/4` 是保守近似（对中文会低估，但低估意味着提前触发压缩，更安全）。

真实生产环境用 `tiktoken` 库做精确计数，但我们有意不引入额外依赖。

### 知识点 2: 滑动窗口 + 摘要压缩

这是 ChatGPT、Claude 管理超长对话的核心策略：

```
压缩前:
  [system] [msg1] [msg2] [msg3] [msg4] [msg5] [msg6] [msg7] [msg8]
            |___________可压缩窗口___________|  |___保留最近N条___|

压缩后:
  [system] [📝摘要消息] [msg7] [msg8]
```

关键参数（[interfaces.py](interfaces.py) `SummarizationConfig`）：
- `max_tokens` (默认 4000): 超过此值触发压缩
- `keep_last_n` (默认 6): 最近 N 条消息不参与压缩，保证当前对话连贯
- `system_prompt_tokens` (默认 500): 预留给 system prompt 的空间

**为什么用 LLM 做摘要而不是简单截断？** 截断丢失中间上下文——用户第 3 轮说的偏好，第 10 轮还要用。LLM 摘要压缩能提取关键信息（"用户喜欢骑行，住在北京"），丢弃噪声（"好的"、"谢谢"）。

### 知识点 3: 优雅降级

```python
try:
    response = client.chat.completions.create(...)
    summary = response.choices[0].message.content
except Exception:
    summary = "(摘要生成失败)"
```

如果摘要 API 调用失败，不崩溃，用截断策略继续。这是 Agent 可靠性的基本原则：**每个非核心路径都应有 fallback**。

---

## 三、长期记忆（LongTermMemory）

**文件**: [long_term_memory.py](long_term_memory.py), [in_memory_vector_store.py](in_memory_vector_store.py)

### 知识点 4: 向量存储的抽象（IVectorStore）— ABC 名义子类型

**文件**: [interfaces.py](interfaces.py) `IVectorStore` 类

这是**依赖倒置原则（Dependency Inversion Principle, DIP）** 的直接应用：

```python
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
```

**设计意图逐方法分析**:

| 方法 | 返回值 | 设计考量 |
|------|--------|---------|
| `add(entry) → str` | 新条目的 ID | 调用方需要 ID 做后续操作；ID 由存储层生成，调用方不关心生成逻辑 |
| `search(query, top_k) → list[...]` | 带 score 的排序结果 | `top_k` 由调用方按场景控制：注入 prompt 用 3 条，recall_fact 工具用 5 条 |
| `delete(id) → None` | 无 | 宽松契约：ID 不存在时静默忽略，不抛异常 |
| `size() → int` | 计数 | 同步方法——暗示实现者这不需要 I/O |
| `get_all() → list[...]` | 全量 | 与 `search()` 语义分离：前者用于管理操作，后者用于语义检索 |

```
LongTermMemory ──依赖──▶ IVectorStore (抽象/契约)
                              ▲
                              │ 实现
              ┌───────────────┼───────────────┐
              │               │               │
    InMemoryVectorStore  ChromaDBStore   FAISSStore
    (关键词Jaccard)    (真实embedding)  (真实embedding)
```

#### 深入：ABC（名义子类型）vs Protocol（结构化子类型）

`IVectorStore` 用 `ABC` 定义接口。Python 还有另一种方式：`typing.Protocol`（Python 3.8+），即**结构化子类型**，也叫"静态鸭子类型"。

核心区别：**ABC 要求显式继承，Protocol 只要求方法签名匹配**。

```python
from typing import Protocol

class VectorStore(Protocol):
    async def add(self, entry: MemoryEntry) -> str: ...
    async def search(self, query: str, top_k: int) -> list[MemorySearchResult]: ...

# 没有继承任何东西——但 mypy 认可它"是" VectorStore
class InMemoryStore:
    async def add(self, entry: MemoryEntry) -> str: ...
    async def search(self, query: str, top_k: int) -> list[MemorySearchResult]: ...

store: VectorStore = InMemoryStore()  # ✅ 类型检查通过
```

| | ABC (`abc.ABC`) | Protocol (`typing.Protocol`) |
|---|---|---|
| 关系建立 | **显式继承** `class X(ABCBase)` | **隐式匹配**，有相同签名即符合 |
| 运行时 `isinstance` | 正常可用 | 默认报错 `TypeError`，需 `@runtime_checkable` |
| 实例化未实现子类时 | 立即报错 `Can't instantiate abstract class` | 不报错，只在类型检查时报错 |
| 哲学 | "你必须是这个家族的人"（名义子类型） | "你会做这些事就够了"（结构化子类型） |
| 适合场景 | 框架基础设施、自己的接口体系 | 描述第三方库的接口、轻量契约 |

Protocol 想表达的是：关系存在于**结构相似性**中，不在对象图里。它本质上是一张**给类型检查器看的声明便签**——"如果某个类有这些方法签名，把它当作这个类型来对待"。它和实现者之间没有任何运行时链接（没有 `__mro__` 条目、没有 `__subclasses__` 记录）。

如果想让 Protocol 在运行时也能 `isinstance`，需要 `@runtime_checkable`——它只检查方法存不存在，不看签名细节：

```python
from typing import runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None: ...

assert isinstance(Circle(), Drawable)  # ✅ True
```

**为什么 Day 5 选 ABC？**
1. 核心基础设施 — 自己的接口体系，不是和第三方对接
2. 可能需要运行时验证依赖
3. ABC 在实例化时就报错 `Can't instantiate abstract class`，比运行时 `AttributeError` 更友好

### 知识点 5: Jaccard 相似度（无 embedding 的检索方案）

```python
def _jaccard(a: list[str], b: list[str]) -> float:
    set_a, set_b = set(a), set(b)
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0
```

**Jaccard 系数 = 交集 / 并集**。值域 [0, 1]，越大越相似。

示例计算：
```
query: "户外活动推荐"        → 关键词: {户外, 活动, 推荐}
entry: "用户喜欢户外徒步和骑行" → 关键词: {用户, 喜欢, 户外, 徒步, 骑行}

交集 = {户外}         (1个)
并集 = {用户, 喜欢, 户外, 徒步, 骑行, 活动, 推荐}  (7个)
Jaccard = 1/7 ≈ 0.14  ← 低分
```

这就是 demo 对话 2 中 `recall_fact("用户所在城市")` 召回效果不佳的原因——关键词重叠太少。

#### 深入：Embedding 向量为什么能捕获语义相似性

Jaccard 的根本局限在于**只看字符重叠**。同一个意思换一种说法就完全不匹配：

```
A: "我昨天买了一辆汽车"     → 关键词: {昨天, 买了, 一辆, 汽车}
B: "我昨天购买了一台轿车"   → 关键词: {昨天, 购买, 一台, 轿车}

Jaccard(A, B) = {"昨天"} / 7 = 0.14  ← 低分，但意思完全相同
```

"买"≠"购买"、"汽车"≠"轿车"——字符层面不重叠，得分为 0。

**Embedding 模型把文字映射到高维向量空间中的坐标**。空间中距离近 = 语义相似：

```
"汽车" → [0.23, -0.45, 0.78, ..., 0.12]
"轿车" → [0.21, -0.43, 0.80, ..., 0.15]   ← 距离很近
"香蕉" → [-0.67, 0.31, -0.22, ..., -0.88] ← 距离很远
```

这不是查词典匹配出来的，而是模型从训练中学到的：**"汽车"和"轿车"在无数真实文本中出现在相同的上下文位置，模型被训练得把它们在向量空间中拉近**。

"我最近很沮丧"和"心情不太好"——字符层面零重叠，但 embedding 能识别出情绪语义相近。

#### 深入：Embedding 映射的训练过程

**这个映射是如何建立起来的？** 不是人工编写规则，而是通过自监督训练从大规模数据中学出来的。

**第一代：Word2Vec（2013）— "一个词的含义由它周围的词决定"**

```
训练语料: "我 今天 开 汽车 去 上班"

以"汽车"为中心，窗口=2：
  输入上下文: ("开", "去")
  预测目标:   "汽车"
```

模型是浅层神经网络。训练完毕后丢弃输出层，取隐藏层权重矩阵 `W_input` 的每一行——那就是每个词的 embedding 向量。为什么"汽车"和"轿车"接近？因为训练中它们总是出现在相同的上下文窗口里（"开 _ 去"），梯度下降迫使它们产生相似的隐藏层激活值。

**第二代：BERT（2018）— 上下文相关的向量**

Word2Vec 缺陷："银行"只有一个固定向量，不管在"存钱"还是"河岸"的语境。BERT 用 **Masked Language Model** + Transformer 自注意力：

```
输入:  "我 昨天 [MASK] 了一辆汽车" → 预测 [MASK] = "买"
```

整个句子过 Transformer，每个词都融入了周围词的上下文。同一个"银行"在不同句子里算出的向量完全不同。

**第三代：对比学习（2020s）— 专为相似度搜索优化**

现代 embedding 模型（OpenAI `text-embedding-3`、BGE 等）直接以相似度为目标训练：

```
训练数据: (query, 正例, 负例)
query:  "如何在Python中读取文件"
正例:   "使用open()函数打开文件并读取内容"
负例:   "巴西在2022年世界杯中止步八强"

训练目标（InfoNCE Loss）:
  cos_sim(query, 正例) → 尽可能接近 1
  cos_sim(query, 负例) → 尽可能接近 0
```

数十亿对训练后，编码器权重固定。推理时就是执行这个固定映射：`"汽车" → 编码器 → [0.23, -0.45, ...]`。

**这个局限本身就是教学内容——让你理解 embedding 的必要性。** Jaccard 先搭骨架，Day 6 用真实 embedding 替换即可体验语义搜索的威力。

### 知识点 6: 中文分词与停用词

**文件**: [in_memory_vector_store.py](in_memory_vector_store.py)

```python
STOP_WORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", ...
    "a", "an", "the", "is", "are", "was", ...
}

_SEGMENT_RE = re.compile(r"[\s，,。.！!？?：:；;、()（）\[\]《》<>...]+")
```

**为什么需要分词？** 英文天然有空格分词，中文则没有。"我喜欢户外徒步"没有标点时就变成一整坨 `["我喜欢户外徒步"]`。向量检索需要把文本拆成可比较的单元。

**为什么需要停用词？** 高频但无语义价值的词（"的"、"是"、"the"、"a"）必须过滤——两段无关文本可能因为都包含"的"而获得假的高相似度。

#### 深入：专业分词器的内部机制（词典 + 统计模型）

Day 5 用正则分词是零依赖简化。真实系统用 jieba 等专业分词器，其核心是三层策略叠加。

**第一层：前缀词典 → 最大概率路径**

jieba 内置前缀词典（数百万词条 + 词频），先找出所有可能的切法构成有向无环图（DAG）：

```
句子: "我们研究生物学"

所有可能的边:
  0→1(我), 0→2(我们)
  1→2(们)
  2→3(研), 2→4(研究)
  3→4(究)
  4→6(生物), 4→7(生物学)
```

动态规划找最大概率路径：

```
路径A: 我们/研究/生物学  →  logP(我们) + logP(研究) + logP(生物学) = -24.6
路径B: 我/们/研究/生/物/学  →  ... = -69.7

路径A 胜出 → ["我们", "研究", "生物学"]
```

单字的词频极低，推动模型倾向组合成高频常见词。

**第二层：HMM（隐马尔可夫模型）— 处理词典没有的词**

人名、地名、新词不可能全在词典里。HMM 做**字位标注**，把每个字标记为四种状态：B (Begin, 词首)、M (Middle, 词中)、E (End, 词尾)、S (Single, 单字成词)。

从已分词语料统计转移概率和发射概率，然后用 Viterbi 动态规划找全局最优状态序列：

```
输入: "小明"（词典未登录）

转移概率: B→E: 0.62, B→M: 0.38, S→S: 0.21, ...
发射概率: P(字="明"|状态=E) 很高（"说明""光明""小明"中"明"常做词尾）
         P(字="小"|状态=B) 很高（"小明""小心"中"小"常做词首）

候选状态序列:
  序列1: B(小) → E(明)    得分高 → "小明"是一个词
  序列2: S(小) → S(明)    得分低
  序列3: B(小) → M(明)    不可能（两字序列末尾不能是M）

序列1 胜出 → ["小明"]
```

**三层协作关系**：

```
输入: "小明在微软写代码"

1. 前缀词典扫描 → 识别已知: [???] [在] [微软] [写] [代码]
2. HMM 处理未登录 "小明" → "小|B 明|E" → "小明"是一个词
3. 最终输出: ["小明", "在", "微软", "写", "代码"]
```

### 知识点 7: LLM 事实提取（Extract-Facts Pattern）

```python
async def extract_facts_from_conversation(self, client, model, messages):
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": "你是一个知识提取器。"
                   "只提取可跨会话保留的长期信息。以 JSON 数组格式返回。"
                   "只返回 JSON 数组，不要其他文字。"},
                  {"role": "user", "content": text}],
    )
    facts = json.loads(response.choices[0].message.content)
```

**这是 Agent 记忆的核心模式**: 不让用户手动管理记忆，而是对话结束后自动"反思"提取重要信息。

类比人类记忆：
- 短期记忆 = 海马体（维持当前对话）
- 长期记忆提取 = 睡眠中的记忆巩固（海马体 → 皮层）

**Prompt 工程要点**:
- `"只返回 JSON 数组，不要其他文字"` → 防止 LLM 加"好的，以下是提取的事实："前缀导致 `json.loads` 解析失败
- `"只提取可跨会话保留的长期信息"` → 防止提取"用户这次问了天气"这种临时信息
- 代码里还有 `json_str.replace("```json", "").replace("```", "")` 的防御性清洗

### 知识点 8: 去重策略 — 向量存储的固有难题

```python
existing = await self._store.search(content, 1)
if existing and existing[0].score > 0.8:
    return existing[0].entry.id  # 跳过，不重复添加
```

Demo 结果显示 8 条记忆而非 4 条——Jaccard 对语义相近但字面不同的文本（"用户叫小明" vs "用户名叫小明"）判断不准确。

#### 深入：为什么 embedding 也不能完美去重

核心矛盾在于 **相似 ≠ 重复**：

```
A: "我在北京工作，做后端开发"
B: "我从事后端开发，工作地点在北京"      ← 真正重复
C: "我在上海做前端开发，但想转后端"      ← 相关但不是重复
```

Embedding 余弦相似度：`cos_sim(A, B) = 0.96`（确实应该去重），但 `cos_sim(A, C) = 0.87` 也很高——A 和 C 共享"工作""开发""后端"这些语义锚点。反过来的情况也存在：

```
D: "今天心情特别好"
E: "今日情绪非常愉悦"     ← 重复但短文本+字面差异大，可能只有 0.72
```

去重需要**明确的二值边界**（重复/不重复），但 embedding 只能给出**连续梯度**。不存在一个能完美分离两者的阈值：

```
0.0 ─── 0.5 ─── 0.7 ─── 0.85 ─── 1.0
无关    略相关   相似    高度相似   完全一致
                    ↑ 阈值设在哪？没有完美位置
```

- 设 0.95：漏掉换说法的重复 → 记忆库膨胀
- 设 0.80：误杀相关但不同的事 → 信息丢失
- 设 0.90：两边的错误都有

工程中的折中方案：

| 方案 | 思路 | 代价 |
|------|------|------|
| 高阈值 embedding | 设 0.95+ | 漏掉换说法的重复 |
| embedding + 关键词交集 | 两关都过才算 | 双倍阈值调试 |
| 内容哈希 | 规范化文本做 hash | 换一个字就失效 |
| LLM 判断 | 逐对判断是否有重叠 | 贵、慢、O(N²) |
| 不去重靠降权 | 检索惩罚冗余 | 记忆库膨胀 |

Day 5 的 `InMemoryVectorStore.add` 直接 `append` 不做去重——这也是 demo 最终有 8 条记录的原因，多条同一事实以不同表述重复入库。

---

## 四、工作记忆（WorkingMemory）

**文件**: [working_memory.py](working_memory.py)

### 知识点 9: 暂存器模式（Scratchpad Pattern）

```python
class WorkingMemory:
    def set(self, key, value): ...    # 写入命名槽位
    def get(self, key): ...           # 读取指定槽位
    def start_task(self, task_id):    # 新任务自动清空
```

本质是一个带生命周期的 `dict`。但它解决的问题很关键：

**没有工作记忆时**: Agent 调了 5 个工具，结果散落在 messages 的 `tool` role 消息中。最后一步整合时，LLM 要扫描整个历史去找，容易遗漏。

**有工作记忆时**: 每步结果显式写入命名槽位，最后一步直接读取。

这对应了人类认知中的 working memory——你在心算 `23 × 47` 时，脑中暂存 `23×40=920` 和 `23×7=161`，最后求和 `920+161=1081`。每个中间值就是 scratchpad 的一个 entry。

#### 深入：工作记忆的三个核心机制

**机制 1 — Agent 主动控制写入**

Agent 通过 `scratchpad` 工具**自己决定**什么时候存、存什么、叫什么名：

```
Turn 2: scratchpad("weather_data", "北京, 25°C, 晴天, 适合户外")
Turn 4: scratchpad("search_results", "八达岭长城(推荐)/香山/颐和园")
```

不是自动的，是 Agent 在推理中主动决策的。

**机制 2 — 每轮注入系统提示词**

[memory_manager.py](day5-memory-system/memory_manager.py#L67-L69) 每轮调用 `_build_augmented_system_prompt`，把当前工作记忆全部内容注入系统提示词：

```
[系统提示词]

[长期记忆召回]

[工作记忆 - 当前任务]
task: 规划周末行程
  weather_data: 北京, 6月7日, 晴天, 25°C, 适合户外
  search_results: 八达岭长城(推荐)/香山/颐和园

[对话历史...]
```

Agent 每一轮都能在系统提示词开头看到最新状态——**不需要回去翻消息历史**。

**机制 3 — 压缩与可覆盖**

Agent 不存原始返回（可能 2000 token），而写自己摘要后的版本（一行）。后续步骤只看到这一行。也可以覆盖更新：`scratchpad("weather_data", "更正：26°C有雨")`。

#### 工作记忆 vs 直接靠对话历史

| | 对话历史中的 tool result | 工作记忆命名槽位 |
|---|---|---|
| 位置 | 散落在多条消息里 | 集中注入系统提示词开头 |
| 结构 | 原始返回数据，可能很长 | Agent 自己摘要后写入 |
| token 开销 | 每条 tool result 都占空间 | 只有最新版本的一句 |
| 可修改 | 不可修改（历史固定） | 可以覆盖更新 |

#### 在 demo 对话 3 中的完整链路

```
用户: "先查天气，搜户外景点，用 scratchpad 暂存，最后推荐方案"

Turn 1: Agent 调用 get_weather("北京")
Turn 2: Agent 调用 scratchpad("weather_data", "25°C晴天")
Turn 3: Agent 调用 search("北京户外景点")  
Turn 4: Agent 调用 scratchpad("search_results", "八达岭长城...")
Turn 5: Agent 看到系统提示词中已有完整天气+景点数据 → stop (输出方案)
```

### 知识点 10: 三种记忆的生命周期对比

| 记忆类型 | 生命周期 | 清空时机 | 人类类比 |
|---------|---------|---------|---------|
| 工作记忆 | 单次任务 | `finalize()` 调用 `end_task()` | 脑子里正在想的事 |
| 短期记忆 | 单次会话 | `initialize()` 调用 `clear()` | 这次聊天记得的事 |
| 长期记忆 | 跨会话持久化 | 手动 `delete_fact()` | 你一直记得的事 |

---

## 五、架构设计模式

### 知识点 11: MemoryManager 编排器模式（Facade Pattern）

**文件**: [memory_manager.py](memory_manager.py)

```python
class MemoryManager:
    async def initialize(self, system_prompt): ...  # 会话开始时
    async def pre_process(self, user_input): ...     # 每次 LLM 调用前
    def post_process(self, response): ...            # 每次 LLM 调用后
    async def finalize(self): ...                    # 会话结束时
```

Agent 不需要知道三种记忆的内部实现，只调用这四个钩子。这是**门面模式（Facade）**——隐藏子系统复杂度，暴露简单接口。

**核心方法 `_build_augmented_system_prompt()`**: 把长期记忆 + 工作记忆动态注入 system prompt，让 LLM 无感知地"拥有记忆"：

```
[原始 system prompt]

[关于用户的长期记忆]
- 用户喜欢户外徒步和骑行
- 用户住在北京

[工作记忆 - 当前任务]
  weather_data: 晴，25°C
  search_results: 故宫、长城、颐和园
```

### 知识点 12: 记忆工具 — Agent 驱动的记忆管理

**文件**: [tools/save_fact_tool.py](tools/save_fact_tool.py), [tools/recall_fact_tool.py](tools/recall_fact_tool.py), [tools/scratchpad_tool.py](tools/scratchpad_tool.py)

两种记忆管理模式并存：

| 模式 | 触发方式 | 时机 | 例子 |
|------|---------|------|------|
| 自动（隐式） | `finalize()` 钩子 | 对话结束 | LLM 扫描全文，提取事实存为 JSON |
| 工具驱动（显式） | Agent 调用 memory tools | 对话中途 | Agent 意识到重要信息，主动 `save_fact` |

工具驱动模式的优势：对话中途保存关键信息，不等到 finalize。两种模式互补——自动提取兜底，显式工具精准控制。

这些工具遵循 Day 4 的 `Tool` 抽象基类模式，注册到 `ToolRegistry`，Agent 像调用天气/搜索一样调用它们。

---

## 六、Day 5 完整知识地图

```
Agent 记忆系统
├── 短期记忆（会话级）
│   ├── Token 估算（字符/4 近似）
│   ├── 滑动窗口 + LLM 摘要压缩
│   └── 优雅降级（摘要失败→截断）
├── 长期记忆（跨会话级）
│   ├── IVectorStore 抽象
│   │   ├── ABC 名义子类型（显式继承，运行时检查）
│   │   └── Protocol 结构化子类型（签名匹配，静态检查）
│   ├── 检索算法
│   │   ├── Jaccard 关键词（字符重叠 → 零语义理解）
│   │   └── Embedding 语义向量
│   │       ├── 原理：语义相似 = 向量空间近邻
│   │       └── 训练：Word2Vec → BERT → 对比学习
│   ├── 中文处理
│   │   ├── 正则分词（零依赖简化）
│   │   ├── jieba 分词（前缀词典 DAG + HMM + Viterbi）
│   │   └── 停用词过滤
│   ├── LLM 事实提取（Extract-Facts 模式）
│   └── 去重策略
│       ├── 核心矛盾：相似 ≠ 重复
│       └── 工程折中：高阈值 / 多条件 / LLM 判断 / 不去重
├── 工作记忆（任务级）
│   ├── 命名槽位（Agent 主动读写）
│   ├── 每轮系统提示词注入
│   ├── 压缩与覆盖（vs 对话历史的不可变性）
│   └── 任务生命周期（start_task / end_task）
└── MemoryManager（门面模式）
    ├── initialize / pre_process / post_process / finalize
    └── 记忆工具（save_fact / recall_fact / scratchpad）
```

---

## 七、Demo 运行结果解读

运行 `python day5-memory-system/demo.py`，观察三阶段输出：

**对话 1 — 建立画像**: Agent 主动调用 `save_fact` 4 次，保存"用户叫小明"、"住北京"等事实。`finalize()` 时 LLM 又自动提取 4 条，最终 8 条记忆（有重复 → Jaccard 去重不完美）。

**对话 2 — 记忆召回**: Agent 调用 `recall_fact("用户所在城市")` → Jaccard 得分低，返回空 → Agent 回答"我还不太了解你"。这**不是 bug，是教学点**：关键词匹配的固有局限，Day 6 用 embedding 解决。

**对话 3 — 工作记忆**: Agent 按指令依次调用 `get_weather` → `scratchpad(write)` → `search` → `scratchpad(write)` → `scratchpad(list)` → 整合输出详细行程方案。完美展示工作记忆在多步任务中的作用。

---

## 八、下一步（Day 6 预告）

Day 6 RAG 将解决 Day 5 的核心局限：

- **text-embedding-3-small** 替代 Jaccard：真正的语义向量
- **ChromaDB** 替代 InMemoryVectorStore：只需实现 `IVectorStore` 接口
- **文档分块（chunking）**: 怎么切文档才不丢上下文
- **完整的 RAG 问答 pipeline**: embed → 检索 → 增强 prompt → 生成回答
