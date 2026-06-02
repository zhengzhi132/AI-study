# AI Agent 开发工程师 — 2 周速成学习路径

> 前提假设：你已有 Python 基础和 CS 专业背景（厦大），每天可投入 8-10 小时全职学习。

---

## 总体思路

**不学框架先学原理。** Agent 开发的核心不是 LangChain/CrewAI 这些框架，而是：LLM 调用 → 工具使用 → 推理循环 → 记忆管理。框架只是封装，原理懂了框架自然上手快。

每天分三个块：
- **上午（3h）**：学概念 + 读论文/文档
- **下午（3h）**：动手写代码
- **晚上（2h）**：复盘 + 整理到 GitHub

---

## 第一周：Agent 核心原理

### Day 1 — LLM API 基础
- Anthropic Messages API（Claude）和 OpenAI Chat Completions API 各写一个 demo
- 理解 system prompt vs user message、temperature、max_tokens、stop sequences
- **写**：一个简单的命令行对话脚本，支持多轮对话
- **读**：[Anthropic API 文档](https://docs.anthropic.com/en/docs) 的 Messages 部分

### Day 2 — Function Calling / Tool Use
- 这是 Agent 的灵魂。理解 tool definition schema、tool_choice 参数
- **写**：定义一个 `get_weather(city)` 工具，让 LLM 自动决定何时调用它并解析结果
- **写**：扩展为 3 个工具（天气、搜索、计算器），LLM 自动路由
- **关键点**：理解 LLM 不执行工具，它只输出"我要调用哪个工具+参数"的 JSON，由你的代码去执行

### Day 3 — ReAct 模式（推理-行动循环）
- **读**：[ReAct 论文](https://arxiv.org/abs/2210.03629)（看前 5 页即可，理解 Thought-Action-Observation 循环）
- **写**：从零实现一个 ReAct Agent，不借助任何框架
  ```
  Loop:
    1. LLM 输出 Thought（推理）+ Action（调用哪个工具）
    2. 你的代码执行工具，得到 Observation
    3. Observation 喂回 LLM，继续循环
    4. 直到 LLM 输出 Final Answer
  ```
- **测试场景**："北京今天的天气适合户外运动吗？如果适合，推荐 3 个活动"

### Day 4 — 工具系统深化
- **写**：一个可扩展的工具注册系统（ToolRegistry）
  - 支持动态添加/移除工具
  - 工具描述自动生成 schema
  - 工具执行错误处理与重试
- **写**：实现一个 `code_executor` 工具（用 subprocess 安全沙箱执行 Python）
- **注意**：沙箱安全，限制 import、限制执行时间、限制内存

### Day 5 — 记忆系统
- 三种记忆类型：
  - **短期记忆**：对话历史（就是 message list）
  - **长期记忆**：向量数据库存储 + 检索（ChromaDB 或 FAISS）
  - **工作记忆**：Agent 当前任务的中间状态（scratchpad）
- **写**：给 Day 3 的 ReAct Agent 加上对话历史管理和摘要压缩
- **写**：用 ChromaDB 实现长期记忆，存储用户偏好并在后续对话中检索

### Day 6 — RAG 基础
- **理解**：Embedding → 向量检索 → 上下文增强 的完整链路
- **写**：一个文档问答 Agent
  1. 将 3-5 篇技术文章/PDF 分块（chunking）
  2. 用 text-embedding-3-small 生成 embedding
  3. 存入 ChromaDB
  4. 用户提问时检索相关 chunk，注入 prompt
- **重点**：chunk size 策略（太小缺上下文，太大检索不精确）

### Day 7 — 综合项目：个人知识库 Agent
- 整合前 6 天的所有内容，做一个完整项目：
  - 支持上传 PDF/网页 → 自动分块 → 存入向量库
  - 用户自然语言提问 → ReAct 循环 → 先检索知识库，不够再调用搜索工具
  - 对话记忆 + 长期偏好记忆
- **目标**：一个能真正用起来的个人 AI 助手雏形

---

## 第二周：工程化与进阶

### Day 8 — MCP 协议（Model Context Protocol）
- MCP 是 Anthropic 提出的 Agent-工具标准化协议，目前行业主流方向
- **理解**：MCP Server、MCP Client、Resource/Tool/Prompt 三大原语
- **写**：实现一个 MCP Server（用 Python SDK 或 FastMCP）
  - 提供文件系统操作工具（读/写/列目录）
- **写**：MCP Client 端，让 Claude 通过 MCP 操作你的本地文件

### Day 9 — Agent 框架速览（只看不深究）
- **LangChain**：过一遍 AgentExecutor 源码（重点看 AgentExecutor._take_next_step 方法）
- **LangGraph**：理解状态图（StateGraph）概念，跑一遍官方 Quickstart
- **CrewAI**：理解 Role-Based Agent，跑一个 3-agent 协作 demo
- **别深究**：目的是了解业界方案，知道什么时候用/不用。两周后你会发现 Day 3 手写的更灵活

### Day 10 — 多 Agent 协作
- **模式**：顺序流水线、并行+汇总、辩论（debate）、层级委派
- **写**：一个双 Agent 系统
  - Agent A（研究员）：搜索 + 收集信息
  - Agent B（写手）：基于 Agent A 的产出写报告
  - 两者通过结构化输出（Pydantic 模型）通信
- **写**：扩展为辩论模式 — 两个 Agent 对同一问题给出不同方案，第三个 Agent 评判

### Day 11 — Agent 评估
- Agent 开发的难点不是写代码，是**你不知道它对不对**
- **写**：构建一个 eval 系统
  - 准备 20 个测试用例（包含预期工具调用和预期最终答案）
  - 自动化跑 Agent → 检查工具调用链路是否正确 → 检查最终答案质量
  - 用 LLM-as-Judge 做答案质量评分（给另一个 LLM 打分标准，让它判分）
- **重点**：eval 不是一次性的，每次改 prompt 或工具都要跑

### Day 12 — 结构化输出与可控性
- **Pydantic + Instructor 库**：让 LLM 输出严格符合 schema 的 JSON
- **写**：用 Instructor 改造 Day 3 的 Agent，所有中间输出都用 Pydantic 模型约束
- **写**：实现 Guardrails
  - 输出格式校验 → 格式不对自动重试
  - 内容安全检查 → 敏感内容拒绝执行
  - 预算控制 → token/费用超限自动停止

### Day 13 — 生产化基础
- **Docker 化**：把你的 Agent 打包成 Docker 镜像
- **API 化**：用 FastAPI 把 Agent 封装成 REST API
  - 异步调用（Agent 可能跑很久，需要异步任务队列）
  - 基本的认证（API Key）
- **可观测性**：
  - 日志记录每次 LLM 调用（prompt、response、token 数、延迟）
  - 用 Langfuse 或自定义方案追踪 Agent 执行链路

### Day 14 — 终极项目 & 简历输出
- **写**：一个完整的开源项目，建议方向（选一个）：
  1. **代码审查 Agent**：给定 GitHub PR 链接，自动审查代码质量/安全问题，输出 review 报告
  2. **智能客服 Agent**：RAG + 多轮对话 + 工单系统 API 调用，解决用户问题
  3. **数据分析 Agent**：用户自然语言描述需求 → Agent 自动写 SQL/写 Python 分析脚本 → 返回图表
- **输出**：
  - GitHub 仓库（README 完整 + 架构图 + Demo 视频/GIF）
  - 一篇技术博客（掘金/知乎/个人博客）
  - 更新简历项目经历（STAR 法则）

---

## 关键资源清单

| 资源 | 用于 |
|------|------|
| [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook) | Day 1-4 的代码参考，最好的 Agent 学习材料 |
| [Anthropic Agent 文档](https://docs.anthropic.com/en/docs/agents-and-tools) | Agent 设计的官方最佳实践 |
| [MCP 官方文档](https://modelcontextprotocol.io) | Day 8 MCP 协议 |
| [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/) | Day 9 过一遍即可 |
| [ChromaDB](https://docs.trychroma.com/) | Day 5-6 向量存储 |
| [Instructor](https://python.useinstructor.com/) | Day 12 结构化输出 |

---

## 每天必做的习惯

1. **早上 10 分钟**：浏览 [Hacker News](https://news.ycombinator.com/) 和 AI 相关 subreddit，了解行业动态
2. **晚上 15 分钟**：把当天的代码 push 到 GitHub，写 3 句话的 learning log
3. **遇到报错**：先自己 debug 20 分钟，再问 Claude/Google。Debug 能力是最好的老师

---

## 两周后的里程碑

- 能不看文档从零写出一个 ReAct Agent（~200 行）
- 理解 MCP 协议并写过 Server + Client
- 有一个完整的个人 Agent 项目在 GitHub 上
- Agent eval 流程跑通，知道怎么衡量 Agent 质量
- 简历上能写出两个有深度的项目经历

如果两周后还有余力，下一步方向：**Agent 的可靠性工程**（如何在不确定的 LLM 输出上构建确定性系统），这是目前业界最前沿也最缺人的方向。
