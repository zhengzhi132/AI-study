"""Day 5 记忆系统演示 — 三层记忆（短期/长期/工作）多对话展示."""

import asyncio
import os
from openai import OpenAI
from in_memory_vector_store import InMemoryVectorStore
from memory_manager import MemoryManager
from interfaces import SummarizationConfig
from react_agent import react_agent, build_day5_registry

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY", "sk-xx"),
)

MODEL = "deepseek-v4-flash"


async def main():
    vector_store = InMemoryVectorStore()
    memory_manager = MemoryManager(
        client=client,
        model=MODEL,
        vector_store=vector_store,
        short_term_config=SummarizationConfig(
            max_tokens=4000,
            keep_last_n=6,
            system_prompt_tokens=500,
        ),
    )

    print("\n🧠 Day 5 — 记忆系统演示（Python 版）\n")

    # ============================================================
    # 对话 1: 建立用户画像
    # ============================================================
    await react_agent(
        client,
        MODEL,
        memory_manager,
        build_day5_registry(memory_manager),
        "你好！我叫小明，我是一名TypeScript全栈工程师，住在北京。"
        "我喜欢户外徒步和骑行。我不喜欢太啰嗦的回答，请保持简洁。",
        "对话 1: 建立用户画像（长期记忆提取）",
    )

    # ============================================================
    # 对话 2: 记忆召回 → 个性化推荐
    # ============================================================
    await react_agent(
        client,
        MODEL,
        memory_manager,
        build_day5_registry(memory_manager),
        "推荐一个这周末的活动吧",
        "对话 2: 模糊提问 → 长期记忆召回 → 个性化推荐",
    )

    # ============================================================
    # 对话 3: 工作记忆 — scratchpad 暂存
    # ============================================================
    await react_agent(
        client,
        MODEL,
        memory_manager,
        build_day5_registry(memory_manager),
        "帮我规划一个周末行程：先查北京天气，然后搜户外景点，"
        "最后用 scratchpad 暂存每个结果，全部查完后统一给我推荐方案。",
        "对话 3: 工作记忆 — 多步骤任务 + scratchpad",
    )

    # ============================================================
    # 最终统计
    # ============================================================
    print(f"\n{'=' * 60}")
    print("📊 记忆系统最终状态")
    print(f"{'=' * 60}")

    stats = memory_manager.stats()

    print(f"\n短期记忆:")
    print(f"  消息数: {stats['short_term']['message_count']}")
    print(f"  估算 tokens: {stats['short_term']['estimated_tokens']}")
    print(f"  已压缩: {stats['short_term']['is_compressed']}")

    print(f"\n长期记忆 ({stats['long_term']['total_facts']} 条):")
    facts = await memory_manager.long_term.get_all_facts()
    for fact in facts:
        print(f"  - {fact.content} (source: {fact.source})")

    print(f"\n工作记忆:")
    print(f"  条目数: {stats['working']['entry_count']}")
    print(f"  task: {stats['working']['task_id'] or '(无)'}")

    print(f"\n✅ Day 5 记忆系统演示完成！")


if __name__ == "__main__":
    asyncio.run(main())
