"""增强版 ReAct Agent — 集成三层记忆系统."""

import json
from openai import OpenAI
from tool import ToolRegistry
from memory_manager import MemoryManager
from tools.weather_tool import WeatherTool
from tools.search_tool import SearchTool
from tools.save_fact_tool import SaveFactTool
from tools.recall_fact_tool import RecallFactTool
from tools.scratchpad_tool import ScratchpadTool

BASE_SYSTEM_PROMPT = """你是一个有记忆能力的 AI 助手。重要规则：
1. 天气信息必须通过 get_weather 获取，不要编造
2. 推荐具体地点、活动、景点时，必须通过 search 搜索
3. 只有拿到所有工具返回的数据后，才能给出最终答案
4. 当你了解到用户的重要偏好或个人信息时，使用 save_fact 工具保存到长期记忆
5. 在回答个性化问题前，先用 recall_fact 工具检查是否有相关的用户记忆
6. 在多步任务中，用 scratchpad 工具暂存中间结果"""


async def react_agent(
    client: OpenAI,
    model: str,
    memory_manager: MemoryManager,
    registry: ToolRegistry,
    user_input: str,
    conversation_label: str | None = None,
) -> str:
    if conversation_label:
        print(f"\n{'=' * 60}")
        print(f"📝 {conversation_label}")
        print(f"{'=' * 60}")

    print(f"👤 [User] {user_input}")

    await memory_manager.initialize(BASE_SYSTEM_PROMPT)
    await memory_manager.pre_process(user_input)

    while True:
        messages = memory_manager.short_term.get_all()

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=registry.get_all_schemas(),
        )

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        memory_manager.post_process(choice.message.model_dump(exclude_none=True))

        print(f"\n💭 [Thought] (finish_reason: {finish_reason})")

        if finish_reason == "stop":
            answer = choice.message.content or "(无回答)"
            print(f"🤖 [最终回答] {answer}")
            await memory_manager.finalize()
            return answer

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                name = tc.function.name
                args_str = tc.function.arguments
                print(f"🔧 [Action] {name}({args_str})")
                try:
                    args = json.loads(args_str)
                    result = await registry.execute(name, args)
                    preview = result[:200] + "..." if len(result) > 200 else result
                    print(f"👁️  [Observation] {preview}")
                    memory_manager.short_term.add({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                except Exception as err:
                    err_msg = f"工具执行错误: {err}"
                    print(f"❌ [Error] {err_msg}")
                    memory_manager.short_term.add({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": err_msg,
                    })


def build_day5_registry(memory_manager: MemoryManager) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WeatherTool())
    registry.register(SearchTool())
    registry.register(SaveFactTool(memory_manager))
    registry.register(RecallFactTool(memory_manager))
    registry.register(ScratchpadTool(memory_manager))
    return registry
