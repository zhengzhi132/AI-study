import OpenAI from "openai";
import { ToolRegistry } from "./Tool.js"
import { WeatherTool } from "./tools/WeatherTool.js";
import { SearchTool } from "./tools/SearchTool.js";
import { CodeExecutorTool } from "./tools/CodeExecutorTool.js";

const client = new OpenAI({
  baseURL: "https://api.deepseek.com",
  apiKey: process.env.DEEPSEEK_API_KEY ?? "sk-xx",
});


async function reactAgent(registry: ToolRegistry, userInput: string) {

    const messages: OpenAI.ChatCompletionMessageParam[] = [
        {
        role: "system",
        content: "你是一个有工具调用能力的 AI 助手。重要规则：1) 天气信息必须通过 getWeather 获取；2) 推荐具体地点、活动、景点时，必须通过 search 搜索(除非没有search工具），禁止仅凭训练知识推荐；3) 只有拿到所有工具返回的数据后，才能给出最终答案。"
        }
    ]

    messages.push({
        role: "user",
        content: userInput
    })

    while (true) {
        const response = await client.chat.completions.create({
            model: "deepseek-v4-flash",
            messages,
            tools: registry.getAllSchemas()
        })

        const choice = response.choices[0];
        console.log(`\n💭 [Thought] LLM 开始思考... (finish_reason: ${choice.finish_reason})`);
        if (choice.finish_reason === "stop") {
            
            console.log(`[最终回答] ${choice.message.content}`);
            break;
        }

        messages.push(choice.message);

        if (choice.message.tool_calls) {
            
            for (const tc of choice.message.tool_calls) {
                console.log(`\n🔧 [Action] 模型决定调用 ${tc.function.name}(${tc.function.arguments})`);
                const result = await registry.execute(tc.function.name, JSON.parse(tc.function.arguments));
                console.log(`👁️  [Observation] 结果: ${result}`);
                messages.push({
                role: "tool",
                tool_call_id: tc.id,
                content: result,
                })
            }
        }
    }
}

async function main() {
    const registry = new ToolRegistry();
    // registry.register(new WeatherTool());
    // registry.register(new SearchTool());
    
    // await reactAgent(registry, "北京今天的天气适合户外运动吗？如果适合，推荐 3 个活动");

    // registry.unregister("search");
    // await reactAgent(registry, "上海今天的天气适合户外运动吗？如果适合，推荐 3 个活动");
    registry.register(new CodeExecutorTool());
    await reactAgent(registry, "帮我算 123 * 456，用 code_executor 计算");
}

main();

   