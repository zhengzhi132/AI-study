// day2-task-singletool.ts
// Day 2 动手任务: 让 LLM 自动决定何时调用 get_weather（单工具）
// 运行: npx tsx day2-tool-use/day2-task-singletool.ts

import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://api.deepseek.com",
  apiKey: process.env.DEEPSEEK_API_KEY ?? "sk-xxx",
});

async function ask(question: string) {
  // Step 1: 发消息 + 工具定义
  const resp = await client.chat.completions.create({
    model: "deepseek-chat",
    messages: [{ role: "user", content: question }],
    tools: [{
      type: "function",
      function: {
        name: "get_weather",
        description: "获取指定城市的天气信息",
        parameters: {
          type: "object",
          properties: { city: { type: "string", description: "城市名" } },
          required: ["city"],
        },
      },
    }],
  });

  const msg = resp.choices[0].message;

  // Step 2: LLM 决定要不要调工具
  if (msg.tool_calls) {
    const tc = msg.tool_calls[0];
    if (tc.type !== "function") return;
    const args = JSON.parse(tc.function.arguments);
    console.log(`[LLM 想调工具] ${tc.function.name}("${args.city}")`);

    // Step 3: 你的代码执行工具
    const result = `城市: ${args.city}, 天气: 晴, 温度: 22°C`;
    console.log(`[工具返回] ${result}`);

    // Step 4: 结果喂回 LLM
    const resp2 = await client.chat.completions.create({
      model: "deepseek-chat",
      messages: [
        { role: "user", content: question },
        msg,
        { role: "tool", tool_call_id: tc.id, content: result },
      ],
    });
    console.log(`[最终回复] ${resp2.choices[0].message.content}`);
  } else {
    console.log(`[直接回复] ${msg.content}`);
  }
}

async function main() {
  // 天气相关 → 调工具；闲聊 → 直接回
  await ask("北京今天天气怎么样？");
  console.log("---");
  await ask("你好，打个招呼吧");
}

main();
