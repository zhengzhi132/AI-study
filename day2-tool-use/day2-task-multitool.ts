// day2-task-multitool.ts
// 扩展为 3 个工具（天气、搜索、计算器），LLM 自动路由
// 运行: npx tsx day2-tool-use/day2-task-multitool.ts

import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://api.deepseek.com",
  apiKey: process.env.DEEPSEEK_API_KEY ?? "sk-xxx",
});

const tools: OpenAI.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "获取指定城市的实时天气信息",
      parameters: {
        type: "object",
        properties: { city: { type: "string", description: "城市名称" } },
        required: ["city"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "search_web",
      description: "搜索网页，获取最新信息、新闻或百科知识",
      parameters: {
        type: "object",
        properties: { query: { type: "string", description: "搜索关键词" } },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "calculator",
      description: "执行数学计算，支持加减乘除和复杂表达式",
      parameters: {
        type: "object",
        properties: { expression: { type: "string", description: "数学表达式，如 '3 * (4 + 2)'" } },
        required: ["expression"],
      },
    },
  },
];

function execute(name: string, args: Record<string, string>): string {
  switch (name) {
    case "get_weather": {
      const m: Record<string, string> = {
        北京: "晴，25°C，湿度 40%",
        上海: "多云，28°C，湿度 65%",
      };
      return m[args.city] ?? `${args.city}: 晴，22°C，湿度 50%`;
    }
    case "search_web":
      return `关于"${args.query}"的搜索结果: DeepSeek 是深度求索公司开发的大语言模型...`;
    case "calculator":
      try {
        return `${args.expression} = ${eval(args.expression)}`;
      } catch {
        return "计算错误";
      }
    default:
      return "未知工具";
  }
}

async function ask(question: string) {
  console.log(`\n[用户] ${question}`);

  const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "user", content: question },
  ];

  const resp = await client.chat.completions.create({
    model: "deepseek-chat",
    messages,
    tools,
  });

  const msg = resp.choices[0].message;
  messages.push(msg);

  if (msg.tool_calls) {
    for (const tc of msg.tool_calls) {
      const args = JSON.parse(tc.function.arguments);
      console.log(`  → 路由: ${tc.function.name}(${JSON.stringify(args)})`);

      const result = execute(tc.function.name, args);
      console.log(`  → 结果: ${result}`);

      messages.push({ role: "tool", tool_call_id: tc.id, content: result });
    }

    const resp2 = await client.chat.completions.create({
      model: "deepseek-chat",
      messages,
    });

    console.log(`[最终] ${resp2.choices[0].message.content}`);
  } else {
    console.log(`[直接] ${msg.content}`);
  }
}

async function main() {
  await ask("北京今天天气怎么样？");
  await ask("搜索一下 DeepSeek 是什么");
  await ask("帮我计算 (15 + 7) * 3");
  await ask("你好，打个招呼吧");
}

main();
