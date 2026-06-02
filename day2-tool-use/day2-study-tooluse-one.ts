// day2-tool-use.ts
// Day 2: Function Calling / Tool Use — Agent 的灵魂
// 运行: npx tsx day2-tool-use.ts

import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://api.deepseek.com",
  apiKey: process.env.DEEPSEEK_API_KEY ?? "sk-xxx",
});

// ===========================================================================
// 第一部分: Tool Definition Schema 详解
// ===========================================================================
// 每个 tool 包含三个核心字段:
//
//   name         — 工具名（LLM 用它来决定调用哪个，命名要语义化）
//   description  — 工具用途说明（LLM 靠这个判断"什么时候该用这个工具"）
//   parameters   — JSON Schema，描述入参的结构和约束
//
// LLM 做的事: 读 user message → 理解意图 → 如果匹配某工具 → 输出 { name, arguments }
// 你的代码做的事: 收到这个输出 → 实际执行工具(get_weather) → 结果喂回 LLM

// ---------------------------------------------------------------------------
// 1.1 单个工具: get_weather
// ---------------------------------------------------------------------------
async function singleTool() {
  console.log("=== 1. 单工具: get_weather ===\n");

  const resp = await client.chat.completions.create({
    model: "deepseek-chat",
    messages: [
      { role: "user", content: "北京今天天气怎么样？" },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "获取指定城市的实时天气信息，返回温度、湿度、天气状况",
          parameters: {
            type: "object",
            properties: {
              city: {
                type: "string",
                description: "城市名称，中文或英文",
              },
            },
            required: ["city"],
          },
        },
      },
    ],
  });

  const msg = resp.choices[0].message;
  console.log("finish_reason:", resp.choices[0].finish_reason);
  console.log("message.content:", msg.content); // tool call 时 content 为 null

  if (msg.tool_calls) {
    for (const tc of msg.tool_calls) {
      console.log("\n[tool_call]");
      console.log("  id:  ", tc.id);
      console.log("  name:", tc.function.name);
      console.log("  args:", tc.function.arguments); // 这是 JSON 字符串，需要 parse
    }
  }
}

// ---------------------------------------------------------------------------
// 1.2 tool_choice 参数 — 控制工具调用行为
// ---------------------------------------------------------------------------
//
// "auto" (默认)   — LLM 自己判断是否需要调工具，也可能直接文字回复
// "none"          — 禁止调工具，即使定义了 tools 也只返回 text
// "required"      — 强制调工具，LLM 必须选一个工具调用
// { type: "function", function: { name: "xxx" } } — 强制调用指定工具
//
async function toolChoiceDemo() {
  console.log("\n=== 2. tool_choice 参数对比 ===\n");

  // --- "none": 有工具定义但禁止调用 ---
  const a = await client.chat.completions.create({
    model: "deepseek-chat",
    messages: [{ role: "user", content: "北京天气怎么样？" }],
    tools: [{
      type: "function" as const,
      function: {
        name: "get_weather",
        description: "获取城市天气",
        parameters: {
          type: "object",
          properties: { city: { type: "string" } },
          required: ["city"],
        },
      },
    }],
    tool_choice: "none", // 禁止调用工具
  });
  console.log("[tool_choice=none]  finish:", a.choices[0].finish_reason);
  console.log("  content:", a.choices[0].message.content?.slice(0, 80));

  // --- "required": 强制调用工具 ---
  const b = await client.chat.completions.create({
    model: "deepseek-chat",
    messages: [{ role: "user", content: "打个招呼" }],
    tools: [{
      type: "function" as const,
      function: {
        name: "get_weather",
        parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
      },
    }],
    tool_choice: "required", // 即使"打个招呼"不需要工具，也强制调用
  });
  console.log("[tool_choice=required] finish:", b.choices[0].finish_reason);
  console.log("  强制调用了:", b.choices[0].message.tool_calls?.[0]?.function.name);

  // --- 指定工具名 ---
  const c = await client.chat.completions.create({
    model: "deepseek-chat",
    messages: [{ role: "user", content: "北京天气怎么样？" }],
    tools: [
      { type: "function" as const, function: { name: "get_weather", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } } },
      { type: "function" as const, function: { name: "search_web", parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } } },
    ],
    tool_choice: { type: "function", function: { name: "search_web" } }, // 强制走 search_web，忽略 get_weather
  });
  console.log("[tool_choice=search_web] 实际调用:", c.choices[0].message.tool_calls?.[0]?.function.name);
}

// ===========================================================================
// 第二部分: 完整的 Tool Loop — LLM 说"我要调"，你来执行
// ===========================================================================
//
// 核心循环:
//   User 提问 → LLM 判断 → 输出 tool_call 或 text
//     ↓ tool_call               ↓ text
//   你的代码执行工具             直接返回
//     ↓ 结果喂回 LLM
//   LLM 可能继续 tool_call 或输出最终答案
//
async function completeToolLoop() {
  console.log("\n=== 3. 完整 Tool Loop: 工具结果回传 ===\n");

  // 模拟的工具实现（在生产代码中这些是真正的 API 调用）
  function get_weather(city: string) {
    const mock: Record<string, string> = {
      "北京": "晴，25°C，湿度 40%，风力 3 级",
      "上海": "多云，28°C，湿度 65%，风力 2 级",
    };
    return mock[city] ?? `${city}: 晴，22°C，湿度 50%`;
  }

  // Step 1: 用户提问，LLM 返回 tool_call
  const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "user", content: "北京今天天气怎么样？适合户外运动吗？" },
  ];

  const resp1 = await client.chat.completions.create({
    model: "deepseek-chat",
    messages,
    tools: [{
      type: "function" as const,
      function: {
        name: "get_weather",
        description: "获取指定城市的实时天气信息",
        parameters: {
          type: "object",
          properties: { city: { type: "string", description: "城市名" } },
          required: ["city"],
        },
      },
    }],
  });

  const msg1 = resp1.choices[0].message;
  console.log("Step 1 — LLM 返回:");
  console.log("  role:", msg1.role);
  console.log("  content:", msg1.content); // null
  console.log("  tool_calls:", msg1.tool_calls?.length, "个");

  // Step 2: 你的代码执行工具
  // 关键: LLM 没有执行工具！它只是输出了 JSON。真正执行的是你的代码。
  messages.push(msg1); // 把 assistant 的 tool_call 消息加入历史

  for (const tc of msg1.tool_calls!) {
    const args = JSON.parse(tc.function.arguments);
    console.log(`\nStep 2 — 执行工具: ${tc.function.name}(${args.city})`);

    const result = get_weather(args.city);
    console.log("  工具返回:", result);

    // 工具结果以 role: "tool" 消息形式回传
    messages.push({
      role: "tool",
      tool_call_id: tc.id,
      content: result,
    });
  }

  // Step 3: LLM 收到工具结果，生成最终回答
  console.log("\nStep 3 — 把工具结果喂回 LLM...");
  const resp2 = await client.chat.completions.create({
    model: "deepseek-chat",
    messages,
  });

  console.log("  最终回答:", resp2.choices[0].message.content);
}

// ===========================================================================
// 第三部分: 多工具自动路由
// ===========================================================================
async function multiToolRouting() {
  console.log("\n=== 4. 多工具自动路由 ===\n");

  // 定义 2 个工具
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

  // 三个不同意图的问题，LLM 自动路由到正确工具
  const queries = [
    "上海今天天气如何？",
    "计算 (15 + 7) * 3 的结果",
  ];

  for (const q of queries) {
    const resp = await client.chat.completions.create({
      model: "deepseek-chat",
      messages: [{ role: "user", content: q }],
      tools,
    });

    const tc = resp.choices[0].message.tool_calls?.[0];
    console.log(`"${q}"`);
    console.log(`  → 路由到: ${tc?.function.name}(${tc?.function.arguments})`);
    console.log();
  }

  console.log("要点: LLM 根据 description 自动判断该用哪个工具。");
  console.log("好的 description = 好的路由准确率。");
}

// ===========================================================================
// 启动
// ===========================================================================
async function main() {
  await singleTool();
  await toolChoiceDemo();
  await completeToolLoop();
  await multiToolRouting();
}

main().catch(console.error);
