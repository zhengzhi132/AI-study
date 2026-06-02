// deepseek-demo.ts
// DeepSeek Chat API — OpenAI SDK 兼容模式
// 安装: npm install openai
// 运行: DEEPSEEK_API_KEY=sk-xxx npx tsx deepseek-demo.ts

import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://api.deepseek.com",
  apiKey: process.env.DEEPSEEK_API_KEY ?? "sk-xx",
});

// ---------------------------------------------------------------------------
// 1. 非流式请求
// ---------------------------------------------------------------------------
async function chat() {
  const resp = await client.chat.completions.create({
    model: "deepseek-v4-flash",
    max_tokens: 512,
    messages: [
      { role: "system", content: "用中文回答，语气简洁。" },
      { role: "user", content: "一句话解释什么是 Transformer。" },
    ],
  });

  const choice = resp.choices[0];
  console.log(
    "[非流式]",
    resp.id,
    resp.model,
    `tokens(in=${resp.usage?.prompt_tokens}, out=${resp.usage?.completion_tokens})`,
  );
  console.log(choice.message.content);
}

// ---------------------------------------------------------------------------
// 2. 流式请求
// ---------------------------------------------------------------------------
async function streamChat() {
  const stream = await client.chat.completions.create({
    model: "deepseek-v4-flash",
    max_tokens: 256,
    stream: true,
    messages: [
      { role: "system", content: "用中文回答，一句话简明扼要。" },
      { role: "user", content: "什么是 RAG？" },
    ],
  });

  process.stdout.write("[流式] ");
  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content;
    if (delta) process.stdout.write(delta);
  }
  process.stdout.write("\n");
}

// ---------------------------------------------------------------------------
// 3. 多轮对话
// ---------------------------------------------------------------------------
async function multiTurn() {
  const resp = await client.chat.completions.create({
    model: "deepseek-chat",
    max_tokens: 256,
    messages: [
      { role: "user", content: "我叫小明。" },
      { role: "assistant", content: "你好小明！有什么可以帮你的？" },
      { role: "user", content: "我叫什么名字？" },
    ],
  });

  console.log("[多轮]", resp.choices[0].message.content);
}

// ---------------------------------------------------------------------------
// 4. Function Calling
// ---------------------------------------------------------------------------
async function toolUse() {
  const resp = await client.chat.completions.create({
    model: "deepseek-chat",
    max_tokens: 256,
    tools: [
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "获取指定城市的天气",
          parameters: {
            type: "object",
            properties: { city: { type: "string", description: "城市名" } },
            required: ["city"],
          },
        },
      },
    ],
    messages: [{ role: "user", content: "北京今天天气怎么样？" }],
  });

  const msg = resp.choices[0].message;
  if (msg.tool_calls?.length) {
    for (const tc of msg.tool_calls) {
      if (tc.type === "function") {
        console.log("[Tool] 模型想调用:", tc.function.name, tc.function.arguments);
      }
    }
  } else {
    console.log("[Tool] 文本:", msg.content);
  }
  console.log(`(finish_reason: ${resp.choices[0].finish_reason})`);
}

// ---------------------------------------------------------------------------
// 5. JSON 模式输出（DeepSeek 支持 json_object，不支持 json_schema）
// ---------------------------------------------------------------------------
async function jsonOutput() {
  const resp = await client.chat.completions.create({
    model: "deepseek-chat",
    max_tokens: 256,
    messages: [
      {
        role: "user",
        content:
          '推荐三本编程书籍。以 JSON 数组返回，每项含 title、author、reason 字段。只输出 JSON，不要其他文字。',
      },
    ],
    response_format: { type: "json_object" },
  });

  console.log("[JSON 输出]", resp.choices[0].message.content);
}

// ---------------------------------------------------------------------------
// 启动
// ---------------------------------------------------------------------------
async function main() {
  await chat();
  await streamChat();
  await multiTurn();
  await toolUse();
  await jsonOutput();
}

main().catch(console.error);
