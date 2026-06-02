// anthropic-demo.ts
// Anthropic Messages API — Claude 的非流式 + 流式对话示例
// 安装: npm install @anthropic-ai/sdk
// 运行: npx tsx anthropic-demo.ts

import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY ?? "sk-ant-xxx",
});

// ---------------------------------------------------------------------------
// 1. 非流式请求 (Non-streaming)
// ---------------------------------------------------------------------------
async function chat() {
  const resp = await client.messages.create({
    model: "claude-sonnet-4-6",          // 或 claude-opus-4-8 / claude-haiku-4-5
    max_tokens: 1024,
    system: "用中文回答，语气简洁。",
    messages: [{ role: "user", content: "一句话解释什么是 Transformer。" }],
  });

  // resp.content 是一个 ContentBlock 数组，text 类型取 .text
  const text = resp.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("\n");

  console.log("[非流式]", resp.id, resp.model, `tokens(in=${resp.usage.input_tokens}, out=${resp.usage.output_tokens})`);
  console.log(text);
}

// ---------------------------------------------------------------------------
// 2. 流式请求 (Streaming)
// ---------------------------------------------------------------------------
async function streamChat() {
  const stream = client.messages.stream({
    model: "claude-sonnet-4-6",
    max_tokens: 512,
    system: "用中文回答，一句话简明扼要。",
    messages: [{ role: "user", content: "什么是 RAG？" }],
  });

  process.stdout.write("[流式] ");
  for await (const event of stream) {
    // event 类型收窄：只处理 text_delta
    if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
      process.stdout.write(event.delta.text);
    }
  }
  process.stdout.write("\n");

  // 流结束后可拿到最终的 message 对象（含 usage）
  const final = await stream.finalMessage();
  console.log(`(tokens: in=${final.usage.input_tokens}, out=${final.usage.output_tokens})`);
}

// ---------------------------------------------------------------------------
// 3. 多轮对话
// ---------------------------------------------------------------------------
async function multiTurn() {
  const resp = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 512,
    messages: [
      { role: "user", content: "我叫小明。" },
      { role: "assistant", content: "你好小明！有什么可以帮你的？" },
      { role: "user", content: "我叫什么名字？" },
    ],
  });

  const text = resp.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("\n");

  console.log("[多轮]", text);
}

// ---------------------------------------------------------------------------
// 4. Tool Use（函数调用）
// ---------------------------------------------------------------------------
async function toolUse() {
  const resp = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 512,
    tools: [
      {
        name: "get_weather",
        description: "获取指定城市的天气",
        input_schema: {
          type: "object",
          properties: { city: { type: "string", description: "城市名" } },
          required: ["city"],
        },
      },
    ],
    messages: [{ role: "user", content: "北京今天天气怎么样？" }],
  });

  for (const block of resp.content) {
    if (block.type === "tool_use") {
      console.log("[Tool Use] 模型想调用:", block.name, block.input);
    } else if (block.type === "text") {
      console.log("[Tool Use] 文本:", block.text);
    }
  }
  console.log(`(stop_reason: ${resp.stop_reason})`);
}

// ---------------------------------------------------------------------------
// 启动
// ---------------------------------------------------------------------------
async function main() {
  await chat();
  await streamChat();
  await multiTurn();
  await toolUse();
}

main().catch(console.error);
