// cli-chat.ts
// 命令行多轮对话脚本 — Day 1 动手任务
// 运行: npx tsx cli-chat.ts

import * as readline from "node:readline";
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://api.deepseek.com",
  apiKey: process.env.DEEPSEEK_API_KEY ?? "sk-xx",
});

const messages: OpenAI.ChatCompletionMessageParam[] = [
  { role: "system", content: "用中文回答，语气简洁。" },
];

async function chat(userInput: string) {
  messages.push({ role: "user", content: userInput });

  const stream = await client.chat.completions.create({
    model: "deepseek-chat",
    messages,
    stream: true,
  });

  process.stdout.write("AI: ");
  let fullReply = "";
  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content;
    if (delta) {
      process.stdout.write(delta);
      fullReply += delta;
    }
  }
  process.stdout.write("\n");

  messages.push({ role: "assistant", content: fullReply });
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function loop() {
  rl.question("You: ", async (input) => {
    if (input.trim().toLowerCase() === "exit") {
      console.log("再见！");
      rl.close();
      return;
    }
    await chat(input);
    if (!rl.closed) loop();
  });
}

console.log("多轮对话已启动，输入 exit 退出。");
loop();
