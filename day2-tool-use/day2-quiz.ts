// day2-quiz.ts
// Day1 + Day2 综合测验 — 手写代码
// 运行: npx tsx day2-tool-use/day2-quiz.ts

import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://api.deepseek.com",
  apiKey: process.env.DEEPSEEK_API_KEY ?? "sk-xxx",
});

// ===========================================================================
// 题目 1: 补全工具 — 新增 translate 工具，支持中英互译
// ===========================================================================
// 要求: tools 数组中补一个 translate 工具定义，并在 execute() 里加对应分支
async function quiz1() {
  console.log("=== 题目 1: 补全 translate 工具 ===\n");

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
    // TODO: 在这里补 translate 工具定义
    // {
    //   type: "function",
    //   function: {
    //     name: "translate",
    //     description: "_______________",
    //     parameters: {
    //       ...
    //     },
    //   },
    // },
    {
      type:"function",
      function: {
        name: "translate",
        description: "翻译文本，支持中英互译",
        parameters: {
          type: "object",
          properties: {
            text: {type: "string", description: "要翻译的文本"}    
          },
          required: ["text"],
        }

      }
    }
  ];

  function execute(name: string, args: Record<string, string>): string {
    switch (name) {
      case "get_weather": {
        const m: Record<string, string> = { 北京: "晴，25°C" };
        return m[args.city] ?? `${args.city}: 晴，22°C`;
      }
      // TODO: 补 translate 的 mock 实现
      case "translate": {
        const m: Record<string, string> = {
          "Hello, how are you?": "你好，你怎么样？",
          "你好，你怎么样？": "Hello, how are you?",
        };
        return m[args.text] ?? `翻译(${args.text})`;
      }
    }
    return "";
  }

  const resp = await client.chat.completions.create({
    model: "deepseek-chat",
    messages: [{ role: "user", content: "把 'Hello, how are you?' 翻译成中文" }],
    tools,
  });

  const msg = resp.choices[0].message;
  if (msg.tool_calls) {
    const tc = msg.tool_calls[0];
    const args = JSON.parse(tc.function.arguments);
    console.log(`路由: ${tc.function.name}(${JSON.stringify(args)})`);
    console.log(`结果: ${execute(tc.function.name, args)}`);
  } else {
    console.log("LLM 没调工具，检查你的 tool 定义是否足够清晰");
  }
}

// ===========================================================================
// 题目 2: 修复 Bug — 下面这段 Tool Loop 有什么问题？
// ===========================================================================
async function quiz2() {
  console.log("\n=== 题目 2: 修复 Bug ===\n");

  async function brokenAsk(question: string) {
    const resp = await client.chat.completions.create({
      model: "deepseek-chat",
      messages: [{ role: "user", content: question }],
      tools: [{
        type: "function",
        function: {
          name: "calculator",
          description: "执行数学计算",
          parameters: {
            type: "object",
            properties: { expression: { type: "string" } },
            required: ["expression"],
          },
        },
      }],
    });

    const msg = resp.choices[0].message;
    if (msg.tool_calls) {
      const tc = msg.tool_calls[0];
      const args = JSON.parse(tc.function.arguments);
      const result = eval(args.expression); // 用 eval 仅为演示

      // BUG 在这里: tool 结果没有回传给 LLM，LLM 看到的只有初始的 user 消息
      // LLM 需要知道工具的执行结果才能生成带答案的回复
      console.log(`[BUG 版本] 工具结果: ${result} — 但 LLM 看不到这个结果`);
      return `工具算出来了: ${result}`;
    }
    return msg.content;
  }

  console.log("思考题: 上面的 brokenAsk 少了什么步骤？");
  console.log("答: 没有将工具处理的结果再次回传给LLM\n");

  const r = await brokenAsk("计算 3 + 5");
  console.log(r);
}

// ===========================================================================
// 题目 3: tool_choice 实战
// ===========================================================================
async function quiz3() {
  console.log("\n=== 题目 3: tool_choice 选择题 ===\n");

  const scenarios = [
    { q: "北京天气好吗？", intent: "LLM 自己判断要不要调工具" ,tool_choice: "auto"},
    { q: "你好，打个招呼", intent: "禁止调工具，即使定义了 tools" ,tool_choice: "none"},
    { q: "北京天气好吗？", intent: "必须调工具，不回答文字" ,tool_choice: "required"},
  ];

  console.log("匹配场景和 tool_choice:");
  for (const s of scenarios) {
    console.log(`  "${s.q}"  → 意图: ${s.intent}  → tool_choice: ${s.tool_choice}`);
  }
  console.log("  选项: auto, none, required");
}

// ===========================================================================
// 题目 4: 多工具链式调用
// ===========================================================================
async function quiz4() {
  console.log("\n=== 题目 4: 多工具链式调用 ===\n");

  console.log("需求: 用户说 '查一下北京天气，顺便搜索一下故宫介绍'");
  console.log("消息顺序: user → assistant(tool_calls) → tool → tool → assistant(最终回复)");
  console.log("");
  console.log("提示: msg.tool_calls 是一个数组，可能包含多个元素");
  console.log("每个 tool_call 都需要一个对应的 role:'tool' 消息");

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
        description: "执行网络搜索，返回简要结果",
          parameters: {
            type: "object",
            properties: { query: { type: "string", description: "搜索关键词" } },
            required: ["query"],
          },
        },
      }
  ];
  const execute = (name: string, args: Record<string, string>): string => {
    switch (name) {
      case "get_weather": {
        const m: Record<string, string> = {
          北京: "晴，25°C，湿度 40%",
          上海: "多云，28°C，湿度 65%",
        };
        return m[args.city] ?? `${args.city}: 晴，22°C，湿度 50%`;
      }
      case "search_web": 
        return `关于"${args.query}"的搜索结果: 故宫是中国的著名景点，位于北京...`;
    }
    return "";
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
    messages.push(msg); // Bug 2 修复: 把 assistant 消息加入历史

    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        if (tc.type !== "function") continue;
        const args = JSON.parse(tc.function.arguments);
        const result = execute(tc.function.name, args);
        console.log(`[工具调用] ${tc.function.name}(${JSON.stringify(args)}) → ${result}`);

        messages.push({ role: "tool", tool_call_id: tc.id, content: result });
      }

      // 所有工具结果回传后，再调一次 LLM 生成最终回复
      const resp2 = await client.chat.completions.create({
        model: "deepseek-chat",
        messages,
      });

      console.log(`[最终回复] ${resp2.choices[0].message.content}`);
    } else {
      console.log(`[最终回复] ${msg.content}`);
    }
  }

  await ask("查一下北京天气，顺便搜索一下故宫介绍");

}
// ===========================================================================
// 题目 5: 设计题 — 写一个好的 tool description
// ===========================================================================
async function quiz5() {
  console.log("\n=== 题目 5: 设计 send_email 工具 ===\n");

  console.log("设计一个 'send_email' 工具:");
  console.log("  name: send_email");
  console.log("  description: 发送邮件或者发送短信的工具，参数里需要包含 recipient（收件人），content（内容）和 type（邮件或短信）");
  console.log("  parameters:");
  console.log("    - _______________");
  console.log("    - _______________");
  console.log("    - _______________");
}

// ===========================================================================
// 题目 6: Day1 回顾 — DeepSeek 适配
// ===========================================================================
async function quiz6() {
  console.log("\n=== 题目 6: Day1 回顾 ===\n");

  console.log("如果要切换到 OpenAI 官方 API，只需要改哪几行？");
  console.log("  1. _______________");
  console.log("  2. _______________");
  console.log("");
  console.log("非流式响应中，取最终回复内容要访问哪个字段路径？");
  console.log("  resp._______________");
}

async function main() {
  await quiz1();
  await quiz2();
  await quiz3();
  await quiz4();
  await quiz5();
  await quiz6();
}

main();
