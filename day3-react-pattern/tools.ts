import OpenAI from "openai";


export const tools: OpenAI.ChatCompletionTool[] = [
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
            name: "search",
            description: "搜索网页，获取最新信息、新闻或百科知识",
            parameters: {
                type: "object",
                properties: { query: { type: "string", description: "搜索关键词" } },
                required: ["query"],
            },
        },
    },
];

export async function execute(name:string, args: Record<string, any>): Promise<string> {
    switch (name) {
        case "get_weather": {
            const m: Record<string, string> = {
                北京: "晴，25°C，湿度 40%",
                上海: "多云，28°C，湿度 65%",
            };
            return m[args.city] ?? `${args.city}: 晴，22°C，湿度 50%`;
        }
        case "search":
            return `关于"${args.query}"的搜索结果: 故宫、长城、颐和园`;
    }
    throw new Error("未知工具");
}


