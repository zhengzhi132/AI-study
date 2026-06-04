import { Tool } from "../Tool.js";

export class SearchTool extends Tool {
    constructor() {
        super("search", "Search the web for the latest information, news, or encyclopedic knowledge.", {
            type: "object",
            properties: {
                query: {
                    type: "string",
                    description: "The search query."
                }
            },
            required: ["query"]
        });
    }

    async execute(args: Record<string, unknown>): Promise<string> {
        return `关于"${args.query as string}"的搜索结果: 故宫、长城、颐和园`;
    }
}