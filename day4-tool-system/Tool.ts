import OpenAI from "openai";


interface JsonSchema {
    type: "object";
    properties: {
        [key: string]: {
            type: string;
            description: string;
        }
    };
    required?: string[];
}

abstract class Tool {
    readonly name: string;
    readonly description: string;
    readonly parameters: JsonSchema;

    constructor(name: string, description: string, parameters: JsonSchema) {
        this.name = name;
        this.description = description;
        this.parameters = parameters;
    }

    abstract execute(args: Record<string, unknown>): Promise<string>;

    toOpenAiSchema() : OpenAI.ChatCompletionTool {
        return {
            type: "function",
            function: {
            name: this.name,
            description: this.description,
            parameters: this.parameters}
        }
    }
}

class ToolRegistry {
    private tools: Map<string, Tool>;

    constructor() {
        this.tools = new Map();
    }

    register(tool: Tool) {
        if (this.tools.has(tool.name)) {
            throw new Error(`Tool with name ${tool.name} is already registered.`);
        }
        this.tools.set(tool.name, tool);
    }

    unregister(toolName: string) {
        this.tools.delete(toolName);
    }

    getAllSchemas(): OpenAI.ChatCompletionTool[] {
        return Array.from(this.tools.values()).map(tool => tool.toOpenAiSchema());
    }

    async execute(toolName: string, args: Record<string, unknown>): Promise<string> {
        const tool = this.tools.get(toolName);
        
        if (!tool) {
            throw new Error(`Tool with name ${toolName} is not registered.`);
        }

        let lastError: unknown;
        for (let attempt = 1; attempt <= 3; attempt++) {
            try {
                return await tool.execute(args);
            } catch (error) {
                lastError = error;
            }
        }
        throw lastError;
    }
    
}


export { Tool, JsonSchema ,ToolRegistry };