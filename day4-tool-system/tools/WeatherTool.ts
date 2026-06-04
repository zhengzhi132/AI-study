import { Tool } from "../Tool.js";
export class WeatherTool extends Tool {
    constructor() {
        super("getWeather", "Get the current weather for a given location.", {
            type: "object",
            properties: {
                location: {
                    type: "string",
                    description: "The location to get the weather for."
                }
            },
            required: ["location"]
        });
    }

    async execute( args: Record<string, unknown>): Promise<string> {
        const m: Record<string, string> = {
            北京: "晴，25°C，湿度 40%",
            上海: "多云，28°C，湿度 65%",
        };
        return m[args.location as string] ?? `${args.location}: 晴，22°C，湿度 50%`;

    }
}

