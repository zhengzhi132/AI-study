import {exec } from "node:child_process";
import { Tool} from "../Tool.js";

export class CodeExecutorTool extends Tool {
    constructor() {
        super("code_executor", "Execute JavaScript code and return the output or error.", {
            type: "object",
            properties: {
                code: {
                    type: "string",
                    description: "The JavaScript code to execute."
                }
            },
            required: ["code"]
        });
    }

    async execute(args: Record<string, unknown>): Promise<string> {

        let code = args.code as string;

        // 自动包裹：如果没有输出语句，把表达式包进 console.log
        if (!code.includes("console.log") && !code.includes("process.stdout")) {
            code = `console.log(${code})`;
        }

        return new Promise((resolve) => {
            exec(`node -e ${JSON.stringify(code)}`,{
                timeout: 5000, // 设置执行超时时间，防止无限循环等问题
                maxBuffer: 100 * 1024  // 设置最大输出缓冲区，防止输出过大导致问题
            }, (error, stdout, stderr) => {
                if (error) {
                    resolve(`执行出错: ${error.message}\n${stderr || ""}`);
                }  else {
                    resolve(stdout || stderr || "（无输出）");
                }
            });
        });

    }


}