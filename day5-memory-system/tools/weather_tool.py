"""天气查询工具 — 模拟天气数据."""

from tool import Tool

MOCK_WEATHER = {
    "北京": "晴，25°C，湿度 40%，适合户外活动",
    "上海": "多云，28°C，湿度 65%",
    "广州": "雷阵雨，30°C，湿度 80%",
    "深圳": "阴，29°C，湿度 75%",
    "杭州": "小雨，22°C，湿度 70%",
    "成都": "阴，24°C，湿度 60%",
    "西安": "晴，27°C，湿度 35%",
    "武汉": "多云，26°C，湿度 55%",
}


class WeatherTool(Tool):
    def __init__(self):
        super().__init__(
            name="get_weather",
            description="获取指定城市的实时天气信息",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"],
            },
        )

    async def execute(self, args: dict) -> str:
        city = args.get("city", "")
        return MOCK_WEATHER.get(city, f"{city}: 晴，22°C，湿度 50%")
