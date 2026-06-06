"""短期记忆 — 对话历史管理 + Token 计数 + LLM 摘要压缩."""

from typing import Any
from openai import OpenAI
from interfaces import SummarizationConfig


class ShortTermMemory:
    def __init__(self, config: SummarizationConfig | None = None):
        self._messages: list[dict[str, Any]] = []
        self._config = config or SummarizationConfig()
        self._compressed = False

    def add(self, message: dict[str, Any]) -> None:
        self._messages.append(message)

    def get_all(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def set_system_prompt(self, content: str) -> None:
        msg = {"role": "system", "content": content}
        if self._messages and self._messages[0]["role"] == "system":
            self._messages[0] = msg
        else:
            self._messages.insert(0, msg)

    def estimate_tokens(self) -> int:
        chars = sum(self._msg_chars(m) for m in self._messages)
        return (chars + 3) // 4  # ceil(chars / 4)

    def _msg_chars(self, msg: dict[str, Any]) -> int:
        content = msg.get("content", "")
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            return sum(part.get("text", "") if isinstance(part, dict) else 0 for part in content)
        return 0

    def is_over_threshold(self) -> bool:
        return self.estimate_tokens() > self._config.max_tokens

    async def summarize(self, client: OpenAI, model: str) -> None:
        system_idx = next((i for i, m in enumerate(self._messages) if m["role"] == "system"), -1)
        start_idx = system_idx + 1 if system_idx >= 0 else 0
        compress_end = max(start_idx, len(self._messages) - self._config.keep_last_n)

        if compress_end <= start_idx:
            return

        compressible = self._messages[start_idx:compress_end]
        conv_text = "\n".join(
            f"[{m['role']}]: {m.get('content', '')}" for m in compressible
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "将以下对话压缩为简洁的要点摘要。保留关键事实、决定和用户偏好。用中文输出。",
                    },
                    {"role": "user", "content": conv_text},
                ],
                max_tokens=500,
            )
            summary = response.choices[0].message.content or "(摘要生成失败)"
        except Exception:
            summary = "(摘要生成失败)"

        summary_msg = {"role": "system", "content": f"[对话历史摘要]\n{summary}"}

        head = [self._messages[system_idx]] if system_idx >= 0 else []
        tail = self._messages[compress_end:]
        self._messages = head + [summary_msg] + tail
        self._compressed = True

    def clear(self) -> None:
        self._messages.clear()
        self._compressed = False

    def clear_except_system(self) -> None:
        sys_msg = self._messages[0] if self._messages and self._messages[0]["role"] == "system" else None
        self._messages = [sys_msg] if sys_msg else []
        self._compressed = False

    def stats(self) -> dict:
        return {
            "message_count": len(self._messages),
            "estimated_tokens": self.estimate_tokens(),
            "is_compressed": self._compressed,
        }
