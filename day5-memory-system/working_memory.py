"""工作记忆 — 当前任务的键值暂存器."""

from interfaces import WorkingMemoryEntry


class WorkingMemory:
    def __init__(self):
        self._store: dict[str, WorkingMemoryEntry] = {}
        self._task_id: str | None = None

    def set(self, key: str, value: str) -> None:
        if not key:
            raise ValueError("键不能为空")
        import time
        self._store[key] = WorkingMemoryEntry(key=key, value=value, timestamp=time.time())

    def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        return entry.value if entry else None

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def get_all(self) -> list[WorkingMemoryEntry]:
        return list(self._store.values())

    def start_task(self, task_id: str) -> None:
        self._task_id = task_id
        self._store.clear()

    def end_task(self) -> None:
        self._task_id = None
        self._store.clear()

    def format_for_prompt(self) -> str:
        if not self._store:
            return ""
        lines = ["[工作记忆 - 当前任务]"]
        if self._task_id:
            lines.append(f"task: {self._task_id}")
        for entry in self._store.values():
            lines.append(f"  {entry.key}: {entry.value}")
        return "\n".join(lines)

    def stats(self) -> dict:
        return {"task_id": self._task_id, "entry_count": len(self._store)}
