# asyncio 协程编程指南

## 协程基础

协程（coroutine）是 Python 异步编程的核心。使用 `async def` 定义，用 `await` 暂停等待。事件循环是 asyncio 的心脏，维护一个任务队列，轮流执行就绪的协程。

## 并发执行

```python
import asyncio

async def fetch_data(url):
    await asyncio.sleep(1)  # 模拟网络请求
    return f"Data from {url}"

async def main():
    # 并发执行，总时间约 1 秒而不是 3 秒
    results = await asyncio.gather(
        fetch_data("url1"),
        fetch_data("url2"),
        fetch_data("url3"),
    )
```

## async/await 与 GIL

asyncio 在单线程中运行，通过协程切换实现并发。因为不需要线程切换，asyncio 天然不受 GIL 的限制。这就是为什么 Python 中 asyncio 是 I/O 密集型任务的首选方案。

## 常见陷阱

在协程中调用同步阻塞函数（如 `time.sleep()`）会卡住整个事件循环。必须使用异步等价物（如 `asyncio.sleep()`）。忘记 `await` 也是常见错误——`asyncio.sleep(1)` 返回一个 coroutine 对象但不执行。

## 最佳实践

1. I/O 密集用 asyncio，CPU 密集用 multiprocessing
2. 使用 `asyncio.create_task()` 创建后台任务
3. 用 `asyncio.wait_for()` 设置超时
4. 使用 `asyncio.Semaphore` 限制并发数
