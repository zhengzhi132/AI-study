# Python GIL 详解

## 什么是 GIL？

GIL（Global Interpreter Lock，全局解释器锁）是 CPython 解释器中的一个互斥锁。它确保同一时刻只有一个线程在执行 Python 字节码。

## 为什么需要 GIL？

CPython 的内存管理（引用计数）不是线程安全的。如果没有 GIL，两个线程同时修改同一个对象的引用计数会导致计数错误，对象可能被提前释放或永不释放。GIL 是解决这个问题的最简单方案。

## GIL 的影响

CPU 密集型任务在多线程下不但不能加速，反而因为锁竞争和上下文切换而变慢。I/O 密集型任务受 GIL 影响很小，因为线程在等待 I/O 时会释放 GIL。

## 绕过 GIL 的方法

1. multiprocessing 模块：每个进程有独立的 Python 解释器和 GIL
2. C 扩展：用 C/C++ 编写的扩展可以在释放 GIL 后执行计算，NumPy 就是典型例子
3. 异步编程（asyncio）：使用单线程事件循环，天然不受 GIL 影响

## Python 3.13 的变化

Python 3.13 引入了实验性的无 GIL 模式（PEP 703），通过 `--disable-gil` 编译选项启用。这是 Python 社区多年努力的成果。

## 总结

GIL 只影响 CPU 密集型的多线程程序。I/O 密集型和异步程序基本不受影响。使用 multiprocessing、C 扩展或 asyncio 可以绕过 GIL。Python 3.13+ 开始支持无 GIL 模式。
