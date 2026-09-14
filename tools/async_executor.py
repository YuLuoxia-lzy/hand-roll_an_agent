"""
异步工具执行器 - 已适配 function calling
"""

import asyncio
import concurrent.futures
from typing import List
from ..core.typedefs import ToolCall
from .registry import ToolRegistry


class AsyncToolExecutor:
    """异步工具执行器 - 把同步的工具函数丢进线程池并发执行"""

    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def execute_async(self, call: ToolCall) -> str:
        """异步执行单个工具调用"""
        # 必须用 get_running_loop(): get_event_loop() 在 3.10+ 已废弃
        loop = asyncio.get_running_loop()

        def _execute():
            # registry.execute 是同步的, 放到线程池里跑, 不阻塞事件循环
            return self.registry.execute(call)

        try:
            return await loop.run_in_executor(self.executor, _execute)
        except Exception as e:
            return f"错误: 工具 '{call.name}' 异步执行失败: {e}"

    async def execute_many(self, calls: List[ToolCall]) -> List[str]:
        """
        并行执行模型一轮请求的多个工具调用。
        Args:
            calls: 模型这一轮返回的全部 tool_calls
        Returns:
            结果列表, 顺序与 calls 严格一一对应(方便 zip 回 tool_call_id)
        """
        if not calls:
            return []

        return await asyncio.gather(*(self.execute_async(call) for call in calls))

    def close(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==================== 便捷函数 ====================
async def run_parallel_tools(
    registry: ToolRegistry,
    calls: List[ToolCall],
    max_workers: int = 4,
) -> List[str]:
    """便捷函数: 并行执行多个工具调用"""
    executor = AsyncToolExecutor(registry, max_workers)
    try:
        return await executor.execute_many(calls)
    finally:
        executor.close()


def execute_many_sync(
    registry: ToolRegistry,
    calls: List[ToolCall],
    max_workers: int = 4,
) -> List[str]:
    """
    并行执行多个工具调用, 同步版本 。

    返回结果严格按 calls 的顺序排列(不是完成顺序):
    结果要靠 zip 回 tool_call_id 配对, 顺序错位就会把 A 工具的结果当成 B 的。
    """
    if not calls:
        return []

    # max_workers 不能超过任务数, 否则白白开线程; 也不能为 0, 否则 ValueError
    workers = max(1, min(max_workers, len(calls)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(registry.execute, call) for call in calls]

        results = []
        for call, future in zip(calls, futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append(f"错误: 工具 '{call.name}' 执行失败: {e}")
        return results


def run_parallel_tools_sync(
    registry: ToolRegistry,
    calls: List[ToolCall],
    max_workers: int = 4,
) -> List[str]:
    """
    同步版本, 供非 async 环境使用。
    """
    return execute_many_sync(registry, calls, max_workers)


# ==================== 旧版本- 注释保留, 仅供参考 ====================
# 旧版的问题:
# 1. 走字符串接口 execute_tool(name, input_data), 传不了结构化参数。
# 2. 任务用 dict {"tool_name", "input_data"} 表达, 应改为 list[ToolCall]。
# 3. asyncio.get_event_loop() 在 Python 3.10+ 已废弃。
# 4. 【隐藏 bug】它自称"并行执行", 其实完全是串行的:
#       async_task = self.execute_tool_async(...)   # 只是创建协程对象, 还没开始跑
#       ...
#       result = await async_task                   # 逐个 await, 等上一个跑完才轮到下一个
#    创建协程对象并不会执行它, 必须 await 才开始。所以整段是"一个一个来"。
#    真正的并行要用 asyncio.gather 一次性调度(新版已修正)。
#
# class AsyncToolExecutor:
#     """异步工具执行器"""
#
#     def __init__(self, registry: ToolRegistry, max_workers: int = 4):
#         self.registry = registry
#         self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
#
#     async def execute_tool_async(self, tool_name: str, input_data: str) -> str:
#         """异步执行单个工具"""
#         loop = asyncio.get_event_loop()
#
#         def _execute():
#             return self.registry.execute_tool(tool_name, input_data)
#
#         try:
#             result = await loop.run_in_executor(self.executor, _execute)
#             return result
#         except Exception as e:
#             return f"❌ 工具 '{tool_name}' 异步执行失败: {e}"
#
#     async def execute_tools_parallel(self, tasks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
#         """并行执行多个工具(实际是串行)"""
#         async_tasks = []
#         for i, task in enumerate(tasks):
#             tool_name = task.get("tool_name")
#             input_data = task.get("input_data", "")
#             if not tool_name:
#                 continue
#             async_task = self.execute_tool_async(tool_name, input_data)
#             async_tasks.append((i, task, async_task))
#
#         results = []
#         for i, task, async_task in async_tasks:
#             try:
#                 result = await async_task          # ← 逐个 await, 串行
#                 results.append({
#                     "task_id": i,
#                     "tool_name": task["tool_name"],
#                     "input_data": task["input_data"],
#                     "result": result,
#                     "status": "success"
#                 })
#             except Exception as e:
#                 results.append({
#                     "task_id": i,
#                     "tool_name": task["tool_name"],
#                     "input_data": task["input_data"],
#                     "result": str(e),
#                     "status": "error"
#                 })
#         return results
#
#     async def execute_tools_batch(self, tool_name: str, input_list: List[str]) -> List[Dict[str, Any]]:
#         """批量执行同一个工具"""
#         tasks = [{"tool_name": tool_name, "input_data": x} for x in input_list]
#         return await self.execute_tools_parallel(tasks)
#
#
# async def run_batch_tool(registry, tool_name, input_list, max_workers=4):
#     """(旧)批量执行同一个工具 —— function calling 下模型会直接发多个 tool_calls, 不再需要"""
#     async with AsyncToolExecutor(registry, max_workers) as executor:
#         return await executor.execute_tools_batch(tool_name, input_list)
#
#
# def run_batch_tool_sync(registry, tool_name, input_list, max_workers=4):
#     return asyncio.run(run_batch_tool(registry, tool_name, input_list, max_workers))
