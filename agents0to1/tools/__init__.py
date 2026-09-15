"""工具系统"""

from .base import Tool, ToolParameter
from .registry import ToolRegistry, global_registry

# 内置工具
# 之前注释掉是因为 search.py 里 import 了一个不存在的 HelloAgentsException, 一导入就报错。
# 那个 import 已经修好, 所以这里可以放开了。
from .builtin.search import SearchTool
from .builtin.calculator import CalculatorTool
# 知识检索工具: 需要自己传一个 SemanticMemory 进来, 所以不在 builtin 里自动实例化
from .builtin.knowledge import KnowledgeSearchTool

# 高级功能
# 注意: 旧的 create_research_chain / create_simple_chain 已废弃(写死了不存在的工具名),
# 新版用 chain.add_step(工具名, {参数名: 值}) 自行搭建
from .chain import ToolChain, ToolChainManager, ChainStep
from .async_executor import (
    AsyncToolExecutor,
    execute_many_sync,
    run_parallel_tools,
    run_parallel_tools_sync,
)

__all__ = [
    # 基础工具系统
    "Tool",
    "ToolParameter",
    "ToolRegistry",
    "global_registry",

    # 内置工具
    "SearchTool",
    "CalculatorTool",
    "KnowledgeSearchTool",

    # 工具链功能
    "ToolChain",
    "ToolChainManager",
    "ChainStep",

    # 异步执行功能
    "AsyncToolExecutor",
    "execute_many_sync",
    "run_parallel_tools",
    "run_parallel_tools_sync",
]
