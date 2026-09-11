"""
工具注册表 - 已适配 function calling
"""

from typing import Optional
from ..core.exceptions import *
from ..core.types import ToolCall          # 【新】模型发出的工具调用请求(结构化)
from .base import Tool
from ..utils.logging import get_logger

logger = get_logger(__name__)

class ToolRegistry:
    """
    工具注册表

    提供工具的注册、管理和执行功能。

    只有一种注册方式: register_tool(Tool 实例)。
    正则时代还支持"直接注册一个 Callable[[str], str] 函数", 那套机制已随
    execute_tool 一起删掉了 —— 原因见文件末尾的说明。
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """
        注册Tool对象

        Args:
            tool: Tool实例
        """
        if tool.name in self._tools:
            logger.warning("工具 '%s' 已存在, 将被覆盖。", tool.name)

        self._tools[tool.name] = tool
        logger.info("工具 '%s' 已注册。", tool.name)

    def unregister(self, name: str):
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            logger.info("工具 '%s' 已注销。", name)
        else:
            logger.warning("工具 '%s' 不存在, 无需注销。", name)

    def get_tool(self, name: str) -> Optional[Tool]:
        """获取Tool对象"""
        return self._tools.get(name)

    # ==================== 新: function calling 主路径 ====================

    def get_tools_schema(self) -> list[dict]:
        """
        【新】获取所有工具的 JSON Schema 列表 —— Agent 循环的入口

        用法:
            response = llm.invoke(messages, tools=registry.get_tools_schema())

        注意返回的是 list: OpenAI 的 tools 参数要"工具列表", 哪怕只有一个工具也要包在 [] 里
        (传单个 dict 会报 400: tools: invalid type: map, expected a sequence)
        """
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def execute(self, call: ToolCall) -> str:
        """
        【新】执行模型请求的工具调用

        Args:
            call: 模型返回的结构化工具调用(含 name 和已解析成字典的 arguments)

        Returns:
            工具执行结果字符串。失败时返回"错误: xxx"而不是抛异常 ——
            因为返回值会作为 tool 消息回传给模型, 模型看到错误能自己换个参数重试。
        """
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(self._tools.keys()) or "无"
            return f"错误:未找到名为 '{call.name}' 的工具。可用工具: {available}"

        # 参数校验: 缺参数时不执行, 直接把"缺什么"告诉模型
        error = tool.validate_arguments(call.arguments)
        if error:
            return f"错误:{error}"

        try:
            # 参数是结构化字典(不再是"2 + 2"这种字符串), 直接整体传给工具
            return tool.run(call.arguments)
        except Exception as e:
            return f"错误：执行工具 '{call.name}' 时发生异常: {str(e)}"

    # ==================== 旧: 工具描述文本(被 get_tools_schema 取代) ====================
    # 这是正则时代的做法: 把工具描述拼成一段文字塞进提示词, 让模型"照着文字说"要用哪个工具。
    # function calling 下工具的说明走 API 的 tools 参数(JSON Schema), 不再需要这段文字。
    # def get_tools_description(self) -> str:
    #     """
    #     获取所有可用工具的格式化描述字符串
    #
    #     Returns:
    #         工具描述字符串，用于构建提示词
    #     """
    #     descriptions = []
    #
    #     # Tool对象描述
    #     for tool in self._tools.values():
    #         descriptions.append(f"- {tool.name}: {tool.description}")
    #
    #     # 函数工具描述
    #     for name, info in self._functions.items():
    #         descriptions.append(f"- {name}: {info['description']}")
    #
    #     return "\n".join(descriptions) if descriptions else "暂无可用工具"

    def list_tools(self) -> list[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())

    def get_all_tools(self) -> list[Tool]:
        """获取所有Tool对象"""
        return list(self._tools.values())

    def clear(self):
        """清空所有工具"""
        self._tools.clear()
        logger.info("所有工具已清空。")

# 全局工具注册表
#
# 【注意】全局单例是隐式共享状态。多个 Agent 想要不同工具集时, 它们会互相污染 ——
# A Agent 注册的工具, B Agent 也会看到, 而且模型真的会去调它。
# 新代码请自己 ToolRegistry() 然后显式传给 Agent, 这个全局的只留给一次性小脚本用。
global_registry = ToolRegistry()


# ==================== 已删除: 函数式注册 + 字符串参数执行 ====================
# 这里原本还有 register_function / get_function / execute_tool 三个方法和一个
# _functions 字典, 本次一并删掉了。理由:
#
# 1. 它们在 function calling 架构下**根本跑不通**。模型要调用一个工具, 前提是
#    API 的 tools 参数里有它的 JSON Schema; 而 get_tools_schema() 只遍历 _tools,
#    _functions 里的工具对模型完全不可见 —— 模型永远不会去调它, 于是那段代码
#    永远等不到调用方。原来的注释写着"保留能用", 其实"不能用"。
#
# 2. execute_tool 走的是字符串参数 tool.run({"input": input_text}), 只能表达
#    "一个字符串进、一个字符串出"。参数一旦是结构化的(比如计算器的 expression
#    + 精度), 这么塞就会把参数名填错。
#
# 想注册一个简单工具, 现在统一写成:
#
#     from Agent_0_to_1.tools import Tool, ToolParameter
#
#     class EchoTool(Tool):
#         def __init__(self):
#             super().__init__(name="echo", description="原样返回输入")
#         def get_parameters(self):
#             return [ToolParameter(name="text", type="string", description="要回显的文本")]
#         def run(self, params):
#             return params.get("text", "")
#
#     registry.register_tool(EchoTool())
