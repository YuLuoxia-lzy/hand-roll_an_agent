"""
工具注册表 - 已适配 function calling
"""

from typing import Optional
from ..core.exceptions import *
from ..core.typedefs import ToolCall          # 【新】模型发出的工具调用请求(结构化)
from .base import Tool
from ..utils.logging import get_logger

logger = get_logger(__name__)

class ToolRegistry:
    """
    工具注册表

    提供工具的注册、管理和执行功能。

    只有一种注册方式: register_tool(Tool 实例)。
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
        获取所有工具的 JSON Schema 列表 —— Agent 循环的入口
            response = llm.invoke(messages, tools=registry.get_tools_schema())
        注意返回的是 list: OpenAI 的 tools 参数要"工具列表", 哪怕只有一个工具也要包在 [] 里
        """
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def execute(self, call: ToolCall) -> str:
        """
        执行模型请求的工具调用
            call: 模型返回的结构化工具调用(含 name 和已解析成字典的 arguments)
        Returns:
            工具执行结果字符串。失败时返回"错误: xxx"而不是抛异常 ——
            因为返回值会作为 tool 消息回传给模型, 模型看到错误能自己换个参数重试。

        需要知道"这次到底成功没有"的调用方(比如工具链)请用 execute_with_status,
        **别去读返回值的前缀** —— 见那个方法的说明。
        """
        return self._execute(call)[0]

    def execute_with_status(self, call: ToolCall) -> tuple:
        """
        和 execute() 一样, 但额外告诉你成功还是失败。返回 (结果, ok)。

        【为什么必须多这一个方法, 而不是让调用方 startswith("错误")】
        前缀是**内容**, 不是协议。拿它当判据两头都会错:
          - 假失败: 工具本来就可能返回一段以"错误"开头的正文(检索到的报错日志、
                    一段讲错误处理的文档)。链会就此中断, 而那次调用明明是成功的。
          - 假成功: 计算器失败时返回 "计算失败: division by zero" —— 不匹配前缀。
                    链会拿着这个字符串当结果继续往下跑, 把错值喂给下一步,
                    最后交出一个**看起来算过、其实基于垃圾**的答案。
        所以判断只能来自执行路径本身, 不能来自读结果。工具自己吞掉异常的那些
        (计算器就是), 由它声明 error_prefixes —— 见 tools/base.py。
        """
        return self._execute(call)

    def _execute(self, call: ToolCall) -> tuple:
        """execute / execute_with_status 的唯一实现 —— 判定逻辑只能有一份。"""
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(self._tools.keys()) or "无"
            return f"错误:未找到名为 '{call.name}' 的工具。可用工具: {available}", False

        # 参数校验: 缺参数时不执行, 直接把"缺什么"告诉模型
        error = tool.validate_arguments(call.arguments)
        if error:
            return f"错误:{error}", False

        try:
            raw = tool.run(call.arguments)
        except Exception as e:
            return f"错误：执行工具 '{call.name}' 时发生异常: {str(e)}", False

        # 长度上限在这里施加, 而不是让每个工具自己记得调 —— 这是所有工具结果的
        # 唯一出口, 放在这儿才不会有漏网的。必须在结果进入 messages **之前**做:
        # 一旦写进轨迹, 它就成了"模型当年看到的"，事后再裁剪会让轨迹和真实请求对不上。
        result = tool.truncate_output(raw)

        # 工具自己吞掉的那个失败(它声明了 error_prefixes)也算失败
        if tool.looks_like_error(result):
            return result, False
        return result, True

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
# 这里原本还有 register_function / get_function / execute_tool 三个方法、
# 一个 _functions 字典、以及一个 get_tools_description()。都已删除,
# 原文和删除理由见本地归档 docs/archived-code.md 的《registry.py》一节。
#
# 想注册一个简单工具, 现在统一写成:
#
#     from agents0to1.tools import Tool, ToolParameter
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
