"""
工具链管理器 - 已适配 function calling
"""

from typing import List, Dict, Any, Optional
from ..core.types import ToolCall
from .registry import ToolRegistry


class ChainStep:
    """工具链的一步: 调哪个工具 + 各参数从哪来"""

    def __init__(self, tool_name: str, arguments: Dict[str, Any], output_key: Optional[str] = None):
        self.tool_name = tool_name
        self.arguments = arguments
        # 结果的存放键名, 供后续步骤用 "{键名}" 引用
        self.output_key = output_key or f"step_{tool_name}"


class ToolChain:
    """工具链 - 按固定顺序执行多个工具调用"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: List[ChainStep] = []

    def add_step(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        output_key: Optional[str] = None,
    ):
        """
        添加一个步骤

        Args:
            tool_name: 工具名(必须与注册表里的工具名一致)
            arguments: 参数字典。值可以是字面量, 也可以是 "{占位符}" 引用前面的结果
            output_key: 本步结果的键名, 供后续步骤引用

        例(注意 search 和 python_calculator 的参数名都叫 input, 不是 query/expression):
            chain.add_step("search", {"input": "{question}"}, output_key="search_result")
            chain.add_step("python_calculator", {"input": "{search_result}"})

        其中 "{question}" 由 execute(registry, {"question": "..."}) 的初始参数提供。
        """
        step = ChainStep(tool_name, arguments, output_key)
        self.steps.append(step)
        return self

    def validate(self, registry: ToolRegistry) -> Optional[str]:
        """
        校验所有步骤: 工具是否已注册 + 必填参数名是否齐全。
        返回 None 表示通过, 否则返回错误信息。

        建议在注册工具之后、执行之前调用一次 —— 这样"工具名写错""参数名写错"
        在启动时就能发现, 而不是跑到那一步才崩。

        参数名也校验的原因: 光看工具在不在是不够的。参数名写错时 registry.execute
        不会报错, 它会把"缺少必填参数"当成**工具结果**回传给模型 ——
        模型于是拿着一条自己看不懂的错误, 一头雾水地重试。
        这种错在 validate 阶段拦住最省事。
        """
        for i, step in enumerate(self.steps, 1):
            tool = registry.get_tool(step.tool_name)
            if tool is None:
                available = ", ".join(registry.list_tools()) or "无"
                return f"第 {i} 步引用的工具 '{step.tool_name}' 未注册。可用工具: {available}"

            declared = tool.get_parameters()
            missing = [
                p.name for p in declared
                if p.required and p.name not in step.arguments
            ]
            if missing:
                names = ", ".join(p.name for p in declared) or "无"
                return (
                    f"第 {i} 步 [{step.tool_name}] 缺少必填参数 {missing}; "
                    f"该工具声明的参数是: {names}"
                )
        return None

    def execute(self, registry: ToolRegistry, initial_arguments: Optional[Dict[str, Any]] = None) -> str:
        """
        执行工具链

        Args:
            registry: 工具注册表
            initial_arguments: 初始参数, 供第一步用 "{键名}" 引用

        Returns:
            最后一步的执行结果。任何一步失败就停止, 并返回错误信息。
        """
        if not self.steps:
            return "错误: 工具链为空, 无法执行"

        # 上下文 = 初始参数 + 各步骤结果, 占位符就从这个字典里取值
        context: Dict[str, Any] = dict(initial_arguments or {})
        final_result = ""

        for i, step in enumerate(self.steps, 1):
            # 1. 把参数字典里的 "{占位符}" 换成实际值
            #    捕获 ValueError(而不是只有 KeyError): 占位符写错有好几种形式,
            #    _resolve_arguments 统一用 ValueError 报出来, 这里一处接住。
            try:
                arguments = self._resolve_arguments(step.arguments, context)
            except ValueError as e:
                return f"错误: 第 {i} 步的参数占位符无法解析 -> {e}"

            # 2. 走 function calling 的标准执行路径
            call = ToolCall(
                id=f"{self.name}_step_{i}",
                name=step.tool_name,
                arguments=arguments,
            )
            result = registry.execute(call)

            # 3. 工具执行失败(registry 把错误作为字符串返回) -> 停止链条
            if isinstance(result, str) and result.startswith("错误"):
                return f"错误: 第 {i} 步 [{step.tool_name}] 执行失败 -> {result}"

            context[step.output_key] = result
            final_result = result

        return final_result

    def _resolve_arguments(self, value: Any, context: Dict[str, Any]) -> Any:
        """
        递归替换参数里的占位符。

        两种替换方式:
        - 整个值就是 "{key}"  -> 原类型透传(dict/list/数字都能直接用)
        - 值里混有 "{key}"     -> 字符串插值(结果一定是字符串)

        占位符有问题时统一抛 ValueError, 由 execute 接住变成错误串 ——
        这样调用方只需要认识一种异常。
        """
        if isinstance(value, str):
            if value.startswith("{") and value.endswith("}") and value.count("{") == 1:
                key = value[1:-1]
                if key not in context:
                    raise ValueError(
                        f"引用了不存在的占位符 '{key}'; 当前可用的有: {self._context_keys(context)}"
                    )
                return context[key]

            try:
                return value.format(**context)
            except KeyError as e:
                raise ValueError(
                    f"引用了不存在的占位符 {e}; 当前可用的有: {self._context_keys(context)}"
                ) from e
            except (IndexError, ValueError) as e:
                # "{0}" 这类位置占位符、或者花括号没配对 —— .format 抛的是
                # IndexError / ValueError, **不是** KeyError。
                # 原来只捕获 KeyError, 这两种会一路冒到调用方把整条链带崩。
                raise ValueError(f"占位符写法有误: {value!r} ({e})") from e

        if isinstance(value, dict):
            return {k: self._resolve_arguments(v, context) for k, v in value.items()}

        if isinstance(value, list):
            return [self._resolve_arguments(v, context) for v in value]

        return value

    @staticmethod
    def _context_keys(context: Dict[str, Any]) -> str:
        """把上下文里已有的键名列出来, 方便对着错误信息改占位符"""
        return ", ".join(str(k) for k in context) or "无(初始参数为空, 且还没有步骤产出了结果)"


class ToolChainManager:
    """工具链管理器 - 管理多条工具链"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: Dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain) -> Optional[str]:
        """
        注册工具链, 同时校验它的工具名是否都已注册。
        返回 None 表示成功; 否则返回错误信息且不注册。
        """
        error = chain.validate(self.registry)
        if error:
            return f"工具链 '{chain.name}' 校验失败: {error}"

        self.chains[chain.name] = chain
        return None

    def execute_chain(self, chain_name: str, initial_arguments: Optional[Dict[str, Any]] = None) -> str:
        """执行指定的工具链"""
        chain = self.chains.get(chain_name)
        if chain is None:
            return f"错误: 工具链 '{chain_name}' 不存在"
        return chain.execute(self.registry, initial_arguments)

    def list_chains(self) -> List[str]:
        """列出所有已注册的工具链"""
        return list(self.chains.keys())

    def get_chain_info(self, chain_name: str) -> Optional[Dict[str, Any]]:
        """获取工具链信息"""
        chain = self.chains.get(chain_name)
        if chain is None:
            return None

        return {
            "name": chain.name,
            "description": chain.description,
            "steps": len(chain.steps),
            "step_details": [
                {
                    "tool_name": step.tool_name,
                    "arguments": step.arguments,
                    "output_key": step.output_key,
                }
                for step in chain.steps
            ],
        }


# ==================== 旧版本(正则时代) - 注释保留, 仅供参考 ====================
# 旧版的问题:
# 1. 参数走字符串接口 execute_tool(name, input_text), 底层传 {"input": ...},
#    但 function calling 的工具参数名各不相同(计算器是 expression, 搜索是 query)。
# 2. 变量替换 input_template.format(**context) 只能把上一步的整个输出塞进下一步,
#    无法"从结果里取出某个字段"填进对应的参数名。
# 3. 便捷函数写死了不存在的工具名 "my_calculator", 一跑就报"未找到工具"。
#
# class ToolChain:
#     """工具链 - 支持多个工具的顺序执行"""
#
#     def __init__(self, name: str, description: str):
#         self.name = name
#         self.description = description
#         self.steps: List[Dict[str, Any]] = []
#
#     def add_step(self, tool_name: str, input_template: str, output_key: str = None):
#         step = {
#             "tool_name": tool_name,
#             "input_template": input_template,
#             "output_key": output_key or f"step_{len(self.steps)}_result"
#         }
#         self.steps.append(step)
#         print(f"✅ 工具链 '{self.name}' 添加步骤: {tool_name}")
#
#     def execute(self, registry: ToolRegistry, input_data: str, context: Dict[str, Any] = None) -> str:
#         if not self.steps:
#             return "❌ 工具链为空，无法执行"
#
#         print(f"🚀 开始执行工具链: {self.name}")
#
#         if context is None:
#             context = {}
#         context["input"] = input_data
#
#         final_result = input_data
#
#         for i, step in enumerate(self.steps):
#             tool_name = step["tool_name"]
#             input_template = step["input_template"]
#             output_key = step["output_key"]
#
#             try:
#                 actual_input = input_template.format(**context)
#             except KeyError as e:
#                 return f"❌ 模板变量替换失败: {e}"
#
#             try:
#                 result = registry.execute_tool(tool_name, actual_input)
#                 context[output_key] = result
#                 final_result = result
#             except Exception as e:
#                 return f"❌ 工具 '{tool_name}' 执行失败: {e}"
#
#         return final_result
#
#
# class ToolChainManager:
#     """工具链管理器"""
#
#     def __init__(self, registry: ToolRegistry):
#         self.registry = registry
#         self.chains: Dict[str, ToolChain] = {}
#
#     def register_chain(self, chain: ToolChain):
#         self.chains[chain.name] = chain
#         print(f"✅ 工具链 '{chain.name}' 已注册")
#
#     def execute_chain(self, chain_name: str, input_data: str, context: Dict[str, Any] = None) -> str:
#         if chain_name not in self.chains:
#             return f"❌ 工具链 '{chain_name}' 不存在"
#         chain = self.chains[chain_name]
#         return chain.execute(self.registry, input_data, context)
#
#
# # 便捷函数: 写死了不存在的工具名 "my_calculator"(实际是 "python_calculator"), 已废弃
# def create_research_chain() -> ToolChain:
#     """创建一个研究工具链：搜索 -> 计算 -> 总结"""
#     chain = ToolChain(name="research_and_calculate", description="搜索信息并进行相关计算")
#     chain.add_step(tool_name="search", input_template="{input}", output_key="search_result")
#     chain.add_step(tool_name="my_calculator", input_template="2 + 2", output_key="calc_result")
#     return chain
#
#
# def create_simple_chain() -> ToolChain:
#     """创建一个简单的工具链示例"""
#     chain = ToolChain(name="simple_demo", description="简单的工具链演示")
#     chain.add_step(tool_name="my_calculator", input_template="{input}", output_key="result")
#     return chain
