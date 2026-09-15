"""
工具链管理器 - 已适配 function calling
"""

from typing import List, Dict, Any, Optional
from ..core.typedefs import ToolCall
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
            (需要区分"结果"和"错误串"的调用方请用 execute_with_status。)
        """
        return self.execute_with_status(registry, initial_arguments)[0]

    def execute_with_status(
        self,
        registry: ToolRegistry,
        initial_arguments: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """
        和 execute() 一样, 但额外告诉你成功还是失败。返回 (结果, ok)。

        【为什么不是读结果前缀】
        初版的判据是 `result.startswith("错误")` —— 那是拿**内容**当协议,
        两头都会错:
          - 假失败: 某个工具正常返回一段以"错误"开头的正文(检索到的报错日志、
                    讲异常处理的文档), 链会就此中断, 而那次调用是成功的。
          - 假成功: 计算器失败时返回 "计算失败: division by zero" —— 不匹配
                    前缀。链会把这个字符串当结果继续往下传, 后面每一步都基于
                    一个错值算, 最后交出一个**看起来算过、其实全是垃圾**的答案。
        真正该问的是"这次执行失败了吗", 答案只有执行路径本身知道 ——
        所以走 registry.execute_with_status()。
        """
        if not self.steps:
            return "错误: 工具链为空, 无法执行", False

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
                return f"错误: 第 {i} 步的参数占位符无法解析 -> {e}", False

            # 2. 走 function calling 的标准执行路径
            call = ToolCall(
                id=f"{self.name}_step_{i}",
                name=step.tool_name,
                arguments=arguments,
            )
            result, ok = registry.execute_with_status(call)

            # 3. 这一步失败了 -> 停止链条。**判据来自执行路径, 不是读结果字符串**
            if not ok:
                return f"错误: 第 {i} 步 [{step.tool_name}] 执行失败 -> {result}", False

            context[step.output_key] = result
            final_result = result

        return final_result, True

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
        return self.execute_chain_with_status(chain_name, initial_arguments)[0]

    def execute_chain_with_status(
        self,
        chain_name: str,
        initial_arguments: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """
        和 execute_chain() 一样, 但额外告诉你成功还是失败。返回 (结果, ok)。

        多这一层的原因和 ToolChain.execute_with_status 完全一样: 上一层要是
        还靠读前缀判断, 下面的修复就白做了 —— 错误信号在传上去的路上又丢了。
        """
        chain = self.chains.get(chain_name)
        if chain is None:
            return f"错误: 工具链 '{chain_name}' 不存在", False
        return chain.execute_with_status(self.registry, initial_arguments)

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


# ==================== 旧版本(正则时代) ====================
# 原文和逐条问题见本地归档 docs/archived-code.md 的《chain.py》一节。一句话版:
# 参数走字符串接口、模板替换只能整段塞、便捷函数写死了不存在的工具名。
