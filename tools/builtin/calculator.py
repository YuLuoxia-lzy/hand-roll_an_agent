"""计算器工具"""

import ast
import operator
import math
from typing import Dict, Any

from ..base import Tool, ToolParameter
from ...utils.logging import get_logger

logger = get_logger(__name__)

class CalculatorTool(Tool):
    """Python计算器工具"""

    # 乘方护栏: 指数绝对值超过这个数就拒绝计算。
    # 为什么必须有: 9**9**9 的指数是 387420489, Python 会老老实实去构造一个
    # 3.8 亿位的大整数 —— 进程直接卡死, 内存飙升, 几百秒都没有反应。
    # 模型很容易一本正经地生成这种表达式(它在纸上算不出来, 就交给工具算),
    # 所以必须在**开始算之前**拦住, 算到一半再拦已经来不及了。
    MAX_EXPONENT = 1000

    # 支持的操作符
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        # 整除和取余: 模型的输出里很常见(AI 算"每3人一组能分几组"就会写 //),
        # 之前不在表里, 会撞出一个 KeyError, 报错信息对模型毫无帮助。
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        # 注意: 这里**故意不放** ast.BitXor。'^' 在 Python 里是按位异或,
        # 而几乎所有人都以为它是乘方, 见 _eval_node 里的处理。
        ast.USub: operator.neg,
    }
    
    # 支持的函数
    FUNCTIONS = {
        'abs': abs,
        'round': round,
        'max': max,
        'min': min,
        'sum': sum,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'exp': math.exp,
        'pi': math.pi,
        'e': math.e,
    }
    
    def __init__(self):
        super().__init__(
            name="python_calculator",
            description="执行数学计算。支持基本运算、数学函数等。例如：2+3*4, sqrt(16), sin(pi/2)等。"
        )
    
    def run(self, parameters: Dict[str, Any]) -> str:
        """
        执行计算

        Args:
            parameters: 包含input参数的字典

        Returns:
            计算结果
        """
        # 支持两种参数格式：input 和 expression
        expression = parameters.get("input", "") or parameters.get("expression", "")
        if not expression:
            return "错误：计算表达式不能为空"

        logger.debug("正在计算: %s", expression)

        try:
            # 解析表达式
            node = ast.parse(expression, mode='eval')
            result = self._eval_node(node.body)
            result_str = str(result)
            logger.info("计算 %s = %s", expression, result_str)
            return result_str
        except Exception as e:
            # 失败也返回字符串而不是抛异常 —— 这条信息会作为工具结果回传给模型,
            # 模型看了能自己改表达式重试。整条链路的设计是"把错误变成模型的输入"。
            error_msg = f"计算失败: {str(e)}"
            logger.warning("%s (表达式: %s)", error_msg, expression)
            return error_msg

    def _eval_node(self, node):
        """递归计算AST节点"""
        if isinstance(node, ast.Constant):  # Python 3.8+ 数字/字符串都是 Constant
            return node.value
        # 这里原来还有一条 `elif isinstance(node, ast.Num): return node.n`(给 Python 3.8 以前用)。
        # 已删除: ast.Num 从 3.12 起标记废弃、3.14 会彻底移除, 光是**引用**它就会发
        # DeprecationWarning —— 本项目的测试配置把废弃警告当错误, 于是每个非常量节点
        # (比如 2+3*4 里的 BinOp)都会在走到这一行时直接炸。
        # 本项目要求 Python >= 3.9, ast.Constant 已经完全覆盖, 这个分支是纯粹的死代码。
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)

            # '^' 在 Python 里是【按位异或】: 2^3 算出 1, 不是 8。
            # 模型说"计算 2 的 3 次方"时经常写成 2^3, 直接算出 1 会让它
            # 拿着一个错答案继续往下推理, 而且完全看不出哪里错了。
            # 所以宁可报错, 把正确写法告诉它。
            if op_type is ast.BitXor:
                raise ValueError(
                    f"'^' 在 Python 里是按位异或, 不是乘方。"
                    f"要算乘方请把 '{ast.unparse(node)}' 写成 "
                    f"'{ast.unparse(node.left)} ** {ast.unparse(node.right)}'"
                )

            if op_type is ast.Pow:
                # 先算指数(它本身可能是个表达式), 再决定要不要算下去
                exponent = self._eval_node(node.right)
                if isinstance(exponent, (int, float)) and abs(exponent) > self.MAX_EXPONENT:
                    raise ValueError(
                        f"指数 {exponent} 超出上限 {self.MAX_EXPONENT}, "
                        f"结果会是天文数字, 拒绝计算以免卡死进程"
                    )
                return self._eval_node(node.left) ** exponent

            func = self.OPERATORS.get(op_type)
            if func is None:
                # 用 .get 而不是 []: 遇到没收录的运算符(比如 a << b)时,
                # 原来的 KeyError 只会打印一个 <class 'ast.LShift'>,
                # 对模型来说等于什么都没说。
                raise ValueError(f"不支持的运算符: {op_type.__name__}")
            return func(
                self._eval_node(node.left),
                self._eval_node(node.right)
            )
        elif isinstance(node, ast.UnaryOp):
            return self.OPERATORS[type(node.op)](self._eval_node(node.operand))
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name in self.FUNCTIONS:
                args = [self._eval_node(arg) for arg in node.args]
                return self.FUNCTIONS[func_name](*args)
            else:
                raise ValueError(f"不支持的函数: {func_name}")
        elif isinstance(node, ast.Name):
            if node.id in self.FUNCTIONS:
                return self.FUNCTIONS[node.id]
            else:
                raise ValueError(f"未定义的变量: {node.id}")
        else:
            raise ValueError(f"不支持的表达式类型: {type(node)}")
    
    def get_parameters(self):
        """获取工具参数定义"""
        return [
            ToolParameter(
                name="input",
                type="string",
                description="要计算的数学表达式，支持基本运算和数学函数",
                required=True
            )
        ]

# 便捷函数
def calculate(expression: str) -> str:
    """
    执行数学计算

    Args:
        expression: 数学表达式

    Returns:
        计算结果字符串
    """
    tool = CalculatorTool()
    return tool.run({"input": expression})
