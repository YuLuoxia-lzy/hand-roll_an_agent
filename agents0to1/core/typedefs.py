"""框架的核心数据类型"""

from dataclasses import dataclass, field
import json
from typing import Literal, Optional



@dataclass
class ToolCall:
    """工具调用所用格式"""
    id: str          # 调用ID, 
    name: str        # 工具名
    arguments: dict  # 参数: 解析成字典

@dataclass
class LLMResponse:
    """大模型回应所需格式"""
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        """模型是否请求调用工具"""
        return len(self.tool_calls) > 0


"""
事件类型
    type 取值 与 对应字段:
        "text"        -> text              最终答案的正文增量, 逐块吐, 拼起来是本轮完整正文
        "thinking"    -> text              模型在发起工具调用前说的话(整段, 不是增量)
        "tool_call"   -> call              模型请求了一次工具调用
        "tool_result" -> call + result     该工具的执行结果
        "final"       -> answer            最终答案(整段, 不是增量)
"""
EventType = Literal["text", "tool_call", "tool_result", "thinking", "final"]


@dataclass
class AgentEvent:
    """
    流式运行 Agent。

    type 取值 与 对应字段:
        "text"        -> text              最终答案的正文增量, 逐块吐, 拼起来是本轮完整正文
        "thinking"    -> text              模型在发起工具调用前说的话(整段, 不是增量)
        "tool_call"   -> call              模型请求了一次工具调用
        "tool_result" -> call + result     该工具的执行结果
        "final"       -> answer            最终答案(整段, 不是增量)

    例如 ReAct 一轮里既有"思考正文"又有"工具调用",
    直接吐字符串的话调用方分不清哪段是过程、哪段是结论, 也没法在后续网页上把工具调用渲染成一张卡片。

    UI 的常规做法是把 text 事件累积拼成最终答案。
    如果把"我先搜一下…"这种过程叙述也塞进 text, 它会混进最终答案里把结果搞脏。
        text     是**增量**, 调用方自己拼接
        thinking 是**整段**, 直接展示 —— 它和写进 messages 的那份必须逐字节相同,
                 逐块吐的话屏幕上看到的内容会取决于有没有丢块, 又变成两份真相了
    """
    type: EventType
    text: Optional[str] = None
    call: Optional[ToolCall] = None
    result: Optional[str] = None
    answer: Optional[str] = None


def safe_parse_arguments(raw: Optional[str]) -> dict:
    """
    把模型给的参数 JSON 字符串解析成字典。**坏 JSON 不抛异常, 退回空字典。**
    退回空字典而不是抛错, 是因为空字典会让 Tool.validate_arguments 报错
    整条链路的设计是"把错误变成模型的输入", 所以这里不能中断。
    """
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}

    # 模型偶尔会输出 JSON 数组、字符串或数字而不是对象, 同样退回空字典
    return parsed if isinstance(parsed, dict) else {}


def parse_tool_calls(raw_tool_calls) -> list[ToolCall]:
    """把SDK返回的原始tool_calls转成框架自己的ToolCall(arguments完成JSON解析)"""
    result = []
    for tc in raw_tool_calls or []:
        function = getattr(tc, "function", None)
        if function is None:        # 结构不完整的条目直接跳过, 不让它把整轮响应带崩
            continue

        result.append(ToolCall(
            id=tc.id,
            name=function.name,
            arguments=safe_parse_arguments(function.arguments),
        ))
    return result
