"""
ReAct Agent实现 - 已适配 function calling
"""

import json
from typing import Iterator, Optional, List
from ..core.agent import Agent
from ..core.llm import Agents0to1
from ..core.config import Config
from ..core.types import AgentEvent, LLMResponse, ToolCall
from ..tools.registry import ToolRegistry
from ..tools.async_executor import execute_many_sync
from ..utils.logging import get_logger

logger = get_logger(__name__)

MAX_STEPS_ANSWER = "抱歉，我无法在限定步数内完成这个任务。"

DEFAULT_REACT_PROMPT = """
你是一个具备推理和行动能力的AI助手，可以调用工具来获取信息，最终给出准确的答案。

## 工作方式
1. 先分析问题，判断需要哪些信息
2. 需要外部信息时，调用合适的工具（一轮可以调用多个）
3. 拿到工具结果后继续推理；信息足够时，直接用自然语言给出最终答案
4. 不要编造工具没有返回的信息
5. 工具返回错误时，可以调整参数重试，或换一个工具

注意：工具调用通过 API 的工具接口进行，你不需要在正文里手写"调用某工具"的格式。
"""


class ReActAgent(Agent):
    """
    ReAct (Reasoning and Acting) Agent

    结合推理和行动的智能体，能够：
    1. 分析问题并制定行动计划
    2. 调用外部工具获取信息
    3. 基于观察结果进行推理
    4. 迭代执行直到得出最终答案

    循环的本质: 调 LLM -> 有 tool_calls 就执行并回传 -> 再调 LLM -> 直到没有 tool_calls。
    """

    def __init__(
        self,
        name: str,
        llm: Agents0to1,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None
    ):
        """
        初始化ReActAgent

        Args:
            name: Agent名称
            llm: LLM实例
            tool_registry: 工具注册表
            system_prompt: 系统提示词(人设)
            config: 配置对象
            max_steps: 最大执行步数(循环上限, 防止模型反复调工具停不下来)
            custom_prompt: 自定义工作方式说明(覆盖 DEFAULT_REACT_PROMPT)
        """
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.prompt_template = custom_prompt if custom_prompt else DEFAULT_REACT_PROMPT

        # 最近一次运行的完整对话, 便于调试时回看"模型当时看到了什么"
        self.last_messages: List[dict] = []

    def _system_content(self) -> str:
        """
        人设 + 工作方式说明, 合成一条 system 消息。
        """
        # join + 过滤空串: 显式传 Config(system_prompt="") 时人设是空的,
        # 直接 f"{base}\n\n{prompt}" 会拼出一个以两个换行开头的 system 消息。
        return "\n\n".join(p for p in (super()._system_content(), self.prompt_template) if p)

    def _tools_schema(self):
        """工具清单走 API 的 tools 参数; 没有工具时传 None, 避免某些服务拒绝空数组"""
        return self.tool_registry.get_tools_schema() or None

    # ==================== 主循环 ====================

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行ReAct Agent

        Args:
            input_text: 用户问题
            **kwargs: 其他参数

        Returns:
            最终答案

        """
        answer = ""
        for event in self.stream_run(input_text, **kwargs):
            if event.type == "final":
                answer = event.answer or ""
        return answer

    def stream_run(self, input_text: str, **kwargs) -> Iterator[AgentEvent]:
        """
        流式运行ReAct Agent, 边跑边吐事件。

        一轮里的事件顺序:
            text*  ->  tool_call+  ->  tool_result+  ->  (下一轮)  ->  final

        【请完整消费这个生成器】提前 break 会让: 本轮工具不执行、历史不记录。
        """
        messages = self._build_messages(input_text)
        tools = self._tools_schema()

        for _step in range(1, self.max_steps + 1):
            # 1. 流式要正文 —— 逐块吐给调用方
            for chunk in self._stream_chat(messages, tools=tools, **kwargs):
                yield AgentEvent(type="text", text=chunk)

            # 2. 流被完整消费后, 基类把完整响应放在 _last_response 上(含 tool_calls)
            response = self._last_response
            if response is None:
                # 正常跑完上面的 for 一定会有值。真为 None 说明服务端一个块都没发,
                # 与其抛异常不如用空答案收尾 —— 至少把历史记上。
                logger.warning("流式响应为空, 提前结束本轮。")
                self._finish(input_text, "", messages)
                yield AgentEvent(type="final", answer="")
                return

            # 3. 模型不再请求工具 -> 这就是最终答案, 循环结束
            if not response.has_tool_calls:
                answer = response.content or ""
                self._finish(input_text, answer, messages)
                yield AgentEvent(type="final", answer=answer)
                return

            # 4. 模型请求了工具 -> 先把它这轮的请求记进对话
            self._append_assistant_tool_calls(messages, response)

            for call in response.tool_calls:
                yield AgentEvent(type="tool_call", call=call)

            # 5. 并行执行工具, 把每个结果作为 tool 消息回传
            pairs = self._execute_tool_calls(messages, response.tool_calls)

            for call, result in pairs:
                yield AgentEvent(type="tool_result", call=call, result=result)

        # 6. 步数用完(模型一直在调工具停不下来) -> 兜底
        self._finish(input_text, MAX_STEPS_ANSWER, messages)
        yield AgentEvent(type="final", answer=MAX_STEPS_ANSWER)

    # ==================== 内部方法 ====================

    def _append_assistant_tool_calls(self, messages: List[dict], response: LLMResponse):
        """
        把模型的工具请求记成一条 assistant 消息。

        格式必须与 API 返回的一致: arguments 要重新序列化成 JSON 字符串。
        这条消息不能省 —— 否则下一轮模型看到 tool 消息时, 会不知道自己什么时候请求过。
        """
        messages.append({
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in response.tool_calls
            ],
        })

    def _execute_tool_calls(self, messages: List[dict], calls: List[ToolCall]) -> List[tuple]:
        """
        执行这一轮的每个工具调用, 各自作为一条 tool 消息回传。
        返回 [(call, result), ...], 顺序与 calls 严格一致 —— stream_run 靠它吐 tool_result 事件。
        """
        # registry.execute 失败时不抛异常, 返回"错误: xxx"字符串 ——
        # 这条错误会原样回传给模型, 模型看了能自己调整参数重试
        results = execute_many_sync(self.tool_registry, calls)

        pairs = []
        for call, result in zip(calls, results):
            logger.info("工具 %s(%s) -> %s", call.name, call.arguments, result)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,   # 靠这个 id 与上面那条 assistant 消息对应
                "content": result,
            })
            pairs.append((call, result))
        return pairs

    def _finish(self, input_text: str, final_answer: str, messages: List[dict]) -> str:
        """收尾: 记录轨迹与对话历史, 返回答案"""
        self.last_messages = messages
        self._record_turn(input_text, final_answer)
        return final_answer


# ==================== 旧版本(正则时代) - 注释保留, 仅供参考 ====================
# 旧版的根本问题:
# 1. 用自然语言当程序协议 —— 提示词要求 "Action: tool[input]", 代码用正则去挖。
#    模型是人不是编译器, 中文冒号、多空格、加句废话都会让正则扑空。
# 2. 工具参数只能是字符串, 传不了结构化数据。
# 3. 终止条件靠猜关键字 "Finish"。
#
# import re
# from typing import Optional, List, Dict, Any, Tuple
#
# DEFAULT_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。
#
# ## 可用工具
# {tools}
#
# ## 工作流程
# 请严格按照以下格式进行回应，每次只能执行一个步骤：
#
# **Thought:** 分析当前问题，思考需要什么信息或采取什么行动。
# **Action:** 选择一个行动，格式必须是以下之一：
# - `{{tool_name}}[{{tool_input}}]` - 调用指定工具
# - `Finish[最终答案]` - 当你有足够信息给出最终答案时
#
# ## 重要提醒
# 1. 每次回应必须包含Thought和Action两部分
# 2. 工具调用的格式必须严格遵循：工具名[参数]
# 3. 只有当你确信有足够信息回答问题时，才使用Finish
# 4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数
#
# ## 当前任务
# **Question:** {question}
#
# ## 执行历史
# {history}
#
# 现在开始你的推理和行动："""
#
#
# class ReActAgent(Agent):
#     """ReAct (Reasoning and Acting) Agent"""
#
#     def __init__(self, name, llm, tool_registry, system_prompt=None,
#                  config=None, max_steps=5, custom_prompt=None):
#         super().__init__(name, llm, system_prompt, config)
#         self.tool_registry = tool_registry
#         self.max_steps = max_steps
#         self.current_history: List[str] = []
#         self.prompt_template = custom_prompt if custom_prompt else DEFAULT_REACT_PROMPT
#
#     def run(self, input_text: str, **kwargs) -> str:
#         self.current_history = []
#         current_step = 0
#
#         print(f"\n🤖 {self.name} 开始处理问题: {input_text}")
#
#         while current_step < self.max_steps:
#             current_step += 1
#             print(f"\n--- 第 {current_step} 步 ---")
#
#             # 构建提示词: 把工具描述和历史都拼成文字塞进提示词
#             tools_desc = self.tool_registry.get_tools_description()
#             history_str = "\n".join(self.current_history)
#             prompt = self.prompt_template.format(
#                 tools=tools_desc, question=input_text, history=history_str
#             )
#
#             messages = [{"role": "user", "content": prompt}]
#             response_text = self.llm.invoke(messages, **kwargs)
#
#             if not response_text:
#                 print("❌ 错误：LLM未能返回有效响应。")
#                 break
#
#             # 正则解析开盲盒
#             thought, action = self._parse_output(response_text)
#             if thought:
#                 print(f"🤔 思考: {thought}")
#             if not action:
#                 print("⚠️ 警告：未能解析出有效的Action，流程终止。")
#                 break
#
#             # 靠猜关键字判断是否结束
#             if action.startswith("Finish"):
#                 final_answer = self._parse_action_input(action)
#                 self.add_message(Message(input_text, "user"))
#                 self.add_message(Message(final_answer, "assistant"))
#                 return final_answer
#
#             tool_name, tool_input = self._parse_action(action)
#             if not tool_name or tool_input is None:
#                 self.current_history.append("Observation: 无效的Action格式，请检查。")
#                 continue
#
#             print(f"🎬 行动: {tool_name}[{tool_input}]")
#
#             # 字符串接口, 参数只能是字符串
#             observation = self.tool_registry.execute_tool(tool_name, tool_input)
#             print(f"👀 观察: {observation}")
#
#             self.current_history.append(f"Action: {action}")
#             self.current_history.append(f"Observation: {observation}")
#
#         print("⏰ 已达到最大步数，流程终止。")
#         final_answer = "抱歉，我无法在限定步数内完成这个任务。"
#         self.add_message(Message(input_text, "user"))
#         self.add_message(Message(final_answer, "assistant"))
#         return final_answer
#
#     def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[str]]:
#         """解析LLM输出，提取思考和行动"""
#         thought_match = re.search(r"Thought: (.*)", text)
#         action_match = re.search(r"Action: (.*)", text)
#         thought = thought_match.group(1).strip() if thought_match else None
#         action = action_match.group(1).strip() if action_match else None
#         return thought, action
#
#     def _parse_action(self, action_text: str) -> Tuple[Optional[str], Optional[str]]:
#         """解析行动文本，提取工具名称和输入"""
#         match = re.match(r"(\w+)\[(.*)\]", action_text)
#         if match:
#             return match.group(1), match.group(2)
#         return None, None
#
#     def _parse_action_input(self, action_text: str) -> str:
#         """解析行动输入"""
#         match = re.match(r"\w+\[(.*)\]", action_text)
#         return match.group(1) if match else ""
