"""
ReAct Agent实现 - 已适配 function calling
"""

import json
from typing import Iterator, Optional, List
from ..core.agent import Agent
from ..core.llm import Agents0to1
from ..core.config import Config
from ..core.typedefs import AgentEvent, ToolCall
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
        custom_prompt: Optional[str] = None,
        hooks: Optional[list] = None,
        agent_id: Optional[str] = None,
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
            hooks: 扩展点。记忆现在是 hooks=[MemoryHook(mem)] —— 它注入进
                   **这一次发出去的请求**, 不进 Turn, 所以历史、快照、下一轮
                   请求都不会带上检索结果(这个性质没变, 变的是它由谁保证:
                   以前靠 _finish 手写还原, 现在靠 hook 返回新列表)。
            agent_id: 身份, 不传就自动生成一个。
        """
        super().__init__(name, llm, system_prompt, config, hooks=hooks, agent_id=agent_id)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.prompt_template = custom_prompt if custom_prompt else DEFAULT_REACT_PROMPT

        # last_messages 现在由基类维护(在 _chat / _stream_chat 出口处存"真正发出去的
        # 那一份"), 这里不再自己定义 —— 否则就是两处各写一份、迟早走岔的经典现场。

    def _snapshot_state(self) -> dict:
        """
        把构造参数交给快照 —— 不存的话, fork 出来的 agent 会悄悄退回默认 max_steps
        和默认提示词, 而"悄悄退回默认值"是最难发现的一类偏差。
        """
        return {"max_steps": self.max_steps, "custom_prompt": self.prompt_template}

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

    def _run(self, input_text: str, **kwargs) -> str:
        """
        运行ReAct Agent

        Args:
            input_text: 用户问题
            **kwargs: 其他参数

        Returns:
            最终答案

        """
        answer = ""
        # ⚠️ 调的是 _stream_run(), **不是 stream_run()**。
        # stream_run() 是模板方法, 它会把当前 ctx 覆盖成新的一轮 ——
        # 于是 after_run 拿到的上下文、记忆检索用的 input_text 全都对不上,
        # 而且 ctx 被覆盖两次、清理只清一次。_stream_run 才是"用现在这个 ctx 接着跑"。
        for event in self._stream_run(input_text, **kwargs):
            if event.type == "final":
                answer = event.answer or ""
        return answer

    def _stream_run(self, input_text: str, **kwargs) -> Iterator[AgentEvent]:
        """
        流式运行ReAct Agent, 边跑边吐事件。

        一轮里的事件顺序:
            没有工具:  text*  ->  final
            有工具:    thinking?  ->  tool_call+  ->  tool_result+  ->  (下一轮)

        """
        tools = self._tools_schema()

        # messages 的前 [0:turn_start) 是**从历史里拼进来的上一轮内容**,
        # turn_start 之后才是本轮真正新产生的。
        messages = self._build_messages(input_text)
        turn_start = len(messages) - 1

        for _step in range(1, self.max_steps + 1):
            # ==================== 先攒着, 流完了再定性 ====================
            #
            # 【为什么不能边收边吐 text】
            # 这一轮的内容到底是"最终答案"还是"工具调用前的过程叙述", 要到**流结束**
            # 才知道 —— 判据是 response.has_tool_calls, 而它只在整个流被消费完之后
            # 才存在(见 core/agent.py:_stream_chat 末尾才写 self._last_response;
            # llm.stream_invoke 同理, tool_calls 是在最后一个块里才拼齐的)。
            #
            # 边收边吐 text 的后果是: 等发现这轮其实要调工具, 那段"我先搜一下…"
            # 已经作为 text 吐出去了 —— 于是同一段内容**既出现在 text 里又出现在
            # thinking 里**。把 text 拼起来当答案的消费方就中招了: 最终答案里混进
            # 一句过程叙述。这正是 core/typedefs.py 里那条警告说的病。
            #
            # 【代价, 写在明处】
            # 正文不再"逐 token 实时"到达, 而是等这一轮流完之后才开始吐。这是把
            # text 和 thinking 分对**必须**付的钱 —— 两者在流结束前根本无法区分,
            # 没有"既实时又分对"的选项。
            # 但事件的**形状**一点没变: 下面仍然按收到的原块边界逐个吐出去, 消费方
            # 的拼接 / 逐块渲染逻辑一个字都不用改, 变的只是时间点。
            chunks: List[str] = []
            for chunk in self._stream_chat(messages, tools=tools, **kwargs):
                chunks.append(chunk)

            # 2. 流被完整消费后, 基类把完整响应放在 _last_response 上(含 tool_calls)
            response = self._last_response
            if response is None:
                # 正常跑完上面的 for 一定会有值。真为 None 说明服务端一个块都没发,
                logger.warning("流式响应为空, 提前结束本轮。")
                # 已经收到的块照吐 —— 改造前它们本来就是边收边吐出去的, 攒着之后
                # 更不能让它们凭空消失(那才是真的"丢字")
                for chunk in chunks:
                    yield AgentEvent(type="text", text=chunk)
                self._append_assistant_message(messages, "")
                self._finish(input_text, "", messages, turn_start)
                yield AgentEvent(type="final", answer="")
                return

            text = response.content or ""

            # 3. 模型不再请求工具 -> 这就是最终答案, 循环结束
            if not response.has_tool_calls:
                for chunk in chunks:
                    yield AgentEvent(type="text", text=chunk)
                self._append_assistant_message(messages, text)
                self._finish(input_text, text, messages, turn_start)
                yield AgentEvent(type="final", answer=text)
                return

            # 4. 模型请求了工具 -> 它这轮说的正文是**过程叙述**, 不是最终答案。
            #    先吐 thinking, 再吐工具事件 —— 和模型"先想后做"的顺序一致。
            #
            #    整段吐一次, 不是逐块: typedefs.py 里 thinking 的定义就是"整段"。
            #    攒下来的 chunks 到这里**整个不用**(否则就是上面说的那种重复)。
            if text:
                yield AgentEvent(type="thinking", text=text)

            # 5. 把这轮的请求记进对话。注意 content 用的是同一个 text 变量,
            #    函数自己也**不能再从 response 里取一遍**(所以签名改成了收 text)。
            self._append_assistant_tool_calls(messages, text, response.tool_calls)

            for call in response.tool_calls:
                yield AgentEvent(type="tool_call", call=call)

            # 6. 并行执行工具, 把每个结果作为 tool 消息回传
            pairs = self._execute_tool_calls(messages, response.tool_calls)

            for call, result in pairs:
                yield AgentEvent(type="tool_result", call=call, result=result)

        # 7. 步数用完(模型一直在调工具停不下来) -> 兜底
        self._append_assistant_message(messages, MAX_STEPS_ANSWER)
        self._finish(input_text, MAX_STEPS_ANSWER, messages, turn_start)
        yield AgentEvent(type="final", answer=MAX_STEPS_ANSWER)

    # ==================== 内部方法 ====================

    def _append_assistant_message(self, messages: List[dict], text: str):
        """
        把模型这轮的最终答复记进工作列表。
        user -> assistant(tool_calls) -> tool -> assistant。
        """
        messages.append({"role": "assistant", "content": text})

    def _append_assistant_tool_calls(self, messages: List[dict], text: str, calls: List[ToolCall]):
        """
        把模型的工具请求记成一条 assistant 消息。

        格式必须与 API 返回的一致: arguments 要重新序列化成 JSON 字符串。
        这条消息不能省 —— 否则下一轮模型看到 tool 消息时, 会不知道自己什么时候请求过。

        """
        messages.append({
            "role": "assistant",
            "content": text,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in calls
            ],
        })

    def _execute_tool_calls(self, messages: List[dict], calls: List[ToolCall]) -> List[tuple]:
        """
        执行这一轮的每个工具调用, 各自作为一条 tool 消息回传。
        返回 [(call, result), ...], 顺序与 calls 严格一致 —— _stream_run 靠它吐 tool_result 事件。

        【after_tool 挂在 append **之前**, 这是唯一说得通的位置】
        tool 消息一旦 append, 它就是下一轮模型看到的东西; 事件也已经 yield 出去了。
        想改结果就只能在这儿改 —— 拿返回值去 append, "hook 改过的"和"模型看到的"
        才是同一份。挂在 append 之后就只是"记录", 不是"拦截"。
        """
        ctx = self._hook_ctx()

        # registry.execute 失败时不抛异常, 返回"错误: xxx"字符串 ——
        # 这条错误会原样回传给模型, 模型看了能自己调整参数重试
        results = execute_many_sync(self.tool_registry, calls)

        pairs = []
        for call, result in zip(calls, results):
            result = self._hooks.after_tool(ctx, call, result)
            logger.info("工具 %s(%s) -> %s", call.name, call.arguments, result)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,   # 靠这个 id 与上面那条 assistant 消息对应
                "content": result,
            })
            pairs.append((call, result))
        return pairs

    def _finish(self, input_text: str, final_answer: str, messages: List[dict], turn_start: int) -> str:
        """
        收尾: 记录轨迹与对话历史, 返回答案。

        【这里原来有个补丁, 现在不需要了】
        改造前 messages[turn_start] 带着本轮的检索结果(记忆注入发生在基类的
        _build_messages 里), 所以存 Turn 之前必须**再手工还原成原始输入** ——
        否则检索结果会被当成"用户说过的话"在下一轮重发, _turn_chars 的预算
        也被它白吃。补丁本身没错, 错的是"注入发生在工作列表里"这件事。

        现在注入由 hook 在 before_llm 阶段做, 而且**返回新列表**:
        messages 从头到尾都是干净的那一份, 没有要还原的东西。
        """
        self._record_turn(input_text, final_answer, messages=messages[turn_start:])
        return final_answer


# ==================== 旧版本 - 注释保留, 仅供参考 ====================
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
