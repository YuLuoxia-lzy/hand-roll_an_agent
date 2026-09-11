"""
Agent基类
用于后续创建自己的agent
可以在此地新增各种新功能
在具体agent处去丰富各种功能
"""


from abc import ABC, abstractmethod
from typing import Iterator, Optional
from .message import Message
from .llm import Agents0to1
from .config import Config
from .types import AgentEvent, LLMResponse


class Agent(ABC):
    """Agent基类"""

    def __init__(
            self,
            name: str,
            llm: Agents0to1,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None
            ):
        self.config = config or Config()
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt or self.config.system_prompt
        self._history : list[Message] = [] #这是内部使用的history

        # 最近一次【流式】调用的完整响应。存在 Agent 上而不是只读 llm.last_response:
        # 同一个 llm 客户端可能被多个 Agent 共用(llm.last_response 里放的是"最后一次流式调用"的结果, 不区分是哪个 Agent 发起的), 存一份在自己的属性上才不会串台。
        self._last_response: Optional[LLMResponse] = None

    #声明必须实现这个方法！！
    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        pass

    def stream_run(self, input_text: str, **kwargs) -> Iterator[AgentEvent]:
        """
        流式运行Agent, 逐块吐出 AgentEvent。

        基类不提供实现(不是所有 Agent 都适合流式), 但把它声明出来,
        调用方就能统一写 `for ev in agent.stream_run(...)`, 不用先 hasattr 试探。
        """
        raise NotImplementedError(f"{type(self).__name__} 暂不支持流式运行。")

    # ==================== 拼消息 ====================

    def _system_content(self) -> str:
        """
        system 消息的内容。**子类可覆盖**, 用来追加自己的工作方式说明。

        例: ReActAgent 覆盖它, 把自己的工作方式提示词拼在人设后面。
        """
        return self.system_prompt

    def _base_messages(self) -> list[dict]:
        """只含 system 的那一条消息(没有 system_prompt 时返回空列表)"""
        content = self._system_content()
        return [{"role": "system", "content": content}] if content else []

    def _history_messages(self) -> list[dict]:
        """把历史记录转成 API 消息格式"""
        return [m.to_dict() for m in self._history]

    def _build_messages(self, input_text: str) -> list[dict]:
        """
        拼入口消息: system + 历史 + 当前问题。

        【为什么它必须在基类】上一版每个 Agent 自己拼消息, 结果 Reflection 和
        PlanSolve 的 Executor 忘了带 system, system_prompt 就成了死参数。
        现在"入口消息"只有这一个来源, 子类想漏也漏不掉。
        """
        return (
            self._base_messages()
            + self._history_messages()
            + [{"role": "user", "content": input_text}]
        )

    def _ensure_system(self, messages: list[dict]) -> list[dict]:
        """
        兜底: 消息列表第一条不是 system, 就在最前面补一条。

        这是"结构上保证 system_prompt 生效"的关键 —— _chat 每次都会调它。
        所以哪怕子类自己拼了一个 [{"role": "user", ...}],
        system 也会被补上。上一版的问题正是"基类提供了便利方法, 但没人强制调用"。
        """
        if messages and messages[0].get("role") == "system":
            return messages
        return self._base_messages() + list(messages)

    # ==================== 调 LLM ====================

    def _chat(self, messages: list[dict], tools=None, **kwargs) -> LLMResponse:
        """
        所有 LLM 调用的唯一出口。

        子类请一律用这个方法, 不要直接 self.llm.invoke —— 直接调就绕过了
        system_prompt 的注入, 那正是上一版的病根。
        """
        return self.llm.invoke(self._ensure_system(messages), tools=tools, **kwargs)

    def _stream_chat(self, messages: list[dict], tools=None, **kwargs) -> Iterator[str]:
        """
        流式版本, 同样强制注入 system。

        注意: 生成器被**完整消费**后, self._last_response 上才有完整响应
        (含 tool_calls); 被提前 break 就不会写入。详见 Agents0to1.stream_invoke。
        """
        yield from self.llm.stream_invoke(self._ensure_system(messages), tools=tools, **kwargs)

        # 只在生成器被完整消费后才会执行到这里 —— 提前 break 的调用方拿不到新值,
        # 这正是"流式调用的固有语义", 不是 bug。
        self._last_response = self.llm.last_response

    # ==================== 历史 ====================

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)
        self._history = self._history[-self.config.max_history_length:]

    def _record_turn(self, input_text: str, output_text: str):
        """一次问答记入历史 —— 各 Agent 的 run() 收尾统一调它"""
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(output_text or "", "assistant"))

    def clear_history(self):
        """清空历史记录"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={getattr(self.llm, 'provider', '?')})"

    def __repr__(self) -> str:
        return self.__str__()
