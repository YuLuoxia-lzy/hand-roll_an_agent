"""简单Agent实现 - 基于OpenAI原生API"""

from typing import Iterator, Optional

from ..core.agent import Agent
from ..core.llm import Agents0to1
from ..core.config import Config
from ..core.types import AgentEvent

class SimpleAgent(Agent):
    """简单的对话Agent"""

    def __init__(
        self,
        name: str,
        llm: Agents0to1,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        #运行目标函数的init的具体方法 可以自己补充
        super().__init__(name, llm, system_prompt, config)

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行简单Agent
        Args:
            input_text: 用户输入
            **kwargs: 其他参数
        Returns:
            模型的回答文本

        【为什么返回 str 而不是整个 LLMResponse】四个 Agent 的 run() 统一返回字符串,
        这样 `for a in agents: print(a.run(q))` 能一把跑完, 不用记住谁返回哪种类型。
        (原来这里返回 LLMResponse, 而基类和另外三个都写 str —— 注解和行为对不上。)
        """
        # system + 历史 + 当前问题, 由基类统一拼 —— 不用再自己遍历 _history
        messages = self._build_messages(input_text)

        response = self._chat(messages, **kwargs) #调用llm 非流式

        self._record_turn(input_text, response.content or "")
        return response.content or ""


    def stream_run(self, input_text: str, **kwargs) -> Iterator[AgentEvent]:
        """
        流式运行Agent

        Args:
            input_text: 用户输入
            **kwargs: 其他参数
        Yields:
            AgentEvent: "text" 是正文增量(逐块), 最后一个是 "final"(含完整答案)

        【注意】如果调用方提前 break, 本轮就不会记入历史 —— 因为记历史在循环之后。
        """
        messages = self._build_messages(input_text)

        full_response = ""
        for chunk in self._stream_chat(messages, **kwargs):
            full_response += chunk
            yield AgentEvent(type="text", text=chunk)

        self._record_turn(input_text, full_response)
        yield AgentEvent(type="final", answer=full_response)
