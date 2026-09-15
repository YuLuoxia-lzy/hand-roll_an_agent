"""简单Agent实现 - 基于OpenAI原生API"""

from typing import Iterator, Optional

from ..core.agent import Agent
from ..core.llm import Agents0to1
from ..core.config import Config
from ..core.typedefs import AgentEvent

class SimpleAgent(Agent):
    """简单的对话Agent"""

    def __init__(
        self,
        name: str,
        llm: Agents0to1,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        hooks: Optional[list] = None,
        agent_id: Optional[str] = None,
    ):
        #运行目标函数的init的具体方法 可以自己补充
        # 前四个是位置参数, 顺序不能动; hooks / agent_id 只能**关键字**传进基类
        # (见 core/agent.py 里"凭什么参数放在 config 后面"那段)
        super().__init__(name, llm, system_prompt, config, hooks=hooks, agent_id=agent_id)

    def _run(self, input_text: str, **kwargs) -> str:
        """
        运行简单Agent
        Args:
            input_text: 用户输入
            **kwargs: 其他参数
        Returns:
            模型的回答文本
        """
        # system + 历史 + 当前问题, 由基类统一拼 —— 不用再自己遍历 _history
        messages = self._build_messages(input_text)

        response = self._chat(messages, **kwargs) #调用llm 非流式

        self._record_turn(input_text, response.content or "")
        return response.content or ""


    def _stream_run(self, input_text: str, **kwargs) -> Iterator[AgentEvent]:
        """
        流式运行Agent

        Args:
            input_text: 用户输入
            **kwargs: 其他参数
        Yields:
            AgentEvent: "text" 是正文增量(逐块), 最后一个是 "final"(含完整答案)
        """
        messages = self._build_messages(input_text)

        full_response = ""
        for chunk in self._stream_chat(messages, **kwargs):
            full_response += chunk
            yield AgentEvent(type="text", text=chunk)

        self._record_turn(input_text, full_response)
        yield AgentEvent(type="final", answer=full_response)
