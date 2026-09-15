"""消息系统"""

from typing import Dict, Any, List, Literal
from datetime import datetime
from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "system", "tool"]


#规定写在外面  pydantic的规定
class Message(BaseModel):
    """消息类"""
    
    content: str
    role: MessageRole
    # default_factory 而不是直接写 datetime.now():
    # 写成默认值的话, 所有在同一个瞬间创建的消息会共享**同一个**时间戳对象。
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(content=content, role=role, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式（OpenAI API格式）。

        【边界】只能表达 role/content 两个字段, 带不了 tool_calls / tool_call_id。

        所以它只负责"没有工具的普通消息" —— SimpleAgent、PlanSolve 的分步提示,
        以及 Turn 里那个兜底的单条消息。

        带工具的完整轨迹由 Agent.add_turn 直接存下原始的messages字典
        """
        return {
            "role": self.role,
            "content": self.content
        }

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"

class Turn(BaseModel):
    """
    一轮完整的对话 —— 历史记录的原子单位。
    模型的工具调用要求 assistant(tool_calls) 与紧随其后的 tool 消息严格配对,
    一旦按"条数"从中间截断, 就会出现"有 tool_call_id 却没有对应 tool_calls"的孤儿消息, API 直接返回 400。
    以 Turn 为单位截断, 切片永远落在轮边界, 配对从**结构上**不可能被撕裂。

    字段说明:
        user     本轮的用户输入;
        messages 本轮完整的上行消息序列
        answer   最终答案
        metadata 步数 / 用过的工具 / 时间戳等
    """

    user: str = ""
    messages: List[dict] = Field(default_factory=list)
    answer: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_valid(self) -> bool:
        """
        烟雾报警器: 只做自检, 不修复。合法返回 True, 否则 False。
        """
        msgs = self.messages
        i = 0
        while i < len(msgs):
            msg = msgs[i]

            # 孤儿 tool: 没有紧跟在一个声明过 tool_calls 的 assistant 后面
            if msg.get("role") == "tool":
                return False

            # 只看带 tool_calls 的 assistant
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # 1. 收集本 assistant 声明的 tool_call id, 顺序保留
                expected_ids = [tc.get("id") for tc in msg["tool_calls"]]
                # 2. 紧跟其后的 N 条必须都是 tool, 且 id 一一对应
                n = len(expected_ids)
                following = msgs[i + 1 : i + 1 + n]
                if len(following) != n:
                    return False
                for tc_id, fm in zip(expected_ids, following):
                    if fm.get("role") != "tool":
                        return False
                    if fm.get("tool_call_id") != tc_id:
                        return False
                i += 1 + n
            else:
                i += 1
        return True
