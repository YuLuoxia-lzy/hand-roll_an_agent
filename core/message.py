"""消息系统"""

from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel

MessageRole = Literal["user", "assistant", "system", "tool"]


#规定写在外面  pydantic的规定
class Message(BaseModel):
    """消息类"""
    
    content: str
    role: MessageRole
    # 注解要写 Optional: 默认值是 None, 只标 datetime 的话类型检查器会认为
    # "这个字段永远是 datetime", 而它其实可能是 None。
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get('timestamp', datetime.now()),
            metadata=kwargs.get('metadata', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式（OpenAI API格式）。

        【已知边界】只能表达 role/content 两个字段, 带不了 tool_calls / tool_call_id。
        所以 _history 里的历史只适合普通对话 —— 工具调用的中间过程(assistant 的
        tool_calls、tool 的返回)不会进历史。

        要完整重放一轮工具调用, 需要消息之间严格配对
        一旦历史被max_history_length 从中间截断, 出现"有 tool_call_id 却没有对应 tool_calls"
        的消息, API 会直接返回 400
        """
        return {
            "role": self.role,
            "content": self.content
        }
    
    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"