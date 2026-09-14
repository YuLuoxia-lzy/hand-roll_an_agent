"""配置管理"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel

class Config(BaseModel):
    """Agents0to1配置类"""
    
    # LLM配置
    # 默认 None = "没指定", 由 Agents0to1 按 provider 挑对应的默认模型
    # (deepseek -> deepseek-chat, openai -> gpt-3.5-turbo, ...)。
    # 这里**不能**写死 "deepseek-chat": 写死之后换 provider 也还是发 deepseek-chat,
    # 报一个看不懂的 400; 而且 LLM_MODEL_ID 环境变量会被这个硬编码值压住, 永远不生效。
    default_model: Optional[str] = None
    # "没指定, 让框架自动检测"。必须写成 Optional[str]: None 是它的正常取值,
    default_provider: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    # 系统配置
    debug: bool = False
    log_level: str = "INFO"
    system_prompt: str = "你是一个全知全能的造物主，无论是什么问题你都可以以极其专业的角度入手并且可以以小白也能够理解的方式解决"
    
    # 其他配置
    max_history_length: int = 100   # 消息条数上限
    max_history_turns: int = 20     # 保留多少轮对话

    # 总字符预算 —— 前两个预算数的是"有几条消息", 这个数的是"一共多大"。
    max_history_chars: int = 40000
    # 旧轮的工具结果被压缩后保留多少字符(要够模型认出"这里原本有个结果")
    compacted_tool_chars: int = 400


    
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置"""
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        # model_dump() 是 pydantic v2 的写法; .dict() 在 v2 里已废弃,
        return self.model_dump()