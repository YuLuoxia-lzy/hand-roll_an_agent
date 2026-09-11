"""配置管理"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel

class Config(BaseModel):
    """Agents0to1配置类"""
    
    # LLM配置
    default_model: str = "deepseek-chat"
    # "没指定, 让框架自动检测"。必须写成 Optional[str]: None 是它的正常取值,
    # 标成 str 只是注解写错了, 类型检查器会误报。
    default_provider: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    # 系统配置
    debug: bool = False
    log_level: str = "INFO"
    system_prompt: str = "你是一个全知全能的人类，无论是什么问题你都可以以极其专业的角度入手并且可以以小白也能够理解的方式解决"
    
    # 其他配置
    max_history_length: int = 100


    
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
        # 2.10 调用会发 DeprecationWarning, 后续版本会被移除。
        return self.model_dump()