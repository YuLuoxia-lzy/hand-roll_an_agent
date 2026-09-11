"""工具基类"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None

class Tool(ABC):
    """工具基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义"""
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数"""
        required_params = [p.name for p in self.get_parameters() if p.required]
        return all(param in parameters for param in required_params)

    def validate_arguments(self, arguments: Dict[str, Any]) -> Optional[str]:
        """
        【新】校验模型给的参数, 供 registry.execute() 调用。

        与上面的 validate_parameters 的区别:
        - validate_parameters 只返回 True/False —— 调用方不知道"缺了什么"
        - validate_arguments 返回 None(通过) 或具体的错误信息字符串。
          这条错误信息会作为工具结果回传给模型, 模型看了就知道该补哪个参数。
        """
        if not isinstance(arguments, dict):
            return f"参数格式错误: 期望字典, 实际是 {type(arguments).__name__}"

        for p in self.get_parameters():
            if p.required and p.name not in arguments:
                return f"缺少必填参数 '{p.name}': {p.description}"
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            # 用 model_dump() 而不是 .dict(): pydantic v2 里 .dict() 已废弃,
            # 2.10 调用它会发 DeprecationWarning, 后续版本直接删掉。
            "parameters": [param.model_dump() for param in self.get_parameters()]
        }
    
    def __str__(self) -> str:
        return f"Tool(name={self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_openai_schema(self) -> dict:
        """把工具定义翻译成 OpenAI function calling 需要的 JSON Schema"""
        properties = {}
        required = []
        for p in self.get_parameters():
            # 1. 参数名 → Schema 属性
            properties[p.name] = {"type": p.type, "description": p.description}
            # 2. required 的收集方式:Schema 用"名字数组",不是每个参数里的 bool
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }
