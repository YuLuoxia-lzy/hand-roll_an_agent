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

    # 单次结果的字符上限, 子类可覆盖。按 2 字符 ≈ 1 token 粗估(中文放宽到 1:1),
    # 6000 字符 ≈ 3000 token —— 够装一份长文档的开头结尾, 又不至于一轮就撑爆上下文。
    max_output_chars: int = 6000

    # 截断保留的尾部长度: 结论、报错、命令输出常常在末尾, 只砍尾巴会丢掉最有用的部分
    _TAIL_CHARS: int = 800

    #: 工具**自己吞掉异常**、把失败当成正常结果返回时, 结果会以这些前缀开头。
    #:
    #: 大多数工具不应该用到它 —— 出错了直接 raise 更干净, registry 会接住并
    #: 标记成失败。但有些工具是**故意**不抛的: 计算器就是这么设计的,
    #: "计算失败: division by zero" 会作为工具结果回传给模型, 让模型自己改
    #: 表达式重试(见 calculator.py 里那段说明)。
    #:
    #: 这种情况下 registry 只看得到一个字符串, 它无从判断这是成功还是失败 ——
    #: 而**猜**别的前缀是不行的: 换个工具完全可能返回一段以"错误"开头的正文
    #: (比如检索到的一段报错日志)。所以由工具自己声明自己的失败词汇。
    error_prefixes: tuple = ()

    def looks_like_error(self, text: str) -> bool:
        """返回值是不是"工具自己吞掉的那个失败"(见 error_prefixes)。"""
        return bool(self.error_prefixes) and isinstance(text, str) and text.startswith(self.error_prefixes)

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @classmethod
    def truncate_output(cls, text: str) -> str:
        """
        超过上限就保留头尾、中间省略, 并且**明确告诉模型这是被截断过的**。
        不标记是最坏的做法: 模型分不清"这就是全部"和"这是被砍过的",
        """
        limit = cls.max_output_chars
        if not isinstance(text, str) or limit <= 0 or len(text) <= limit:
            return text

        omitted = len(text) - limit
        tail_len = min(cls._TAIL_CHARS, limit // 3)
        head_len = max(limit - tail_len, 1)
        head, tail = text[:head_len], text[-tail_len:] if tail_len else ""

        # 尽量切在行边界上, 别把一行/一个 URL 拦腰截断
        cut = head.rfind("\n")
        if cut > head_len * 0.6:
            head = head[:cut]
        cut = tail.find("\n")
        if 0 <= cut < len(tail) * 0.4:
            tail = tail[cut + 1:]

        return f"{head}\n\n...[已截断, 省略 {omitted} 字符]...\n\n{tail}"

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
        校验模型给的参数, 供 registry.execute() 调用。

        与上面的 validate_parameters 的区别:
        - validate_parameters 只返回 True/False —— 调用方不知道"缺了什么"
        - validate_arguments 返回 None(通过) 或具体的错误信息字符串。
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
