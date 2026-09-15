"""异常体系 待补充"""

class Agents0to1Exception(Exception):
    """Agents0to1 常规异常"""
    pass

class LLMException(Agents0to1Exception):
    """大模型相关异常"""
    pass

class AgentException(Agents0to1Exception):
    """智能体相关异常"""
    pass

class ConfigException(Agents0to1Exception):
    """配置相关异常"""
    pass

class ToolException(Agents0to1Exception):
    """工具相关异常"""
    pass

class EmbeddingException(Agents0to1Exception):
    """Embedding 相关异常"""

class VectorStoreException(Agents0to1Exception):
    """向量库相关异常"""

class SemanticMemoryException(Agents0to1Exception):
    """语义记忆相关异常"""

class EpisodicMemoryException(Agents0to1Exception):
    """情景记忆相关异常"""

configException = ConfigException


