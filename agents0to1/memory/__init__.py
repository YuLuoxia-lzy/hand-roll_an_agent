"""记忆层

四块积木, 按依赖顺序排:

    embedding.py     EmbeddingClient —— 文本 -> 向量 (独立于 chat 的一套 provider 配置)
    vector_store.py  VectorStore     —— 向量库 (sqlite + 余弦相似度)
    semantic.py      SemanticMemory  —— 切分 + 入库 + 检索 (跨会话记得"知识")
    episodic.py      EpisodicMemory  —— 一轮轮对话的持久化 (跨会话记得"你说过什么")

前三个是"知识", 第四个是"经历"。两件事, 两个 sqlite 文件。
"""

from .embedding import EmbeddingClient
from .vector_store import (
    VectorStore,
    Scorer,
    cosine_similarity,
    has_numpy,
    DEFAULT_STORAGE,
    STORAGE_BLOB,
    STORAGE_JSON,
)
from .semantic import (
    SemanticMemory,
    MemoryItem,
    chunk_text,
    split_sentences,
)
from .episodic import EpisodicMemory

__all__ = [
    # 客户端
    "EmbeddingClient",

    # 向量库
    "VectorStore",
    "Scorer",              # 换打分公式的那个缝(文档系统 vs 小镇)
    "cosine_similarity",
    "has_numpy",
    "STORAGE_BLOB",
    "STORAGE_JSON",
    "DEFAULT_STORAGE",

    # 语义记忆
    "SemanticMemory",
    "MemoryItem",
    "chunk_text",
    "split_sentences",

    # 情景记忆
    "EpisodicMemory",
]
