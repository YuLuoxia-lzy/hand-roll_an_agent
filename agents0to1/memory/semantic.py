"""
语义记忆 —— 切分 + 入库 + 检索

切分的基本要求:
  - 按**句子边界**切, 不要在句子中间断开。中文按 。！？ 和换行切
  - 有 **overlap**(相邻块重叠一部分), 否则答案正好落在边界上就丢了
  - 带 **metadata**(来源、位置), 否则检索回来你不知道它从哪来, 也没法引用

【绝不自动把 assistant 自己的回答入库】
"""

import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

from ..core.exceptions import *
from ..utils.logging import get_logger
from .embedding import EmbeddingClient
from .vector_store import VectorStore

logger = get_logger(__name__)

# ==================== 检索结果 ====================

class MemoryItem(BaseModel):
    """
    一条检索结果。

    score 必须暴露出来, 因为:
      - 你要能设阈值 —— 分数太低的不该喂给模型, 那只会干扰它
      - 你要能调试 —— "为什么这次没检索到"是 RAG 最高频的问题
    """

    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def citation(self) -> str:
        source = self.metadata.get("source") or "未知来源"
        index = self.metadata.get("chunk_index")
        return f"{source}#{index}" if index is not None else str(source)

    def __str__(self) -> str:
        return f"[{self.score:.3f}] ({self.citation()}) {self.text[:60]}"

# ==================== 切分 ====================

#: 句子结束符。中文的 。！？ , 英文的 .!?          换行。
_SENTENCE_END = "。！？!?；;…\n"


def split_sentences(text: str) -> List[str]:
    """
    按句子边界切开, 保留标点。
    连续结束符算一个边界 "什么？！" 是一句话—
    """
    sentences: List[str] = []
    buf: List[str] = []
    for i, ch in enumerate(text):
        buf.append(ch)
        if ch in _SENTENCE_END:
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt and nxt in _SENTENCE_END:
                continue            # 后面还跟着结束符, 等最后一个
            sentence = "".join(buf).strip()
            if sentence:
                sentences.append(sentence)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _overlap_tail(text: str, target: int) -> str:
    """
    从 text 末尾取约 target 个字符作为下一块的开头。
    """
    if target <= 0 or not text:
        return ""

    tail = text[-target:]
    minimum = max(1, int(target * 0.4))

    for i, ch in enumerate(tail):
        if ch in _SENTENCE_END and len(tail) - (i + 1) >= minimum:
            return tail[i + 1:].lstrip()

    return tail


def _hard_split(sentence: str, chunk_size: int) -> List[str]:
    """超长单句只能硬切 —— 没有边界可用。这会切断语义, 所以记一条警告。"""
    pieces = [sentence[i:i + chunk_size] for i in range(0, len(sentence), chunk_size)]
    logger.warning(
        "有一句话长达 %d 字符, 超过 chunk_size=%d, 只能从中间硬切 —— ",
        len(sentence), chunk_size,
    )
    return pieces


def chunk_text(
    text: str,
    chunk_size: int = 600,
    overlap: float = 0.15,
    min_chunk_chars: int = 80,
) -> List[str]:
    """
    把长文本切成带 overlap 的块。
    Args:
        text:            原文(纯文本 / markdown)
        chunk_size:      单块目标字符数。中文从 400-800 起步, 之后按实际效果调
        overlap:         相邻块重叠比例, 10%-20% 起步
        min_chunk_chars: 最后一块太短就并进上一块 —— 一个 20 字的碎片单独成块,
                         检索出来几乎没有信息量, 却会占掉一个 top_k 名额
    Returns:
        块列表。空输入返回 [], 不会返回 [""]。
    """
    if text is None or not str(text).strip():
        return []
    if chunk_size <= 0:
        raise SemanticMemoryException(f"chunk_size 必须为正整数, 收到 {chunk_size}。")
    text = str(text)

    # 统一换行符: \r\n 会被当成两个边界, 切出来的块带着孤立的 \r
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    overlap_chars = max(0, int(chunk_size * overlap))

    # 先把超长单句拆掉, 后面就可以假设"每句话都塞得下"
    sentences: List[str] = []
    for sentence in split_sentences(normalized):
        if len(sentence) <= chunk_size:
            sentences.append(sentence)
        else:
            sentences.extend(_hard_split(sentence, chunk_size))

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    carry_chars = 0      # current 开头有多少字符是上一块的尾巴

    for sentence in sentences:
        # 只有当这一块里有"新内容"时才允许在此断开 ——
        # 否则会出现"整块都是上一块的尾巴"的退化块(纯重复, 检索出来是噪声)
        if current and (current_len - carry_chars) > 0 and current_len + len(sentence) > chunk_size:
            chunk = "".join(current)
            chunks.append(chunk)

            carry = _overlap_tail(chunk, overlap_chars)
            current = [carry] if carry else []
            current_len = len(carry)
            carry_chars = len(carry)

        current.append(sentence)
        current_len += len(sentence)

    tail = "".join(current).strip()
    if tail:
        # 尾巴太短就并回上一块: 前提是并进去不会让上一块膨胀得离谱
        if chunks and len(tail) < min_chunk_chars and len(chunks[-1]) + len(tail) <= chunk_size * 1.5:
            chunks[-1] = chunks[-1] + tail
        else:
            chunks.append(tail)

    return chunks

# ==================== 语义记忆 ====================

class SemanticMemory:
    """
    语义记忆 —— 跨会话记得"知识"的那一层。

    【和情景记忆的分工】
        语义记忆: 跨会话记得**知识**(文档、用户说过的事实)  <- 本类
        情景记忆: 跨会话记得**你说过什么**(一轮轮对话)      <- episodic.py
    两件事, 两张表, 两个文件。

    【分层: 异常在哪抛、在哪吞】
        EmbeddingClient.embed()   -> 抛     (客户端, 失败就是失败)
        SemanticMemory.search()   -> 抛     (编程接口, 调用方有权知道)
        SemanticMemory.build_context() -> **吞** (fail-open, 它正对着 LLM 漏斗)
        Agent 的注入钩子          -> **吞** (兜底, 任何 memory 实现都不能弄挂 agent)

    前两层是"给人用的 API", 出错要立刻暴露; 后两层是"给 LLM 用的
    增强", 它们一旦抛异常, 整个 agent 就挂了 —— 而记忆只是锦上添花,

    用法:
        mem = SemanticMemory()                      # 默认走 EmbeddingClient()
        mem.remember("用户叫小明, 住在杭州")
        mem.ingest_file("docs/手册.md")
        for item in mem.search("用户住哪"):
            print(item.score, item.text)

        agent = SimpleAgent("a", llm, memory=mem)    # 接进 Agent 循环见第五步
    """

    DEFAULT_CONTEXT_BUDGET = 2000

    def __init__(
        self,
        embedder: Optional[EmbeddingClient] = None,
        path: str = "data/semantic.sqlite3",
        table: str = "semantic",
        chunk_size: int = 600,
        overlap: float = 0.15,
        top_k: int = 5,
        min_score: float = 0.0,
        context_budget: int = DEFAULT_CONTEXT_BUDGET,
        storage: Optional[str] = None,
        embedder_id: Optional[str] = None,
    ):
        """
        Args:
            embedder:       不传就自建一个 EmbeddingClient()。测试时传假 embedder
            path:           sqlite 文件。/ 默认与情景记忆分开存(生命周期不同,
                            以后想单独备份/清理时混在一起会很别扭)
            chunk_size/overlap: 见 chunk_text
            top_k:          默认检索条数
            min_score:      相似度下限。低于它的结果不喂给模型 —— 那只会干扰它
            context_budget: 检索结果拼成上下文时的字符上限
            embedder_id:    写进库的"模型指纹", 默认取 embedder.model_id
        """
        self.embedder = embedder or EmbeddingClient()
        self.store = VectorStore(path=path, table=table, storage=storage)

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.min_score = min_score
        self.context_budget = context_budget

        self.embedder_id = embedder_id or getattr(self.embedder, "model_id", None) or "unknown"

    # ==================== 写入路径(必须显式设计) ====================

    def remember(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        手动存一条。
        Args:
            text:     要记住的内容。**只能是用户说的话或外部资料**,
            metadata: 来源等信息, 会自动补上 source/ingested_at
        Returns:
            实际入库的块数
        """
        return self.index_texts([text], [metadata or {}])

    def ingest_file(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        读文件 -> 切分 -> 批量入库。只处理纯文本和 markdown。
        (pdf/docx 的解析不在本次范围内 —— 那属于"文档格式解析", 是另一个话题)
        Returns:
            实际入库的块数
        """
        path = str(path)
        if not os.path.isfile(path):
            raise SemanticMemoryException(f"文件不存在: {path}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        meta = {"source": os.path.basename(path), "path": path}
        meta.update(metadata or {})
        return self.index_texts([content], [meta])

    def index_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> int:
        """
        批量入库(会被切分)。embedding 请求按 batch_size 攒批发 ——
        """
        metas = list(metadatas) if metadatas is not None else [{} for _ in texts]

        all_chunks: List[str] = []
        all_meta: List[Dict[str, Any]] = []

        for text, meta in zip(texts, metas):
            chunks = chunk_text(text, chunk_size=self.chunk_size, overlap=self.overlap)
            if not chunks:
                continue
            # 记位置: 没有它, 检索回来你既不知道它从哪来, 也没法引用
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_meta.append({
                    **meta,
                    "source": meta.get("source") or "手动记录",
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "ingested_at": time.time(),
                })

        if not all_chunks:
            logger.warning("没有可入库的内容(全是空白?), 已跳过。")
            return 0

        vectors = self.embedder.embed(all_chunks)       # 内部自动攒批
        self.store.add(all_chunks, vectors, all_meta, model=self.embedder_id)
        logger.info("已入库 %d 块 (来自 %d 段文本)", len(all_chunks), len(texts))
        return len(all_chunks)

    # ==================== 检索 ====================

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[MemoryItem]:
        """
        检索。**这个方法是会抛异常的**(embedding 挂了就抛) ——
        要 fail-open 的调用方请用 build_context()。
        Args:
            query:     查询文本
            top_k:     返回条数, 默认用 self.top_k
            min_score: 相似度下限, 默认用 self.min_score
        Returns:
            MemoryItem 列表(含 score 和 metadata), 按相似度从高到低。空查询返回 []。
        """
        if not query or not query.strip():
            return []

        limit = self.top_k if top_k is None else top_k
        threshold = self.min_score if min_score is None else min_score

        vector = self.embedder.embed_one(query)
        rows = self.store.search(vector, top_k=max(limit * 3, limit), model=self.embedder_id)

        items = [
            MemoryItem(text=text, score=score, metadata=meta)
            for score, text, meta in rows
            if score >= threshold
        ]
        return items[:limit]

    # ==================== 给 LLM 看的形态 ====================

    def build_context(
        self,
        query: str,
        budget: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """
        检索并拼成一段可以直接塞进 prompt 的文本。**fail-open**:
        任何失败都返回 "" 并记一条日志, 绝不抛。

        这是记忆层正对 LLM 漏斗的那一层。它抛异常的后果不是"少了一段上下文",
        而是整个 agent 挂掉 —— 记忆只是增强, 不该有这个权力。
        """
        try:
            items = self.search(query, top_k=top_k)
        except Exception as e:
            # 用 warning 而不是 exception: 这在生产里是会反复发生的事
            # (独立 provider、独立 key、DeepSeek 还没有 embedding 端点),
            # 每次都打一整个堆栈会把日志刷爆。
            logger.warning("记忆检索失败, 本轮按『没有记忆』继续: %s", e)
            return ""

        if not items:
            return ""
        return self.format_items(items, budget=budget)

    def format_items(
        self,
        items: Sequence[MemoryItem],
        budget: Optional[int] = None,
    ) -> str:
        """
        把检索结果排成文本, 按预算裁剪。

        【为什么裁剪必须在这里做, 而不是交给外层的 truncate_output】
        1. Tool.max_output_chars 的截断是"保头 + 保尾", 尾部会接到前面去,
           相关性顺序就乱了 —— 而顺序是检索结果的灵魂, 最相关的必须在最前面
        2. 工具路径和上下文路径共用这一个函数, 才能保证模型看到的是**同一个形状**
        """
        if not items:
            return ""

        limit = self.context_budget if budget is None else budget

        lines: List[str] = []
        used = 0
        dropped = 0
        for i, item in enumerate(items, 1):
            block = (
                f"[{i}] 相似度 {item.score:.3f} | 来源: {item.citation()}\n"
                f"{item.text}"
            )
            # 第一块无论如何都留下 —— 否则"预算太小"会表现为"什么都没有",
            # 而真正的原因藏在预算里
            if lines and used + len(block) > limit:
                dropped = len(items) - i + 1
                break
            lines.append(block)
            used += len(block)

        body = "\n\n".join(lines)
        if dropped:
            # 明确告诉模型"还有但没给你", 而不是让它以为这就是全部 ——
            # 它至少可以说一句"资料可能不全"
            body += f"\n\n...(还有 {dropped} 条相关片段超出预算未展示)"
        return body

    # ==================== 杂项 ====================

    def count(self) -> int:
        return self.store.count()

    def stats(self) -> Dict[str, Any]:
        info = self.store.stats()
        info["embedder"] = self.embedder_id
        return info

    def clear(self) -> int:
        return self.store.clear()

    def close(self) -> None:
        self.store.close()

    def __len__(self) -> int:
        return self.count()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return (
            f"SemanticMemory(path={self.store.path}, table={self.store.table}, "
            f"embedder={self.embedder_id}, count={len(self)})"
        )


# ==================== 验证(指南第四步, 用一个假 embedder 就不花钱) ====================
#
# 假 embedder 把文本按字符哈希成固定长度的向量, 于是:
#     相同文本 -> 相同向量      不同文本 -> 不同向量
# 完全不花钱、不联网。完整版见 tests/test_semantic.py:
#     python tests/test_semantic.py

