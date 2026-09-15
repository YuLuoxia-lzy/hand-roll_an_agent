"""
向量库 
接口定好了, 之后换 faiss / sqlite-vec / chroma 就是换一个类
cos(q, v) = dot(q, v) / (|q| * |v|)
三个设计点:
  ① 预存模长 —— |v| 在入库时算一次存下来, 查询时只算 query 的模长
  ② 向量存储格式 —— array('f').tobytes() 存进 BLOB, 比 JSON 文本快很多
  ③ 模型和维度必须存进每一行 —— 跨模型的向量不可比, 而且**不会报错**
"""
import json
import os
import sqlite3
import threading
import time
from array import array
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.exceptions import *
from ..utils.logging import get_logger

logger = get_logger(__name__)

# ==================== numpy 可选加速 ====================
#

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:      # 没装 numpy 完全不影响功能
    _np = None
    _HAS_NUMPY = False


def has_numpy() -> bool:
    """当前进程有没有 numpy 加速"""
    return _HAS_NUMPY

# ==================== 存储格式开关 ====================
#
# "blob": array('f').tobytes() —— 快 人类不友好
# "json": json.dumps(list)     —— 慢 人类友好
#
# 改这个常量(或设环境变量 VECTOR_STORAGE)就能切。**两种格式可以在同一张表里共存**:
# 每行都记了自己是怎么存的, 读的时候按行分派 —— 所以切换开关不会把老数据读坏,
# 可以入库一批 JSON 的, 切开关再入库一批 BLOB 的, 直接对比。
#

STORAGE_BLOB = "blob"
STORAGE_JSON = "json"

#: 默认存储格式。可用环境变量覆盖: set VECTOR_STORAGE=json
DEFAULT_STORAGE = os.getenv("VECTOR_STORAGE", STORAGE_BLOB).strip().lower()

# ==================== 向量数学 ====================

def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    """
    点积。 整个模块唯一的一处 numpy 分叉 。
    numpy 只是加速, 不是依赖: 没装就走纯 Python, 结果一样, 只是慢 。
    """
    if _HAS_NUMPY:
        return float(_np.dot(a, b))
    return sum(x * y for x, y in zip(a, b))

def _norm(v: Sequence[float]) -> float:
    """L2 模长"""
    if _HAS_NUMPY:
        return float(_np.linalg.norm(v))
    return sum(x * x for x in v) ** 0.5

def cosine_similarity(
    query: Sequence[float],
    vector: Sequence[float],
    query_norm: Optional[float] = None,
    vector_norm: Optional[float] = None,
) -> float:
    """
    余弦相似度。模长允许外部传进来(见"预存模长")
    两个模长里只要有一个是 0(全零向量), 相似度就不再有定义 —— 返回 0.0
    """
    qn = query_norm if query_norm is not None else _norm(query)
    vn = vector_norm if vector_norm is not None else _norm(vector)
    if qn == 0 or vn == 0:
        return 0.0
    return _dot(query, vector) / (qn * vn)


class VectorStore:
    """
    基于 sqlite 的向量库。

    一行 = 一条文本 + 一个向量 + 它的来源信息。检索就是全表扫描算余弦,
    取分数最高的 top_k 条。

    性能量级参考: 10k 条 × 1024 维, 纯 Python 每次查询约 1 秒级, numpy 约 1 毫秒级。
    到几万条再考虑索引(HNSW / IVF)。

    用法:
        store = VectorStore("data/vectors.sqlite3")
        store.add(["文本一", "文本二"], [[0.1, 0.2], [0.3, 0.4]], model="openai/x")
        for score, text, meta in store.search([0.1, 0.2], top_k=3, model="openai/x"):
            print(score, text)
    """

    #: 表结构。vector 一列同时装 BLOB 和 TEXT —— sqlite 是弱类型的, 两种都收得下。
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS {table} (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        text       TEXT    NOT NULL,
        metadata   TEXT    NOT NULL DEFAULT '{{}}',
        model      TEXT    NOT NULL,
        dim        INTEGER NOT NULL,
        norm       REAL    NOT NULL,
        storage    TEXT    NOT NULL,
        vector     BLOB    NOT NULL,
        created_at REAL    NOT NULL
    )
    """

    def __init__(
        self,
        path: str = ":memory:",
        table: str = "vectors",
        storage: Optional[str] = None,
    ):
        """
        Args:
            path:    sqlite 文件路径, 或 ":memory:"(进程内, 关掉就没了)
            table:   表名。**一个 sqlite 文件可以放多张表**, 但不同 embedder
                     的向量最好分表或分库, 见下面 _assert_same_model
            storage: blob / json, 默认取 DEFAULT_STORAGE
        """
        self.path = str(path)
        self.table = table
        self.storage = (storage or DEFAULT_STORAGE).lower()
        if self.storage not in (STORAGE_BLOB, STORAGE_JSON):
            raise VectorStoreException(
                f"storage 只能是 '{STORAGE_BLOB}' 或 '{STORAGE_JSON}', 收到 '{self.storage}'。"
            )

        # ============ sqlite 的线程问题 ============
        #   1. check_same_thread=False  允许跨线程使用同一个连接
        #   2. 一把 threading.Lock 包住所有读写, 因为 sqlite3 连接本身不是线程安全的
        # 第二条绝对不能省: 只关掉 check_same_thread 而不加锁, 换来的是"偶尔"的
        # 数据损坏和 InterfaceError, 比稳定报错难查得多。
        self._lock = threading.RLock()

        if self.path != ":memory:":
            parent = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(parent, exist_ok=True)

        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        with self._lock:
            # WAL: 读写不互相阻塞, 多个线程各开各的连接时必须开它
            self._conn.execute("PRAGMA journal_mode=WAL")
            # 每个线程共用同一个连接时, 关掉 sqlite 自己的同步等待, 由我们的锁负责
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute(self._SCHEMA.format(table=self.table))
            self._conn.commit()

        logger.info(
            "向量库就绪: path=%s table=%s storage=%s numpy=%s",
            self.path, self.table, self.storage, _HAS_NUMPY,
        )

    # ==================== 编码 / 解码 ====================

    #: array('f') 是 float32 —— 一半的字节数, 精度损失在检索场景可以忽略
    _ARRAY_TYPECODE = "f"

    @classmethod
    def _encode(cls, vector: Sequence[float], storage: str):
        """向量 -> 存进 sqlite 的载荷"""
        if storage == STORAGE_BLOB:
            # sqlite3.Binary 让 sqlite 把它当 BLOB 而不是尝试当文本处理
            return sqlite3.Binary(array(cls._ARRAY_TYPECODE, vector).tobytes())
        return json.dumps(list(vector))

    @classmethod
    def _decode(cls, payload: Any, storage: str) -> List[float]:
        """sqlite 里的载荷 -> 向量列表"""
        if storage == STORAGE_BLOB:
            # 从 BLOB 读回来是 bytes, 需要 frombytes 还原
            buf = array(cls._ARRAY_TYPECODE)
            buf.frombytes(bytes(payload))
            return list(buf)
        return json.loads(payload)

    # ==================== 写 ====================

    def add(
        self,
        texts: Iterable[str],
        vectors: Iterable[Sequence[float]],
        metadatas: Optional[Iterable[Optional[Dict[str, Any]]]] = None,
        model: str = "unknown",
    ) -> List[int]:
        """
        批量入库。

        Args:
            texts:     文本
            vectors:   与 texts 一一对应的向量
            metadatas: 可选, 与 texts 一一对应
            model:     产生这些向量的模型 跨模型的向量不可比

        Returns:
            新插入行的 id 列表
        """
        texts = list(texts)
        vectors = [list(v) for v in vectors]
        metas = list(metadatas) if metadatas is not None else [None] * len(texts)

        if not texts:
            return []
        if not (len(texts) == len(vectors) == len(metas)):
            raise VectorStoreException(
                f"长度不一致: texts={len(texts)} vectors={len(vectors)} metadatas={len(metas)}。"
            )

        dim = len(vectors[0])
        if dim == 0:
            raise VectorStoreException("向量维度为 0, 拒绝入库。")
        for i, v in enumerate(vectors):
            if len(v) != dim:
                raise VectorStoreException(
                    f"第 {i} 条向量维度是 {len(v)}, 期望 {dim}。同一批必须同维度。"
                )

        rows = []
        now = time.time()
        for text, vector, meta in zip(texts, vectors, metas):
            rows.append((
                text,
                json.dumps(meta or {}, ensure_ascii=False),
                model,
                dim,
                _norm(vector),                      # ← ① 预存模长, 查询时就不用再开方
                self.storage,
                self._encode(vector, self.storage),
                now,
            ))

        sql = (
            f"INSERT INTO {self.table} "
            f"(text, metadata, model, dim, norm, storage, vector, created_at) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        with self._lock:
            self._conn.executemany(sql, rows)
            self._conn.commit()
            last = self._conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
            ids = list(range(last - len(rows) + 1, last + 1))

        logger.info("向量库入库 %d 条 (model=%s, dim=%d, storage=%s)", len(rows), model, dim, self.storage)
        return ids

    # ==================== 读 ====================

    def _assert_same_model(self, model: Optional[str], dim: Optional[int]) -> None:
        """
        跨模型 / 跨维度直接拒绝。
        """
        with self._lock:
            rows = self._conn.execute(
                f"SELECT DISTINCT model FROM {self.table}"
            ).fetchall()
            dims = self._conn.execute(
                f"SELECT DISTINCT dim FROM {self.table}"
            ).fetchall()

        stored_models = [r["model"] for r in rows]
        stored_dims = [r["dim"] for r in dims]

        if model is not None and stored_models and model not in stored_models:
            raise VectorStoreException(
                f"跨模型的向量不可比：库里的向量来自 {stored_models}，"
                f"这次查询用的是 '{model}'。请换用产生这些向量的同一个模型来查询，"
                f"或者把这批向量重新灌一遍。"
            )
        if dim is not None and stored_dims and dim not in stored_dims:
            raise VectorStoreException(
                f"库里的向量维度是 {stored_dims}, 这次查询是 {dim} 维。维度不同无法计算相似度。"
            )

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        model: Optional[str] = None,
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        """
        全表扫描算余弦, 返回最像的 top_k 条。
        Args:
            query_vector: 查询向量
            top_k:        返回条数
            model:        查询向量的模型指纹, 用于跨模型校验
        Returns:
            [(相似度, 文本, metadata), ...], 按相似度**从高到低**。
            相似度在 [-1, 1]: 1 = 完全同向, 0 = 正交(无关), -1 = 反向。
        """
        query = list(query_vector)
        if not query:
            return []
        if top_k <= 0:
            return []

        self._assert_same_model(model, len(query))
        q_norm = _norm(query)

        with self._lock:
            rows = self._conn.execute(
                f"SELECT text, metadata, norm, storage, vector FROM {self.table}"
            ).fetchall()

        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for row in rows:
            vector = self._decode(row["vector"], row["storage"])
            if len(vector) != len(query):
                logger.warning("跳过一条维度不符的记录(库里 %d 维, 查询 %d 维)", len(vector), len(query))
                continue

            score = cosine_similarity(query, vector, query_norm=q_norm, vector_norm=row["norm"])
            try:
                meta = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                meta = {}
            scored.append((score, row["text"], meta))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[:top_k]

    # ==================== 杂项 ====================

    def count(self) -> int:
        """库里有多少条"""
        with self._lock:
            return self._conn.execute(f"SELECT COUNT(*) AS n FROM {self.table}").fetchone()["n"]

    def stats(self) -> Dict[str, Any]:
        """一眼看清库里存了什么 —— 排查"为什么搜不到"时先看它"""
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) AS n, COUNT(DISTINCT model) AS models, "
                f"COUNT(DISTINCT dim) AS dims, COUNT(DISTINCT storage) AS storages "
                f"FROM {self.table}"
            ).fetchone()
            models = [r["model"] for r in self._conn.execute(
                f"SELECT DISTINCT model FROM {self.table}"
            ).fetchall()]
            storages = [r["storage"] for r in self._conn.execute(
                f"SELECT DISTINCT storage FROM {self.table}"
            ).fetchall()]
        return {
            "count": row["n"],
            "models": models,
            "storages": storages,
            "path": self.path,
            "table": self.table,
        }

    def clear(self) -> int:
        """清空(表结构保留), 返回删掉多少条"""
        with self._lock:
            n = self.count()
            self._conn.execute(f"DELETE FROM {self.table}")
            self._conn.commit()
        logger.warning("向量库已清空: %s.%s (删除 %d 条)", self.path, self.table, n)
        return n

    def close(self) -> None:
        """关连接。用完记得调 —— 不关的话 Windows 上文件会一直被占着。"""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __len__(self) -> int:
        return self.count()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        try:
            n = self.count()
        except Exception:
            n = "?"
        return f"VectorStore(path={self.path}, table={self.table}, storage={self.storage}, count={n})"


# ==================== 验证(指南第三步, 不需要 API key) ====================
#
# 构造几个已知向量, 手算余弦, 对比:
#     正交向量 -> 0      同向向量 -> 1      反向向量 -> -1
#
# 完整的版本见 tests/test_vector_store.py, 直接跑:
#     python tests/test_vector_store.py


