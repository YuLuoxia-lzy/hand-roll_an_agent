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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
    **ignored,
) -> float:
    """
    余弦相似度。模长允许外部传进来(见"预存模长")
    两个模长里只要有一个是 0(全零向量), 相似度就不再有定义 —— 返回 0.0

    【**ignored 不是摆设】
    它是默认 scorer, 而 search() 调 scorer 时会多传 created_at / metadata
    (见下面的 Scorer 说明)。多出来的关键字在这里被丢掉, 于是"默认打分"和
    "自定义打分"能被同一处代码调用 —— 否则就得写个 if scorer is None 的分支,
    两条路的调用姿势还不一样。
    """
    qn = query_norm if query_norm is not None else _norm(query)
    vn = vector_norm if vector_norm is not None else _norm(vector)
    if qn == 0 or vn == 0:
        return 0.0
    return _dot(query, vector) / (qn * vn)


#: 打分函数的签名。
#:
#: 默认是纯余弦 —— 这是**文档系统**的正确选择: 一份 2023 年的手册不该因为"旧"就沉下去。
#: 小镇的记忆流要的是 recency + importance + relevance 三因子, 那是**另一套**公式。
#: 两者冲突, 框架不该替应用选, 所以留这条缝:
#:
#:     def memory_stream_scorer(query, vector, *, query_norm, vector_norm,
#:                              created_at, metadata, **kw):
#:         relevance  = cosine_similarity(query, vector, query_norm, vector_norm)
#:         recency    = 0.99 ** ((time.time() - created_at) / 3600)   # 指数衰减
#:         importance = metadata.get("importance", 5) / 10            # LLM 打的分存这
#:         return 1.0 * recency + 1.0 * importance + 1.5 * relevance
#:
#:     for score, text, meta in store.search(vec, scorer=memory_stream_scorer): ...
#:
#: 返回的分数**越大越靠前**(search 按 -score 排序)。
Scorer = Callable[..., float]


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
            self._has_json1 = self._probe_json1()

        logger.info(
            "向量库就绪: path=%s table=%s storage=%s numpy=%s json1=%s",
            self.path, self.table, self.storage, _HAS_NUMPY, self._has_json1,
        )

    def _probe_json1(self) -> bool:
        """
        探一次 json1 扩展在不在。

        filters(元数据过滤)和 delete(filters=...) 靠 json_extract 实现, 它需要
        sqlite 的 json1 扩展。Python 3.10 自带的 sqlite 实测都有, 但**这不是保证**
        —— 编译选项不同的发行版可能没有。所以探一次, 把结果记下来。

        探到没有不会报错, 而是走**退化路径**(取回后在 Python 里筛), 并且每次退化都
        留一条 warning —— 静默退化比报错更难查。在这里探而不是等用户第一次传
        filters 时才崩在一条 sqlite 的 OperationalError 上, 也是同一个理由。
        """
        try:
            self._conn.execute("SELECT json_extract('{\"a\":1}', '$.a')")
            return True
        except sqlite3.OperationalError:
            return False

    def _require_open(self) -> None:
        """
        关掉之后再用, 给一句人话。

        不加这个的话报的是 `AttributeError: 'NoneType' object has no attribute
        'execute'` —— 它既没说"这个库已经关了", 也没说是谁关的, 而且是**在**
        with 块外面才炸(往往是几层调用之后), 找起来很费劲。
        """
        if self._conn is None:
            raise VectorStoreException(
                f"向量库已经关闭(path={self.path}, table={self.table}), 不能再用了。"
                f"close() 之后连 count() 都不行 —— 想要继续用就重新建一个实例。"
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
        self._require_open()
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

        【表级校验, 它给的是一条更清楚的报错, 不是唯一的防线】
        它只在"这个模型在库里**完全没出现过**"时拦下来。库里混进两个模型的向量时
        (换 embedding provider 之后复用同一个库就会发生), 它**放行** —— 所以
        search() 里还有一条行级的 `WHERE model = ?`, 两者缺一不可。见 2.4。
        """
        self._require_open()
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
        filters: Optional[Dict[str, Any]] = None,
        scorer: Optional[Scorer] = None,
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        """
        全表扫描算分, 返回最像的 top_k 条。

        Args:
            query_vector: 查询向量
            top_k:        返回条数
            model:        查询向量的模型指纹。**行级过滤** —— 不只是校验, 见下
            filters:      元数据等值过滤, 如 {"source": "手册.md"}。
                          **在 SQL 层过滤**(和向量检索同一次扫描), 不是取回来再筛 ——
                          top_k 是在全库上截断的, 先截断再筛可能一条不剩,
                          而真正匹配的还躺在第 6 到第 50 名。
            scorer:       打分函数, 默认纯余弦。见 Scorer 的说明
        Returns:
            [(分数, 文本, metadata), ...], 按分数**从高到低**(大者靠前)。
            默认打分下分数在 [-1, 1]: 1 = 完全同向, 0 = 正交(无关), -1 = 反向。
            换成自定义 scorer 之后分数就是它自己的量纲 —— **返回值永远是三元组**,
            recency 之类的效果已经编码在分数里, 不额外往外传(零破坏)。
        """
        query = list(query_vector)
        if not query:
            return []
        if top_k <= 0:
            return []

        # 表级校验: 模型在库里完全没出现过时, 给一句更清楚的报错
        self._assert_same_model(model, len(query))
        q_norm = _norm(query)

        where_parts, params = [], []
        if model is not None:
            # ← 行级过滤, 和表级校验是**两回事**, 两个都要。
            # 表级只在"这个模型在库里一条都没有"时拦下来; 库里混进两个模型的向量时
            # (换 embedding provider 后复用同一个库就会发生) 它会**放行**,
            # 于是旧模型的行也参与打分、还可能拿满分。不报错, 只给错结果。
            where_parts.append("model = ?")
            params.append(model)

        # filters: 能用 SQL 就用 SQL, 不能就退化(见 _probe_json1)
        use_sql_filters = bool(filters) and self._has_json1
        if use_sql_filters:
            where, filter_params = self._where_clause(filters)
            where_parts.append(where.replace(" WHERE ", ""))
            params.extend(filter_params)
        elif filters:
            logger.warning(
                "本机 sqlite 没有 json1 扩展, filters 退化成『取回后在内存里筛』: %s。"
                "结果可能比预期少 —— top_k 仍然是在全库上截断的。",
                filters,
            )

        sql = f"SELECT id, text, metadata, norm, storage, vector, created_at FROM {self.table}"
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        score_fn = scorer or cosine_similarity

        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for row in rows:
            vector = self._decode(row["vector"], row["storage"])
            if len(vector) != len(query):
                logger.warning("跳过一条维度不符的记录(库里 %d 维, 查询 %d 维)", len(vector), len(query))
                continue

            try:
                meta = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                meta = {}

            if filters and not use_sql_filters and not self._match_filters(meta, filters):
                continue

            score = score_fn(
                query, vector,
                query_norm=q_norm,
                vector_norm=row["norm"],
                # 下面两个是给自定义 scorer 的额外信号(recency 要用 created_at,
                # importance 之类走 metadata)。默认的 cosine_similarity 用 **ignored 丢掉它们。
                created_at=row["created_at"],
                metadata=meta,
            )
            scored.append((score, row["text"], meta))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[:top_k]

    # ==================== filters: 元数据过滤 ====================

    @staticmethod
    def _check_filter_keys(filters: Dict[str, Any]) -> None:
        """
        键必须白名单校验 —— 它是**拼进 SQL 字符串**的, 不是绑定参数。

        虽然 key 落在 json_extract 的路径字面量里, 注入面比看起来小,
        但"看起来安全"不是安全。

        规则和报错信息说的一致: 只收 ASCII 字母数字下划线。
        (⚠️ str.isalnum() 对中文也返回 True, 所以必须再加 isascii ——
         光靠 isalnum 会把 '$.中文' 这种路径放进 SQL。)
        """
        for key in filters:
            if not isinstance(key, str):
                raise VectorStoreException(
                    f"filters 的键必须是字符串, 收到 {type(key).__name__}。"
                )
            bare = key.replace("_", "")
            if not (bare.isascii() and bare.isalnum()):
                raise VectorStoreException(
                    f"filters 的键只能是字母数字下划线, 收到 '{key}'。"
                )

    def _where_clause(self, filters: Optional[Dict[str, Any]]) -> Tuple[str, list]:
        """
        把 {k: v} 翻译成 SQL 的 WHERE 片段。**等值匹配**, 需要范围查询再加。

        返回 (" WHERE a = ? AND b = ?", [值, 值]) —— 值走**参数绑定**, 不拼进 SQL。
        """
        if not filters:
            return "", []

        self._check_filter_keys(filters)

        clauses, params = [], []
        for key, value in filters.items():
            clauses.append(f"json_extract(metadata, '$.{key}') = ?")
            params.append(value)
        return " WHERE " + " AND ".join(clauses), params

    @staticmethod
    def _match_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        json1 不可用时的退化路径: 取回之后在内存里筛。

        语义刻意和 SQL 版对齐(等值), 包括"键不存在就算不匹配"这一条 ——
        SQL 那边键不存在时 json_extract 给 NULL, `NULL = ?` 是 NULL(假),
        所以这里也要求 k 必须**在** metadata 里, 而不是拿 .get() 的 None 去比。
        """
        return all(k in metadata and metadata[k] == v for k, v in filters.items())

    # ==================== delete: 按条件删除 ====================

    def delete(
        self,
        ids: Optional[Sequence[int]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        按 id 或元数据条件删除, 返回删了几条。

        【安全设计: 两个都不传就拒绝】
        不带条件的 delete 和 clear() 长得一样, 但语义完全不同 ——
        前者是"我确定要删这些", 后者是"我要清库"。
        允许无参调用, 就等于给了一次"手滑清库"的机会。

        【为什么需要它】
        今天只有 clear()(全清)。想"只更新变了的那份文档"就只能全量重建,
        连别的 source 一起清掉 —— 所以"增量入库"在原来的 API 上**做不到**。
        """
        self._require_open()
        if ids is None and filters is None:
            raise VectorStoreException(
                "delete 必须给出 ids 或 filters 之一。"
                "要清空整张表请显式调用 clear() —— 它会在日志里留一条 warning。"
            )

        if ids is not None:
            ids = list(ids)

        asked_filters = filters          # 只给日志用 —— 退化路径会把它折进 ids
        if filters and not self._has_json1:
            # 没有 json1 就没法在 SQL 里筛元数据 —— 退化成"先扫出 id, 再按 id 删"。
            # 结果**是对的**(和 SQL 版等值语义一致), 只是多一次全表扫描。
            logger.warning("本机 sqlite 没有 json1 扩展, delete(filters=...) 退化成先查 id 再删: %s", filters)
            matched = set(self._scan_for_filters(filters))
            ids = sorted(matched if ids is None else matched & set(ids))
            filters = None

        clauses, params = [], []
        if ids:
            clauses.append(f"id IN ({','.join('?' * len(ids))})")
            params.extend(ids)
        if filters:
            where, filter_params = self._where_clause(filters)
            # 这里的片段不带 " WHERE " 前缀, 因为要拼进 AND 串里
            clauses.append(where.replace(" WHERE ", ""))
            params.extend(filter_params)

        if not clauses:
            return 0

        sql = f"DELETE FROM {self.table} WHERE " + " AND ".join(clauses)
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
        logger.info("向量库删除 %d 条 (ids=%s filters=%s)", cur.rowcount, ids, asked_filters)
        return cur.rowcount

    def _scan_for_filters(self, filters: Dict[str, Any]) -> List[int]:
        """json1 不可用时的兜底: 全表取回, 在内存里按 filters 筛出匹配行的 id"""
        self._check_filter_keys(filters)
        with self._lock:
            rows = self._conn.execute(f"SELECT id, metadata FROM {self.table}").fetchall()

        matched = []
        for row in rows:
            try:
                meta = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                meta = {}
            if self._match_filters(meta, filters):
                matched.append(row["id"])
        return matched

    # ==================== 杂项 ====================

    def count(self) -> int:
        """库里有多少条"""
        self._require_open()
        with self._lock:
            return self._conn.execute(f"SELECT COUNT(*) AS n FROM {self.table}").fetchone()["n"]

    def stats(self) -> Dict[str, Any]:
        """一眼看清库里存了什么 —— 排查"为什么搜不到"时先看它"""
        self._require_open()
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
        self._require_open()
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


