"""语义记忆离线测试 —— 用假 embedder, 不花钱不联网

    python tests/test_semantic.py

测的是第四步那句"写入路径必须显式设计": remember / ingest_file 真的能被调用,
而且 search 真的能搜到。**只设计 search() 而没人调 add() 的话, 库永远是空的。**
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import FakeEmbedder, TempDir, run_tests            # noqa: E402

from agents0to1.memory.semantic import (                         # noqa: E402
    MemoryItem,
    SemanticMemory,
    SemanticMemoryException,
)


def _memory(dim: int = 64, **kwargs):
    """一个连着假 embedder 的语义记忆 —— 每次调用都用一个全新的临时库"""
    return SemanticMemory(embedder=FakeEmbedder(dim=dim), path=":memory:", **kwargs)


# ==================== 写入路径 ====================

def test_remember_then_search():
    mem = _memory()
    mem.remember("小明住在杭州西湖区。")
    mem.remember("公司团建定在下个月三号。")

    got = mem.search("小明住哪里")
    assert got, "刚存进去就搜不到了 —— 检查 embedder 是不是恒等返回了同一个向量"
    assert all(isinstance(i, MemoryItem) for i in got)
    mem.close()


def test_search_returns_most_similar_first():
    mem = _memory()
    mem.remember("向量检索的核心是余弦相似度。")
    mem.remember("今天中午吃的是番茄炒蛋。")
    mem.remember("余弦相似度等于点积除以两个模长的乘积。")

    got = mem.search("余弦相似度怎么算")
    assert len(got) >= 2
    assert "余弦" in got[0].text, f"最相关的没排最前: {[i.text for i in got]}"
    # 分数必须真的递减
    scores = [i.score for i in got]
    assert scores == sorted(scores, reverse=True), scores
    mem.close()


def test_long_text_is_chunked_on_write():
    """remember 一条长文本会存成多块 —— 否则整篇文档只有一个向量, 什么都搜不准"""
    mem = _memory()
    n = mem.remember("第一句话在这里。" * 200)
    assert n > 1, f"长文本没有被切分, 只存了 {n} 块"
    assert mem.count() == n
    mem.close()


def test_ingest_file():
    with TempDir() as d:
        path = d / "手册.md"
        path.write_text(
            "# 部署手册\n\n第一步先装依赖。\n第二步配置密钥。\n第三步启动服务。\n",
            encoding="utf-8",
        )
        mem = _memory()
        n = mem.ingest_file(str(path))
        assert n >= 1
        got = mem.search("怎么配置密钥")
        assert got, "入库之后搜不到"
        assert got[0].metadata["source"] == "手册.md", got[0].metadata
        mem.close()


def test_ingest_missing_file_raises():
    mem = _memory()
    try:
        mem.ingest_file("这个文件不存在.txt")
    except SemanticMemoryException:
        mem.close()
        return
    raise AssertionError("文件不存在应该明确报错")


def test_ingest_blank_file_is_skipped():
    """空文件不该产生空块, 也不该报错"""
    with TempDir() as d:
        path = d / "空.txt"
        path.write_text("   \n\n  ", encoding="utf-8")
        mem = _memory()
        assert mem.ingest_file(str(path)) == 0
        assert mem.count() == 0
        mem.close()


def test_metadata_carries_position():
    """没有位置信息, 检索回来你既不知道它从哪来, 也没法引用"""
    mem = _memory()
    mem.remember("一些内容。" * 100)
    items = mem.search("一些内容")
    assert items[0].metadata["chunk_index"] == 0
    assert items[0].metadata["chunk_total"] >= 1
    assert "source" in items[0].metadata
    assert items[0].citation(), "citation() 不该是空的"
    mem.close()


# ==================== 检索边界 ====================

def test_empty_query_returns_nothing():
    mem = _memory()
    mem.remember("有内容。")
    assert mem.search("") == []
    assert mem.search("   ") == []
    assert mem.search(None) == []
    mem.close()


def test_search_on_empty_store_returns_nothing():
    mem = _memory()
    assert mem.search("随便问点什么") == []
    mem.close()


def test_min_score_threshold_filters():
    """分数太低的不该喂给模型, 那只会干扰它"""
    mem = _memory()
    mem.remember("完全不相关的内容在这里, 讲的是做饭和买菜。")

    assert mem.search("余弦相似度", min_score=-1.0), "阈值放到 -1 应该能搜出来"
    assert mem.search("余弦相似度", min_score=0.99) == [], "阈值 0.99 应该滤掉它"
    mem.close()


def test_search_does_not_write():
    """search 是只读的 —— 检索一次就往库里写东西的话, 库会被自己的查询污染"""
    mem = _memory()
    mem.remember("一条内容。")
    before = mem.count()
    for _ in range(5):
        mem.search("一条内容")
    assert mem.count() == before
    mem.close()


# ==================== 批量: 不要一条条发 ====================

def test_embedding_is_batched_not_per_chunk():
    """
    一次请求发 32 条 vs 发 32 次请求, 差几十倍 —— 这是 embedding 和 chat
    最大的调用习惯差异。所以入库时**必须**是"攒一批发一次", 不是每块发一次。
    """
    with TempDir() as d:
        path = d / "长文.md"
        path.write_text("这是第%d句话, 用来把文档撑长。" % 1 + "内容各异的一句话。" * 300, encoding="utf-8")

        mem = _memory()
        n = mem.ingest_file(str(path))
        assert n > 5, f"文档没被切成足够多的块: {n}"

        calls = mem.embedder.embed_calls
        assert len(calls) == 1, (
            f"入库应该只发 1 次 embedding 请求(内部攒批), 实际发了 {len(calls)} 次 —— "
            f"退化成『一块一发』了"
        )
        assert len(calls[0]) == n
        mem.close()


# ==================== 给 LLM 看的形态 ====================

def test_format_items_empty():
    mem = _memory()
    assert mem.format_items([]) == ""
    mem.close()


def test_build_context_includes_score_and_source():
    """score 必须暴露出来: 你要能设阈值, 也要能调试"为什么这次没检索到" """
    mem = _memory()
    mem.remember("部署流程是先装依赖再配密钥。", metadata={"source": "手册.md"})
    ctx = mem.build_context("部署流程")
    assert "相似度" in ctx and "手册.md" in ctx, ctx
    mem.close()


def test_build_context_respects_budget():
    """超预算时必须在**这里**裁好, 不能留给外层的 truncate_output ——
    那个是"保头保尾", 会把最后一块的尾巴接到前面, 相关性顺序就乱了。"""
    mem = _memory(context_budget=180)
    for i in range(8):
        mem.remember(f"第{i}份资料讲的是各不相同的主题内容。", metadata={"source": f"doc{i}.md"})

    ctx = mem.build_context("资料")
    assert "超出预算未展示" in ctx, f"预算太小却没触发裁剪:\n{ctx}"
    assert len(ctx) < 600, f"裁剪没生效, 长度 {len(ctx)}"
    mem.close()


def test_build_context_keeps_best_first_when_truncated():
    mem = _memory(context_budget=150)
    mem.remember("余弦相似度是向量检索的核心指标。")
    for i in range(6):
        mem.remember(f"第{i}条讲的完全是别的事情。")

    ctx = mem.build_context("余弦相似度")
    assert "余弦" in ctx[:200], f"最相关的那条被裁掉了:\n{ctx[:300]}"
    mem.close()


# ==================== 分层: 谁抛、谁吞 ====================

def test_search_raises_when_embedder_fails():
    """search 是**编程接口**, 出错要让调用方知道 —— 异常在这里不被吞掉"""
    mem = _memory()
    mem.embedder.fail = True
    try:
        mem.search("随便问")
    except Exception as e:
        assert "FakeEmbedder" in str(e) or "失败" in str(e), e
        mem.close()
        return
    raise AssertionError("embedder 挂了, search 必须抛 —— 它不该偷偷返回空列表")


def test_build_context_fails_open():
    """build_context 正对着 LLM 漏斗, **必须** fail-open:

    它抛异常的后果不是"少了一段上下文", 而是整个 agent 挂掉。
    记忆只是增强, 不该有这个权力。
    """
    mem = _memory()
    mem.remember("有内容。")
    mem.embedder.fail = True

    ctx = mem.build_context("随便问")
    assert ctx == "", f"应该退化成空字符串, 实际: {ctx!r}"
    mem.close()


def test_fail_open_logs_warning_not_traceback():
    """
    fail-open 用的必须是一条 warning, **不是整个堆栈** ——
    这在生产里是会反复发生的事(embedding 独立 provider、独立 key、DeepSeek
    压根没有 embedding 端点), 每次都打一整个堆栈会把日志刷爆。

    【上一版这条是装样子的】它只调了一遍 build_context, 没有任何东西观察
    logging —— 实现改成 logger.exception 打整个堆栈, 它照样绿。
    所以这里挂一个 handler 上去, 真的把记录抓下来看。
    """
    import logging

    from agents0to1.memory.vector_store import VectorStoreException   # noqa: F401  仅为断言用

    records = []

    class Grab(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("agents0to1.memory.semantic")
    handler = Grab(level=logging.DEBUG)
    old_level = logger.level
    logger.addHandler(handler)
    # 别让外部配置(LOG_LEVEL=ERROR)把这条用例弄红 —— 它测的是"打的是什么级别",
    # 不是"当前环境开没开日志"
    logger.setLevel(logging.DEBUG)
    try:
        mem = _memory()
        mem.embedder.fail = True
        assert mem.build_context("随便问") == ""
        mem.close()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)

    assert records, "fail-open 时应该留下一条日志, 否则出问题没人看得见"
    assert all(r.levelno == logging.WARNING for r in records), (
        f"fail-open 只该 warning, 收到: {[(r.levelname, r.getMessage()) for r in records]}"
    )


# ==================== 缝: store 可注入 / scorer / 遗忘 ====================

class _FakeStore:
    """
    一个**自己实现**的向量库。

    刻意不继承 VectorStore —— 那会把"换一个类"变成"换一个子类",
    而接口承诺的是"有那几个方法就行"。
    """

    def __init__(self):
        self.rows = []              # [{"text", "vector", "meta"}]
        self.search_args = []

    def add(self, texts, vectors, metadatas=None, model="unknown"):
        metas = list(metadatas) if metadatas is not None else [{} for _ in texts]
        start = len(self.rows)
        for text, vector, meta in zip(texts, vectors, metas):
            self.rows.append({"text": text, "vector": list(vector), "meta": dict(meta or {})})
        return list(range(start, len(self.rows)))

    def search(self, query_vector, top_k=5, model=None, filters=None, scorer=None):
        self.search_args.append({"top_k": top_k, "model": model, "filters": filters})
        out = []
        for row in self.rows:
            if filters and not all(row["meta"].get(k) == v for k, v in filters.items()):
                continue
            score = 1.0
            if scorer:
                score = scorer(query_vector, row["vector"], metadata=row["meta"],
                               created_at=0.0, query_norm=None, vector_norm=None)
            out.append((score, row["text"], row["meta"]))
        out.sort(key=lambda item: -item[0])
        return out[:top_k]

    def count(self):
        return len(self.rows)

    def stats(self):
        return {"count": len(self.rows)}

    def clear(self):
        n = len(self.rows)
        self.rows = []
        return n

    def close(self):
        self.closed = True


def test_semantic_memory_accepts_a_custom_store():
    """**"换 faiss / chroma 就是换一个类"这条承诺的兑现处。**

    改造前 store 是写死的 VectorStore(...), 文档却一直声称可换 —— 换不了。
    """
    store = _FakeStore()
    mem = SemanticMemory(embedder=FakeEmbedder(), store=store)

    mem.remember("小明住在杭州西湖区。")
    assert store.count() == 1, "入库必须走的是注入进来的那个 store"
    assert mem.count() == 1

    got = mem.search("小明住哪里", top_k=3)
    assert [i.text for i in got] == ["小明住在杭州西湖区。"]

    # 检索参数也要真的传到 store 上(尤其是 filters)
    assert store.search_args[-1]["model"] == mem.embedder_id
    mem.close()
    assert store.closed, "close() 要穿透到注入的 store"


def test_custom_store_needs_the_full_contract():
    """缺方法要在**构造时**就报错 —— 而不是等第一次检索才炸"""
    class Half:
        def add(self, *a, **k): ...
        def search(self, *a, **k): ...

    try:
        SemanticMemory(embedder=FakeEmbedder(), store=Half())
    except TypeError as e:
        assert "count" in str(e)
        return
    raise AssertionError("store 缺 count/clear/close 应该在构造时就拒绝")


def test_scorer_can_be_swapped_at_memory_level():
    """小镇用法: **不继承、不改框架**, 传一个 scorer 就换掉了打分公式。

    文档系统保持默认余弦(旧手册不该因为旧就沉下去), 小镇要三因子 ——
    两者冲突, 所以框架把选择权交出来。
    """
    def by_rank(query, vector, *, metadata, **kw):
        return float(metadata.get("rank", 0))

    mem = _memory(scorer=by_rank)
    mem.remember("甲", {"rank": 1})
    mem.remember("乙", {"rank": 3})
    mem.remember("丙", {"rank": 2})

    got = [i.text for i in mem.search("任意查询", top_k=3)]
    assert got == ["乙", "丙", "甲"], f"打分该由 scorer 决定(rank 3/2/1), 实际 {got}"

    # 单次调用可以覆盖实例上的默认值
    def reversed_rank(query, vector, *, metadata, **kw):
        return -float(metadata.get("rank", 0))

    # ⚠️ 必须显式放低 min_score: min_score 比的是 **scorer 的输出**,
    # 默认 0.0 会把 -1/-2/-3 全部滤掉 —— 换打分公式就得跟着调阈值。
    got = [i.text for i in mem.search("任意查询", top_k=3, scorer=reversed_rank, min_score=-10)]
    assert got == ["甲", "丙", "乙"], f"单次调用的 scorer 该覆盖实例默认值, 实际 {got}"
    mem.close()


def test_filters_reach_the_store():
    mem = _memory()
    mem.remember("甲来源", {"source": "甲.md"})
    mem.remember("乙来源", {"source": "乙.md"})

    got = mem.search("来源", top_k=5, filters={"source": "乙.md"})
    assert [i.text for i in got] == ["乙来源"]
    mem.close()


def test_forget_by_source():
    """小镇做"遗忘"的入口"""
    with TempDir() as d:
        mem = SemanticMemory(embedder=FakeEmbedder(), path=str(Path(d) / "s.sqlite3"))
        mem.remember("甲说的话", {"source": "甲.md"})
        mem.remember("乙说的话", {"source": "乙.md"})
        assert mem.count() == 2

        assert mem.forget({"source": "甲.md"}) == 1
        assert mem.count() == 1
        assert [i.text for i in mem.search("说的话", top_k=5)] == ["乙说的话"]
        mem.close()


def test_reindex_file_replaces_instead_of_duplicating():
    """**增量入库**: 没有 delete 就只能全量重建, 连别的来源一起清掉。"""
    with TempDir() as d:
        src = Path(d) / "手册.md"
        src.write_text("第一版的内容。", encoding="utf-8")

        mem = SemanticMemory(embedder=FakeEmbedder(), path=str(Path(d) / "s.sqlite3"))
        mem.ingest_file(str(src))
        first = mem.count()
        assert first == 1

        # 同一个文件再 ingest 一次 -> 重复入库。这就是那个坑, 也是 reindex 存在的理由
        mem.ingest_file(str(src))
        assert mem.count() == first * 2, "重复 ingest 会重复入库(所以更新文件要用 reindex_file)"

        src.write_text("第二版的内容。", encoding="utf-8")
        assert mem.reindex_file(str(src)) == first
        assert mem.count() == first, "reindex 之后该只剩新的一版, 而不是两版叠加"

        got = mem.search("内容", top_k=10)
        assert len(got) == 1 and "第二版" in got[0].text
        mem.close()


# ==================== 杂项 ====================

def test_stats_and_clear():
    mem = _memory()
    mem.remember("一些内容。")
    info = mem.stats()
    assert info["count"] >= 1
    assert info["embedder"] == mem.embedder_id
    assert mem.clear() >= 1 and mem.count() == 0
    mem.close()


def test_close_is_idempotent_and_then_fails_clearly():
    """
    关两次不该炸(幂等), 但关掉**之后**再用必须报一句人话。

    报 `AttributeError: 'NoneType' object has no attribute 'execute'` 的话,
    既没说"这个库已经关了", 也没说谁关的 —— 而且往往是在 with 块外面几层才炸。
    """
    from agents0to1.memory.vector_store import VectorStoreException

    mem = _memory()
    mem.close()
    mem.close()                                   # 幂等

    try:
        mem.count()
    except VectorStoreException as e:
        assert "关闭" in str(e), f"报错信息里没说清是「已关闭」: {e}"
        return
    raise AssertionError("close() 之后 count() 必须报一句清楚的错, 而不是 AttributeError")


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "语义记忆"))
