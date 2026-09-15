"""向量库离线测试 —— **完全不需要 API key, 随时可跑**

    python tests/test_vector_store.py

指南第三步的验证在这儿: 构造已知向量, 手算余弦, 对比。
再加一条第三步特意点名的: **并发安全**(单线程测一百遍都测不出来)。
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import TempDir, run_tests                      # noqa: E402

from agents0to1.memory.vector_store import (                 # noqa: E402
    STORAGE_BLOB,
    STORAGE_JSON,
    VectorStore,
    VectorStoreException,
    cosine_similarity,
    has_numpy,
)


# ==================== 余弦相似度本身 ====================

def test_cosine_orthogonal_is_zero():
    assert abs(cosine_similarity([1, 0, 0], [0, 1, 0])) < 1e-12


def test_cosine_same_direction_is_one():
    assert abs(cosine_similarity([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-12


def test_cosine_opposite_is_minus_one():
    assert abs(cosine_similarity([1, 2, 3], [-1, -2, -3]) + 1.0) < 1e-12


def test_cosine_zero_vector_is_safe():
    """全零向量没有方向, 相似度没有定义 —— 返回 0 而不是 ZeroDivisionError。

    库里混进一条全零向量不该让整个查询崩掉。
    """
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0
    assert cosine_similarity([1, 2, 3], [0, 0, 0]) == 0.0


def test_cosine_precomputed_norm_gives_same_answer():
    """预存模长只是个优化, 结果必须和现算的一模一样"""
    a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    from agents0to1.memory.vector_store import _norm
    assert abs(
        cosine_similarity(a, b) - cosine_similarity(a, b, _norm(a), _norm(b))
    ) < 1e-12


# ==================== 入库 / 检索 ====================

def test_search_returns_most_similar_first():
    store = VectorStore(":memory:")
    store.add(
        ["正交的", "最像的", "同向的"],
        [[0, 1, 0], [0.99, 0.1, 0], [1, 0, 0]],
        model="m",
    )
    got = store.search([1, 0, 0], top_k=3, model="m")
    assert [t for _, t, _ in got] == ["同向的", "最像的", "正交的"], [t for _, t, _ in got]
    assert got[0][0] > got[1][0] > got[2][0]
    store.close()


def test_search_respects_top_k():
    store = VectorStore(":memory:")
    store.add([f"t{i}" for i in range(10)], [[1.0, i / 10] for i in range(10)], model="m")
    assert len(store.search([1, 0], top_k=3, model="m")) == 3
    assert len(store.search([1, 0], top_k=100, model="m")) == 10
    assert store.search([1, 0], top_k=0, model="m") == []
    store.close()


def test_metadata_roundtrip():
    store = VectorStore(":memory:")
    store.add(["x"], [[1, 0]], metadatas=[{"source": "手册.md", "chunk_index": 3}], model="m")
    _, _, meta = store.search([1, 0], top_k=1, model="m")[0]
    assert meta["source"] == "手册.md" and meta["chunk_index"] == 3
    store.close()


def test_empty_inputs_are_noop():
    store = VectorStore(":memory:")
    assert store.add([], [], model="m") == []
    assert store.search([1, 0], model="m") == []      # 空库
    assert store.search([], model="m") == []          # 空查询
    assert store.count() == 0
    store.close()


def test_add_length_mismatch_raises():
    store = VectorStore(":memory:")
    try:
        store.add(["a", "b"], [[1, 0]], model="m")
    except VectorStoreException:
        store.close()
        return
    raise AssertionError("长度不一致应该抛异常 —— 硬塞进去会让文本和向量错位")


def test_add_dim_mismatch_raises():
    store = VectorStore(":memory:")
    try:
        store.add(["a", "b"], [[1, 0], [1, 0, 0]], model="m")
    except VectorStoreException:
        store.close()
        return
    raise AssertionError("同一批里维度不一致应该抛异常")


# ==================== 第 ③ 条: 跨模型 / 跨维度直接拒绝 ====================

def test_cross_model_search_rejected():
    """**这是本文件最该有的一条测试。**

    两个不同模型产生的向量, 余弦相似度算出来是纯噪声 —— 而且它**不会报错**,
    只会给你错误的结果。宁可不给结果, 也不给垃圾。
    """
    store = VectorStore(":memory:")
    store.add(["a"], [[1, 0]], model="openai/text-embedding-3-small")
    try:
        store.search([1, 0], model="dashscope/text-embedding-v3")
    except VectorStoreException as e:
        assert "不可比" in str(e)
        store.close()
        return
    raise AssertionError("跨模型检索必须直接拒绝, 而不是算出一个看起来正常的垃圾分数")


def test_cross_dim_search_rejected():
    store = VectorStore(":memory:")
    store.add(["a"], [[1, 0, 0]], model="m")
    try:
        store.search([1, 0], model="m")
    except VectorStoreException:
        store.close()
        return
    raise AssertionError("维度不同无法计算相似度, 必须拦下来")


def test_same_model_still_works_after_other_model_added():
    """拒绝的是"查询和库里对不上", 不是"库里有两种模型就全废" """
    store = VectorStore(":memory:")
    store.add(["a"], [[1, 0]], model="m1")
    store.add(["b"], [[0, 1]], model="m2")
    got = store.search([1, 0], model="m1")      # 维度相同, m1 在库里 -> 放行, 不抛
    assert [t for _, t, _ in got] == ["a"]      # 但只该拿到 m1 自己的那条
    store.close()


def test_mixed_model_rows_do_not_participate_in_scoring():
    """**表级校验拦不住的那种混库。**

    换 embedding provider 之后复用同一个库就会发生: 库里有 m1 和 m2 两种向量,
    拿 m1 查询时表级校验会**放行**(m1 确实在库里)。如果 SELECT 不带行级过滤,
    m2 的行也会一起参与打分 —— 而且可能拿到接近满分。

    实测过的原症状(修复前):
        库里模型: ['openai/text-embedding-3-small', 'dashscope/text-embedding-v3']
           1.0000  乙: 牛顿定律      ← dashscope 自己的
           1.0000  甲: 苹果是水果     ← openai 生成的, 却被打了满分

    不报错, 只给错结果。
    """
    store = VectorStore(":memory:")
    store.add(["甲: 苹果是水果"], [[1, 0]], model="openai/emb")
    store.add(["乙: 牛顿定律"], [[1, 0]], model="dashscope/emb")   # 故意用同一个向量

    got = store.search([1, 0], top_k=10, model="openai/emb")

    texts = [t for _, t, _ in got]
    assert texts == ["甲: 苹果是水果"], f"旧模型的向量不该参与打分, 却拿到了: {texts}"
    # 分数也必须是"余弦"而不是"被别的行撑起来的"
    assert got[0][0] > 0.99
    store.close()


# ==================== 三处缝: scorer / filters / delete ====================
#
# 这三条都是"文档声称可换、实际换不了"的缺口。共同点是: 改动只有几十行,
# 但决定了应用层(文档系统 / 小镇)能不能自己选。

def test_scorer_replaces_the_ranking():
    """传进去的 scorer, 它的分数**就是**最终排序 —— 框架不掺一脚。

    默认余弦对"文档系统"是对的; "小镇"要的是 recency + importance + relevance。
    两者冲突, 所以框架不该替应用选。
    """
    store = VectorStore(":memory:")
    store.add(
        ["甲", "乙", "丙"],
        [[1, 0], [1, 0], [1, 0]],                    # 向量故意全一样: 排序只可能来自 scorer
        [{"rank": 3}, {"rank": 1}, {"rank": 2}],
        model="m",
    )

    def by_rank(query, vector, *, metadata, **kw):
        return float(metadata["rank"])

    got = [t for _, t, _ in store.search([1, 0], top_k=3, model="m", scorer=by_rank)]
    assert got == ["甲", "丙", "乙"], f"排序应该完全由 scorer 决定(rank 3/2/1), 实际 {got}"
    store.close()


def test_scorer_receives_created_at_and_metadata():
    """recency 要用 created_at, importance 走 metadata —— 两个额外信号都得送到。

    它们**不往外传**(返回值仍是三元组), 所以 scorer 是拿到它们的唯一途径。
    """
    seen = {}

    def spy_scorer(query, vector, *, created_at, metadata, query_norm, vector_norm, **kw):
        seen["created_at"] = created_at
        seen["metadata"] = metadata
        seen["norms"] = (query_norm, vector_norm)
        return 1.0

    store = VectorStore(":memory:")
    before = time.time()
    store.add(["x"], [[1, 0]], [{"importance": 7}], model="m")
    store.search([1, 0], model="m", scorer=spy_scorer)

    assert isinstance(seen["created_at"], float) and before <= seen["created_at"] <= time.time()
    assert seen["metadata"] == {"importance": 7}
    assert seen["norms"][0] is not None and seen["norms"][1] is not None
    store.close()


def test_default_scorer_tolerates_the_extra_signals():
    """默认 scorer 和自定义 scorer 走的是同一处调用 —— 多传的信号它要能丢掉。

    不这么做的话就得写 `if scorer is None` 两条分支, 两条路的调用姿势还不一样。
    """
    store = VectorStore(":memory:")
    store.add(["x"], [[1, 0]], model="m")
    got = store.search([1, 0], model="m", scorer=cosine_similarity)   # 显式传默认的
    assert len(got) == 1 and got[0][0] > 0.99
    store.close()


def test_filters_apply_before_top_k_truncation():
    """**过滤必须发生在 SQL 层。**

    在 Python 里筛是**错的**: top_k 是在全库上先截断的, 筛完之后可能一条不剩,
    而真正匹配的还躺在第 6 到第 50 名。这条测试构造的正是那个场景 ——
    top_k=1 时, 不过滤拿到的是别的来源, 过滤之后才拿到自己那一条。
    """
    store = VectorStore(":memory:")
    store.add(
        ["最像的(别的来源)", "次像的(本来源)", "不像的(本来源)"],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        [{"source": "other"}, {"source": "target"}, {"source": "target"}],
        model="m",
    )

    assert [t for _, t, _ in store.search([1.0, 0.0], top_k=1, model="m")] == ["最像的(别的来源)"]

    got = [t for _, t, _ in store.search([1.0, 0.0], top_k=1, model="m", filters={"source": "target"})]
    assert got == ["次像的(本来源)"], f"过滤器要在 SQL 层生效, 实际 {got}"
    store.close()


def test_filters_missing_key_matches_nothing():
    """库里没有这个键 -> 一条都不匹配(和 SQL 的 NULL = ? 语义对齐)"""
    store = VectorStore(":memory:")
    store.add(["x"], [[1, 0]], [{"source": "a"}], model="m")
    assert store.search([1, 0], model="m", filters={"nope": "a"}) == []
    assert store.search([1, 0], model="m", filters={"source": "b"}) == []
    store.close()


def test_filter_key_must_be_whitelisted():
    """键是**拼进 SQL 字符串**的, 不是绑定参数 —— 不能靠"看起来安全"。"""
    store = VectorStore(":memory:")
    for bad in ("a'; DROP TABLE vectors; --", "中文键", "a b", ""):
        try:
            store.search([1, 0], model="m", filters={bad: 1})
        except VectorStoreException as e:
            assert "字母数字下划线" in str(e)
            continue
        store.close()
        raise AssertionError(f"键 {bad!r} 应该被白名单拦下来")
    store.close()


def test_filters_still_work_without_json1():
    """json1 不可用时的退化路径: 取回后在内存里筛, **结果必须和 SQL 版一致**。

    本机的 sqlite 都有 json1, 所以这里手动把开关掰掉来验退化路径 ——
    它是文档明确要求存在的兜底, 不验就等于没有。
    """
    store = VectorStore(":memory:")
    store.add(
        ["甲", "乙"],
        [[1.0, 0.0], [0.0, 1.0]],
        [{"source": "a"}, {"source": "b"}],
        model="m",
    )
    store._has_json1 = False                      # 假装这台机器的 sqlite 没有 json1

    got = [t for _, t, _ in store.search([1.0, 0.0], top_k=5, model="m", filters={"source": "a"})]
    assert got == ["甲"], f"退化路径的结果该和 SQL 版一样, 实际 {got}"
    store.close()


def test_delete_by_ids():
    store = VectorStore(":memory:")
    ids = store.add(["甲", "乙", "丙"], [[1, 0], [0, 1], [1, 1]], model="m")

    assert store.delete(ids=[ids[1]]) == 1
    assert store.count() == 2
    assert [t for _, t, _ in store.search([0, 1], top_k=5, model="m")] == ["丙", "甲"]
    store.close()


def test_delete_by_filters():
    """**没有它就做不了"增量入库"** —— 只能全量重建, 连别的来源一起清掉。"""
    store = VectorStore(":memory:")
    store.add(
        ["甲-1", "甲-2", "乙-1"],
        [[1, 0], [0.9, 0.1], [0, 1]],
        [{"source": "甲.md"}, {"source": "甲.md"}, {"source": "乙.md"}],
        model="m",
    )

    assert store.delete(filters={"source": "甲.md"}) == 2
    assert store.count() == 1
    assert [t for _, t, _ in store.search([0, 1], top_k=5, model="m")] == ["乙-1"]   # 乙没被牵连
    store.close()


def test_delete_ids_and_filters_are_intersected():
    """两个都给 = 交集, 不是并集"""
    store = VectorStore(":memory:")
    ids = store.add(
        ["甲", "乙", "丙"],
        [[1, 0], [0, 1], [1, 1]],
        [{"source": "x"}, {"source": "x"}, {"source": "y"}],
        model="m",
    )
    assert store.delete(ids=[ids[0], ids[2]], filters={"source": "x"}) == 1   # 只有甲同时满足
    assert store.count() == 2
    store.close()


def test_delete_by_filters_still_works_without_json1():
    """json1 不可用时的退化路径: 先扫出 id 再删。**结果必须和 SQL 版一致。**"""
    store = VectorStore(":memory:")
    ids = store.add(
        ["甲", "乙", "丙"],
        [[1, 0], [0, 1], [1, 1]],
        [{"source": "x"}, {"source": "x"}, {"source": "y"}],
        model="m",
    )
    store._has_json1 = False                      # 假装这台机器的 sqlite 没有 json1

    # 和 ids 一起给的时候仍然是**交集**
    assert store.delete(ids=[ids[0], ids[2]], filters={"source": "x"}) == 1
    assert store.count() == 2
    assert store.delete(filters={"source": "x"}) == 1
    assert store.count() == 1
    store.close()


def test_delete_without_args_raises():
    """无参 delete 和 clear() 只差一个"手滑"的距离 —— 必须拒绝"""
    store = VectorStore(":memory:")
    store.add(["甲"], [[1, 0]], model="m")
    try:
        store.delete()
    except VectorStoreException as e:
        assert "clear()" in str(e), "报错要顺手告诉用户清库该调哪个方法"
        assert store.count() == 1, "拒绝之后一条都不许少"
        store.close()
        return
    store.close()
    raise AssertionError("无参 delete 必须拒绝, 否则它就是第二个 clear()")


def test_delete_empty_ids_is_noop():
    """空 ids 是"没有要删的", 不是"删全部" —— 这个区别值一条测试"""
    store = VectorStore(":memory:")
    store.add(["甲"], [[1, 0]], model="m")
    assert store.delete(ids=[]) == 0
    assert store.count() == 1
    store.close()


def test_delete_after_close_raises_clearly():
    store = VectorStore(":memory:")
    store.close()
    try:
        store.delete(ids=[1])
    except VectorStoreException as e:
        assert "已经关闭" in str(e)
        return
    raise AssertionError("关掉之后再用应该给一句人话, 而不是 NoneType 的 AttributeError")


# ==================== 两种存储格式 ====================

def test_both_storages_return_same_order():
    """BLOB 走 float32, JSON 走 float64 —— 分数会有极小的差异, 但排序必须一致。

    这个差异不是 bug, 是精度换速度。差异有多大, 跑 tests/bench_storage.py 看。
    """
    texts = ["a", "b", "c"]
    vectors = [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]
    query = [1.0, 0.0]

    results = {}
    for storage in (STORAGE_BLOB, STORAGE_JSON):
        store = VectorStore(":memory:", table=f"t_{storage}", storage=storage)
        store.add(texts, vectors, model="m")
        results[storage] = store.search(query, top_k=3, model="m")
        store.close()

    assert [t for _, t, _ in results[STORAGE_BLOB]] == [t for _, t, _ in results[STORAGE_JSON]]
    for (s1, *_), (s2, *_) in zip(results[STORAGE_BLOB], results[STORAGE_JSON]):
        assert abs(s1 - s2) < 1e-6, f"两种格式分数差太多: {s1} vs {s2}"


def test_storage_formats_can_coexist_in_one_table():
    """切换开关不会把老数据读坏 —— 每行都记了自己是怎么存的"""
    store = VectorStore(":memory:", storage=STORAGE_JSON)
    store.add(["json 进的"], [[1, 0]], model="m")
    store.storage = STORAGE_BLOB                      # 模拟"改了开关"
    store.add(["blob 进的"], [[0, 1]], model="m")

    assert store.count() == 2
    got = store.search([1, 0], top_k=2, model="m")
    assert [t for _, t, _ in got] == ["json 进的", "blob 进的"]
    store.close()


def test_bad_storage_name_raises():
    try:
        VectorStore(":memory:", storage="parquet")
    except VectorStoreException:
        return
    raise AssertionError("不认识的存储格式应该早早报错, 而不是跑起来才发现")


# ==================== 持久化 ====================

def test_persists_across_reopen():
    with TempDir() as d:
        path = str(d / "v.sqlite3")
        s1 = VectorStore(path)
        s1.add(["记住我"], [[1, 0]], model="m")
        s1.close()

        s2 = VectorStore(path)
        assert s2.count() == 1
        score, text, _ = s2.search([1, 0], top_k=1, model="m")[0]
        assert text == "记住我" and score > 0.99
        s2.close()


def test_clear_and_stats():
    store = VectorStore(":memory:")
    store.add(["a", "b"], [[1, 0], [0, 1]], model="m")
    info = store.stats()
    assert info["count"] == 2 and info["models"] == ["m"]
    assert store.clear() == 2 and store.count() == 0
    store.close()


# ==================== 并发安全(第三步点名的那条) ====================

def test_concurrent_search_is_safe():
    """
    【这条只有并发测得出】

    sqlite 连接默认 check_same_thread=True, 而 tools/async_executor.py 用
    ThreadPoolExecutor 跑工具 —— 一轮里模型完全可能同时发两个 knowledge_search。
    第二个线程一进来就 ProgrammingError。

    单线程测一百遍都测不出来, 上线才炸。所以这里必须专门测。
    """
    store = VectorStore(":memory:")
    n = 200
    store.add(
        [f"文本{i}" for i in range(n)],
        [[1.0, i / n] for i in range(n)],
        model="m",
    )

    errors = []

    def worker():
        try:
            for _ in range(15):
                store.search([1.0, 0.5], top_k=5, model="m")
        except Exception as e:                    # noqa: BLE001 — 就是要抓住任何异常
            errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    store.close()
    assert not errors, f"并发检索出错(这就是那个只有并发才暴露的坑): {errors}"


def test_concurrent_write_and_read_is_safe():
    """读写混在一起 —— 这才是工具池真实的负载形状"""
    store = VectorStore(":memory:")
    store.add(["种子"], [[1.0, 0.0]], model="m")

    errors = []

    def writer():
        try:
            for i in range(30):
                store.add([f"新{i}"], [[1.0, i / 30]], model="m")
        except Exception as e:
            errors.append(f"writer {type(e).__name__}: {e}")

    def reader():
        try:
            for _ in range(30):
                store.search([1.0, 0.0], top_k=3, model="m")
        except Exception as e:
            errors.append(f"reader {type(e).__name__}: {e}")

    threads = [threading.Thread(target=writer) for _ in range(2)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    count = store.count()
    store.close()
    assert not errors, f"并发读写出错: {errors}"
    assert count == 61, f"写入丢了: 期望 61 条, 实际 {count}"


def test_concurrent_across_two_connections():
    """两个连接同时读同一个文件 —— WAL 模式管的就是这件事"""
    with TempDir() as d:
        path = str(d / "concurrent.sqlite3")
        s1 = VectorStore(path)
        s2 = VectorStore(path)
        s1.add([f"t{i}" for i in range(50)], [[1.0, i / 50] for i in range(50)], model="m")

        errors = []

        def worker(store):
            try:
                for _ in range(10):
                    store.search([1.0, 0.3], top_k=3, model="m")
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")

        threads = [threading.Thread(target=worker, args=(s,)) for s in (s1, s2, s1, s2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        s1.close()
        s2.close()
        assert not errors, f"多连接并发出错: {errors}"


if __name__ == "__main__":
    print(f"(numpy 加速: {'开' if has_numpy() else '关 —— 纯 Python, 功能不受影响'})")
    sys.exit(run_tests(globals(), "向量库"))
