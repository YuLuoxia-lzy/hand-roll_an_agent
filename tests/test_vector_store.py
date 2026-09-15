"""向量库离线测试 —— **完全不需要 API key, 随时可跑**

    python tests/test_vector_store.py

指南第三步的验证在这儿: 构造已知向量, 手算余弦, 对比。
再加一条第三步特意点名的: **并发安全**(单线程测一百遍都测不出来)。
"""

import sys
import threading
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
    assert len(store.search([1, 0], model="m1")) == 2      # 维度相同, m1 在库里 -> 放行
    store.close()


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
