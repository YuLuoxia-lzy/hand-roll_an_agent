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


def test_fail_open_does_not_print_traceback():
    """fail-open 用 warning 而不是 exception: 这在生产里会反复发生,
    每次都打一整个堆栈会把日志刷爆"""
    mem = _memory()
    mem.embedder.fail = True
    mem.build_context("随便问")
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


def test_close_is_idempotent():
    mem = _memory()
    mem.close()
    mem.close()


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "语义记忆"))
